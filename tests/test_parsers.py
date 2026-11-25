"""Tests for hafiz.core.parsers — Protocol contract, concrete parsers,
registry behavior, and entry-point discovery.

The Protocol tests target behavior (units, kinds, registration),
not implementation details. Concrete parsers are exercised with small
inline fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hafiz.core.parsers import (
    ParsedUnit,
    Parser,
    ParseResult,
    ParserRegistry,
    get_registry,
    reset_registry,
)
from hafiz.core.parsers.prose import ProseParser
from hafiz.core.parsers.whole_file import WholeFileParser

# ---------------------------------------------------------------------------
# Protocol contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parser",
    [ProseParser(), WholeFileParser()],
)
def test_parser_satisfies_protocol(parser):
    assert isinstance(parser, Parser)
    assert isinstance(parser.name, str) and parser.name
    assert isinstance(parser.languages, list) and parser.languages
    assert callable(parser.parse)


@pytest.mark.parametrize(
    "parser,path,content",
    [
        (ProseParser(), Path("t.md"), "# Hi\nHello.\n"),
        (WholeFileParser(), Path("t.xyz"), "anything\n"),
    ],
)
def test_parse_returns_parse_result(parser, path, content):
    result = parser.parse(path, content)
    assert isinstance(result, ParseResult)
    assert all(isinstance(u, ParsedUnit) for u in result.units)


def test_store_source_tag_declaration_wins_over_name_heuristic():
    """`store._source_tag` prefers an explicit `source_tag` attribute, and
    falls back to the `'ast' in name` heuristic when it's absent."""
    from hafiz.core.store import _source_tag

    # Explicit declaration wins even without 'ast' in the name.
    class _Declared:
        name = "third_party"
        languages = [".go"]
        source_tag = "ast"

        def parse(self, path, content):
            return ParseResult()

    assert _source_tag(_Declared()) == "ast"

    # Fallback: no source_tag attribute, name decides.
    class _NamedAst:
        name = "go_ast"
        languages = [".go"]

        def parse(self, path, content):
            return ParseResult()

    class _NoEdges:
        name = "prose"
        languages = [".md"]

        def parse(self, path, content):
            return ParseResult()

    assert _source_tag(_NamedAst()) == "ast"
    assert _source_tag(_NoEdges()) == "parser"


# ---------------------------------------------------------------------------
# ProseParser
# ---------------------------------------------------------------------------


def test_prose_heading_tree():
    md = (
        "# Top\n"
        "Intro text.\n"
        "\n"
        "## Section A\n"
        "Content A.\n"
        "\n"
        "### Subsection A1\n"
        "Sub content.\n"
        "\n"
        "## Section B\n"
        "Content B.\n"
    )
    result = ProseParser().parse(Path("doc.md"), md)
    headings = [u for u in result.units if u.kind == "doc.heading"]
    names = [u.name for u in headings]
    assert "Top" in names
    assert "Top > Section A" in names
    assert "Top > Section A > Subsection A1" in names
    assert "Top > Section B" in names


def test_prose_no_headings_gives_doc_body():
    result = ProseParser().parse(Path("note.md"), "Just plain text.\nNo headings.\n")
    assert len(result.units) == 1
    assert result.units[0].kind == "doc.body"


def test_prose_emits_paragraph_children():
    md = "# Intro\n\nFirst para.\nstill first para.\n\nSecond para.\n"
    result = ProseParser().parse(Path("n.md"), md)
    paragraphs = [u for u in result.units if u.kind == "doc.paragraph"]
    assert len(paragraphs) == 2
    assert all(p.parent_name == "Intro" for p in paragraphs)


def test_prose_ignores_headings_in_code_blocks():
    md = "# Real\n\n```python\n# Not a heading\n```\n"
    result = ProseParser().parse(Path("a.md"), md)
    headings = [u.name for u in result.units if u.kind == "doc.heading"]
    assert headings == ["Real"]


# ---------------------------------------------------------------------------
# WholeFileParser
# ---------------------------------------------------------------------------


def test_whole_file_emits_one_unit():
    result = WholeFileParser().parse(Path("mystery.bin"), "opaque content\n")
    assert len(result.units) == 1
    u = result.units[0]
    assert u.kind == "file.raw"
    assert u.name == "mystery"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_routes_by_extension():
    r = ParserRegistry()
    r.register(ProseParser())
    r.register(WholeFileParser())

    assert r.for_path(Path("a.md")).name == "prose"
    assert r.for_path(Path("a.MD")).name == "prose"  # case-insensitive
    # Unknown extensions — code included — fall through to the fallback
    assert r.for_path(Path("a.xyz")).name == "whole_file"
    assert r.for_path(Path("a.py")).name == "whole_file"


def test_registry_last_registered_wins():
    """Third-party parsers installed via entry points can override
    in-tree parsers for the same extension."""
    r = ParserRegistry()
    r.register(ProseParser())

    class OverrideMd:
        name = "override_md"
        languages = [".md"]

        def parse(self, path, content):
            return ParseResult()

    r.register(OverrideMd())
    assert r.for_path(Path("a.md")).name == "override_md"


def test_registry_no_parser_without_fallback_raises():
    r = ParserRegistry()
    r.register(ProseParser())
    with pytest.raises(LookupError):
        r.for_path(Path("a.xyz"))


def test_get_registry_is_cached():
    reset_registry()
    try:
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
    finally:
        reset_registry()


def test_default_registry_covers_shipped_parsers():
    reset_registry()
    try:
        r = get_registry()
        names = {p.name for p in r.all_parsers()}
        assert {"prose", "whole_file"} <= names
    finally:
        reset_registry()


# ---------------------------------------------------------------------------
# Entry-point discovery
# ---------------------------------------------------------------------------


def test_entry_point_parser_is_registered(monkeypatch):
    """A fake `hafiz.parsers` entry point should be picked up by
    _build_registry and its parser registered by extension."""

    class FakeLangParser:
        name = "fakelang"
        languages = [".fake"]

        def parse(self, path, content):
            return ParseResult(
                units=[
                    ParsedUnit(
                        kind="fakelang.thing",
                        name=path.stem,
                        content=content,
                    )
                ]
            )

    class _FakeEntryPoint:
        name = "fakelang"

        def load(self):
            return FakeLangParser

    from importlib import metadata as im

    def fake_entry_points(*, group=None, **_kwargs):
        if group == "hafiz.parsers":
            return [_FakeEntryPoint()]
        return []

    monkeypatch.setattr(im, "entry_points", fake_entry_points)
    reset_registry()
    try:
        r = get_registry()
        assert r.for_path(Path("x.fake")).name == "fakelang"
    finally:
        reset_registry()


def test_entry_point_parser_that_fails_to_load_is_skipped(monkeypatch, caplog):
    """A broken entry point must not take down the whole registry."""

    class _BadEntryPoint:
        name = "broken"

        def load(self):
            raise RuntimeError("intentional")

    from importlib import metadata as im

    def fake_entry_points(*, group=None, **_kwargs):
        if group == "hafiz.parsers":
            return [_BadEntryPoint()]
        return []

    monkeypatch.setattr(im, "entry_points", fake_entry_points)
    reset_registry()
    try:
        r = get_registry()
        # In-tree parsers still present.
        names = {p.name for p in r.all_parsers()}
        assert "prose" in names
    finally:
        reset_registry()
