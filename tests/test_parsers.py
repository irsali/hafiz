"""Tests for hafiz.core.parsers — Protocol contract, concrete parsers,
registry behavior, and entry-point discovery.

The Protocol tests target behavior (units, edges, kinds, registration),
not implementation details. Concrete parsers are exercised with small
inline fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hafiz.core.parsers import (
    Parser,
    ParsedEdge,
    ParsedUnit,
    ParseResult,
    ParserRegistry,
    get_registry,
    reset_registry,
)
from hafiz.core.parsers.prose import ProseParser
from hafiz.core.parsers.python_ast import PythonAstParser
from hafiz.core.parsers.whole_file import WholeFileParser


# ---------------------------------------------------------------------------
# Protocol contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "parser",
    [PythonAstParser(), ProseParser(), WholeFileParser()],
)
def test_parser_satisfies_protocol(parser):
    assert isinstance(parser, Parser)
    assert isinstance(parser.name, str) and parser.name
    assert isinstance(parser.languages, list) and parser.languages
    assert callable(parser.parse)


@pytest.mark.parametrize(
    "parser,path,content",
    [
        (PythonAstParser(), Path("t.py"), "x = 1\n"),
        (ProseParser(), Path("t.md"), "# Hi\nHello.\n"),
        (WholeFileParser(), Path("t.xyz"), "anything\n"),
    ],
)
def test_parse_returns_parse_result(parser, path, content):
    result = parser.parse(path, content)
    assert isinstance(result, ParseResult)
    assert all(isinstance(u, ParsedUnit) for u in result.units)
    assert all(isinstance(e, ParsedEdge) for e in result.edges)


# ---------------------------------------------------------------------------
# PythonAstParser
# ---------------------------------------------------------------------------

def test_python_module_unit_always_emitted():
    result = PythonAstParser().parse(Path("mod.py"), "x = 1\n")
    kinds = [u.kind for u in result.units]
    assert "code.module" in kinds
    module = next(u for u in result.units if u.kind == "code.module")
    assert module.name == "mod"


def test_python_function_class_method_kinds():
    code = (
        "def top_level():\n"
        "    pass\n"
        "\n"
        "class Foo:\n"
        "    def bar(self):\n"
        "        return 1\n"
    )
    result = PythonAstParser().parse(Path("a.py"), code)
    kinds_by_name = {u.name: u.kind for u in result.units}
    assert kinds_by_name["top_level"] == "code.function"
    assert kinds_by_name["Foo"] == "code.class"
    assert kinds_by_name["Foo.bar"] == "code.method"


def test_python_inherits_edge():
    code = "class Base:\n    pass\n\nclass Child(Base):\n    pass\n"
    result = PythonAstParser().parse(Path("a.py"), code)
    inherits = [e for e in result.edges if e.relation == "inherits"]
    assert any(
        e.source_name == "Child" and e.target_name == "Base"
        for e in inherits
    )


def test_python_imports_edges():
    code = "import os\nfrom pathlib import Path\n"
    result = PythonAstParser().parse(Path("mod.py"), code)
    imports = [e for e in result.edges if e.relation == "imports"]
    targets = {e.target_name for e in imports}
    assert "os" in targets
    assert "pathlib.Path" in targets
    # All imports attribute to the module as source
    assert all(e.source_name == "mod" for e in imports)


def test_python_calls_attribute_to_nearest_caller():
    code = (
        "def helper():\n"
        "    pass\n"
        "\n"
        "def user():\n"
        "    helper()\n"
        "\n"
        "helper()\n"  # module-level call
    )
    result = PythonAstParser().parse(Path("mod.py"), code)
    calls = [e for e in result.edges if e.relation == "calls"]
    # One call from user -> helper, one from module -> helper.
    sources = {e.source_name for e in calls if e.target_name == "helper"}
    assert "user" in sources
    assert "mod" in sources


def test_python_syntax_error_degrades_to_module_unit():
    result = PythonAstParser().parse(Path("bad.py"), "def missing_colon()\n    pass\n")
    assert len(result.units) == 1
    assert result.units[0].kind == "code.module"
    assert result.edges == []


def test_python_async_function():
    code = "async def fetch():\n    return 1\n"
    result = PythonAstParser().parse(Path("a.py"), code)
    assert any(
        u.kind == "code.function" and u.name == "fetch" for u in result.units
    )


def test_python_nested_class():
    code = "class Outer:\n    class Inner:\n        def deep(self):\n            pass\n"
    result = PythonAstParser().parse(Path("a.py"), code)
    names = {u.name for u in result.units}
    assert "Outer" in names
    assert "Outer.Inner" in names
    assert "Outer.Inner.deep" in names


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
    md = (
        "# Intro\n"
        "\n"
        "First para.\n"
        "still first para.\n"
        "\n"
        "Second para.\n"
    )
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
    r.register(PythonAstParser())
    r.register(ProseParser())
    r.register(WholeFileParser())

    assert r.for_path(Path("a.py")).name == "python_ast"
    assert r.for_path(Path("a.md")).name == "prose"
    assert r.for_path(Path("a.MD")).name == "prose"  # case-insensitive
    # Unknown extension falls through to the wildcard fallback
    assert r.for_path(Path("a.xyz")).name == "whole_file"


def test_registry_last_registered_wins():
    """Third-party parsers installed via entry points can override
    in-tree parsers for the same extension."""
    r = ParserRegistry()
    r.register(PythonAstParser())

    class OverridePy:
        name = "override_py"
        languages = [".py"]

        def parse(self, path, content):
            return ParseResult()

    r.register(OverridePy())
    assert r.for_path(Path("a.py")).name == "override_py"


def test_registry_no_parser_without_fallback_raises():
    r = ParserRegistry()
    r.register(PythonAstParser())
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
        assert "python_ast" in names
        assert "prose" in names
        assert "whole_file" in names
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
        assert "python_ast" in names
    finally:
        reset_registry()
