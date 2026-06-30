"""Parser Protocol, data types, and registry for the structural layer.

Parsers turn a file's content into (units, edges). They are the deterministic
half of Hafiz's knowledge extraction: AST / prose structure / generic
whole-file fallback. Agents own the semantic half (annotations, concepts,
patterns) and plug in via the separate `extract import` path.

Parsers are **capabilities, not toggles** — there is no enable/disable
config. A file's language is handled by whichever parser is registered for
its extension; unregistered languages fall through to `WholeFileParser`.

In-tree parsers register via module import. Third-party parsers plug in via
the `hafiz.parsers` Python entry-point group — `pip install hafiz-parser-go`
is how you turn on Go AST for that language; no config edit required.

See workitems/active/structural-grounding.md for the design.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger("hafiz.parsers")


# ---------------------------------------------------------------------------
# Data types — what a parser produces
# ---------------------------------------------------------------------------


@dataclass
class ParsedUnit:
    """One addressable unit inside a file.

    `kind` is namespaced `domain.subtype` by convention (`code.function`,
    `doc.heading`, `mail.message`, `file.raw`, …). `name` is qualified
    within the file (e.g. `Foo.bar` for a method). Line numbers are
    1-indexed and inclusive. `content` is the raw body the ingest layer
    will hash and embed.
    """

    kind: str
    name: str
    parent_name: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    content: str = ""
    language: str | None = None


@dataclass
class ParsedEdge:
    """One relation between units.

    `source_name` and `target_name` are parser-local identifiers (the
    qualified names the parser emits for units). The edge resolver in
    Phase 4 binds these to `unit_id`s; external targets (stdlib,
    third-party imports) stay unresolved with `target_name` populated.
    """

    source_name: str
    target_name: str
    relation: str
    evidence: str | None = None
    line: int | None = None


@dataclass
class ParseResult:
    """A parser's output for one file."""

    units: list[ParsedUnit] = field(default_factory=list)
    edges: list[ParsedEdge] = field(default_factory=list)
    language: str | None = None


# ---------------------------------------------------------------------------
# Protocol — the contract every parser satisfies
# ---------------------------------------------------------------------------


@runtime_checkable
class Parser(Protocol):
    """Protocol for file parsers. Implementations need not inherit — just
    expose `name`, `languages`, and `parse(path, content) -> ParseResult`.

    Attributes:
        name: Stable identifier, e.g. `"python_ast"`, `"prose"`.
        languages: File extensions claimed (each with leading dot,
            lowercase, e.g. `[".py"]`), or the sentinel `["*"]` to
            register as the universal fallback.
    Optional, not part of the structural Protocol check:
        source_tag: The `unit_revisions.source` / `edges.source` value the
            store should record for this parser's output. A structural
            parser that emits `code.*` edges declares
            `source_tag = "ast"` as a plain class attribute (the AST layer
            owns structure). Parsers that emit no edges (prose, whole-file)
            omit it. The store reads it via `getattr` and falls back to a
            name heuristic when absent, so it stays an opt-in convention —
            it is deliberately NOT listed below, to keep `isinstance(p,
            Parser)` from requiring it.
    """

    name: str
    languages: list[str]

    def parse(self, path: Path, content: str) -> ParseResult: ...


# ---------------------------------------------------------------------------
# Registry — extension -> Parser, with fallback
# ---------------------------------------------------------------------------


class ParserRegistry:
    """Maps file extensions to parsers. First-class lookup for ingest;
    introspection surface for `hafiz parsers list`.

    Registration semantics: last-registered wins for a given extension.
    This lets third-party parsers (entry-point-discovered) override
    in-tree parsers if the user installs one. The fallback slot
    (`languages=["*"]`) is replaced on re-registration too.
    """

    def __init__(self) -> None:
        self._by_ext: dict[str, Parser] = {}
        self._fallback: Parser | None = None

    def register(self, parser: Parser) -> None:
        for lang in parser.languages:
            if lang == "*":
                self._fallback = parser
                continue
            ext = lang if lang.startswith(".") else f".{lang}"
            self._by_ext[ext.lower()] = parser

    def for_path(self, path: Path) -> Parser:
        ext = path.suffix.lower()
        if ext in self._by_ext:
            return self._by_ext[ext]
        if self._fallback is not None:
            return self._fallback
        raise LookupError(
            f"No parser registered for extension {ext!r} and no fallback available. "
            "This is a bug: WholeFileParser should always be registered as fallback."
        )

    def all_parsers(self) -> list[Parser]:
        """Deduplicated list of every registered parser. Used by
        `hafiz parsers list` (Phase 7)."""
        seen: list[Parser] = []
        for p in list(self._by_ext.values()) + (
            [self._fallback] if self._fallback is not None else []
        ):
            if p not in seen:
                seen.append(p)
        return seen

    def extensions_for(self, parser: Parser) -> list[str]:
        """Extensions this parser claims in this registry."""
        exts = [ext for ext, p in self._by_ext.items() if p is parser]
        if self._fallback is parser:
            exts.append("*")
        return exts


# ---------------------------------------------------------------------------
# Discovery — in-tree + entry-point
# ---------------------------------------------------------------------------


def _build_registry() -> ParserRegistry:
    """Assemble a fresh registry: in-tree parsers first, then entry-point
    discovered parsers on top (so third-party can override)."""
    registry = ParserRegistry()

    # In-tree parsers — imported here to avoid circular import at module load.
    from hafiz.core.parsers.prose import ProseParser
    from hafiz.core.parsers.python_ast import PythonAstParser
    from hafiz.core.parsers.whole_file import WholeFileParser

    registry.register(PythonAstParser())
    registry.register(ProseParser())

    # JS/TS is an optional capability: its tree-sitter deps live in the
    # `hafiz[js]` extra. Register only when the grammars imported cleanly,
    # so a base install never carries the dependency and .js/.ts files fall
    # through to WholeFileParser instead.
    from hafiz.core.parsers.tree_sitter_js import AVAILABLE as _JS_AVAILABLE
    from hafiz.core.parsers.tree_sitter_js import TreeSitterJsParser

    if _JS_AVAILABLE:
        registry.register(TreeSitterJsParser())

    registry.register(WholeFileParser())

    # Entry-point discovered parsers
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="hafiz.parsers")
        for ep in eps:
            try:
                parser_cls_or_obj = ep.load()
                parser = (
                    parser_cls_or_obj()
                    if isinstance(parser_cls_or_obj, type)
                    else parser_cls_or_obj
                )
                if not isinstance(parser, Parser):
                    logger.warning(
                        "Entry-point parser %r does not satisfy Parser Protocol; skipping.",
                        ep.name,
                    )
                    continue
                registry.register(parser)
                logger.debug("Registered entry-point parser %r", ep.name)
            except Exception as exc:
                logger.warning("Failed to load entry-point parser %r: %s", ep.name, exc)
    except Exception as exc:
        logger.debug("entry_points lookup failed: %s", exc)

    return registry


_registry: ParserRegistry | None = None


def get_registry() -> ParserRegistry:
    """Return the process-wide parser registry (lazily built)."""
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry


def reset_registry() -> None:
    """Drop the cached registry. For tests that monkey-patch entry points."""
    global _registry
    _registry = None


__all__ = [
    "ParsedUnit",
    "ParsedEdge",
    "ParseResult",
    "Parser",
    "ParserRegistry",
    "get_registry",
    "reset_registry",
]
