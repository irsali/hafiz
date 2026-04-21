"""Whole-file fallback parser.

Registered with `languages = ["*"]` — the registry uses it for any file
whose extension isn't claimed by a specialized parser. Emits a single
`file.raw` unit containing the entire file contents. No edges.

This is the universal safety net: every file gets represented, even if
we don't know how to parse it. Ingest still gets a unit + content to
embed; agents can still attach annotations.
"""

from __future__ import annotations

from pathlib import Path

from hafiz.core.parsers import ParsedUnit, ParseResult


class WholeFileParser:
    name = "whole_file"
    languages = ["*"]

    def parse(self, path: Path, content: str) -> ParseResult:
        return ParseResult(
            units=[
                ParsedUnit(
                    kind="file.raw",
                    name=path.stem or path.name,
                    line_start=1,
                    line_end=max(content.count("\n") + 1, 1),
                    content=content,
                    language=None,
                )
            ],
        )
