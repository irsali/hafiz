"""Markdown / prose parser — heading tree + paragraph children.

Emits:
    doc.heading    one per Markdown heading (#, ##, ### …). Nested by level;
                   each heading's `content` covers the full section it owns.
    doc.paragraph  one per blank-line-separated text block inside a heading
                   section (parent_name = heading qualified name).
    doc.body       fallback when a file has no headings — one unit for the
                   whole file.

Qualified heading names use ` > ` as a separator to distinguish from the
`.` used by code parsers: `Intro > Section A > Subsection`. This keeps
downstream name-resolution unambiguous about what namespace it's in.
"""

from __future__ import annotations

import re
from pathlib import Path

from hafiz.core.parsers import ParsedUnit, ParseResult

# Match ATX headings (# through ######) only. Setext headings (underlines)
# are rare in agent-authored docs; we can add them if they show up.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class ProseParser:
    name = "prose"
    languages = [".md", ".markdown", ".txt", ".rst"]

    def parse(self, path: Path, content: str) -> ParseResult:
        lines = content.splitlines(keepends=True)
        headings = _find_headings(lines)

        if not headings:
            return ParseResult(
                units=[
                    ParsedUnit(
                        kind="doc.body",
                        name=path.stem,
                        line_start=1,
                        line_end=max(len(lines), 1),
                        content=content,
                        language="markdown",
                    )
                ],
                language="markdown",
            )

        units: list[ParsedUnit] = []
        stack: list[tuple[int, str]] = []  # (level, text)

        for i, (lineno, level, text) in enumerate(headings):
            while stack and stack[-1][0] >= level:
                stack.pop()

            parent_name = " > ".join(name for _, name in stack) if stack else None
            qualified = f"{parent_name} > {text}" if parent_name else text

            # Section extent: up to the next heading at same-or-higher level,
            # else EOF.
            end_line = len(lines)
            for j in range(i + 1, len(headings)):
                if headings[j][1] <= level:
                    end_line = headings[j][0] - 1
                    break

            section_content = "".join(lines[lineno - 1 : end_line])
            units.append(
                ParsedUnit(
                    kind="doc.heading",
                    name=qualified,
                    parent_name=parent_name,
                    line_start=lineno,
                    line_end=end_line,
                    content=section_content,
                    language="markdown",
                )
            )

            # Paragraph children: blank-line-separated blocks inside this
            # heading's section, excluding the heading line itself and
            # nested headings (they'll be their own units).
            for para_start, para_end, para_text in _extract_paragraphs(lines, lineno + 1, end_line):
                units.append(
                    ParsedUnit(
                        kind="doc.paragraph",
                        name=f"{qualified} #p{para_start}",
                        parent_name=qualified,
                        line_start=para_start,
                        line_end=para_end,
                        content=para_text,
                        language="markdown",
                    )
                )

            stack.append((level, text))

        return ParseResult(units=units, language="markdown")


def _find_headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return (lineno, level, text) for every ATX heading. 1-indexed."""
    out: list[tuple[int, int, str]] = []
    in_code_block = False
    for idx, line in enumerate(lines, start=1):
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        m = _HEADING_RE.match(line)
        if m:
            out.append((idx, len(m.group(1)), m.group(2).strip()))
    return out


def _extract_paragraphs(lines: list[str], start: int, end: int) -> list[tuple[int, int, str]]:
    """Split the range (1-indexed, inclusive) into paragraph blocks.
    Skips sub-heading lines so they don't leak into parent paragraphs."""
    paragraphs: list[tuple[int, int, str]] = []
    current: list[str] = []
    current_start = start

    for offset in range(start, end + 1):
        line = lines[offset - 1] if offset - 1 < len(lines) else ""

        is_heading = bool(_HEADING_RE.match(line))
        is_blank = line.strip() == ""

        if is_heading or is_blank:
            if current:
                paragraphs.append((current_start, offset - 1, "".join(current)))
                current = []
            current_start = offset + 1
        else:
            if not current:
                current_start = offset
            current.append(line)

    if current:
        paragraphs.append((current_start, end, "".join(current)))

    return paragraphs
