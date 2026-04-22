"""File walking (gitignore-aware) and embedding-part preparation.

After the structural-grounding reshape, this module is narrow on purpose.
Parsing is the Parser Protocol's job (see :mod:`hafiz.core.parsers`);
storage is the store's; embedding is the embedding service's. This module
handles the two things that sit *between* those layers:

  1. ``walk_files(root, …)`` — yield absolute Paths from a directory,
     respecting `.gitignore` and `.hafizignore` at every level plus
     config-level ignore patterns. Binary-looking files are skipped so
     the embedder isn't fed garbage.

  2. ``prepare_embedding_parts(content, …)`` — split a unit's body into
     one or more ``EmbeddingPart`` records. Small units get one part;
     oversized units (long docs, huge functions, whole-file fallback on
     large files) get many. Each part is content-hashed independently so
     a partial edit re-embeds only the affected parts. Feeds the
     ``embeddings`` table (see :mod:`hafiz.core.database.Embedding`).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def compute_hash(content: str) -> str:
    """SHA-256 hex digest of content. Used for unit content_hash and
    embedding part content_hash."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Embedding-part preparation
# ---------------------------------------------------------------------------

# nomic-embed-text-v1.5 supports ~8k tokens; at ~3 chars per token that's
# ~24k chars, which we use as the soft max. Most real units fit in one
# part; only long docs and whole-file fallback on big files split.
DEFAULT_MAX_PART_CHARS = 24_000


@dataclass
class EmbeddingPart:
    """One embeddable slice of a unit's body.

    ``part_index`` is 0 for single-part units, 0..N for split units.
    ``token_span_start`` / ``token_span_end`` are character offsets into
    the source unit's content (not token indices — naming is historical
    and kept for schema compatibility with the ``embeddings`` table).
    """

    content: str
    part_index: int
    token_span_start: int | None = None
    token_span_end: int | None = None


def prepare_embedding_parts(
    content: str, *, max_chars: int = DEFAULT_MAX_PART_CHARS
) -> list[EmbeddingPart]:
    """Split a unit's content into embedding parts.

    Most units fit in a single part. For oversized content, split on
    newline boundaries when possible to keep semantic coherence within
    each part.
    """
    if len(content) <= max_chars:
        return [
            EmbeddingPart(
                content=content,
                part_index=0,
                token_span_start=0,
                token_span_end=len(content),
            )
        ]

    parts: list[EmbeddingPart] = []
    start = 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        if end < len(content):
            # Prefer to split at a newline, but only if we won't throw away
            # more than ~half the window (avoid tiny leftover parts).
            nl = content.rfind("\n", start + max_chars // 2, end)
            if nl > start:
                end = nl + 1
        parts.append(
            EmbeddingPart(
                content=content[start:end],
                part_index=len(parts),
                token_span_start=start,
                token_span_end=end,
            )
        )
        start = end
    return parts


# ---------------------------------------------------------------------------
# File walking
# ---------------------------------------------------------------------------

# If the first chunk of a file contains a null byte, treat it as binary and
# skip. Images, compiled artifacts, archives all get caught by this.
_BINARY_PROBE_BYTES = 8192


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            blob = f.read(_BINARY_PROBE_BYTES)
    except OSError:
        return True
    return b"\x00" in blob


def _load_ignore_patterns(directory: Path) -> list[str]:
    """Load gitignore-style patterns from .gitignore and .hafizignore in a
    directory."""
    patterns: list[str] = []
    for name in (".gitignore", ".hafizignore"):
        path = directory / name
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    patterns.append(stripped)
    return patterns


def _normalize_pattern(pattern: str, rel_dir: str) -> list[str]:
    """Convert a subdirectory ignore pattern to root-relative patterns.

    In gitignore, a pattern like ``*.py`` in ``subdir/.gitignore`` matches
    any ``.py`` file at any depth under ``subdir/``. We normalize to
    root-relative patterns so all rules live in one PathSpec where later
    (deeper) rules correctly override earlier (shallower) ones.
    """
    negation = pattern.startswith("!")
    p = pattern[1:] if negation else pattern
    prefix = "!" if negation else ""

    if p.startswith("/"):
        return [f"{prefix}{rel_dir}{p}"]
    if "/" in p:
        return [f"{prefix}{rel_dir}/{p}"]
    return [f"{prefix}{rel_dir}/**/{p}"]


def walk_files(
    root: Path,
    *,
    ignore_patterns: list[str] | None = None,
    skip_binary: bool = True,
) -> Iterator[Path]:
    """Walk a directory tree and yield absolute Paths to text-ish files,
    respecting `.gitignore` / `.hafizignore` at every level plus config
    ignore patterns.

    If ``root`` is a single file, yield just that file (still binary-
    checked). Subdirectory ignore files can override parent rules —
    matching real git semantics (pathspec last-match-wins).
    """
    import pathspec

    if root.is_file():
        if not skip_binary or not _looks_binary(root):
            yield root
        return

    config_ignore = ignore_patterns or [
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
    ]

    all_patterns: list[str] = list(config_ignore) + _load_ignore_patterns(root)
    spec = pathspec.PathSpec.from_lines("gitwildmatch", all_patterns)

    for dirpath_str, dirnames, filenames in os.walk(root, topdown=True):
        dirpath = Path(dirpath_str)
        rel_dir = dirpath.relative_to(root)

        if dirpath != root:
            local_patterns = _load_ignore_patterns(dirpath)
            if local_patterns:
                rel_str = str(rel_dir)
                for p in local_patterns:
                    all_patterns.extend(_normalize_pattern(p, rel_str))
                spec = pathspec.PathSpec.from_lines(
                    "gitwildmatch", all_patterns
                )

        dirnames[:] = [
            d
            for d in sorted(dirnames)
            if not spec.match_file(
                str(rel_dir / d) if str(rel_dir) != "." else d
            )
        ]

        for filename in sorted(filenames):
            rel_path = (
                str(rel_dir / filename) if str(rel_dir) != "." else filename
            )
            if spec.match_file(rel_path):
                continue

            filepath = dirpath / filename
            if skip_binary and _looks_binary(filepath):
                continue
            yield filepath
