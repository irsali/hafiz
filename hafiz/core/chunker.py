"""File chunking using LlamaIndex SentenceSplitter and CodeSplitter.

Auto-detects language by file extension and selects the appropriate splitter.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.node_parser import CodeSplitter, SentenceSplitter

# File extension -> (language name, chunk_type)
# CodeSplitter requires tree-sitter language names
LANGUAGE_MAP: dict[str, tuple[str, str]] = {
    ".py": ("python", "code"),
    ".js": ("javascript", "code"),
    ".jsx": ("javascript", "code"),
    ".ts": ("typescript", "code"),
    ".tsx": ("typescript", "code"),
    ".go": ("go", "code"),
    ".rs": ("rust", "code"),
    ".java": ("java", "code"),
    ".rb": ("ruby", "code"),
    ".php": ("php", "code"),
    ".c": ("c", "code"),
    ".cpp": ("cpp", "code"),
    ".h": ("c", "code"),
    ".hpp": ("cpp", "code"),
    ".cs": ("c_sharp", "code"),
    ".swift": ("swift", "code"),
    ".kt": ("kotlin", "code"),
    ".scala": ("scala", "code"),
    ".sh": ("bash", "code"),
    ".bash": ("bash", "code"),
    ".sql": ("sql", "code"),
    ".html": ("html", "code"),
    ".css": ("css", "code"),
    ".scss": ("scss", "code"),
    ".yaml": ("yaml", "doc"),
    ".yml": ("yaml", "doc"),
    ".toml": ("toml", "doc"),
    ".json": ("json", "doc"),
    ".md": ("markdown", "doc"),
    ".rst": ("rst", "doc"),
    ".txt": ("text", "doc"),
}

# Languages that CodeSplitter supports via tree-sitter
CODE_SPLITTER_LANGUAGES = {
    "python", "javascript", "typescript", "go", "rust", "java", "ruby",
    "php", "c", "cpp", "c_sharp", "swift", "kotlin", "scala", "bash",
    "html", "css",
}


@dataclass
class ChunkResult:
    """A single chunk extracted from a file."""

    content: str
    source_file: str
    line_start: int | None = None
    line_end: int | None = None
    chunk_type: str = "code"
    language: str | None = None
    checksum: str = ""
    metadata: dict = field(default_factory=dict)


def compute_checksum(content: str) -> str:
    """SHA-256 checksum of content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def detect_language(file_path: Path) -> tuple[str | None, str]:
    """Detect language and chunk type from file extension.

    Returns (language, chunk_type).
    """
    ext = file_path.suffix.lower()
    if ext in LANGUAGE_MAP:
        return LANGUAGE_MAP[ext]
    return None, "doc"


def chunk_file(
    file_path: Path,
    *,
    relative_to: Path | None = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> list[ChunkResult]:
    """Chunk a single file into pieces.

    Uses CodeSplitter for recognized code files, SentenceSplitter for everything else.
    """
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    if not text.strip():
        return []

    language, chunk_type = detect_language(file_path)
    source = str(file_path.relative_to(relative_to)) if relative_to else str(file_path)

    # Select the appropriate splitter
    if language and language in CODE_SPLITTER_LANGUAGES:
        try:
            splitter = CodeSplitter(
                language=language,
                max_chars=chunk_size,
                chunk_lines=40,
                chunk_lines_overlap=5,
            )
        except Exception:
            # Fall back to sentence splitter if tree-sitter language isn't available
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
    else:
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # Split the text
    try:
        chunks = splitter.split_text(text)
    except Exception:
        # Last resort: split by character count
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]

    results = []
    current_pos = 0
    lines = text.split("\n")

    for chunk_text in chunks:
        if not chunk_text.strip():
            continue

        # Estimate line numbers
        chunk_start_pos = text.find(chunk_text, current_pos)
        if chunk_start_pos >= 0:
            line_start = text[:chunk_start_pos].count("\n") + 1
            line_end = line_start + chunk_text.count("\n")
            current_pos = chunk_start_pos + 1
        else:
            line_start = None
            line_end = None

        results.append(
            ChunkResult(
                content=chunk_text,
                source_file=source,
                line_start=line_start,
                line_end=line_end,
                chunk_type=chunk_type,
                language=language,
                checksum=compute_checksum(chunk_text),
            )
        )

    return results


def _load_ignore_patterns(directory: Path) -> list[str]:
    """Load gitignore-style patterns from .gitignore and .hafizignore in a directory."""
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
    any ``.py`` file at any depth under ``subdir/``.  We normalize this to
    root-relative patterns so all rules can live in a single PathSpec where
    later (deeper) rules correctly override earlier (shallower) ones.
    """
    negation = pattern.startswith("!")
    p = pattern[1:] if negation else pattern
    prefix = "!" if negation else ""

    if p.startswith("/"):
        # Anchored to the ignore-file's directory
        return [f"{prefix}{rel_dir}{p}"]

    if "/" in p:
        # Contains a path separator — treat as relative to the directory
        return [f"{prefix}{rel_dir}/{p}"]

    # No slash — matches at any depth under the directory
    # subdir/**/*.py matches subdir/foo.py AND subdir/a/b/foo.py
    return [f"{prefix}{rel_dir}/**/{p}"]


def walk_and_chunk(
    root: Path,
    *,
    ignore_patterns: list[str] | None = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> list[ChunkResult]:
    """Walk a directory and chunk all recognized files.

    Respects ``.gitignore`` and ``.hafizignore`` at every directory level
    (including negation patterns like ``!important.log``).  Subdirectory
    ignore files override parent rules — matching real git semantics.
    Config-level ``workspace.ignore`` patterns are merged in.

    Returns a flat list of ChunkResults.
    """
    import os

    import pathspec

    if root.is_file():
        return chunk_file(
            root, relative_to=root.parent, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    config_ignore = ignore_patterns or [".git", "node_modules", "__pycache__", ".venv"]

    # All patterns collected root-relative.  Ordered shallowest → deepest so
    # that deeper rules override shallower ones (pathspec last-match-wins).
    all_patterns: list[str] = list(config_ignore) + _load_ignore_patterns(root)
    spec = pathspec.PathSpec.from_lines("gitwildmatch", all_patterns)

    results: list[ChunkResult] = []

    for dirpath_str, dirnames, filenames in os.walk(root, topdown=True):
        dirpath = Path(dirpath_str)
        rel_dir = dirpath.relative_to(root)

        # Pick up subdirectory ignore patterns and rebuild the spec
        if dirpath != root:
            local_patterns = _load_ignore_patterns(dirpath)
            if local_patterns:
                rel_str = str(rel_dir)
                for p in local_patterns:
                    all_patterns.extend(_normalize_pattern(p, rel_str))
                spec = pathspec.PathSpec.from_lines("gitwildmatch", all_patterns)

        # Prune ignored directories (modifying dirnames in-place skips them)
        dirnames[:] = [
            d
            for d in sorted(dirnames)
            if not spec.match_file(str(rel_dir / d) if str(rel_dir) != "." else d)
        ]

        # Process files
        for filename in sorted(filenames):
            rel_path = str(rel_dir / filename) if str(rel_dir) != "." else filename

            if spec.match_file(rel_path):
                continue

            # Only chunk recognized file types
            filepath = dirpath / filename
            if (
                filepath.suffix.lower() not in LANGUAGE_MAP
                and filepath.suffix.lower() not in {".txt", ".cfg", ".ini", ".env.example"}
            ):
                continue

            chunks = chunk_file(
                filepath,
                relative_to=root,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            results.extend(chunks)

    return results
