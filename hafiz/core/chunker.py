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


def should_ignore(path: Path, ignore_patterns: list[str]) -> bool:
    """Check if a path matches any ignore pattern."""
    path_str = str(path)
    for pattern in ignore_patterns:
        if pattern.startswith("*."):
            if path.suffix == pattern[1:]:
                return True
        elif pattern in path.parts:
            return True
    return True if path.name.startswith(".") and path.is_dir() else False


def walk_and_chunk(
    root: Path,
    *,
    ignore_patterns: list[str] | None = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
) -> list[ChunkResult]:
    """Walk a directory and chunk all recognized files.

    Returns a flat list of ChunkResults.
    """
    ignore = ignore_patterns or [".git", "node_modules", "__pycache__", ".venv"]
    results: list[ChunkResult] = []

    if root.is_file():
        return chunk_file(root, relative_to=root.parent, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if should_ignore(path, ignore):
            continue
        # Skip binary files by checking extension
        if path.suffix.lower() in LANGUAGE_MAP or path.suffix.lower() in {".txt", ".cfg", ".ini", ".env.example"}:
            chunks = chunk_file(
                path,
                relative_to=root,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            results.extend(chunks)

    return results
