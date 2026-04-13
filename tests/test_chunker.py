"""Tests for hafiz.core.chunker."""

from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

from hafiz.core.chunker import (
    ChunkResult,
    chunk_file,
    compute_checksum,
    detect_language,
    should_ignore,
)


def test_detect_language_python():
    lang, ctype = detect_language(Path("main.py"))
    assert lang == "python"
    assert ctype == "code"


def test_detect_language_markdown():
    lang, ctype = detect_language(Path("README.md"))
    assert lang == "markdown"
    assert ctype == "doc"


def test_detect_language_typescript():
    lang, ctype = detect_language(Path("app.tsx"))
    assert lang == "typescript"
    assert ctype == "code"


def test_detect_language_unknown():
    lang, ctype = detect_language(Path("data.xyz"))
    assert lang is None
    assert ctype == "doc"


def test_compute_checksum():
    cs1 = compute_checksum("hello world")
    cs2 = compute_checksum("hello world")
    cs3 = compute_checksum("different text")
    assert cs1 == cs2
    assert cs1 != cs3
    assert len(cs1) == 16


def test_should_ignore_git():
    assert should_ignore(Path(".git"), [".git"])


def test_should_ignore_node_modules():
    assert should_ignore(Path("project/node_modules/pkg"), ["node_modules"])


def test_chunk_file_creates_results(tmp_path):
    """Chunking a simple Python file should return ChunkResult objects."""
    py_file = tmp_path / "example.py"
    py_file.write_text(dedent("""\
        def hello():
            print("Hello, world!")

        def goodbye():
            print("Goodbye, world!")
    """))

    results = chunk_file(py_file, relative_to=tmp_path)
    assert len(results) > 0
    assert all(isinstance(r, ChunkResult) for r in results)
    assert all(r.language == "python" for r in results)
    assert all(r.chunk_type == "code" for r in results)


def test_chunk_file_empty(tmp_path):
    """Empty files should return no chunks."""
    empty = tmp_path / "empty.py"
    empty.write_text("")
    results = chunk_file(empty)
    assert results == []


def test_chunk_file_markdown(tmp_path):
    """Markdown files should be chunked as docs."""
    md_file = tmp_path / "notes.md"
    md_file.write_text("# Title\n\nSome content about architecture decisions.\n\n## Section\n\nMore text here.")
    results = chunk_file(md_file, relative_to=tmp_path)
    assert len(results) > 0
    assert all(r.chunk_type == "doc" for r in results)
