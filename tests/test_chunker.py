"""Tests for hafiz.core.chunker — walk_files + prepare_embedding_parts.

The chunker is narrow by design. These tests cover what matters:

  - ``walk_files`` yields only text-ish files, honors .gitignore, and
    skips binary content.
  - ``prepare_embedding_parts`` keeps short content as one part, splits
    oversized content into multiple parts with preserved offsets, and
    prefers newline boundaries when splitting.
  - The default ``DEFAULT_MAX_PART_CHARS`` is the safe CPU default we
    ship with — not the old 24_000. The registry is the runtime source
    of truth; this constant is the fallback.
"""

from __future__ import annotations

from pathlib import Path

from hafiz.core.chunker import (
    DEFAULT_MAX_PART_CHARS,
    EmbeddingPart,
    compute_hash,
    prepare_embedding_parts,
    walk_files,
)


# ── compute_hash ────────────────────────────────────────────────────────


def test_compute_hash_is_deterministic():
    assert compute_hash("hello world") == compute_hash("hello world")


def test_compute_hash_distinguishes_inputs():
    assert compute_hash("hello world") != compute_hash("goodbye world")


def test_compute_hash_is_sha256_hex():
    h = compute_hash("x")
    assert len(h) == 64  # sha256 hex digest length
    int(h, 16)  # valid hex


# ── prepare_embedding_parts ─────────────────────────────────────────────


def test_short_content_yields_single_part():
    parts = prepare_embedding_parts("hello")
    assert len(parts) == 1
    assert parts[0].content == "hello"
    assert parts[0].part_index == 0
    assert parts[0].token_span_start == 0
    assert parts[0].token_span_end == 5


def test_oversized_content_splits():
    content = "a" * (DEFAULT_MAX_PART_CHARS * 2 + 10)
    parts = prepare_embedding_parts(content)
    assert len(parts) >= 2
    # Every part must be an EmbeddingPart with monotonically increasing index.
    for i, p in enumerate(parts):
        assert isinstance(p, EmbeddingPart)
        assert p.part_index == i
    # Concatenating all part bodies must recover the original content.
    assert "".join(p.content for p in parts) == content
    # Spans must be contiguous and cover the content.
    assert parts[0].token_span_start == 0
    assert parts[-1].token_span_end == len(content)
    for a, b in zip(parts, parts[1:]):
        assert a.token_span_end == b.token_span_start


def test_split_prefers_newline_boundary():
    """When a newline exists in the back half of the window, the split
    should land just after it so parts end at a logical boundary."""
    # Build content with a newline well inside the second half of the window.
    head = "a" * (DEFAULT_MAX_PART_CHARS - 10)
    tail = "b" * (DEFAULT_MAX_PART_CHARS + 50)
    # Place a newline ~3/4 through the first window.
    newline_pos = DEFAULT_MAX_PART_CHARS * 3 // 4
    content = head[:newline_pos] + "\n" + head[newline_pos:] + tail
    parts = prepare_embedding_parts(content)
    assert len(parts) >= 2
    # First part should end right after the newline we inserted.
    assert parts[0].content.endswith("\n")


def test_max_chars_override():
    parts = prepare_embedding_parts("x" * 500, max_chars=100)
    assert len(parts) == 5
    assert all(len(p.content) == 100 for p in parts)


def test_default_is_cpu_safe():
    """Regression guard: the old 24_000 default OOM-killed ingest on a
    16 GB laptop. Don't silently raise without updating the Tunable's
    description and the work item."""
    assert DEFAULT_MAX_PART_CHARS == 2_000


# ── walk_files ──────────────────────────────────────────────────────────


def test_walk_files_yields_text_files(tmp_path: Path):
    (tmp_path / "a.py").write_text("print('a')\n")
    (tmp_path / "b.md").write_text("# hello\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("text\n")
    found = {p.name for p in walk_files(tmp_path)}
    assert found == {"a.py", "b.md", "c.txt"}


def test_walk_files_skips_binary(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("readable\n")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02\x03")
    found = {p.name for p in walk_files(tmp_path)}
    assert found == {"ok.txt"}


def test_walk_files_respects_gitignore(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("secret.txt\n")
    (tmp_path / "secret.txt").write_text("nope")
    (tmp_path / "keep.txt").write_text("yes")
    found = {p.name for p in walk_files(tmp_path)}
    assert "secret.txt" not in found
    assert "keep.txt" in found


def test_walk_files_single_file(tmp_path: Path):
    f = tmp_path / "one.py"
    f.write_text("x = 1\n")
    found = list(walk_files(f))
    assert found == [f]


def test_walk_files_respects_config_ignore(tmp_path: Path):
    (tmp_path / "keep.py").write_text("x")
    blacklisted = tmp_path / "buildout"
    blacklisted.mkdir()
    (blacklisted / "junk.py").write_text("x")
    found = {p.name for p in walk_files(tmp_path, ignore_patterns=["buildout"])}
    assert "keep.py" in found
    assert "junk.py" not in found
