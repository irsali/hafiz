"""Tests for hafiz.core.export — the sovereignty dump.

These are DB-free: ORM instances are constructed in memory (they're plain
Python objects until a session touches them) and pushed through the
serializers. ``export_brain`` is exercised by monkeypatching ``_gather``
so the atomic-write + manifest path runs without Postgres.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hafiz.core import export as ex
from hafiz.core.database import Annotation


def _ann(content: str, *, kind: str = "decision", **kw) -> Annotation:
    return Annotation(
        id=uuid.uuid4(),
        content=content,
        kind=kind,
        source=kw.get("source", "agent:claude-code"),
        project=kw.get("project"),
        tags=kw.get("tags"),
        confidence=kw.get("confidence", 1.0),
        valid_from=kw.get("valid_from", datetime.now(UTC)),
    )


def _bundle_with(annotations) -> ex._Bundle:
    b = ex._Bundle()
    b.annotations = list(annotations)
    return b


# ── serializer: markdown ────────────────────────────────────────────────────


def test_markdown_groups_observations_by_kind(tmp_path: Path):
    b = _bundle_with([_ann("a decision", kind="decision"), _ann("a fact", kind="fact")])
    ex._render_markdown(b, tmp_path, include_transcripts=False)

    assert (tmp_path / "observations" / "decision.md").exists()
    assert (tmp_path / "observations" / "fact.md").exists()
    assert (tmp_path / "README.md").exists()
    assert "a decision" in (tmp_path / "observations" / "decision.md").read_text()


def test_markdown_readme_warns_about_secrets(tmp_path: Path):
    ex._render_markdown(_bundle_with([_ann("x")]), tmp_path, include_transcripts=False)
    readme = (tmp_path / "README.md").read_text()
    assert "sensitive" in readme.lower()
    assert "Code is intentionally" in readme


def test_markdown_empty_brain_still_writes_readme(tmp_path: Path):
    ex._render_markdown(_bundle_with([]), tmp_path, include_transcripts=False)
    assert (tmp_path / "README.md").exists()
    assert not (tmp_path / "observations").exists()


# ── serializer: json ────────────────────────────────────────────────────────


def test_json_drops_embedding_and_keeps_metadata(tmp_path: Path):
    ex._render_json(_bundle_with([_ann("decided", tags=["t"])]), tmp_path)
    line = (tmp_path / "knowledge" / "annotations.jsonl").read_text().strip()
    rec = json.loads(line)
    assert rec["content"] == "decided"
    assert "embedding" not in rec
    assert "metadata" in rec  # mapped from metadata_ to its DB name


# ── orchestration: export_brain ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_export_brain_rejects_bad_format(tmp_path: Path):
    result = await ex.export_brain(out_dir=tmp_path / "e", fmt="xml")
    assert result["ok"] is False
    assert "unknown format" in result["error"]


@pytest.mark.asyncio
async def test_export_brain_writes_atomically(tmp_path: Path, monkeypatch):
    async def fake_gather(*, project, include_transcripts):
        return _bundle_with([_ann("decided X over Y")])

    monkeypatch.setattr(ex, "_gather", fake_gather)
    out = tmp_path / "brain"
    result = await ex.export_brain(out_dir=out, fmt="md")

    assert result["ok"] is True
    assert Path(result["path"]) == out.resolve()
    assert result["counts"]["annotations"] == 1
    # No stray temp dir left behind.
    assert not out.with_name(out.name + ".tmp").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema_version"] == ex.SCHEMA_VERSION
    assert manifest["format"] == "md"
    assert "code/structure" in manifest["excludes"]


@pytest.mark.asyncio
async def test_export_brain_replaces_existing_dir(tmp_path: Path, monkeypatch):
    async def fake_gather(*, project, include_transcripts):
        return _bundle_with([_ann("only one")])

    monkeypatch.setattr(ex, "_gather", fake_gather)
    out = tmp_path / "brain"
    out.mkdir()
    (out / "stale.txt").write_text("old export artifact")

    await ex.export_brain(out_dir=out, fmt="md")
    # Stale file from the prior export must be gone after the atomic swap.
    assert not (out / "stale.txt").exists()
    assert (out / "README.md").exists()


# ── sovereignty filter: forgotten/expired never gathered ─────────────────────


class _RecordingSession:
    """Minimal async-session stand-in that captures the statement the
    helper builds and returns an empty result. Lets us assert the
    *actual* query carries the sovereignty filters, DB-free."""

    def __init__(self) -> None:
        self.statements: list = []

    async def execute(self, stmt):
        self.statements.append(stmt)
        return _EmptyResult()


class _EmptyScalars:
    def all(self):
        return []


class _EmptyResult:
    def scalars(self):
        return _EmptyScalars()


@pytest.mark.asyncio
async def test_live_annotations_filters_tombstones():
    sess = _RecordingSession()
    await ex._live_annotations(sess, None)
    compiled = str(sess.statements[0])
    assert "valid_until IS NULL" in compiled


@pytest.mark.asyncio
async def test_live_communications_filters_tombstone_and_retention():
    sess = _RecordingSession()
    await ex._live_communications(sess, None, datetime.now(UTC))
    compiled = str(sess.statements[0])
    assert "valid_until IS NULL" in compiled
    assert "retention_until" in compiled


@pytest.mark.asyncio
async def test_live_annotations_scopes_by_project():
    sess = _RecordingSession()
    await ex._live_annotations(sess, "hafiz")
    compiled = str(sess.statements[0])
    assert "project" in compiled
