"""Tests for the git-axis layer (Phase 5):

  - ``commit_metadata`` / ``changed_files_since`` in
    :mod:`hafiz.core.git_context`
  - ``upsert_commit`` / ``latest_indexed_commit`` in
    :mod:`hafiz.core.store`

The metadata + diff helpers run against the hafiz repo itself (same
pattern as ``tests/test_git_context.py``). The store-level tests run
against a live Postgres and skip gracefully otherwise.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from hafiz.core.database import (
    File,
    close_engine,
    get_session_factory,
)
from hafiz.core.git_context import (
    changed_files_since,
    commit_metadata,
    is_git_repo,
)
from hafiz.core.store import latest_indexed_commit, upsert_commit

# ── commit_metadata ────────────────────────────────────────────────────────


def test_commit_metadata_of_head_has_expected_shape():
    repo = Path.cwd()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    meta = commit_metadata(head, repo)
    assert meta is not None
    assert isinstance(meta["author"], str) and meta["author"]
    assert meta["committed_at"] is not None
    assert isinstance(meta["summary"], str) and meta["summary"]


def test_commit_metadata_of_unknown_sha_returns_none():
    # "deadbeef..." is valid-looking but almost certainly not in the repo.
    fake = "deadbeef" * 5  # 40 chars
    assert commit_metadata(fake, Path.cwd()) is None


def test_commit_metadata_outside_git_returns_none(tmp_path: Path):
    assert commit_metadata("abc", tmp_path) is None


# ── changed_files_since ────────────────────────────────────────────────────


def test_changed_files_since_on_unreachable_base_returns_none():
    fake = "deadbeef" * 5
    assert changed_files_since(fake, Path.cwd()) is None


def test_changed_files_since_head_of_head_is_empty_or_uncommitted_only():
    """Diffing HEAD against itself yields no committed changes; any
    returned paths must be uncommitted working-tree edits."""
    repo = Path.cwd()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    changed = changed_files_since(head, repo, include_uncommitted=False)
    assert changed == set()


def test_changed_files_since_between_two_commits():
    """Walk the last two commits and verify their diff contains
    something — sanity check that the parse pipeline is alive."""
    repo = Path.cwd()
    # HEAD and HEAD~1 — previous commit may not exist in a brand-new
    # repo; skip if so.
    prev = subprocess.run(
        ["git", "rev-parse", "HEAD~1"],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if prev.returncode != 0:
        pytest.skip("Only one commit in the repo")
    prev_sha = prev.stdout.strip()
    changed = changed_files_since(prev_sha, repo, include_uncommitted=False)
    assert isinstance(changed, set)
    # At least one file differs between any two distinct commits.
    assert len(changed) >= 1


def test_is_git_repo_true_here_false_in_tmp(tmp_path: Path):
    assert is_git_repo(Path.cwd()) is True
    assert is_git_repo(tmp_path) is False


# ── upsert_commit / latest_indexed_commit (DB-backed) ─────────────────────


async def _db_available() -> bool:
    try:
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text("SELECT 1 FROM commits LIMIT 1"))
        return True
    except Exception:
        return False


async def _cleanup():
    factory = get_session_factory()
    async with factory() as s:
        await s.execute(text("DELETE FROM commits"))
        await s.execute(text("DELETE FROM files"))
        await s.commit()


@pytest.fixture(autouse=True)
async def _skip_and_clean():
    if not await _db_available():
        pytest.skip("Postgres not reachable")
    await _cleanup()
    yield
    await close_engine()


@pytest.mark.asyncio
async def test_upsert_commit_records_head_metadata():
    repo = Path.cwd()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    row = await upsert_commit(head, project="hafiz-test", cwd=repo)
    assert row is not None
    assert row.hash == head
    assert row.author
    assert row.committed_at is not None
    assert row.summary


@pytest.mark.asyncio
async def test_upsert_commit_is_idempotent():
    repo = Path.cwd()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    await upsert_commit(head, project="p1", cwd=repo)
    await upsert_commit(head, project="p1", cwd=repo)

    factory = get_session_factory()
    async with factory() as s:
        result = await s.execute(text("SELECT COUNT(*) FROM commits"))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_upsert_commit_unknown_sha_noop():
    fake = "deadbeef" * 5
    row = await upsert_commit(fake, project="p1", cwd=Path.cwd())
    assert row is None
    factory = get_session_factory()
    async with factory() as s:
        result = await s.execute(text("SELECT COUNT(*) FROM commits"))
        assert result.scalar() == 0


@pytest.mark.asyncio
async def test_latest_indexed_commit_returns_mode_of_last_seen():
    """The latest-indexed commit is whichever SHA appears most often in
    the project's current files — the 'project is indexed at this
    commit' signal used by diff-driven ingest."""
    factory = get_session_factory()
    async with factory() as s:
        # Three files: two at sha_a, one at sha_b. sha_a wins.
        sha_a = "a" * 40
        sha_b = "b" * 40
        s.add_all(
            [
                File(
                    id=uuid.uuid4(),
                    project="p1",
                    path=f"/t/{i}.py",
                    last_seen_commit=sha_a if i < 2 else sha_b,
                )
                for i in range(3)
            ]
        )
        await s.commit()

    latest = await latest_indexed_commit("p1")
    assert latest == sha_a


@pytest.mark.asyncio
async def test_latest_indexed_commit_none_when_no_files():
    assert await latest_indexed_commit("nonexistent") is None
