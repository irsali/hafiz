"""Tests for Phase 5b — rewrite resilience.

Covers:
  - ``git_operation_in_progress`` detects .git/<marker> files
  - ``is_commit_reachable`` distinguishes reachable from orphaned
  - ``reconcile_orphaned_commits`` marks stale rows with rewritten_at
  - ``hafiz hooks install`` writes all three templates including
    post-rewrite
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from hafiz.core.database import Commit, close_engine, get_session_factory
from hafiz.core.git_context import (
    git_operation_in_progress,
    is_commit_reachable,
)
from hafiz.core.store import reconcile_orphaned_commits

# ── git_operation_in_progress ──────────────────────────────────────────────


def test_git_operation_in_progress_returns_none_here():
    assert git_operation_in_progress(Path.cwd()) is None


def test_git_operation_in_progress_outside_git_returns_none(tmp_path: Path):
    assert git_operation_in_progress(tmp_path) is None


def test_git_operation_in_progress_detects_marker(tmp_path: Path, monkeypatch):
    """Create a fake .git dir with a marker file and verify detection."""
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "t@t"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "t"],
        check=True,
        capture_output=True,
    )
    # Create an initial commit so HEAD exists.
    (tmp_path / "f.txt").write_text("hi")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "init", "-q"],
        check=True,
        capture_output=True,
    )
    # Synthesize a rebase marker.
    (tmp_path / ".git" / "MERGE_HEAD").write_text("dead")

    assert git_operation_in_progress(tmp_path) == "MERGE_HEAD"


# ── is_commit_reachable ────────────────────────────────────────────────────


def test_is_commit_reachable_head_is_true():
    assert is_commit_reachable("HEAD", Path.cwd()) is True


def test_is_commit_reachable_unknown_sha_is_false():
    assert is_commit_reachable("deadbeef" * 5, Path.cwd()) is False


def test_is_commit_reachable_outside_git_is_false(tmp_path: Path):
    assert is_commit_reachable("HEAD", tmp_path) is False


# ── reconcile_orphaned_commits (DB-backed) ────────────────────────────────


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
        await s.commit()


@pytest.fixture(autouse=True)
async def _skip_and_clean():
    if not await _db_available():
        pytest.skip("Postgres not reachable")
    await _cleanup()
    yield
    await close_engine()


@pytest.mark.asyncio
async def test_reconcile_marks_unreachable_commits():
    """A commit hash that isn't reachable in git gets rewritten_at set."""
    factory = get_session_factory()
    async with factory() as s:
        s.add(
            Commit(
                hash="deadbeef" * 5,  # 40 chars, definitely not in git
                project="hafiz-test",
                summary="fake",
            )
        )
        await s.commit()

    reconciled = await reconcile_orphaned_commits("hafiz-test", Path.cwd())
    assert reconciled == 1

    async with factory() as s:
        row = await s.get(Commit, "deadbeef" * 5)
        assert row is not None
        assert row.rewritten_at is not None


@pytest.mark.asyncio
async def test_reconcile_leaves_reachable_commits_alone():
    """Real HEAD stays un-rewritten."""
    import subprocess

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    factory = get_session_factory()
    async with factory() as s:
        s.add(
            Commit(
                hash=head,
                project="hafiz-test",
                summary="real",
            )
        )
        await s.commit()

    reconciled = await reconcile_orphaned_commits("hafiz-test", Path.cwd())
    assert reconciled == 0

    async with factory() as s:
        row = await s.get(Commit, head)
        assert row is not None
        assert row.rewritten_at is None


@pytest.mark.asyncio
async def test_reconcile_scoped_to_project():
    """A commit belonging to a different project stays untouched."""
    factory = get_session_factory()
    async with factory() as s:
        s.add_all(
            [
                Commit(hash="a" * 40, project="p1", summary="fake a"),
                Commit(hash="b" * 40, project="p2", summary="fake b"),
            ]
        )
        await s.commit()

    reconciled = await reconcile_orphaned_commits("p1", Path.cwd())
    assert reconciled == 1

    async with factory() as s:
        row_b = await s.get(Commit, "b" * 40)
        assert row_b is not None
        assert row_b.rewritten_at is None


# ── Hook installer covers post-rewrite ────────────────────────────────────


def test_hooks_install_includes_post_rewrite(tmp_path: Path):
    """`hafiz hooks install` writes all three templates including post-rewrite."""
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)

    from hafiz.commands.hooks import run_hooks_install

    run_hooks_install(str(tmp_path), project="demo")

    hooks_dir = tmp_path / ".git" / "hooks"
    for name in ("post-commit", "post-merge", "post-rewrite"):
        hook_file = hooks_dir / name
        assert hook_file.exists(), f"{name} hook not installed"
        content = hook_file.read_text()
        assert "hafiz ingest" in content
        assert "demo" in content
