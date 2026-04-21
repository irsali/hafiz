"""Smoke tests for ``hafiz.core.git_context.current_git_context``.

The hafiz repo is itself a git repo, so we can verify shape + types
without mocking subprocess. A separate case covers the non-git path.
"""

from pathlib import Path

from hafiz.core.git_context import current_git_context


def test_inside_git_repo_returns_expected_keys():
    ctx = current_git_context()
    assert set(ctx) == {"commit_hash", "branch", "is_dirty"}
    assert isinstance(ctx["commit_hash"], str)
    assert isinstance(ctx["branch"], str)
    assert isinstance(ctx["is_dirty"], bool)
    # HEAD should resolve to a 40-char SHA under normal conditions.
    assert len(ctx["commit_hash"]) == 40


def test_outside_git_repo_returns_empty(tmp_path: Path):
    assert current_git_context(tmp_path) == {}
