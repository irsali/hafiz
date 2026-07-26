"""Tests for the git hooks `hafiz hooks install` generates.

The defect: `--project` was optional, and omitting it produced

    nohup hafiz ingest --git-hook > /dev/null 2>&1 &

with no repo path and no project. `ingest --git-hook` then walked cwd and
tagged everything `project=NULL`. Since `files` is unique on `(project, path)`,
that built a *second, untagged copy* of the repo rather than updating the real
rows — measured at 1,951 untagged files across four hooked repos, whose
project-tagged rows meanwhile sat 31-64 commits behind.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hafiz.commands import hooks

HOOK_NAMES = ("post-commit", "post-merge", "post-rewrite")


@pytest.fixture
def repo(tmp_path) -> Path:
    subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    """Default to "project not indexed anywhere" so tests stay DB-free."""
    monkeypatch.setattr(hooks, "_conflicting_root", lambda project, repo: None)


def _hook(repo: Path, name: str) -> str:
    return (repo / ".git" / "hooks" / name).read_text(encoding="utf-8")


# ── Project + repo are always pinned ────────────────────────────────────


def test_project_defaults_to_the_repo_directory_name(repo, capsys):
    hooks.run_hooks_install(str(repo))
    body = _hook(repo, "post-commit")
    assert f"HAFIZ_PROJECT={repo.name}" in body


def test_no_hook_is_ever_generated_without_a_project(repo):
    """The exact regression: an untagged ingest builds a shadow index."""
    hooks.run_hooks_install(str(repo))
    for name in HOOK_NAMES:
        body = _hook(repo, name)
        assert "# No project specified" not in body
        assert "--project" in body
        assert "HAFIZ_PROJECT=" in body


def test_repo_path_is_baked_into_every_hook(repo):
    """A hook's cwd isn't guaranteed to be the work tree (worktrees, `git -C`)."""
    hooks.run_hooks_install(str(repo), project="proj")
    for name in HOOK_NAMES:
        body = _hook(repo, name)
        assert f"HAFIZ_REPO={repo}" in body
        assert 'hafiz ingest "$HAFIZ_REPO"' in body


def test_explicit_project_overrides_the_directory_name(repo):
    hooks.run_hooks_install(str(repo), project="Admin Portal")
    assert "HAFIZ_PROJECT='Admin Portal'" in _hook(repo, "post-commit")


def test_project_with_spaces_is_shell_quoted(repo):
    """Unquoted, `--project Admin Portal` would parse as two arguments."""
    hooks.run_hooks_install(str(repo), project="Admin Portal")
    body = _hook(repo, "post-commit")
    assert "HAFIZ_PROJECT='Admin Portal'" in body
    assert '--project "$HAFIZ_PROJECT"' in body


def test_repo_path_with_spaces_is_shell_quoted(tmp_path):
    spaced = tmp_path / "my repo"
    spaced.mkdir()
    subprocess.run(["git", "init", "-q", "."], cwd=spaced, check=True, capture_output=True)
    hooks.run_hooks_install(str(spaced), project="p")
    assert f"HAFIZ_REPO='{spaced}'" in _hook(spaced, "post-commit")


def test_generated_hooks_are_valid_bash(repo):
    hooks.run_hooks_install(str(repo), project="proj")
    for name in HOOK_NAMES:
        result = subprocess.run(
            ["bash", "-n", str(repo / ".git" / "hooks" / name)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{name}: {result.stderr}"


def test_generated_hooks_are_executable(repo):
    hooks.run_hooks_install(str(repo), project="proj")
    for name in HOOK_NAMES:
        assert (repo / ".git" / "hooks" / name).stat().st_mode & 0o111


def test_all_three_hooks_are_installed(repo):
    hooks.run_hooks_install(str(repo), project="proj")
    for name in HOOK_NAMES:
        assert (repo / ".git" / "hooks" / name).exists()


# ── Coexistence with foreign hooks ──────────────────────────────────────


def test_appends_to_a_foreign_hook_without_clobbering_it(repo):
    target = repo / ".git" / "hooks"
    target.mkdir(exist_ok=True)
    (target / "post-commit").write_text("#!/bin/sh\necho mine\n", encoding="utf-8")

    hooks.run_hooks_install(str(repo), project="proj")
    body = _hook(repo, "post-commit")
    assert "echo mine" in body
    assert "HAFIZ_PROJECT=proj" in body


def test_appended_block_is_valid_bash(repo):
    target = repo / ".git" / "hooks"
    target.mkdir(exist_ok=True)
    (target / "post-commit").write_text("#!/usr/bin/env bash\necho mine\n", encoding="utf-8")
    hooks.run_hooks_install(str(repo), project="proj")
    result = subprocess.run(
        ["bash", "-n", str(target / "post-commit")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_reinstall_is_idempotent(repo):
    hooks.run_hooks_install(str(repo), project="proj")
    first = _hook(repo, "post-commit")
    hooks.run_hooks_install(str(repo), project="proj")
    assert _hook(repo, "post-commit") == first


# ── Refusing a scope mismatch ───────────────────────────────────────────


def test_refuses_when_the_project_is_indexed_under_another_root(repo, monkeypatch):
    """Committing here would re-index — and tombstone against — another tree."""
    monkeypatch.setattr(hooks, "_conflicting_root", lambda project, repo: "/elsewhere/other")
    with pytest.raises(SystemExit) as exc:
        hooks.run_hooks_install(str(repo), project="hafiz")
    assert exc.value.code == 2
    assert not (repo / ".git" / "hooks" / "post-commit").exists()


def test_force_allows_repointing_a_moved_project(repo, monkeypatch):
    monkeypatch.setattr(hooks, "_conflicting_root", lambda project, repo: "/elsewhere/other")
    hooks.run_hooks_install(str(repo), project="hafiz", force=True)
    assert "HAFIZ_PROJECT=hafiz" in _hook(repo, "post-commit")


def test_a_root_containing_the_repo_is_not_a_conflict(repo, monkeypatch):
    """Re-installing into an already-correctly-indexed repo must just work."""
    monkeypatch.setattr(hooks, "_conflicting_root", hooks._conflicting_root)

    async def _roots():
        return {"proj": str(repo)}

    monkeypatch.setattr("hafiz.core.store.indexed_root_per_project", _roots)
    monkeypatch.setattr("hafiz.core.database.close_engine", _noop)
    hooks.run_hooks_install(str(repo), project="proj")
    assert (repo / ".git" / "hooks" / "post-commit").exists()


async def _noop():
    return None


def test_db_failure_does_not_block_installation(repo, monkeypatch):
    """The scope check is advisory — a DB outage must not stop a hook install."""
    monkeypatch.setattr(hooks, "_conflicting_root", hooks._conflicting_root)

    async def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr("hafiz.core.store.indexed_root_per_project", _boom)
    monkeypatch.setattr("hafiz.core.database.close_engine", _noop)
    hooks.run_hooks_install(str(repo), project="proj")
    assert (repo / ".git" / "hooks" / "post-commit").exists()


# ── Guards ──────────────────────────────────────────────────────────────


def test_non_repo_is_rejected(tmp_path):
    with pytest.raises(SystemExit) as exc:
        hooks.run_hooks_install(str(tmp_path))
    assert exc.value.code == 1


def test_git_hook_ingest_forwards_the_path_it_was_given(monkeypatch):
    """`--git-hook` used to hardcode '.', ignoring the repo the hook names."""
    seen = {}

    def _fake_run_ingest(path, *, project=None, **kw):
        seen["path"] = path
        seen["project"] = project

    monkeypatch.setattr("hafiz.commands.ingest.run_ingest", _fake_run_ingest)
    from hafiz.commands.ingest import run_git_hook_ingest_cmd

    run_git_hook_ingest_cmd("/repos/Admin Portal", project="Admin Portal")
    assert seen == {"path": "/repos/Admin Portal", "project": "Admin Portal"}


def test_git_hook_ingest_defaults_to_cwd(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "hafiz.commands.ingest.run_ingest",
        lambda path, **kw: seen.update(path=path),
    )
    from hafiz.commands.ingest import run_git_hook_ingest_cmd

    run_git_hook_ingest_cmd(None, project="p")
    assert seen["path"] == "."


def test_git_hook_ingest_warns_when_untagged(monkeypatch, capsys):
    """A project-less ingest is the shadow-index bug; say so out loud."""
    monkeypatch.setattr("hafiz.commands.ingest.run_ingest", lambda path, **kw: None)
    from hafiz.commands.ingest import run_git_hook_ingest_cmd

    run_git_hook_ingest_cmd(".", project=None)
    assert "duplicate untagged index" in capsys.readouterr().out
