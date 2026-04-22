"""Capture git state for observation metadata and for the diff-driven
ingest pipeline.

One subprocess call per field, with graceful degradation — a non-git cwd,
a detached HEAD, or a missing git binary all yield an empty dict or a
safe fallback rather than raising.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def is_git_repo(cwd: Path) -> bool:
    """True iff ``cwd`` is inside a git work tree."""
    return _git(["rev-parse", "--is-inside-work-tree"], cwd) == "true"


def current_git_context(cwd: Path | None = None) -> dict:
    """Return a dict describing the current git HEAD, or {} if not in a repo.

    Fields, when returned:
      - commit_hash: full SHA of HEAD (may be empty in pathological cases).
      - branch: current branch name, or "HEAD" when detached.
      - is_dirty: True if the working tree has uncommitted changes.
    """
    repo = Path(cwd) if cwd else Path.cwd()

    if not is_git_repo(repo):
        return {}

    commit_hash = _git(["rev-parse", "HEAD"], repo)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo) or "HEAD"
    is_dirty = bool(_git(["status", "--porcelain"], repo))

    return {
        "commit_hash": commit_hash,
        "branch": branch,
        "is_dirty": is_dirty,
    }


# ── Commit metadata (Phase 5 — git-axis as first-class) ────────────────────


def commit_metadata(sha: str, cwd: Path) -> dict | None:
    """Return ``{author, committed_at, summary}`` for ``sha``, or None if
    the commit isn't reachable from cwd (e.g. rebased away)."""
    if not sha or not is_git_repo(cwd):
        return None
    # `git show --no-patch --format=...` is a single round trip.
    fmt = "%an <%ae>%x1f%cI%x1f%s"
    raw = _git(
        ["show", "--no-patch", f"--format={fmt}", sha],
        cwd,
    )
    if not raw:
        return None
    parts = raw.split("\x1f")
    if len(parts) < 3:
        return None
    author, iso_dt, summary = parts[0], parts[1], parts[2]
    try:
        committed_at = datetime.fromisoformat(iso_dt)
        if committed_at.tzinfo is None:
            committed_at = committed_at.replace(tzinfo=timezone.utc)
    except ValueError:
        committed_at = None
    return {
        "author": author,
        "committed_at": committed_at,
        "summary": summary,
    }


def changed_files_since(
    base_sha: str, cwd: Path, *, include_uncommitted: bool = True
) -> set[Path] | None:
    """Return absolute paths changed between ``base_sha`` and HEAD.

    Includes uncommitted changes in the working tree when
    ``include_uncommitted`` is True so `hafiz ingest` on a dirty checkout
    still picks up your WIP. Returns None if ``base_sha`` isn't reachable
    (typical after a rebase / force-push) — callers should fall back to
    a full walk.
    """
    if not is_git_repo(cwd):
        return None
    # `git merge-base --is-ancestor` returns exit 0 if reachable.
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_sha, "HEAD"],
        cwd=str(cwd),
        capture_output=True,
        timeout=5,
    )
    if ancestor.returncode != 0:
        return None

    diff_raw = _git(
        ["diff", "--name-only", f"{base_sha}..HEAD"],
        cwd,
    )
    changed: set[Path] = set()
    for line in diff_raw.splitlines():
        if line.strip():
            changed.add((cwd / line.strip()).resolve())

    if include_uncommitted:
        # Uncommitted staged + unstaged. `--porcelain=v1 -uall` lists every
        # file with a short status; we want the paths regardless of what
        # happened to them (mod/add/del) so downstream tombstoning and
        # re-parse still kick in.
        porcelain = _git(["status", "--porcelain"], cwd)
        for line in porcelain.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Porcelain format: "XY path" (or "XY path -> path2" for renames).
            # Slice off the two-char status and optional space.
            _, _, rest = line.partition(" ")
            if not rest:
                rest = stripped[3:] if len(stripped) > 3 else stripped
            path_part = rest.split(" -> ")[-1].strip()
            if path_part:
                changed.add((cwd / path_part).resolve())

    return changed
