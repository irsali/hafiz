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


# Filesystem markers git creates while a rewrite-class operation is in
# flight. If any exist, the tree is in an intermediate state and ingesting
# it would capture garbage. Used by hafiz ingest's race-safety guard.
_REWRITE_IN_PROGRESS_MARKERS = (
    "rebase-apply",
    "rebase-merge",
    "MERGE_HEAD",
    "CHERRY_PICK_HEAD",
    "REVERT_HEAD",
    "BISECT_LOG",
)


def git_operation_in_progress(cwd: Path) -> str | None:
    """Return the name of the in-flight git operation, or None.

    Checks for the well-known filesystem markers git writes during
    rebase / merge / cherry-pick / revert / bisect. Ingesting while one
    of these is set would capture an intermediate tree state.
    """
    if not is_git_repo(cwd):
        return None
    git_dir_raw = _git(["rev-parse", "--git-dir"], cwd)
    if not git_dir_raw:
        return None
    git_dir = Path(git_dir_raw)
    if not git_dir.is_absolute():
        git_dir = (cwd / git_dir).resolve()
    for marker in _REWRITE_IN_PROGRESS_MARKERS:
        if (git_dir / marker).exists():
            return marker
    return None


def is_commit_reachable(sha: str, cwd: Path) -> bool:
    """True iff ``sha`` is reachable from any ref in ``cwd``.

    Used by reconcile-on-ingest to detect hashes orphaned by rebase /
    force-push. Uses ``git cat-file -e`` (object exists) AND checks for
    reachability via refs — a rewritten commit may still be in the
    object db but unreachable from any branch.
    """
    if not sha or not is_git_repo(cwd):
        return False
    # First: does the object exist at all?
    exists = subprocess.run(
        ["git", "cat-file", "-e", sha],
        cwd=str(cwd),
        capture_output=True,
        timeout=5,
    )
    if exists.returncode != 0:
        return False
    # Second: reachable from any ref? `git for-each-ref --contains <sha>`
    # lists refs that have <sha> in their history; empty output = orphaned.
    reachable = _git(
        ["for-each-ref", "--contains", sha, "--count=1", "--format=%(refname)"],
        cwd,
    )
    return bool(reachable)


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
