"""Capture the current git state for observation metadata.

One subprocess call per field, with graceful degradation — a non-git cwd,
a detached HEAD, or a missing git binary all yield an empty dict or a
safe fallback rather than raising.
"""

from __future__ import annotations

import subprocess
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


def current_git_context(cwd: Path | None = None) -> dict:
    """Return a dict describing the current git HEAD, or {} if not in a repo.

    Fields, when returned:
      - commit_hash: full SHA of HEAD (may be empty in pathological cases).
      - branch: current branch name, or "HEAD" when detached.
      - is_dirty: True if the working tree has uncommitted changes.
    """
    repo = Path(cwd) if cwd else Path.cwd()

    if _git(["rev-parse", "--is-inside-work-tree"], repo) != "true":
        return {}

    commit_hash = _git(["rev-parse", "HEAD"], repo)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], repo) or "HEAD"
    is_dirty = bool(_git(["status", "--porcelain"], repo))

    return {
        "commit_hash": commit_hash,
        "branch": branch,
        "is_dirty": is_dirty,
    }
