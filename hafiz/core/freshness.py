"""How far each project's index trails its repo.

Search results used to carry no freshness signal at all — no indexed commit, no
distance from HEAD — so a caller could not tell a current result from one 28
commits stale. The only safe policy was therefore a blanket one, and a real
integrator adopted it: *"the code index is deliberately NOT used: it runs 30-64
commits behind every repo, so it returns code that no longer exists."*

Measured on that same deployment, that rule was over-broad — 4.4% of live files
had changed since their indexed commit and 3 had been deleted outright. But it
was pointing at something real: the project being **actively edited** was 80.5%
stale, because staleness concentrates exactly where the work is. Volume is low;
correlation with what you're asking about is high.

So the fix isn't to make the index fresher (hooks do that) — it's to stop making
the caller guess. This module is the shared probe behind ``status`` and the
``staleness`` block on search results.
"""

from __future__ import annotations

from pathlib import Path


async def index_staleness(
    projects: list[str] | None = None,
    *,
    last_commit: dict[str | None, str] | None = None,
) -> dict[str, dict]:
    """Per-project: indexed commit, repo HEAD, and the distance between them.

    ``projects`` narrows both DB queries; pass the projects present in a result
    set rather than sweeping the whole store (~40ms on a 15-project index).
    ``last_commit`` lets a caller that already computed the map (``status``)
    hand it over instead of paying for it twice.

    Every field degrades to ``None`` rather than raising. Two reasons: an
    operator reaches for this when something is already wrong and it must still
    print, and a git failure must never take down a search.

    The untagged (``project IS NULL``) bucket is always excluded. Its files span
    every repo a project-less hook ever walked, so the derived root comes out as
    ``/`` and "how far behind HEAD" is not a meaningful question for it — see
    ``prune --untagged``.
    """
    from hafiz.core.git_context import commits_behind_head, is_git_repo
    from hafiz.core.store import indexed_root_per_project, last_indexed_commit_per_project

    scope = [p for p in projects if p is not None] if projects is not None else None
    if scope is not None and not scope:
        return {}

    if last_commit is None:
        last_commit = await last_indexed_commit_per_project(scope)
    roots = await indexed_root_per_project(scope)

    out: dict[str, dict] = {}
    for project, indexed_sha in last_commit.items():
        if project is None:
            continue
        root = roots.get(project)
        entry: dict = {
            "repo_path": root,
            "indexed_commit": indexed_sha,
            "head_commit": None,
            "commits_behind": None,
            "is_ancestor": None,
        }
        if root and is_git_repo(Path(root)):
            entry.update(commits_behind_head(indexed_sha, Path(root)))
        out[project] = entry
    return out


def stale_projects(staleness: dict[str, dict]) -> dict[str, dict]:
    """Just the entries a caller should be warned about.

    "Behind by N" and "diverged" are both actionable; "unknown" is not — it
    usually means the repo isn't on this machine, which is not a data problem.
    """
    return {
        project: entry
        for project, entry in staleness.items()
        if entry.get("commits_behind") or entry.get("is_ancestor") is False
    }
