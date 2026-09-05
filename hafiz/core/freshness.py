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

from datetime import UTC, datetime, timedelta
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


async def capture_freshness(*, settle_minutes: int = 30) -> dict[str, dict]:
    """Per-agent: when the source layer last received a transcript, and
    how many *settled* transcripts are sitting on disk newer than that.

    This exists because the absence of a signal here cost real data.
    Transcript capture for ``claude-code`` stopped on 2026-06-30 and was
    not noticed until 2026-09-04 — ``status`` reported
    ``retention.overdue: 0`` throughout, which looks like health but only
    ever meant "the sweep is keeping up", and a store receiving nothing
    is trivially swept. Silence is not health.

    ``pending_on_disk`` counts sessions whose **id is not in the store at
    all**, per agent, via that agent's own ``pending_on_disk`` probe —
    only an importer knows where its harness keeps sessions. Probes read
    just enough to identify a session (a JSONL head, a SQLite id column)
    rather than parsing it: `status` is on the hot path and must stay
    cheap. For claude-code specifically, the filename stem is *not* usable
    as the id — it matches only a session's first file, and 124 of 200
    files disagreed with it.

    Deliberately *not* "mtime newer than the last capture", which was the
    first cut and was wrong: ``last_captured_at`` is ``max(started_at)``,
    i.e. when the newest captured session *began*. Any session that ran for
    more than a moment therefore has an mtime later than its own start and
    reported as pending while being fully captured. Never-captured is the
    signal that has a remedy; "grew since capture" is handled by the hook
    re-importing, and quantified precisely by ``import --dry-run``.

    ``settle_minutes`` excludes transcripts touched very recently, which is
    what keeps the *current* session out of the count — it is unknown to
    the store until it is first captured, and it is being appended to as
    you read this. Without the window the warning would fire on every
    ``status`` during any active session, and a warning that is always on
    is a warning nobody reads — the exact failure mode this signal exists
    to correct. In-progress sessions are not uncaptured work; they are in
    progress, and are reported separately as ``active_on_disk``.

    Every field degrades to ``None``/absent rather than raising — same
    contract as :func:`index_staleness`.
    """
    from sqlalchemy import func, select

    from hafiz.core.database import Communication, get_session_factory

    out: dict[str, dict] = {}
    try:
        factory = get_session_factory()
        async with factory() as session:
            rows = await session.execute(
                select(
                    Communication.agent,
                    func.max(Communication.started_at),
                    func.count(),
                    func.count(Communication.valid_until),
                ).group_by(Communication.agent)
            )
            for agent, last_started, total, tombstoned in rows.all():
                last = last_started
                out[agent] = {
                    "last_captured_at": last.isoformat() if last else None,
                    "days_since": (
                        (datetime.now(UTC) - last).days
                        if last is not None and last.tzinfo
                        else None
                    ),
                    "communications": total,
                    "live": total - tombstoned,
                    "tombstoned": tombstoned,
                }
    except Exception:
        return {}

    # Each agent's own importer knows where its harness keeps sessions and
    # how to recognise one; freshness just asks. ChatGPT has no probe on
    # purpose — an export is a file the user downloads, not a store that
    # can fall behind.
    from hafiz.core.importers import PENDING_PROBES, pending_probe

    settled_before = datetime.now(UTC) - timedelta(minutes=settle_minutes)
    for agent in PENDING_PROBES:
        probe = pending_probe(agent)
        if probe is None:
            continue
        entry = out.setdefault(agent, {"last_captured_at": None, "days_since": None})
        try:
            async with factory() as session:
                rows = await session.execute(
                    select(Communication.external_id).where(
                        Communication.agent == agent,
                        Communication.external_id.is_not(None),
                    )
                )
                known = {e for (e,) in rows.all()}
            pending, active = probe(known, settled_before)
            entry["pending_on_disk"] = pending
            entry["active_on_disk"] = active
        except Exception:
            # An unreadable store is not a reason to fail `status`; the
            # operator reaches for it when something is already wrong.
            entry["pending_on_disk"] = None

    return out


def stale_captures(capture: dict[str, dict]) -> dict[str, dict]:
    """Just the agents a caller should be warned about: those with
    transcripts waiting on disk.

    Deliberately *only* that condition, because it is the only one with a
    remedy — ``hafiz import <agent>`` fixes it. A long-quiet agent with
    nothing pending is either one you stopped using or one whose files
    already rotated away; warning about it every ``status`` teaches the
    operator to ignore the block, which is how the original signal gap
    got expensive in the first place.
    """
    return {
        agent: entry for agent, entry in capture.items() if (entry.get("pending_on_disk") or 0) > 0
    }


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
