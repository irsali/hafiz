"""Store health as data: counts, index freshness, retention, capture.

Extracted from ``commands/maintenance.py``, where it lived inside
``run_status`` interleaved with Rich rendering. Two callers now need the same
numbers — the CLI and the ``hafiz_status`` MCP tool — and the alternative to
extracting was a second implementation that would drift from the first.

**This module does not manage engine lifecycle**, and that is load-bearing
rather than incidental. The code it came from ended in
``finally: await close_engine()``, which is right for a one-shot CLI process
and actively wrong for a long-lived server: the MCP server would dispose the
connection pool it depends on, on every status call. Disposal stays with the
caller that owns the process.
"""

from __future__ import annotations

from typing import Any

from hafiz.core.config import get_settings


def embedding_device_summary() -> dict:
    """Sync, DB-independent summary of the embedding-device selection."""
    from hafiz.core import device_state as dstate

    settings = get_settings()
    sticky = dstate.load_state()
    configured = settings.embedding.device

    if configured in ("cpu", "gpu"):
        source = "config"
        effective = configured
    elif sticky is not None:
        source = "sticky"
        effective = sticky.device
    else:
        source = "not-probed"
        effective = "(not probed)"

    return {
        "configured": configured,
        "source": source,
        "effective": effective,
        "sticky_probed_at": sticky.probed_at if sticky else None,
        "sticky_reason_category": sticky.reason_category if sticky else None,
    }


async def index_staleness_for(last_commit_per_project: dict[str | None, str]) -> dict:
    """For each project, how far its indexed commit trails the repo's HEAD.

    Fills the gap that let four repos drift 31-64 commits behind while hooks
    were installed and firing: nothing ever compared the index to the repo.

    Thin wrapper over :func:`hafiz.core.freshness.index_staleness`, which
    search results share — the probe belongs in core, and two copies would
    drift. The already-computed map is handed over rather than re-derived.
    """
    from hafiz.core.freshness import index_staleness

    return await index_staleness(list(last_commit_per_project), last_commit=last_commit_per_project)


def is_stale(entry: dict) -> bool:
    """True when a project's index has actually fallen behind its repo.

    ``is_ancestor is False`` means the indexed commit is not in HEAD's history
    at all — rebased away or force-pushed over — which is a worse state than
    being merely behind, and one where a commit count would be meaningless.
    """
    return bool(entry.get("commits_behind")) or entry.get("is_ancestor") is False


async def collect_status(verbose: bool = False) -> dict[str, Any]:
    """Every number ``hafiz status`` reports, as plain data.

    Args:
        verbose: True returns the full payload, byte-identical to
            ``hafiz status --json``. False — the default, and what an agent
            gets — returns the same keys with ``staleness`` filtered to only
            projects that are actually stale or diverged, plus a
            ``staleness_summary`` saying how many were checked.

            The saving is real but varies with health, and is not the main
            point — measured on one developer store with 8 of 11 projects
            stale it was only 12%, because almost nothing was droppable.
            What trimming reliably buys is *signal*: the stale projects are
            named and counted, instead of a caller having to scan every
            entry to work out which three of eleven matter.

            The summary is not optional politeness: an empty ``staleness``
            with nothing beside it cannot be told apart from "freshness was
            never checked", and this codebase has already paid for one signal
            whose silence was mistaken for health.

    ``retention`` and ``capture`` are never trimmed. They are the "is the
    retention guarantee actually being met" and "is anything still arriving"
    signals, and the second one going unread is what let transcript capture
    die unnoticed for two months.
    """
    from sqlalchemy import func, select

    from hafiz.core.communications import count_overdue_communications
    from hafiz.core.database import (
        Annotation,
        Commit,
        Edge,
        Embedding,
        File,
        Unit,
        UnitRevision,
        get_session_factory,
    )
    from hafiz.core.freshness import capture_freshness
    from hafiz.core.store import last_indexed_commit_per_project
    from hafiz.core.telemetry import count_overdue_retrievals

    session_factory = get_session_factory()
    async with session_factory() as session:

        async def count(entity, *where) -> int:
            stmt = select(func.count()).select_from(entity)
            if where:
                stmt = stmt.where(*where)
            return (await session.execute(stmt)).scalar() or 0

        # ── Current-state counts (tombstoned / superseded excluded) ─
        files_count = await count(File, File.valid_until.is_(None))
        units_count = await count(Unit, Unit.valid_until.is_(None))
        current_revisions_count = await count(UnitRevision, UnitRevision.superseded_at.is_(None))
        embeddings_count = await count(Embedding)
        edges_count = await count(Edge, Edge.superseded_at.is_(None))
        annotations_count = await count(Annotation)
        commits_count = await count(Commit)

        # ── Historical totals (include tombstoned for context) ──
        total_units = await count(Unit)
        total_revisions = await count(UnitRevision)

        # ── Breakdowns by project and kind (current only) ──
        project_rows = (
            await session.execute(
                select(File.project, func.count())
                .where(File.valid_until.is_(None))
                .group_by(File.project)
                .order_by(func.count().desc())
            )
        ).all()

        kind_rows = (
            await session.execute(
                select(Unit.kind, func.count())
                .where(Unit.valid_until.is_(None))
                .group_by(Unit.kind)
                .order_by(func.count().desc())
            )
        ).all()

    # ── Most-recent commit per project, and how stale it is ──
    # Ordered by commits.committed_at, NOT max(hash): hashes are hex, so a
    # lexicographic max picks a commit at random and silently masked genuinely
    # stale indexes across six repos for months.
    last_commit_per_project = await last_indexed_commit_per_project()
    staleness = await index_staleness_for(last_commit_per_project)

    # Bounded retention is an outward-facing commitment; the sweep only runs on
    # `import`, which stops firing exactly when it's needed. The count is what
    # makes an unenforced policy visible.
    overdue_comms = await count_overdue_communications()
    overdue_retr = await count_overdue_retrievals()

    # Retention overdue answers "is the sweep keeping up", which a store
    # receiving nothing passes trivially. Capture freshness answers the
    # question that actually went unasked for two months: "is anything still
    # arriving?"
    capture = await capture_freshness()

    stats: dict[str, Any] = {
        "files": files_count,
        "units": units_count,
        "revisions_current": current_revisions_count,
        "revisions_total": total_revisions,
        "units_total": total_units,
        "units_tombstoned": total_units - units_count,
        "embeddings": embeddings_count,
        "edges": edges_count,
        "annotations": annotations_count,
        "commits": commits_count,
        "by_project": {p or "(none)": c for p, c in project_rows},
        "by_kind": {k or "(none)": c for k, c in kind_rows},
        "last_commit_per_project": {p or "(none)": c for p, c in last_commit_per_project.items()},
        "staleness": staleness,
        "retention": {
            "overdue": overdue_comms + overdue_retr,
            "communications": overdue_comms,
            "retrievals": overdue_retr,
        },
        "capture": capture,
        # A project-less ingest can't update a project's rows — `files` is
        # unique on (project, path) — so it writes a parallel untagged copy
        # that search then returns alongside the real one. 1,956 such rows
        # accumulated unnoticed on a real deployment because nothing counted
        # them.
        "untagged": {"files": dict(project_rows).get(None, 0)},
        "embedding_device": embedding_device_summary(),
    }

    if not verbose:
        stale_only = {name: entry for name, entry in staleness.items() if is_stale(entry)}
        stats["staleness"] = stale_only
        stats["staleness_summary"] = {
            "projects_checked": len(staleness),
            "stale": len(stale_only),
            "trimmed": len(staleness) - len(stale_only),
        }

    return stats
