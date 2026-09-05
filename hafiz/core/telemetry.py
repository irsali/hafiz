"""Recording what the store was asked for, so it can be evaluated.

Hafiz could not answer the most basic question about itself. Auditing a 3.5-week
deployment for "is this earning its keep?" required parsing 169 Claude Code
transcripts, because hafiz kept no record of its own reads: it could not say
which annotations had ever been recalled, which surfaced and were useful, or
which had never come up once.

Three design constraints, in priority order:

1. **A failure here must never fail a search.** Recording is best-effort and
   swallows everything. A memory layer that can break the read path gets removed.
2. **The serving path must not be able to bypass it.** So the call sites are in
   ``hafiz/core/`` — ``search_annotations`` and ``vector_search`` — not in the
   CLI commands. ``hafiz/core/daemon.py`` calls ``search_annotations`` directly;
   telemetry wired at the command layer would have silently missed every warm
   request, which is the exact failure class the audit kept finding.
3. **It inherits the source layer's guarantees.** Bounded retention,
   tombstoneable, and off entirely with ``[telemetry] retrieval = false``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

#: Commands as recorded in ``retrievals.command``. Kept as literals rather than
#: derived from argv so the daemon and the CLI agree on the label.
QUERY = "query"
OBSERVATIONS = "query --observations"
#: ``context`` fans out to both layers, so it records one row per layer under
#: its own labels. Without them a context bundle was indistinguishable from a
#: plain ``query`` — the flagship command was invisible in its own telemetry,
#: and "never used" read the same as "never instrumented".
CONTEXT = "context"
CONTEXT_OBSERVATIONS = "context --observations"
RECALL = "recall"


async def record_retrieval(
    *,
    command: str,
    query: str,
    result_ids: list[uuid.UUID | str] | None = None,
    top_score: float | None = None,
    reranked: bool = False,
    filters: dict | None = None,
    source: str | None = None,
) -> None:
    """Append one row describing a search. Never raises.

    ``result_ids`` are annotation ids for the wisdom layer and embedding ids for
    the unit index; ``command`` says which. The session is resolved from the
    ambient cursor so a hook-driven read is attributable without the caller
    threading it through.
    """
    from hafiz.core.config import get_settings

    try:
        settings = get_settings()
        cfg = settings.telemetry
        if not cfg.retrieval:
            return
        cleaned = (query or "").strip()
        if len(cleaned) < cfg.min_query_chars:
            # "yes" / "go on" say nothing about what the store was asked for.
            return

        ids: list[uuid.UUID] = []
        for raw in result_ids or []:
            try:
                ids.append(raw if isinstance(raw, uuid.UUID) else uuid.UUID(str(raw)))
            except (ValueError, AttributeError, TypeError):
                continue  # a non-uuid id is not worth losing the row over

        now = datetime.now(UTC)
        # Resolved *before* the write session opens. Doing it inline in the
        # Retrieval(...) call nested a second session inside an open one, and a
        # failure in the inner one returned a connection to the pool mid-flight
        # — which surfaced much later, in an unrelated caller, as
        # "another operation is in progress".
        session_id = await _ambient_session_id()

        from hafiz.core.database import Retrieval, get_session_factory

        factory = get_session_factory()
        async with factory() as session:
            session.add(
                Retrieval(
                    id=uuid.uuid4(),
                    at=now,
                    command=command,
                    query_text=cleaned,
                    filters={k: v for k, v in (filters or {}).items() if v is not None},
                    result_ids=ids,
                    n_results=len(ids),
                    top_score=top_score,
                    reranked=reranked,
                    session_id=session_id,
                    source=source,
                    retention_until=now + timedelta(days=cfg.retention_days),
                )
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001 — constraint 1: never fail a search
        logger.debug("retrieval telemetry skipped (%s)", exc)


async def _ambient_session_id() -> uuid.UUID | None:
    """The current session's uuid, if a cursor is open. None on any problem."""
    try:
        from hafiz.core.session import resolve_session_tag
        from hafiz.core.sessions import resolve_session_uuid

        tag = resolve_session_tag(None)
        return await resolve_session_uuid(tag) if tag else None
    except Exception:  # noqa: BLE001
        return None


async def retrieval_report(*, since_days: int = 30, limit: int = 20) -> dict:
    """The three questions the store couldn't answer about itself.

    ``never_recalled`` — live annotations that have never appeared in a result
    set. Bucketed by age, because a two-day-old row looks identical to dead
    knowledge otherwise, and because nothing written before telemetry existed
    can be judged at all (``blind_before``).

    ``unanswered`` — queries that returned nothing. The most actionable output
    here: it's the gap between what agents ask for and what the store holds, so
    it says what to write down next. No other signal produces it.

    ``most_recalled`` — what's actually earning its keep.
    """
    from sqlalchemy import Integer, and_, cast, desc, func, literal_column, select, text

    from hafiz.core.database import Annotation, Retrieval, get_session_factory
    from hafiz.core.dialect import backend_of, most_recalled_sql, unnest_ids

    since = datetime.now(UTC) - timedelta(days=since_days)
    factory = get_session_factory()
    async with factory() as session:
        first_row = (await session.execute(select(func.min(Retrieval.at)))).scalar()
        total = (
            await session.execute(
                select(func.count()).select_from(Retrieval).where(Retrieval.at >= since)
            )
        ).scalar() or 0

        backend = backend_of(session)
        recalled = unnest_ids(Retrieval.result_ids, Retrieval, backend)
        never = (
            await session.execute(
                select(func.count())
                .select_from(Annotation)
                .where(Annotation.valid_until.is_(None))
                .where(Annotation.id.not_in(select(recalled.c.id)))
            )
        ).scalar() or 0
        # A genuine subset of never_recalled, not a parallel count: the useful
        # statement is "of the rows never recalled, this many were written before
        # anything was being recorded", so they can't be judged either way.
        blind_before = (
            await session.execute(
                select(func.count())
                .select_from(Annotation)
                .where(Annotation.valid_until.is_(None))
                .where(Annotation.id.not_in(select(recalled.c.id)))
                .where(Annotation.valid_from < (first_row or datetime.now(UTC)))
            )
        ).scalar() or 0

        unanswered = (
            await session.execute(
                select(Retrieval.query_text, func.count().label("n"), func.max(Retrieval.at))
                .where(Retrieval.at >= since)
                .where(Retrieval.n_results == 0)
                .group_by(Retrieval.query_text)
                .order_by(desc(literal_column("n")))
                .limit(limit)
            )
        ).all()

        top = (
            await session.execute(
                text(most_recalled_sql(backend)),
                {"since": since, "limit": limit},
            )
        ).all()

        empty_rate = (
            await session.execute(
                select(
                    func.count().label("n"),
                    func.sum(cast(and_(Retrieval.n_results == 0), Integer)).label("empty"),
                ).where(Retrieval.at >= since)
            )
        ).one()

        # Which entry points are actually in use. Without this the report can
        # say how well retrieval performed but not whether anyone is calling
        # the command that matters — and a label with no reader is not
        # instrumentation. Empty-per-command is here too because a caller that
        # always comes back empty is a different problem from an unused one.
        by_command = (
            await session.execute(
                select(
                    Retrieval.command,
                    func.count().label("n"),
                    func.sum(cast(and_(Retrieval.n_results == 0), Integer)).label("empty"),
                    func.avg(Retrieval.top_score).label("avg_top"),
                )
                .where(Retrieval.at >= since)
                .group_by(Retrieval.command)
                .order_by(desc(literal_column("n")))
            )
        ).all()

    return {
        "since_days": since_days,
        "telemetry_started": first_row.isoformat() if first_row else None,
        "retrievals": total,
        "by_command": [
            {
                "command": c,
                "calls": n,
                "empty": int(e or 0),
                "avg_top_score": round(float(a), 3) if a is not None else None,
            }
            for c, n, e, a in by_command
        ],
        "empty_result_rate": (
            round((empty_rate.empty or 0) / empty_rate.n, 3) if empty_rate.n else None
        ),
        "never_recalled": never,
        "blind_before": blind_before,
        "unanswered": [{"query": q, "times": n, "last": at.isoformat()} for q, n, at in unanswered],
        "most_recalled": [{"id": i, "kind": k, "preview": p, "hits": h} for i, k, p, h in top],
    }


async def count_overdue_retrievals() -> int:
    """Live rows past ``retention_until``. The number that makes an unenforced
    policy visible — the same lesson as ``communications``: the sweep trigger is
    secondary, the count is what gets acted on."""
    from sqlalchemy import func, select

    from hafiz.core.database import Retrieval, get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        return (
            await session.execute(
                select(func.count())
                .select_from(Retrieval)
                .where(Retrieval.valid_until.is_(None))
                .where(Retrieval.retention_until.is_not(None))
                .where(Retrieval.retention_until <= func.now())
            )
        ).scalar() or 0


async def tombstone_expired_retrievals(*, dry_run: bool = False) -> dict:
    """Soft-tombstone every retrieval past its retention window."""
    from sqlalchemy import select, update

    from hafiz.core.database import Retrieval, get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        matched = (
            (
                await session.execute(
                    select(Retrieval.id)
                    .where(Retrieval.valid_until.is_(None))
                    .where(Retrieval.retention_until.is_not(None))
                    .where(Retrieval.retention_until <= datetime.now(UTC))
                )
            )
            .scalars()
            .all()
        )
        if dry_run or not matched:
            return {"matched": len(matched), "tombstoned": 0, "dry_run": dry_run}
        await session.execute(
            update(Retrieval).where(Retrieval.id.in_(matched)).values(valid_until=datetime.now(UTC))
        )
        await session.commit()
        return {"matched": len(matched), "tombstoned": len(matched), "dry_run": False}
