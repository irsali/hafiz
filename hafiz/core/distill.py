"""Distill — surface recent raw captures (notes + transcripts) as
promotable candidates.

Propose, don't auto-apply. This module is a **scanner**: it returns the
ids + content of recent ``obs_type="note"`` rows and transcripts so the
agent / user can read them and decide what (if anything) to promote
into a ``decision`` / ``learning`` / ``pattern`` via a follow-up
``hafiz observe`` call with ``--derived-from``.

Explicitly NOT an LLM call. Hafiz stays sovereign; the distillation
judgement is delegated to whoever reads the candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from hafiz.core.database import Observation, get_session_factory
from hafiz.core.journal import JournalCapture, fetch_captures


@dataclass
class NoteCandidate:
    id: str
    content: str
    valid_from: datetime
    source: str | None
    project: str | None
    tags: list[str] | None
    session_id: str | None
    task: str | None


@dataclass
class DistillBundle:
    window_start: datetime
    window_end: datetime
    notes: list[NoteCandidate] = field(default_factory=list)
    transcripts: list[JournalCapture] = field(default_factory=list)


async def find_distill_candidates(
    *,
    since: timedelta | None = None,
    project: str | list[str] | None = None,
    session_id: str | None = None,
    task: str | None = None,
    include_transcripts: bool = True,
    limit: int = 200,
) -> DistillBundle:
    """Return active ``note`` observations + transcripts in ``[now-since, now]``.

    Expired / superseded notes are excluded — already handled elsewhere,
    don't drag them into fresh distillation.
    """
    now = datetime.now(timezone.utc)
    start = now - (since or timedelta(days=7))
    end = now

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(Observation)
            .where(Observation.obs_type == "note")
            .where(Observation.valid_from >= start)
            .where(Observation.valid_from <= end)
            .where(
                (Observation.valid_until.is_(None))
                | (Observation.valid_until > now)
            )
            .order_by(Observation.valid_from.desc())
            .limit(limit)
        )
        if isinstance(project, list):
            stmt = stmt.where(Observation.project.in_(project))
        elif project:
            stmt = stmt.where(Observation.project == project)
        if session_id:
            stmt = stmt.where(Observation.session_id == session_id)
        if task:
            stmt = stmt.where(Observation.task == task)

        rows = (await session.execute(stmt)).scalars().all()

    notes = [
        NoteCandidate(
            id=str(o.id),
            content=o.content,
            valid_from=o.valid_from,
            source=o.source,
            project=o.project,
            tags=o.tags,
            session_id=o.session_id,
            task=o.task,
        )
        for o in rows
    ]

    transcripts: list[JournalCapture] = []
    if include_transcripts:
        transcripts = await fetch_captures(
            start=start,
            end=end,
            project=project,
            session_id=session_id,
            task=task,
        )

    return DistillBundle(
        window_start=start,
        window_end=end,
        notes=notes,
        transcripts=transcripts,
    )
