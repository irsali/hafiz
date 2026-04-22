"""Distill — surface recent raw captures (notes + transcripts) as
promotable candidates.

Propose, don't auto-apply. This module is a **scanner**: it returns the
ids + content of recent ``kind="note"`` annotations and transcripts so
the agent / user can read them and decide what (if anything) to promote
into a ``decision`` / ``learning`` / ``pattern`` via a follow-up
``hafiz observe`` call with ``--derived-from``.

Explicitly NOT an LLM call. Hafiz stays sovereign; the distillation
judgement is delegated to whoever reads the candidates.

Transcripts list is currently empty: Phase 3b-2 rewires
:mod:`hafiz.core.capture` so transcripts live as units + annotation
links. Until then, only note-kind annotations are surfaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from hafiz.core.database import Annotation, get_session_factory
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
    """Return active ``kind="note"`` annotations + transcripts in
    ``[now-since, now]``.

    Expired / superseded notes are excluded — already handled elsewhere,
    don't drag them into fresh distillation.
    """
    now = datetime.now(timezone.utc)
    start = now - (since or timedelta(days=7))
    end = now

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(Annotation)
            .where(Annotation.kind == "note")
            .where(Annotation.valid_from >= start)
            .where(Annotation.valid_from <= end)
            .where(
                (Annotation.valid_until.is_(None))
                | (Annotation.valid_until > now)
            )
            .order_by(Annotation.valid_from.desc())
            .limit(limit)
        )
        if isinstance(project, list):
            stmt = stmt.where(Annotation.project.in_(project))
        elif project:
            stmt = stmt.where(Annotation.project == project)
        if session_id:
            stmt = stmt.where(Annotation.session_id == session_id)
        if task:
            stmt = stmt.where(Annotation.task == task)

        rows = (await session.execute(stmt)).scalars().all()

    notes = [
        NoteCandidate(
            id=str(a.id),
            content=a.content,
            valid_from=a.valid_from,
            source=a.source,
            project=a.project,
            tags=a.tags,
            session_id=a.session_id,
            task=a.task,
        )
        for a in rows
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
