"""Distill — surface recent raw captures (notes + transcripts + messages)
as promotable candidates.

Propose, don't auto-apply. This module is a **scanner**: it returns the
ids + content of recent ``kind="note"`` annotations, legacy transcripts,
and (Phase 5) source-layer ``communication_messages`` so the agent /
user can read them and decide what (if anything) to promote into a
``decision`` / ``learning`` / ``pattern`` via a follow-up ``hafiz
observe`` call with ``--derived-from``.

Explicitly NOT an LLM call. Hafiz stays sovereign; the distillation
judgement is delegated to whoever reads the candidates.

Phase 5 enrichment: when a session filter is provided (or there's an
active session), source-layer message ids are surfaced as well, so a
distilled decision can cite turns directly via the polymorphic
``annotation_targets`` pivot.

Capture-table transcripts: currently empty (Phase 3b-2 rewires
:mod:`hafiz.core.capture` onto the new schema). Until then, only
note-kind annotations and source-layer messages are surfaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from hafiz.core.database import (
    Annotation,
    Communication,
    CommunicationMessage,
    get_session_factory,
)
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
class MessageCandidate:
    """Source-layer turn surfaced as a distillation source."""

    id: str
    communication_id: str
    seq: int
    role: str
    author: str | None
    content: str
    ts: datetime
    marked_salient: bool


@dataclass
class DistillBundle:
    window_start: datetime
    window_end: datetime
    notes: list[NoteCandidate] = field(default_factory=list)
    transcripts: list[JournalCapture] = field(default_factory=list)
    messages: list[MessageCandidate] = field(default_factory=list)


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
    now = datetime.now(UTC)
    start = now - (since or timedelta(days=7))
    end = now

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(Annotation)
            .where(Annotation.kind == "note")
            .where(Annotation.valid_from >= start)
            .where(Annotation.valid_from <= end)
            .where((Annotation.valid_until.is_(None)) | (Annotation.valid_until > now))
            .order_by(Annotation.valid_from.desc())
            .limit(limit)
        )
        if isinstance(project, list):
            stmt = stmt.where(Annotation.project.in_(project))
        elif project:
            stmt = stmt.where(Annotation.project == project)
        if session_id:
            # Filter on legacy_session_id (the historical text slug). Phase 2
            # will resolve user-supplied slugs to the new uuid FK and union.
            stmt = stmt.where(Annotation.legacy_session_id == session_id)
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
            session_id=a.legacy_session_id or (str(a.session_id) if a.session_id else None),
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

    messages = await _fetch_message_candidates(
        start=start,
        end=end,
        project=project,
        session_slug=session_id,
    )

    return DistillBundle(
        window_start=start,
        window_end=end,
        notes=notes,
        transcripts=transcripts,
        messages=messages,
    )


async def _fetch_message_candidates(
    *,
    start: datetime,
    end: datetime,
    project: str | list[str] | None,
    session_slug: str | None,
    limit: int = 50,
) -> list[MessageCandidate]:
    """Surface source-layer turns in the distillation window.

    Filters:
    - ``ts`` within [start, end]
    - communication is not tombstoned
    - if session_slug is set, restrict to communications belonging to
      that session (so distillation lineage can cite the actual turns)
    - if project filter is set, restrict to communications scoped to it

    Salient turns (``marked_salient=true``) are surfaced regardless of
    embedding state; everything else is included as long as it has
    non-empty content.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        from hafiz.core.sessions import get_session_by_slug

        stmt = (
            select(CommunicationMessage)
            .join(
                Communication,
                Communication.id == CommunicationMessage.communication_id,
            )
            .where(CommunicationMessage.ts >= start)
            .where(CommunicationMessage.ts <= end)
            .where(Communication.valid_until.is_(None))
            .order_by(CommunicationMessage.ts.asc())
            .limit(limit)
        )

        if isinstance(project, list):
            stmt = stmt.where(Communication.scope_value.in_(project))
        elif project:
            stmt = stmt.where(Communication.scope_value == project)

        if session_slug:
            sess = await get_session_by_slug(session_slug)
            if sess is not None:
                stmt = stmt.where(Communication.session_id == sess.id)
            else:
                # Session slug doesn't resolve — surface nothing rather
                # than every message in the window.
                return []

        rows = (await session.execute(stmt)).scalars().all()

    return [
        MessageCandidate(
            id=str(m.id),
            communication_id=str(m.communication_id),
            seq=m.seq,
            role=m.role,
            author=m.author,
            content=m.content,
            ts=m.ts,
            marked_salient=m.marked_salient,
        )
        for m in rows
    ]
