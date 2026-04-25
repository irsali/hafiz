"""Time-bounded digest over annotations — the ``hafiz journal`` feature.

A lightweight view layer: "what did I record between X and Y, grouped by
day". Reads from the ``annotations`` table — no new storage.

Transcript captures are temporarily absent from the bundle: Phase 3b-2
rewires :mod:`hafiz.core.capture` onto the new schema (transcripts
become units + annotation links). Until then, :func:`fetch_captures`
returns an empty list and the bundle is annotation-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from hafiz.core.database import Annotation, get_session_factory


@dataclass
class JournalEntry:
    id: str
    content: str
    kind: str
    source: str | None
    project: str | None
    tags: list[str] | None
    confidence: float
    valid_from: datetime
    valid_until: datetime | None
    session_id: str | None
    task: str | None
    commit_hash: str | None
    metadata: dict


@dataclass
class JournalCapture:
    """One transcript, aggregated from its constituent turns.

    Populated by :func:`fetch_captures` once Phase 3b-2 rewires capture.
    Until then, :class:`JournalBundle.captures` is always an empty list.
    """

    transcript_id: str
    title: str | None
    source_file: str
    turn_count: int
    captured_at: datetime
    source: str | None
    tags: list[str] | None
    project: str | None
    session_id: str | None
    task: str | None
    preview: str


@dataclass
class JournalBundle:
    window_start: datetime
    window_end: datetime
    entries: list[JournalEntry] = field(default_factory=list)
    captures: list[JournalCapture] = field(default_factory=list)

    def grouped_by_day(
        self,
    ) -> list[tuple[str, list[JournalEntry], list[JournalCapture]]]:
        """Return [(YYYY-MM-DD, entries, captures), ...] newest day first."""
        buckets: dict[str, tuple[list[JournalEntry], list[JournalCapture]]] = {}
        for e in self.entries:
            day = e.valid_from.astimezone(timezone.utc).strftime("%Y-%m-%d")
            buckets.setdefault(day, ([], []))[0].append(e)
        for c in self.captures:
            day = c.captured_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
            buckets.setdefault(day, ([], []))[1].append(c)
        return sorted(
            ((d, es, cs) for d, (es, cs) in buckets.items()),
            key=lambda t: t[0],
            reverse=True,
        )


async def build_journal(
    *,
    since: timedelta | None = None,
    day: datetime | None = None,
    project: str | list[str] | None = None,
    source: str | None = None,
    kind: str | None = None,
    session_id: str | None = None,
    task: str | None = None,
    limit: int = 500,
) -> JournalBundle:
    """Build a time-bounded journal bundle.

    Window rules:
      - If ``day`` is given, the window is that UTC day (00:00 → 23:59:59.999999).
      - Otherwise the window is ``[now - since, now]``; ``since`` defaults to 7d.

    ``session_id`` / ``task`` filter both annotations and captures to items
    tagged with the given thread of work.
    """
    now = datetime.now(timezone.utc)
    if day is not None:
        start = day.astimezone(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(days=1) - timedelta(microseconds=1)
    else:
        start = now - (since or timedelta(days=7))
        end = now

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(Annotation)
            .where(Annotation.valid_from >= start)
            .where(Annotation.valid_from <= end)
            .order_by(Annotation.valid_from.desc())
            .limit(limit)
        )
        if isinstance(project, list):
            stmt = stmt.where(Annotation.project.in_(project))
        elif project:
            stmt = stmt.where(Annotation.project == project)
        if source:
            stmt = stmt.where(Annotation.source == source)
        if kind:
            stmt = stmt.where(Annotation.kind == kind)
        if session_id:
            # Filter on legacy_session_id (the historical text slug). Phase 2
            # will resolve user-supplied slugs to the new uuid FK and union.
            stmt = stmt.where(Annotation.legacy_session_id == session_id)
        if task:
            stmt = stmt.where(Annotation.task == task)

        rows = (await session.execute(stmt)).scalars().all()

    entries = [
        JournalEntry(
            id=str(a.id),
            content=a.content,
            kind=a.kind,
            source=a.source,
            project=a.project,
            tags=a.tags,
            confidence=a.confidence,
            valid_from=a.valid_from,
            valid_until=a.valid_until,
            session_id=a.legacy_session_id or (str(a.session_id) if a.session_id else None),
            task=a.task,
            commit_hash=a.commit_hash,
            metadata=a.metadata_ or {},
        )
        for a in rows
    ]

    captures = await fetch_captures(
        start=start,
        end=end,
        project=project,
        source=source,
        session_id=session_id,
        task=task,
    )

    return JournalBundle(
        window_start=start,
        window_end=end,
        entries=entries,
        captures=captures,
    )


async def fetch_captures(
    *,
    start: datetime,
    end: datetime,
    project: str | list[str] | None = None,
    source: str | None = None,
    session_id: str | None = None,
    task: str | None = None,
) -> list[JournalCapture]:
    """Transcripts in ``[start, end]``.

    Temporarily a no-op: Phase 3b-2 rewires :mod:`hafiz.core.capture` so
    transcripts live as units (`chat.turn` or similar) tied to
    annotations. Until that lands the journal is annotation-only —
    returning an empty list here keeps :func:`build_journal` and
    :func:`hafiz.core.distill.find_distill_candidates` unchanged.
    """
    # Suppress unused-arg warnings — the signature is preserved so Phase
    # 3b-2 can drop in a real implementation without touching callers.
    del start, end, project, source, session_id, task
    return []
