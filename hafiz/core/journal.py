"""Time-bounded digest over observations — the ``hafiz journal`` feature.

A lightweight view layer: "what did I record between X and Y, grouped by day".
Reuses the existing ``observations`` table — no new storage, no migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from hafiz.core.database import Chunk, Observation, get_session_factory


@dataclass
class JournalEntry:
    id: str
    content: str
    obs_type: str
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
    """One transcript, aggregated from its constituent chunks."""

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
    obs_type: str | None = None,
    session_id: str | None = None,
    task: str | None = None,
    limit: int = 500,
) -> JournalBundle:
    """Build a time-bounded journal bundle.

    Window rules:
      - If ``day`` is given, the window is that UTC day (00:00 → 23:59:59.999999).
      - Otherwise the window is ``[now - since, now]``; ``since`` defaults to 7d.

    ``session_id`` / ``task`` filter both observations and captures to items
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
            select(Observation)
            .where(Observation.valid_from >= start)
            .where(Observation.valid_from <= end)
            .order_by(Observation.valid_from.desc())
            .limit(limit)
        )
        if isinstance(project, list):
            stmt = stmt.where(Observation.project.in_(project))
        elif project:
            stmt = stmt.where(Observation.project == project)
        if source:
            stmt = stmt.where(Observation.source == source)
        if obs_type:
            stmt = stmt.where(Observation.obs_type == obs_type)
        if session_id:
            stmt = stmt.where(Observation.session_id == session_id)
        if task:
            stmt = stmt.where(Observation.task == task)

        rows = (await session.execute(stmt)).scalars().all()

    entries = [
        JournalEntry(
            id=str(o.id),
            content=o.content,
            obs_type=o.obs_type,
            source=o.source,
            project=o.project,
            tags=o.tags,
            confidence=o.confidence,
            valid_from=o.valid_from,
            valid_until=o.valid_until,
            session_id=o.session_id,
            task=o.task,
            commit_hash=o.commit_hash,
            metadata=o.metadata_ or {},
        )
        for o in rows
    ]

    captures = await _fetch_captures(
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


async def _fetch_captures(
    *,
    start: datetime,
    end: datetime,
    project: str | list[str] | None = None,
    source: str | None = None,
    session_id: str | None = None,
    task: str | None = None,
) -> list[JournalCapture]:
    """Fetch transcripts whose chunks were indexed in ``[start, end]``."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(Chunk)
            .where(Chunk.chunk_type == "transcript")
            .where(Chunk.indexed_at >= start)
            .where(Chunk.indexed_at <= end)
            .order_by(Chunk.indexed_at.desc())
        )
        if isinstance(project, list):
            stmt = stmt.where(Chunk.project.in_(project))
        elif project:
            stmt = stmt.where(Chunk.project == project)
        if source:
            stmt = stmt.where(Chunk.metadata_["source"].astext == source)
        if session_id:
            stmt = stmt.where(Chunk.session_id == session_id)
        if task:
            stmt = stmt.where(Chunk.task == task)
        rows = (await session.execute(stmt)).scalars().all()

    groups: dict[str, list] = {}
    for c in rows:
        tid = (c.metadata_ or {}).get("transcript_id")
        if tid:
            groups.setdefault(tid, []).append(c)

    captures: list[JournalCapture] = []
    for tid, cs in groups.items():
        # Representative "turn 0" chunk drives title, source, and preview.
        first = min(cs, key=lambda c: (c.metadata_ or {}).get("turn_index", 0))
        meta = first.metadata_ or {}
        preview = first.content[:140] + ("..." if len(first.content) > 140 else "")
        captures.append(
            JournalCapture(
                transcript_id=tid,
                title=meta.get("title"),
                source_file=first.source_file,
                turn_count=meta.get("total_turns") or len(cs),
                captured_at=min(c.indexed_at for c in cs),
                source=meta.get("source"),
                tags=meta.get("tags"),
                project=first.project,
                session_id=first.session_id,
                task=first.task,
                preview=preview,
            )
        )

    captures.sort(key=lambda c: c.captured_at, reverse=True)
    return captures
