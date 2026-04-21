"""Time-bounded digest over observations — the ``hafiz journal`` feature.

A lightweight view layer: "what did I record between X and Y, grouped by day".
Reuses the existing ``observations`` table — no new storage, no migration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from hafiz.core.database import Observation, get_session_factory


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
    metadata: dict


@dataclass
class JournalBundle:
    window_start: datetime
    window_end: datetime
    entries: list[JournalEntry] = field(default_factory=list)

    def grouped_by_day(self) -> list[tuple[str, list[JournalEntry]]]:
        """Return [(YYYY-MM-DD, entries), ...] with the newest day first."""
        buckets: dict[str, list[JournalEntry]] = {}
        for e in self.entries:
            day = e.valid_from.astimezone(timezone.utc).strftime("%Y-%m-%d")
            buckets.setdefault(day, []).append(e)
        return sorted(buckets.items(), reverse=True)


async def build_journal(
    *,
    since: timedelta | None = None,
    day: datetime | None = None,
    project: str | list[str] | None = None,
    source: str | None = None,
    obs_type: str | None = None,
    limit: int = 500,
) -> JournalBundle:
    """Build a time-bounded journal bundle.

    Window rules:
      - If ``day`` is given, the window is that UTC day (00:00 → 23:59:59.999999).
      - Otherwise the window is ``[now - since, now]``; ``since`` defaults to 7d.
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
            metadata=o.metadata_ or {},
        )
        for o in rows
    ]

    return JournalBundle(window_start=start, window_end=end, entries=entries)
