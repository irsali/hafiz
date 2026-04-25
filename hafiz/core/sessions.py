"""DB-backed session CRUD (the source-layer ``sessions`` table).

This module owns the **persistent** half of session state. The per-TTY
JSON cursor in :mod:`hafiz.core.session` keeps track of *which* session
this terminal is currently in; the actual session record — slug, name,
agent, scope, started_at, ended_at — lives here.

Phase 2 of workitems/active/communications-and-sessions.md.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select

from hafiz.core.database import Session as SessionRow, get_session_factory


@dataclass
class StoredSession:
    id: uuid.UUID
    slug: str
    name: str | None
    agent: str | None
    scope_kind: str | None
    scope_value: str | None
    task: str | None
    tty: str | None
    started_at: datetime
    ended_at: datetime | None
    metadata: dict


def _to_stored(row: SessionRow) -> StoredSession:
    return StoredSession(
        id=row.id,
        slug=row.slug,
        name=row.name,
        agent=row.agent,
        scope_kind=row.scope_kind,
        scope_value=row.scope_value,
        task=row.task,
        tty=row.tty,
        started_at=row.started_at,
        ended_at=row.ended_at,
        metadata=row.metadata_ or {},
    )


async def create_session(
    *,
    slug: str,
    name: str | None = None,
    agent: str | None = None,
    scope_kind: str | None = None,
    scope_value: str | None = None,
    task: str | None = None,
    tty: str | None = None,
    started_at: datetime | None = None,
    metadata: dict | None = None,
) -> StoredSession:
    """Insert a new session row. Slug must be unique."""
    factory = get_session_factory()
    async with factory() as s:
        row = SessionRow(
            id=uuid.uuid4(),
            slug=slug,
            name=name,
            agent=agent,
            scope_kind=scope_kind,
            scope_value=scope_value,
            task=task,
            tty=tty,
            started_at=started_at or datetime.now(timezone.utc),
            metadata_=metadata or {},
        )
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return _to_stored(row)


async def get_session_by_slug(slug: str) -> StoredSession | None:
    """Look up a session by its human-readable slug."""
    factory = get_session_factory()
    async with factory() as s:
        result = await s.execute(
            select(SessionRow).where(SessionRow.slug == slug)
        )
        row = result.scalar_one_or_none()
        return _to_stored(row) if row else None


async def get_session_by_id(session_id: uuid.UUID) -> StoredSession | None:
    factory = get_session_factory()
    async with factory() as s:
        row = await s.get(SessionRow, session_id)
        return _to_stored(row) if row else None


async def resolve_session_uuid(slug_or_uuid: str | None) -> uuid.UUID | None:
    """Resolve a string (slug or uuid) to a session uuid via the DB.

    Returns None if the input is None/empty or the slug doesn't exist
    yet. Called from ``store_annotation`` and the importer to bind a
    user-facing slug to its persistent uuid.
    """
    if slug_or_uuid is None:
        return None
    raw = str(slug_or_uuid).strip()
    if not raw:
        return None
    try:
        return uuid.UUID(raw)
    except ValueError:
        pass
    found = await get_session_by_slug(raw)
    return found.id if found else None


async def end_session_db(session_id: uuid.UUID) -> StoredSession | None:
    """Set ``ended_at = now`` on a session and return the updated row."""
    factory = get_session_factory()
    async with factory() as s:
        row = await s.get(SessionRow, session_id)
        if row is None:
            return None
        if row.ended_at is None:
            row.ended_at = datetime.now(timezone.utc)
            await s.commit()
            await s.refresh(row)
        return _to_stored(row)


async def list_sessions(
    *,
    agent: str | None = None,
    scope_kind: str | None = None,
    scope_value: str | None = None,
    limit: int = 50,
    include_ended: bool = True,
) -> list[StoredSession]:
    factory = get_session_factory()
    async with factory() as s:
        stmt = (
            select(SessionRow)
            .order_by(SessionRow.started_at.desc())
            .limit(limit)
        )
        if agent:
            stmt = stmt.where(SessionRow.agent == agent)
        if scope_kind:
            stmt = stmt.where(SessionRow.scope_kind == scope_kind)
        if scope_value:
            stmt = stmt.where(SessionRow.scope_value == scope_value)
        if not include_ended:
            stmt = stmt.where(SessionRow.ended_at.is_(None))
        rows = (await s.execute(stmt)).scalars().all()
        return [_to_stored(r) for r in rows]


__all__ = [
    "StoredSession",
    "create_session",
    "get_session_by_slug",
    "get_session_by_id",
    "resolve_session_uuid",
    "end_session_db",
    "list_sessions",
]
