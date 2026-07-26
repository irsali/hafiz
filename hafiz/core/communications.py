"""Source-layer storage for communications (transcripts, chat threads, etc.).

This module is the home of the **selective-embed policy** and the
``communications`` / ``communication_messages`` store helpers.

Design rules (locked in workitems/active/communications-and-sessions.md):

  - **Raw is canonical, embedding is derived.** ``content`` is required
    on every message; ``embedding`` is nullable and populated only when
    the message clears the embed criteria.
  - **Selective embedding**: skip messages under ~30 tokens, skip pure
    tool-result echoes, embed at chunk level for long sessions, and
    honor the ``marked_salient`` override regardless of length.
  - **Default exclusion from query/context.** Source-layer rows surface
    only via ``hafiz recall`` or explicit opt-in flags.
  - **Bounded retention.** Communications default to ``started_at + 90
    days`` for ``retention_until`` unless explicitly overridden.
  - **Append-only**. Messages, once written, are immutable.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select

from hafiz.core.database import (
    Communication,
    CommunicationMessage,
    get_session_factory,
)
from hafiz.core.embeddings import embed_query

# Tunables — kept module-level for now; promote to the tunable registry
# (hafiz/core/tunables.py) when query patterns demand per-host probing.
DEFAULT_RETENTION_DAYS = 90
EMBED_MIN_TOKENS = 30  # below this, skip embedding (degenerate vectors)
TOKEN_RATIO = 4  # rough chars-per-token; we don't need precision here


# ---------------------------------------------------------------------------
# Selective-embed policy
# ---------------------------------------------------------------------------


def _approx_token_count(text: str) -> int:
    """Cheap whitespace-based token approximation. Good enough for thresholds."""
    if not text:
        return 0
    return max(1, len(text) // TOKEN_RATIO)


_TOOL_RESULT_DOMINANCE_RE = re.compile(r"```[\s\S]+?```|<file[^>]*>[\s\S]+?</file>", re.MULTILINE)


def _is_pure_tool_result_echo(role: str, content: str) -> bool:
    """Detect messages that are dominated by a single tool result.

    Heuristic: the message is from the assistant or tool role, and 80%+
    of its content is taken by a single fenced code block or file
    payload. Such rows are already represented in ``units`` for the
    file they reference; double-indexing them dilutes vector search.
    """
    if role not in ("assistant", "tool"):
        return False
    total = len(content or "")
    if total < 200:
        return False
    matches = list(_TOOL_RESULT_DOMINANCE_RE.finditer(content))
    if not matches:
        return False
    largest = max((m.end() - m.start()) for m in matches)
    return largest >= 0.8 * total


def should_embed_message(
    *,
    role: str,
    content: str,
    marked_salient: bool = False,
) -> bool:
    """Apply the selective-embed policy.

    Returns True if this message should be embedded. ``marked_salient``
    overrides every other rule.
    """
    if marked_salient:
        return True
    if not content or not content.strip():
        return False
    if _approx_token_count(content) < EMBED_MIN_TOKENS:
        return False
    if _is_pure_tool_result_echo(role, content):
        return False
    return True


# ---------------------------------------------------------------------------
# Communication / message store helpers
# ---------------------------------------------------------------------------


@dataclass
class MessageInput:
    """A single message about to be stored. ``ts`` should be set by
    the caller — we keep ingest deterministic by trusting source data.

    ``id`` may be prescribed by the caller (e.g. the importer needs
    stable ids so subsequent messages can reference them via
    ``parent_message_id``). When None, the store helper generates a
    fresh uuid.
    """

    seq: int
    role: str
    content: str
    ts: datetime
    id: uuid.UUID | None = None
    author: str | None = None
    content_type: str = "text/markdown"
    tool_calls: list | None = None
    parent_message_id: uuid.UUID | None = None
    marked_salient: bool = False
    metadata: dict | None = None


@dataclass
class StoredCommunication:
    """Lightweight view of a stored communication for callers."""

    id: uuid.UUID
    session_id: uuid.UUID | None
    external_id: str | None
    agent: str
    started_at: datetime
    message_count: int
    embedded_count: int
    skipped_existing: bool


def _default_retention(started_at: datetime) -> datetime:
    return started_at + timedelta(days=DEFAULT_RETENTION_DAYS)


async def upsert_communication(
    *,
    agent: str,
    external_id: str | None = None,
    session_id: uuid.UUID | None = None,
    channel: str | None = None,
    participants: list | None = None,
    scope_kind: str | None = None,
    scope_value: str | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    retention_until: datetime | None = None,
    metadata: dict | None = None,
) -> tuple[Communication, bool]:
    """Insert a communication row, or return the existing one when
    ``(agent, external_id)`` already maps. Returns ``(row, created)``.

    Idempotency by ``(agent, external_id)`` — re-importing the same
    Claude Code session JSONL is a no-op at this layer.
    """
    started_at = started_at or datetime.now(UTC)
    retention = retention_until or _default_retention(started_at)

    factory = get_session_factory()
    async with factory() as session:
        if external_id is not None:
            existing = await session.execute(
                select(Communication).where(
                    Communication.agent == agent,
                    Communication.external_id == external_id,
                )
            )
            row = existing.scalar_one_or_none()
            if row is not None:
                return row, False

        row = Communication(
            id=uuid.uuid4(),
            session_id=session_id,
            external_id=external_id,
            agent=agent,
            channel=channel,
            participants=participants or [],
            scope_kind=scope_kind,
            scope_value=scope_value,
            started_at=started_at,
            ended_at=ended_at,
            retention_until=retention,
            metadata_=metadata or {},
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row, True


async def append_messages(
    communication_id: uuid.UUID,
    messages: Sequence[MessageInput],
    *,
    embed: bool = True,
) -> tuple[int, int]:
    """Append ``messages`` to ``communication_id`` in seq order.

    Applies the selective-embed policy when ``embed`` is True. Skips
    rows whose ``(communication_id, seq)`` already exist (idempotent
    re-import). Returns ``(written, embedded)``.
    """
    if not messages:
        return 0, 0

    factory = get_session_factory()
    written = 0
    embedded = 0

    async with factory() as session:
        # Find which seqs are already present so re-import is idempotent.
        existing_seqs = set()
        if messages:
            seqs = [m.seq for m in messages]
            result = await session.execute(
                select(CommunicationMessage.seq).where(
                    CommunicationMessage.communication_id == communication_id,
                    CommunicationMessage.seq.in_(seqs),
                )
            )
            existing_seqs = {row for (row,) in result.all()}

        # Track in-memory ids that we *won't* write (existing-seq skip).
        # If a later message uses one of these as its
        # parent_message_id, that pointer is invalid (the prescribed
        # uuid never lands in the DB; the previously-imported row has
        # a different id). Reset such pointers to None — parent
        # linkage is best-effort per the work item.
        skipped_ids: set[uuid.UUID] = set()

        for msg in messages:
            if msg.seq in existing_seqs:
                if msg.id is not None:
                    skipped_ids.add(msg.id)
                continue
            embedding = None
            if embed and should_embed_message(
                role=msg.role,
                content=msg.content,
                marked_salient=msg.marked_salient,
            ):
                embedding = await embed_query(msg.content)
                embedded += 1

            parent_id = msg.parent_message_id
            if parent_id is not None and parent_id in skipped_ids:
                parent_id = None

            session.add(
                CommunicationMessage(
                    id=msg.id or uuid.uuid4(),
                    communication_id=communication_id,
                    seq=msg.seq,
                    role=msg.role,
                    author=msg.author,
                    content=msg.content,
                    content_type=msg.content_type,
                    tool_calls=msg.tool_calls,
                    parent_message_id=parent_id,
                    ts=msg.ts,
                    embedding=embedding,
                    marked_salient=msg.marked_salient,
                    metadata_=msg.metadata or {},
                )
            )
            written += 1

        await session.commit()

    return written, embedded


# ---------------------------------------------------------------------------
# Recall — read-side helpers (Phase 4 layers a CLI on top of these)
# ---------------------------------------------------------------------------


@dataclass
class MessageRow:
    id: str
    communication_id: str
    seq: int
    role: str
    author: str | None
    content: str
    ts: datetime
    tool_calls: list | None
    marked_salient: bool
    metadata: dict


@dataclass
class CommunicationRow:
    id: str
    session_id: str | None
    external_id: str | None
    agent: str
    channel: str | None
    started_at: datetime
    ended_at: datetime | None
    retention_until: datetime | None
    valid_until: datetime | None
    message_count: int
    metadata: dict


async def get_communication(
    comm_id: uuid.UUID, *, include_tombstoned: bool = False
) -> CommunicationRow | None:
    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(Communication, comm_id)
        if row is None:
            return None
        if not include_tombstoned and row.valid_until is not None:
            now = datetime.now(UTC)
            if row.valid_until <= now:
                return None
        count_stmt = select(func.count()).where(CommunicationMessage.communication_id == row.id)
        count = (await session.execute(count_stmt)).scalar() or 0
        return CommunicationRow(
            id=str(row.id),
            session_id=str(row.session_id) if row.session_id else None,
            external_id=row.external_id,
            agent=row.agent,
            channel=row.channel,
            started_at=row.started_at,
            ended_at=row.ended_at,
            retention_until=row.retention_until,
            valid_until=row.valid_until,
            message_count=count,
            metadata=row.metadata_ or {},
        )


async def list_messages(
    communication_id: uuid.UUID,
    *,
    role: str | None = None,
    has_tool_call: bool | None = None,
    seq_from: int | None = None,
    seq_to: int | None = None,
    limit: int = 1000,
) -> list[MessageRow]:
    factory = get_session_factory()
    async with factory() as session:
        stmt = (
            select(CommunicationMessage)
            .where(CommunicationMessage.communication_id == communication_id)
            .order_by(CommunicationMessage.seq.asc())
            .limit(limit)
        )
        if role:
            stmt = stmt.where(CommunicationMessage.role == role)
        if has_tool_call is True:
            stmt = stmt.where(CommunicationMessage.tool_calls.isnot(None))
        elif has_tool_call is False:
            stmt = stmt.where(CommunicationMessage.tool_calls.is_(None))
        if seq_from is not None:
            stmt = stmt.where(CommunicationMessage.seq >= seq_from)
        if seq_to is not None:
            stmt = stmt.where(CommunicationMessage.seq <= seq_to)

        rows = (await session.execute(stmt)).scalars().all()
        return [
            MessageRow(
                id=str(r.id),
                communication_id=str(r.communication_id),
                seq=r.seq,
                role=r.role,
                author=r.author,
                content=r.content,
                ts=r.ts,
                tool_calls=r.tool_calls,
                marked_salient=r.marked_salient,
                metadata=r.metadata_ or {},
            )
            for r in rows
        ]


async def search_messages(
    query: str,
    *,
    limit: int = 10,
    agent: str | None = None,
    session_id: uuid.UUID | None = None,
    communication_id: uuid.UUID | None = None,
) -> list[tuple[MessageRow, float]]:
    """Vector search over message embeddings.

    Default callers (``hafiz query``, ``hafiz context``) MUST NOT call
    this — it's the implementation behind ``hafiz recall`` and the
    ``--include-transcripts`` opt-in. The wisdom layer must remain
    primary by default.
    """
    query_embedding = await embed_query(query)
    similarity = (1 - CommunicationMessage.embedding.cosine_distance(query_embedding)).label(
        "similarity"
    )

    factory = get_session_factory()
    async with factory() as session:
        stmt = (
            select(CommunicationMessage, similarity)
            .where(CommunicationMessage.embedding.isnot(None))
            .order_by(CommunicationMessage.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )
        if communication_id is not None:
            stmt = stmt.where(CommunicationMessage.communication_id == communication_id)
        elif agent is not None or session_id is not None:
            stmt = stmt.join(
                Communication,
                Communication.id == CommunicationMessage.communication_id,
            )
            if agent is not None:
                stmt = stmt.where(Communication.agent == agent)
            if session_id is not None:
                stmt = stmt.where(Communication.session_id == session_id)

        rows = (await session.execute(stmt)).all()

    out: list[tuple[MessageRow, float]] = []
    for msg, sim in rows:
        out.append(
            (
                MessageRow(
                    id=str(msg.id),
                    communication_id=str(msg.communication_id),
                    seq=msg.seq,
                    role=msg.role,
                    author=msg.author,
                    content=msg.content,
                    ts=msg.ts,
                    tool_calls=msg.tool_calls,
                    marked_salient=msg.marked_salient,
                    metadata=msg.metadata_ or {},
                ),
                round(float(sim), 4),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Retention sweeper
# ---------------------------------------------------------------------------


async def count_overdue_communications(*, now: datetime | None = None) -> int:
    """Count communications past ``retention_until`` and not yet tombstoned.

    Bounded retention is an outward-facing commitment, and it was only enforced
    when someone remembered to run ``hafiz forget --all-expired``. 358 rows sat
    overdue for four weeks in a real deployment because nothing ever said so.
    Visibility is the actual fix — a trigger that only fires on ``import`` stops
    firing precisely when imports stop, while retention keeps ticking.

    Cheap enough (indexed on ``retention_until``) to call from ``status`` and
    ``doctor`` unconditionally.
    """
    now = now or datetime.now(UTC)
    factory = get_session_factory()
    async with factory() as session:
        stmt = (
            select(func.count())
            .select_from(Communication)
            .where(
                Communication.retention_until.isnot(None),
                Communication.retention_until <= now,
                Communication.valid_until.is_(None),
            )
        )
        return (await session.execute(stmt)).scalar() or 0


async def tombstone_expired_communications(
    *, now: datetime | None = None, dry_run: bool = False
) -> dict:
    """Tombstone communications whose ``retention_until`` has passed.

    Sets ``valid_until = now`` on each row whose ``retention_until <= now``
    and which is not already tombstoned. Messages are kept (cascade
    delete is not used) — they remain queryable for explicit ``hafiz
    forget --hard`` operations.

    Returns ``{"matched": N, "tombstoned": M, "dry_run": bool}``.
    """
    now = now or datetime.now(UTC)
    factory = get_session_factory()
    async with factory() as session:
        stmt = select(Communication).where(
            Communication.retention_until.isnot(None),
            Communication.retention_until <= now,
            Communication.valid_until.is_(None),
        )
        rows = list((await session.execute(stmt)).scalars().all())
        matched = len(rows)
        tombstoned = 0
        if not dry_run:
            for row in rows:
                row.valid_until = now
                tombstoned += 1
            await session.commit()
        return {
            "matched": matched,
            "tombstoned": tombstoned,
            "dry_run": dry_run,
        }


async def forget_communication(comm_id: uuid.UUID, *, hard: bool = False) -> dict:
    """Explicit redaction of a communication.

    ``hard=False`` (default): tombstone only (sets ``valid_until = now``;
    rows survive for audit). ``hard=True``: delete the communication
    and cascade-delete all messages.

    Returns ``{"id": ..., "deleted_messages": N, "hard": bool}``.
    """
    factory = get_session_factory()
    async with factory() as session:
        row = await session.get(Communication, comm_id)
        if row is None:
            return {"id": str(comm_id), "deleted_messages": 0, "found": False}
        if hard:
            count_stmt = select(func.count()).where(
                CommunicationMessage.communication_id == comm_id
            )
            count = (await session.execute(count_stmt)).scalar() or 0
            await session.delete(row)
            await session.commit()
            return {
                "id": str(comm_id),
                "deleted_messages": int(count),
                "hard": True,
                "found": True,
            }
        row.valid_until = datetime.now(UTC)
        await session.commit()
        return {
            "id": str(comm_id),
            "deleted_messages": 0,
            "hard": False,
            "found": True,
        }


__all__ = [
    "MessageInput",
    "MessageRow",
    "CommunicationRow",
    "StoredCommunication",
    "should_embed_message",
    "upsert_communication",
    "append_messages",
    "get_communication",
    "list_messages",
    "search_messages",
    "count_overdue_communications",
    "tombstone_expired_communications",
    "forget_communication",
    "DEFAULT_RETENTION_DAYS",
    "EMBED_MIN_TOKENS",
]
