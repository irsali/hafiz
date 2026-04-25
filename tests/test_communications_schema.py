"""Phase 1 — schema + ORM round-trips for the source-layer tables.

Exercises the new ``sessions``, ``communications``, ``communication_messages``,
and ``annotation_targets`` tables, plus the ``annotations.session_id`` /
``legacy_session_id`` pivot. Tests run against a real Postgres + pgvector
instance configured via ``hafiz.toml`` — same setup other DB-touching tests
use.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from hafiz.core.communications import (
    DEFAULT_RETENTION_DAYS,
    EMBED_MIN_TOKENS,
    MessageInput,
    append_messages,
    forget_communication,
    get_communication,
    list_messages,
    should_embed_message,
    tombstone_expired_communications,
    upsert_communication,
)
from hafiz.core.database import (
    Annotation,
    AnnotationTarget,
    Communication,
    CommunicationMessage,
    Session as SessionRow,
    close_engine,
    get_session_factory,
)


@pytest.fixture(autouse=True)
async def _isolate_engine():
    """Dispose the cached engine before AND after each test so pool
    state doesn't leak across event loops (every async test gets a
    fresh asyncio loop via pytest-asyncio)."""
    await close_engine()
    yield
    await close_engine()


# ---------------------------------------------------------------------------
# Selective-embed policy
# ---------------------------------------------------------------------------


def test_should_embed_skips_short_messages():
    assert should_embed_message(role="user", content="ok") is False
    assert should_embed_message(role="assistant", content="thanks") is False
    long = "alpha beta " * 50
    assert should_embed_message(role="user", content=long) is True


def test_should_embed_marked_salient_overrides_length():
    assert should_embed_message(
        role="user", content="ok", marked_salient=True
    ) is True


def test_should_embed_skips_pure_tool_result_echo():
    big_block = "```\n" + ("x = 1\n" * 200) + "```"
    payload = "Here is the file:\n" + big_block
    assert should_embed_message(role="assistant", content=payload) is False


def test_should_embed_keeps_real_assistant_message():
    content = (
        "I'll start by reading the migration file to understand the existing "
        "shape, then we can plan the conversion path. The tricky bit is the "
        "FK ordering during the rename."
    )
    assert should_embed_message(role="assistant", content=content) is True


def test_token_threshold_constant_is_documented():
    # If this changes, callers may need to re-tune their fixture lengths.
    assert EMBED_MIN_TOKENS == 30
    assert DEFAULT_RETENTION_DAYS == 90


# ---------------------------------------------------------------------------
# Communication / message round-trips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_communication_is_idempotent():
    started = datetime.now(timezone.utc)
    ext = f"test-ext-{uuid.uuid4().hex[:8]}"
    comm1, created1 = await upsert_communication(
        agent="test-agent",
        external_id=ext,
        started_at=started,
    )
    assert created1 is True
    assert comm1.agent == "test-agent"
    assert comm1.external_id == ext
    # default retention 90 days from started_at
    assert comm1.retention_until == started + timedelta(days=DEFAULT_RETENTION_DAYS)

    comm2, created2 = await upsert_communication(
        agent="test-agent",
        external_id=ext,
        started_at=started,
    )
    assert created2 is False
    assert comm2.id == comm1.id


@pytest.mark.asyncio
async def test_append_messages_writes_with_selective_embedding():
    comm, _ = await upsert_communication(
        agent="test-agent",
        external_id=f"embed-test-{uuid.uuid4().hex[:8]}",
    )
    now = datetime.now(timezone.utc)
    messages = [
        MessageInput(seq=0, role="user", content="ok", ts=now),
        MessageInput(
            seq=1,
            role="assistant",
            content="A real explanation that is long enough to embed " * 5,
            ts=now,
        ),
        MessageInput(
            seq=2,
            role="user",
            content="hi",
            ts=now,
            marked_salient=True,  # forces embedding despite short length
        ),
    ]
    written, embedded = await append_messages(comm.id, messages)
    assert written == 3
    assert embedded == 2  # short non-salient skipped

    rows = await list_messages(comm.id)
    assert [m.seq for m in rows] == [0, 1, 2]
    assert [m.role for m in rows] == ["user", "assistant", "user"]


@pytest.mark.asyncio
async def test_append_messages_is_idempotent_per_seq():
    comm, _ = await upsert_communication(
        agent="test-agent",
        external_id=f"idem-test-{uuid.uuid4().hex[:8]}",
    )
    now = datetime.now(timezone.utc)
    payload = [
        MessageInput(seq=0, role="user", content="hello world from test", ts=now),
        MessageInput(seq=1, role="assistant", content="response back to user", ts=now),
    ]
    w1, _ = await append_messages(comm.id, payload, embed=False)
    w2, _ = await append_messages(comm.id, payload, embed=False)
    assert w1 == 2
    assert w2 == 0  # already present, skipped
    rows = await list_messages(comm.id)
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_role_check_constraint_rejects_unknown_role():
    comm, _ = await upsert_communication(
        agent="test-agent",
        external_id=f"role-test-{uuid.uuid4().hex[:8]}",
    )
    factory = get_session_factory()
    async with factory() as session:
        bad = CommunicationMessage(
            id=uuid.uuid4(),
            communication_id=comm.id,
            seq=0,
            role="not_a_real_role",
            content="x",
            ts=datetime.now(timezone.utc),
        )
        session.add(bad)
        with pytest.raises(Exception):  # IntegrityError from CHECK
            await session.commit()


# ---------------------------------------------------------------------------
# annotation_targets pivot
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_annotation_targets_pivot_basic_round_trip():
    factory = get_session_factory()
    async with factory() as session:
        ann = Annotation(
            id=uuid.uuid4(),
            content="A test decision",
            kind="decision",
            source="agent:test",
        )
        session.add(ann)
        await session.commit()
        await session.refresh(ann)

        target_id = uuid.uuid4()
        link = AnnotationTarget(
            id=uuid.uuid4(),
            annotation_id=ann.id,
            target_kind="message",
            target_id=target_id,
            relation="derived_from",
        )
        session.add(link)
        await session.commit()

        result = await session.execute(
            select(AnnotationTarget).where(
                AnnotationTarget.annotation_id == ann.id
            )
        )
        rows = list(result.scalars().all())
        assert len(rows) == 1
        assert rows[0].target_kind == "message"
        assert rows[0].relation == "derived_from"


@pytest.mark.asyncio
async def test_annotation_targets_check_constraint_rejects_bad_kind():
    factory = get_session_factory()
    async with factory() as session:
        ann = Annotation(
            id=uuid.uuid4(),
            content="A test decision (bad kind)",
            kind="decision",
            source="agent:test",
        )
        session.add(ann)
        await session.commit()
        bad = AnnotationTarget(
            id=uuid.uuid4(),
            annotation_id=ann.id,
            target_kind="not_a_real_kind",
            target_id=uuid.uuid4(),
            relation="derived_from",
        )
        session.add(bad)
        with pytest.raises(Exception):
            await session.commit()


# ---------------------------------------------------------------------------
# session_id pivot on annotations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_annotation_legacy_session_id_round_trips_strings():
    """Phase 1 keeps writing the legacy text slug — the new uuid column
    stays empty until Phase 2 wires it up."""
    from hafiz.core.annotations import store_annotation

    slug = f"phase-1-test-{uuid.uuid4().hex[:6]}"
    ann = await store_annotation(
        "Phase 1 session test annotation",
        kind="note",
        source="agent:test",
        session_id=slug,
    )
    assert ann.legacy_session_id == slug
    assert ann.session_id is None  # uuid column not populated yet


@pytest.mark.asyncio
async def test_annotation_session_id_accepts_uuid():
    """When the caller passes a uuid that matches a sessions row, the
    FK column is populated *and* the slug is back-resolved into
    legacy_session_id for human-readable journal/distill output."""
    from hafiz.core.annotations import store_annotation

    slug = f"phase-2-prep-{uuid.uuid4().hex[:6]}"
    factory = get_session_factory()
    async with factory() as s:
        sess = SessionRow(
            id=uuid.uuid4(),
            slug=slug,
            name="Phase 2 prep",
        )
        s.add(sess)
        await s.commit()
        sess_id = sess.id

    ann = await store_annotation(
        "Annotation linked via uuid FK",
        kind="note",
        source="agent:test",
        session_id=sess_id,
    )
    assert ann.session_id == sess_id
    assert ann.legacy_session_id == slug


@pytest.mark.asyncio
async def test_annotation_session_id_uuid_without_session_row_is_rejected():
    """An orphan uuid (one that doesn't match any sessions row) is
    rejected by the FK constraint at insert time. This is the right
    behavior — write-time integrity beats silent corruption."""
    from sqlalchemy.exc import IntegrityError

    from hafiz.core.annotations import store_annotation

    orphan = uuid.uuid4()
    with pytest.raises(IntegrityError):
        await store_annotation(
            "Orphan uuid annotation",
            kind="note",
            source="agent:test",
            session_id=orphan,
        )


# ---------------------------------------------------------------------------
# Retention sweeper + forget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tombstone_expired_communications():
    past_started = datetime.now(timezone.utc) - timedelta(days=120)
    expired_until = datetime.now(timezone.utc) - timedelta(days=1)
    comm, _ = await upsert_communication(
        agent="test-agent",
        external_id=f"retention-test-{uuid.uuid4().hex[:8]}",
        started_at=past_started,
        retention_until=expired_until,
    )
    result = await tombstone_expired_communications()
    assert result["matched"] >= 1
    assert result["tombstoned"] >= 1

    row = await get_communication(comm.id, include_tombstoned=True)
    assert row is not None
    assert row.valid_until is not None

    # Default get_communication hides tombstoned rows.
    row_default = await get_communication(comm.id)
    assert row_default is None


@pytest.mark.asyncio
async def test_forget_communication_soft_then_hard():
    comm, _ = await upsert_communication(
        agent="test-agent",
        external_id=f"forget-test-{uuid.uuid4().hex[:8]}",
    )
    await append_messages(
        comm.id,
        [
            MessageInput(
                seq=0,
                role="user",
                content="forget me please",
                ts=datetime.now(timezone.utc),
            )
        ],
        embed=False,
    )

    soft = await forget_communication(comm.id, hard=False)
    assert soft["found"] is True
    assert soft["hard"] is False
    row = await get_communication(comm.id, include_tombstoned=True)
    assert row is not None
    assert row.valid_until is not None

    hard = await forget_communication(comm.id, hard=True)
    assert hard["found"] is True
    assert hard["hard"] is True
    assert hard["deleted_messages"] == 1
    factory = get_session_factory()
    async with factory() as s:
        result = await s.execute(
            select(Communication).where(Communication.id == comm.id)
        )
        assert result.scalar_one_or_none() is None
