"""Phase 5 — polymorphic ``derived_from`` via the annotation_targets pivot.

Verifies that ``hafiz observe --derived-from <ids>`` writes
annotation_targets rows with the correct ``target_kind``, regardless
of whether the cited id points at an annotation or a source-layer
message. Also verifies that ``hafiz distill`` enriches its output
with message candidates when a session has communications.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from hafiz.core.annotations import store_annotation
from hafiz.core.communications import (
    MessageInput,
    append_messages,
    upsert_communication,
)
from hafiz.core.database import (
    AnnotationTarget,
    close_engine,
    get_session_factory,
)
from hafiz.core.distill import find_distill_candidates
from hafiz.core.sessions import create_session


@pytest.fixture(autouse=True)
async def _isolate_engine():
    # Drop any engine cached by a previous (cross-file) test so this
    # test's loop owns its own asyncpg pool.
    await close_engine()
    yield
    await close_engine()


@pytest.mark.asyncio
async def test_derived_from_classifies_annotation_targets():
    """A note distilled from another annotation produces a pivot row
    with target_kind='annotation'."""
    parent = await store_annotation(
        "Parent note that will be cited",
        kind="note",
        source="agent:test",
    )
    child = await store_annotation(
        "Distilled decision citing parent annotation",
        kind="decision",
        source="agent:test",
        derived_from=[str(parent.id)],
    )
    factory = get_session_factory()
    async with factory() as s:
        rows = (
            (
                await s.execute(
                    select(AnnotationTarget).where(AnnotationTarget.annotation_id == child.id)
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1
    assert rows[0].target_kind == "annotation"
    assert rows[0].target_id == parent.id
    assert rows[0].relation == "derived_from"


@pytest.mark.asyncio
async def test_derived_from_classifies_message_targets():
    """A decision distilled from communication_messages produces
    pivot rows with target_kind='message'."""
    sess = await create_session(
        slug=f"phase-5-msg-{uuid.uuid4().hex[:6]}",
        name="Phase 5 message lineage",
    )
    comm, _ = await upsert_communication(
        agent="claude-code",
        external_id=f"phase-5-{uuid.uuid4().hex[:8]}",
        session_id=sess.id,
    )
    now = datetime.now(UTC)
    await append_messages(
        comm.id,
        [
            MessageInput(
                seq=0,
                role="user",
                content="A user turn that drives the decision in detail",
                ts=now,
            ),
            MessageInput(
                seq=1,
                role="assistant",
                content="An assistant turn proposing the chosen approach",
                ts=now,
                author="claude-opus-4-7",
            ),
        ],
        embed=False,
    )
    factory = get_session_factory()
    async with factory() as s:
        from hafiz.core.database import CommunicationMessage

        msg_ids = [
            str(r[0])
            for r in (
                await s.execute(
                    select(CommunicationMessage.id).where(
                        CommunicationMessage.communication_id == comm.id
                    )
                )
            ).all()
        ]
    assert len(msg_ids) == 2

    decision = await store_annotation(
        "Decision distilled directly from two turns",
        kind="decision",
        source="agent:test",
        derived_from=msg_ids,
    )

    async with factory() as s:
        rows = (
            (
                await s.execute(
                    select(AnnotationTarget).where(AnnotationTarget.annotation_id == decision.id)
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 2
    assert {r.target_kind for r in rows} == {"message"}
    assert {str(r.target_id) for r in rows} == set(msg_ids)


@pytest.mark.asyncio
async def test_derived_from_unknown_uuid_is_skipped_not_raised():
    """An unknown uuid in --derived-from is recorded but does not
    block the annotation write itself. Lineage is best-effort."""
    bogus = str(uuid.uuid4())
    ann = await store_annotation(
        "Decision with one bogus and one real lineage id",
        kind="decision",
        source="agent:test",
        derived_from=[bogus],
    )
    factory = get_session_factory()
    async with factory() as s:
        rows = (
            (
                await s.execute(
                    select(AnnotationTarget).where(AnnotationTarget.annotation_id == ann.id)
                )
            )
            .scalars()
            .all()
        )
    # Bogus uuid skipped — no row written.
    assert rows == []


@pytest.mark.asyncio
async def test_derived_from_metadata_still_recorded_for_back_compat():
    """The legacy ``metadata.derived_from`` list stays populated so
    existing readers (older agents, log inspection) keep working
    during the transition."""
    parent = await store_annotation(
        "Parent for back-compat check",
        kind="note",
        source="agent:test",
    )
    child = await store_annotation(
        "Child with both metadata + pivot lineage",
        kind="decision",
        source="agent:test",
        derived_from=[str(parent.id)],
    )
    assert (child.metadata_ or {}).get("derived_from") == [str(parent.id)]


@pytest.mark.asyncio
async def test_distill_surfaces_message_candidates_for_session():
    """When a session has communications, distill returns
    MessageCandidate rows with seq + role + ts so a follow-up
    ``hafiz observe ... --derived-from <ids>`` can cite the actual
    turns."""
    slug = f"phase-5-distill-{uuid.uuid4().hex[:6]}"
    sess = await create_session(slug=slug, name="Phase 5 distill")
    comm, _ = await upsert_communication(
        agent="claude-code",
        external_id=f"phase-5-distill-{uuid.uuid4().hex[:8]}",
        session_id=sess.id,
        scope_kind="project",
        scope_value="hafiz-test",
    )
    now = datetime.now(UTC)
    await append_messages(
        comm.id,
        [
            MessageInput(
                seq=0,
                role="user",
                content="A long user question with enough content to be salient",
                ts=now,
            ),
            MessageInput(
                seq=1,
                role="assistant",
                content="A long assistant answer that articulates the chosen approach",
                ts=now,
                author="claude-opus-4-7",
            ),
        ],
        embed=False,
    )

    bundle = await find_distill_candidates(session_id=slug)
    assert any(m.seq == 0 and m.role == "user" for m in bundle.messages)
    assert any(m.seq == 1 and m.role == "assistant" for m in bundle.messages)


@pytest.mark.asyncio
async def test_distill_unknown_session_returns_no_messages():
    """An unresolvable session_slug shouldn't surface every message
    in the window — bundle.messages stays empty."""
    bundle = await find_distill_candidates(session_id=f"unknown-session-{uuid.uuid4().hex[:6]}")
    assert bundle.messages == []
