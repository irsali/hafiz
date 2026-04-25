"""Phase 2 — sessions promoted to a DB row; annotation FK populated.

Tests that ``hafiz.core.sessions`` (DB CRUD) and ``hafiz.core.session``
(per-TTY cursor) work together correctly, that slug→uuid resolution
fires from ``store_annotation``, and that the legacy slug is preserved
on annotations for human-readable journal/distill output.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from hafiz.core.annotations import store_annotation
from hafiz.core.database import close_engine
from hafiz.core.sessions import (
    create_session,
    end_session_db,
    get_session_by_id,
    get_session_by_slug,
    list_sessions,
    resolve_session_uuid,
)


@pytest.fixture(autouse=True)
async def _isolate_engine():
    await close_engine()
    yield
    await close_engine()


@pytest.mark.asyncio
async def test_create_and_lookup_session_by_slug():
    slug = f"phase-2-create-{uuid.uuid4().hex[:6]}"
    stored = await create_session(slug=slug, name="Phase 2 create")
    assert stored.slug == slug
    assert stored.name == "Phase 2 create"
    assert stored.id is not None

    found = await get_session_by_slug(slug)
    assert found is not None
    assert found.id == stored.id

    by_id = await get_session_by_id(stored.id)
    assert by_id is not None
    assert by_id.slug == slug


@pytest.mark.asyncio
async def test_list_sessions_filters_by_agent_and_scope():
    slug = f"list-test-{uuid.uuid4().hex[:6]}"
    stored = await create_session(
        slug=slug,
        name="list filter test",
        agent="claude-code",
        scope_kind="project",
        scope_value="hafiz",
    )
    rows = await list_sessions(agent="claude-code", scope_value="hafiz", limit=100)
    assert any(r.id == stored.id for r in rows)
    other_scope = await list_sessions(scope_value="not-a-real-project")
    assert all(r.id != stored.id for r in other_scope)


@pytest.mark.asyncio
async def test_resolve_session_uuid_handles_slug_uuid_and_missing():
    slug = f"resolve-test-{uuid.uuid4().hex[:6]}"
    stored = await create_session(slug=slug, name="resolve test")

    by_slug = await resolve_session_uuid(slug)
    assert by_slug == stored.id

    by_str = await resolve_session_uuid(str(stored.id))
    assert by_str == stored.id

    missing = await resolve_session_uuid("definitely-not-a-real-slug-zzzz")
    assert missing is None

    none_in = await resolve_session_uuid(None)
    assert none_in is None


@pytest.mark.asyncio
async def test_end_session_db_sets_ended_at():
    slug = f"end-test-{uuid.uuid4().hex[:6]}"
    stored = await create_session(slug=slug, name="end test")
    assert stored.ended_at is None

    ended = await end_session_db(stored.id)
    assert ended is not None
    assert ended.ended_at is not None

    # Idempotent — second call returns the same ended_at without re-stamping.
    again = await end_session_db(stored.id)
    assert again is not None
    assert again.ended_at == ended.ended_at


@pytest.mark.asyncio
async def test_store_annotation_resolves_slug_to_uuid_when_session_exists():
    """Phase 2 contract: passing a slug that matches a sessions row
    populates *both* annotations.session_id (uuid) and
    annotations.legacy_session_id (slug)."""
    slug = f"ann-resolve-{uuid.uuid4().hex[:6]}"
    stored = await create_session(slug=slug, name="ann resolve test")

    ann = await store_annotation(
        "Phase 2 annotation that should bind FK",
        kind="note",
        source="agent:test",
        session_id=slug,
    )
    assert ann.session_id == stored.id
    assert ann.legacy_session_id == slug


@pytest.mark.asyncio
async def test_store_annotation_with_unknown_slug_lands_in_legacy_only():
    """Unknown slug = no DB row exists. The slug is preserved for
    audit/display but the FK column stays NULL."""
    bogus_slug = f"unknown-slug-{uuid.uuid4().hex[:6]}"
    ann = await store_annotation(
        "Phase 2 annotation with unknown slug",
        kind="note",
        source="agent:test",
        session_id=bogus_slug,
    )
    assert ann.session_id is None
    assert ann.legacy_session_id == bogus_slug


@pytest.mark.asyncio
async def test_store_annotation_with_uuid_back_resolves_slug_for_display():
    """When a caller passes the canonical uuid, we still populate the
    slug column for human-readable journal/distill output."""
    slug = f"uuid-back-{uuid.uuid4().hex[:6]}"
    stored = await create_session(slug=slug, name="uuid back resolve")

    ann = await store_annotation(
        "Phase 2 uuid in, slug filled for display",
        kind="note",
        source="agent:test",
        session_id=stored.id,
    )
    assert ann.session_id == stored.id
    assert ann.legacy_session_id == slug
