"""Smoke tests for the seven-table structural-grounding schema.

Covers basic CRUD + the load-bearing constraints on each table:
  - units.identity_key UNIQUE
  - unit_revisions partial unique on current revision
  - embeddings (unit_revision_id, part_index) UNIQUE
  - CHECK on unit_revisions.source and edges.source
  - FK cascade semantics

Integration tests — skip gracefully when no DB is reachable.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from hafiz.core.database import (
    Annotation,
    Commit,
    Edge,
    Embedding,
    File,
    Unit,
    UnitRevision,
    close_engine,
    get_session_factory,
)


async def _db_available() -> bool:
    try:
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text("SELECT 1 FROM units LIMIT 1"))
        return True
    except Exception:
        return False


async def _cleanup():
    """Clear all rows from the schema. Called before each test."""
    factory = get_session_factory()
    async with factory() as s:
        # Child-first so FK cascades aren't triggered unexpectedly.
        await s.execute(text("DELETE FROM annotations"))
        await s.execute(text("DELETE FROM edges"))
        await s.execute(text("DELETE FROM embeddings"))
        await s.execute(text("DELETE FROM unit_revisions"))
        await s.execute(text("DELETE FROM units"))
        await s.execute(text("DELETE FROM files"))
        await s.execute(text("DELETE FROM commits"))
        await s.commit()


@pytest.fixture(autouse=True)
async def _maybe_skip_and_clean():
    """Skip if no DB; otherwise clean state before the test."""
    if not await _db_available():
        pytest.skip("Postgres not reachable")
    await _cleanup()
    yield
    await close_engine()


@pytest.mark.asyncio
async def test_commit_crud():
    factory = get_session_factory()
    async with factory() as s:
        c = Commit(hash="abc123", project="hafiz", author="irshad", summary="t")
        s.add(c)
        await s.commit()

        got = await s.get(Commit, "abc123")
        assert got is not None
        assert got.summary == "t"


@pytest.mark.asyncio
async def test_file_unique_project_path():
    factory = get_session_factory()
    async with factory() as s:
        s.add(File(project="hafiz", path="/a.py"))
        await s.commit()

        s.add(File(project="hafiz", path="/a.py"))
        with pytest.raises(IntegrityError):
            await s.commit()


@pytest.mark.asyncio
async def test_unit_identity_key_unique():
    factory = get_session_factory()
    async with factory() as s:
        f = File(project="hafiz", path="/a.py")
        s.add(f)
        await s.flush()

        u1 = Unit(
            file_id=f.id,
            kind="code.function",
            name="foo",
            identity_key="k1",
        )
        s.add(u1)
        await s.commit()

        u2 = Unit(
            file_id=f.id,
            kind="code.function",
            name="bar",
            identity_key="k1",
        )
        s.add(u2)
        with pytest.raises(IntegrityError):
            await s.commit()


@pytest.mark.asyncio
async def test_unit_revision_current_partial_unique():
    """At most one current revision (superseded_at IS NULL) per unit."""
    factory = get_session_factory()
    async with factory() as s:
        f = File(project="hafiz", path="/a.py")
        s.add(f)
        await s.flush()
        u = Unit(
            file_id=f.id, kind="code.function", name="foo", identity_key="k"
        )
        s.add(u)
        await s.flush()

        r1 = UnitRevision(
            unit_id=u.id, content="v1", content_hash="h1", source="ast"
        )
        s.add(r1)
        await s.commit()

        r2 = UnitRevision(
            unit_id=u.id, content="v2", content_hash="h2", source="ast"
        )
        s.add(r2)
        with pytest.raises(IntegrityError):
            await s.commit()


@pytest.mark.asyncio
async def test_unit_revision_source_check():
    """CHECK constraint rejects unknown source values."""
    factory = get_session_factory()
    async with factory() as s:
        f = File(project="hafiz", path="/a.py")
        s.add(f)
        await s.flush()
        u = Unit(
            file_id=f.id, kind="code.function", name="foo", identity_key="k"
        )
        s.add(u)
        await s.flush()

        bad = UnitRevision(
            unit_id=u.id,
            content="v",
            content_hash="h",
            source="bogus",
        )
        s.add(bad)
        with pytest.raises(IntegrityError):
            await s.commit()


@pytest.mark.asyncio
async def test_supersession_then_new_current():
    """After superseding, a new current revision is allowed."""
    factory = get_session_factory()
    async with factory() as s:
        f = File(project="hafiz", path="/a.py")
        s.add(f)
        await s.flush()
        u = Unit(
            file_id=f.id, kind="code.function", name="foo", identity_key="k"
        )
        s.add(u)
        await s.flush()

        r1 = UnitRevision(
            unit_id=u.id, content="v1", content_hash="h1", source="ast"
        )
        s.add(r1)
        await s.commit()

        # Mark r1 superseded.
        await s.execute(
            text(
                "UPDATE unit_revisions SET superseded_at = now() "
                "WHERE id = :id"
            ),
            {"id": r1.id},
        )
        await s.commit()

        r2 = UnitRevision(
            unit_id=u.id, content="v2", content_hash="h2", source="ast"
        )
        s.add(r2)
        await s.commit()  # must not raise


@pytest.mark.asyncio
async def test_embedding_revision_part_unique_and_cascade():
    factory = get_session_factory()
    async with factory() as s:
        f = File(project="hafiz", path="/a.py")
        s.add(f)
        await s.flush()
        u = Unit(
            file_id=f.id, kind="code.function", name="foo", identity_key="k"
        )
        s.add(u)
        await s.flush()
        r = UnitRevision(
            unit_id=u.id, content="v", content_hash="h", source="ast"
        )
        s.add(r)
        await s.flush()

        s.add(
            Embedding(
                unit_revision_id=r.id,
                part_index=0,
                content="v",
                content_hash="h",
            )
        )
        await s.commit()

        # Same (revision, part_index) should conflict.
        s.add(
            Embedding(
                unit_revision_id=r.id,
                part_index=0,
                content="v",
                content_hash="h",
            )
        )
        with pytest.raises(IntegrityError):
            await s.commit()
        await s.rollback()

        # Cascade delete: removing the revision deletes embeddings.
        await s.delete(r)
        await s.commit()

        count = await s.execute(text("SELECT COUNT(*) FROM embeddings"))
        assert count.scalar() == 0


@pytest.mark.asyncio
async def test_edge_source_check():
    factory = get_session_factory()
    async with factory() as s:
        f = File(project="hafiz", path="/a.py")
        s.add(f)
        await s.flush()
        u = Unit(
            file_id=f.id, kind="code.function", name="foo", identity_key="k"
        )
        s.add(u)
        await s.flush()

        bad = Edge(
            source_unit_id=u.id,
            relation="calls",
            source="parser",  # valid for unit_revisions, NOT for edges
        )
        s.add(bad)
        with pytest.raises(IntegrityError):
            await s.commit()


@pytest.mark.asyncio
async def test_edge_unresolved_target():
    """target_unit_id may be null; target_name carries the unresolved ref."""
    factory = get_session_factory()
    async with factory() as s:
        f = File(project="hafiz", path="/a.py")
        s.add(f)
        await s.flush()
        u = Unit(
            file_id=f.id, kind="code.function", name="foo", identity_key="k"
        )
        s.add(u)
        await s.flush()

        e = Edge(
            source_unit_id=u.id,
            target_unit_id=None,
            target_name="requests.get",
            relation="calls",
            source="ast",
        )
        s.add(e)
        await s.commit()

        assert e.id is not None


@pytest.mark.asyncio
async def test_annotation_free_and_linked():
    factory = get_session_factory()
    async with factory() as s:
        f = File(project="hafiz", path="/a.py")
        s.add(f)
        await s.flush()
        u = Unit(
            file_id=f.id, kind="code.function", name="foo", identity_key="k"
        )
        s.add(u)
        await s.flush()

        a_free = Annotation(
            content="free-floating decision",
            kind="decision",
            source="user:irshad",
        )
        a_linked = Annotation(
            content="this function handles auth",
            kind="fact",
            source="agent:claude-code",
            unit_id=u.id,
        )
        s.add_all([a_free, a_linked])
        await s.commit()

        count = await s.execute(text("SELECT COUNT(*) FROM annotations"))
        assert count.scalar() == 2


@pytest.mark.asyncio
async def test_file_cascade_to_units_and_revisions():
    """Deleting a file cascades to its units and their revisions."""
    factory = get_session_factory()
    async with factory() as s:
        f = File(project="hafiz", path="/a.py")
        s.add(f)
        await s.flush()
        u = Unit(
            file_id=f.id, kind="code.function", name="foo", identity_key="k"
        )
        s.add(u)
        await s.flush()
        r = UnitRevision(
            unit_id=u.id, content="v", content_hash="h", source="ast"
        )
        s.add(r)
        await s.commit()

        await s.delete(f)
        await s.commit()

        assert (
            await s.execute(text("SELECT COUNT(*) FROM units"))
        ).scalar() == 0
        assert (
            await s.execute(text("SELECT COUNT(*) FROM unit_revisions"))
        ).scalar() == 0
