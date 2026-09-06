"""Moving a whole store between backends.

This command moves someone's entire second brain, and the failure that
matters is not a crash — it is a copy that *looks* like it worked. So the
tests here are weighted towards the things a green exit code does not prove:
that self-referencing rows still point where they pointed, that retention and
tombstone timestamps were not quietly recomputed, and that the vector
verification can actually fail.

Validated once against the real store before these were written: 421,732 rows
Postgres to SQLite in 81s, 71,194 back-references restored, vectors round-
tripping with a maximum element delta of 0.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, text

from hafiz.core.database import (
    Annotation,
    Communication,
    close_engine,
    get_session_factory,
)
from hafiz.core.migrate import (
    SELF_REFERENCES,
    MigrationError,
    _open,
    _verify_vector,
    migrate_backend,
)

SOURCE_TAG = "agent:test-migrate"
PROJECT = "migrate-test-project"


async def _mock_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic vectors, so the round-trip check compares known values."""
    import hashlib

    return [
        [(b - 128) / 128.0 for b in (hashlib.sha256(t.encode()).digest() * 24)[:768]] for t in texts
    ]


@pytest.fixture(autouse=True)
async def _engine_per_test():
    yield
    async with get_session_factory()() as s:
        await s.execute(text("DELETE FROM annotations WHERE source = :s"), {"s": SOURCE_TAG})
        await s.execute(text("DELETE FROM communications WHERE agent = :a"), {"a": "test-migrate"})
        await s.execute(
            text(
                "DELETE FROM embeddings WHERE unit_revision_id IN (SELECT ur.id FROM "
                "unit_revisions ur JOIN units u ON u.id = ur.unit_id JOIN files f ON "
                "f.id = u.file_id WHERE f.project = :p)"
            ),
            {"p": PROJECT},
        )
        await s.execute(
            text(
                "DELETE FROM unit_revisions WHERE unit_id IN (SELECT u.id FROM units u "
                "JOIN files f ON f.id = u.file_id WHERE f.project = :p)"
            ),
            {"p": PROJECT},
        )
        await s.execute(
            text("DELETE FROM units WHERE file_id IN (SELECT id FROM files WHERE project = :p)"),
            {"p": PROJECT},
        )
        await s.execute(text("DELETE FROM files WHERE project = :p"), {"p": PROJECT})
        await s.commit()
    await close_engine()


@pytest.fixture
def target_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'migrated.db'}"


async def _configured_url() -> str:
    from hafiz.core.config import get_settings

    return get_settings().database.url


async def _seed() -> tuple[uuid.UUID, uuid.UUID, datetime]:
    """One superseded annotation pair, and a communication with a retention window."""
    now = datetime.now(UTC)
    retention = now + timedelta(days=17)  # deliberately not the 90-day default
    old_id, new_id = uuid.uuid4(), uuid.uuid4()

    async with get_session_factory()() as s:
        s.add(
            Annotation(
                id=old_id,
                content="the superseded belief",
                kind="decision",
                source=SOURCE_TAG,
                valid_from=now,
                valid_until=now,  # tombstoned
            )
        )
        await s.flush()
        s.add(
            Annotation(
                id=new_id,
                content="the replacement belief",
                kind="decision",
                source=SOURCE_TAG,
                valid_from=now,
                supersedes_id=old_id,
            )
        )
        s.add(
            Communication(
                id=uuid.uuid4(),
                agent="test-migrate",
                external_id=str(uuid.uuid4()),
                started_at=now,
                retention_until=retention,
            )
        )
        await s.commit()

    # Real units + embeddings, so the vector round-trip has something to check
    # and the source is never empty. Without this the suite passed on an empty
    # SQLite leg by copying nothing into nothing.
    from pathlib import Path

    from hafiz.core.store import index_file

    await index_file(
        Path("/tmp/migrate_fixture.py"),
        "def alpha():\n    return 1\n\n\ndef beta():\n    return 2\n",
        project=PROJECT,
        embed_fn=_mock_embed,
    )
    return old_id, new_id, retention


async def test_a_migration_copies_everything_and_says_so(target_url):
    """Row counts must be verified by the command, not by the caller's optimism."""
    await _seed()
    result = await migrate_backend(source_url=await _configured_url(), target_url=target_url)

    assert result.ok
    assert result.total_rows > 0
    assert {t.name for t in result.tables} >= {"annotations", "communications", "embeddings"}
    for entry in result.tables:
        assert entry.copied == entry.source_rows, f"{entry.name} lost rows"


async def test_self_references_survive_the_two_pass_copy(target_url):
    """The whole reason for a second pass.

    ``supersedes_id`` points at another row of the same table, so a row can
    reference one not yet inserted. Copied naively it is either an FK
    violation or — worse — a NULL nobody notices, which silently un-retires a
    belief that had been superseded.
    """
    old_id, new_id, _ = await _seed()
    await migrate_backend(source_url=await _configured_url(), target_url=target_url)

    engine, factory = _open(target_url)
    try:
        async with factory() as s:
            landed = (
                await s.execute(select(Annotation).where(Annotation.id == new_id))
            ).scalar_one()
            assert landed.supersedes_id == old_id, (
                "the supersedes link was lost in transit; the replaced belief would "
                "read as live again"
            )
    finally:
        await engine.dispose()


async def test_retention_and_tombstones_are_copied_not_recomputed(target_url):
    """The compliance property, and the one a plausible implementation gets wrong.

    Deriving ``retention_until`` from ``started_at + 90 days`` on write would
    look correct and would silently extend the retention window of every
    migrated communication — a regulator-visible change from a command that
    presents itself as plumbing. The seeded window is 17 days precisely so a
    recomputed 90 would stand out.
    """
    _old, _new, retention = await _seed()
    await migrate_backend(source_url=await _configured_url(), target_url=target_url)

    engine, factory = _open(target_url)
    try:
        async with factory() as s:
            comm = (
                await s.execute(select(Communication).where(Communication.agent == "test-migrate"))
            ).scalar_one()
            assert comm.retention_until is not None
            drift = abs((comm.retention_until - retention).total_seconds())
            assert drift < 2, (
                f"retention_until moved by {drift:.0f}s during migration — it was "
                f"recomputed rather than copied"
            )

            tombstoned = (
                (
                    await s.execute(
                        select(Annotation).where(
                            Annotation.source == SOURCE_TAG, Annotation.valid_until.is_not(None)
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert tombstoned, (
                "the tombstoned annotation did not arrive; forgotten rows are the "
                "audit trail and must migrate as tombstoned, not be dropped"
            )
    finally:
        await engine.dispose()


async def test_a_non_empty_target_is_refused(target_url):
    """Copying into an occupied store would merge two brains with no rule for conflicts."""
    await _seed()
    await migrate_backend(source_url=await _configured_url(), target_url=target_url)
    with pytest.raises(MigrationError, match="not empty"):
        await migrate_backend(source_url=await _configured_url(), target_url=target_url)


async def test_migrating_onto_itself_is_refused():
    url = await _configured_url()
    with pytest.raises(MigrationError, match="same database"):
        await migrate_backend(source_url=url, target_url=url)


async def test_a_dry_run_writes_nothing(target_url):
    from pathlib import Path

    result = await migrate_backend(
        source_url=await _configured_url(), target_url=target_url, dry_run=True
    )
    assert result.dry_run
    assert result.tables
    assert not Path(target_url.replace("sqlite:///", "")).exists(), (
        "a dry run created the target database"
    )


async def test_the_vector_check_can_actually_fail(target_url):
    """Guard the guard.

    Vectors are the one column whose storage differs by backend, so they are
    the one column row counts cannot vouch for — and a corrupted vector fails
    nothing, it just returns wrong search results forever. A verification that
    cannot fail would certify exactly that.

    So: migrate, corrupt one vector in the target, and assert the check
    objects.
    """
    await _seed()
    await migrate_backend(source_url=await _configured_url(), target_url=target_url)

    source_engine, source_factory = _open(await _configured_url())
    target_engine, target_factory = _open(target_url)
    try:
        from hafiz.core.database import Embedding

        async with source_factory() as s:
            original = (
                await s.execute(
                    select(Embedding)
                    .where(Embedding.embedding.is_not(None))
                    .order_by(Embedding.id)
                    .limit(1)
                )
            ).scalar_one_or_none()
        assert original is not None, "the fixture should have produced embeddings to check"

        # Sanity: unmodified, the check passes.
        assert await _verify_vector(source_factory, target_factory)

        # Corrupt through the ORM rather than by rewriting raw bytes. An
        # earlier version did the latter and passed on SQLite while failing on
        # Postgres, because a uuid rendered with `str()` carries dashes and
        # SQLAlchemy's Uuid type stores 32-char hex without them — so the raw
        # UPDATE silently matched nothing. Writing a plainly different vector
        # tests the same property and knows nothing about either encoding.
        async with target_factory() as s:
            landed = (
                await s.execute(select(Embedding).where(Embedding.id == original.id))
            ).scalar_one()
            landed.embedding = [-1.0] * len(list(original.embedding))
            await s.commit()

        with pytest.raises(MigrationError, match="changed value in transit|changed length"):
            await _verify_vector(source_factory, target_factory)
    finally:
        await source_engine.dispose()
        await target_engine.dispose()


def test_every_self_reference_is_declared():
    """A self-FK added later and not declared here would be silently nulled.

    The two-pass copy only restores columns listed in ``SELF_REFERENCES``.
    Anything self-referencing that is *not* listed still gets nulled on
    insert — and then never filled in, losing the link with no error at all.
    So the list is derived from the schema rather than trusted.
    """
    from hafiz.core.database import Base
    from hafiz.core.migrate import COPY_ORDER

    by_table = {model.__tablename__: model for model in COPY_ORDER}
    undeclared = []
    for model in COPY_ORDER:
        table = model.__table__
        for column in table.columns:
            for fk in column.foreign_keys:
                if fk.column.table is table and SELF_REFERENCES.get(model) != column.name:
                    undeclared.append(f"{table.name}.{column.name}")
    assert not undeclared, (
        "these columns reference their own table but are not in SELF_REFERENCES, so "
        "migration would null them and never restore them: " + ", ".join(undeclared)
    )
    assert set(Base.metadata.tables) - {"alembic_version"} <= set(by_table), (
        "a table exists in the schema but is not in COPY_ORDER, so migration would silently skip it"
    )


async def test_create_tables_refuses_a_database_it_cannot_actually_target(target_url):
    """The bug this guard exists for was found the expensive way.

    ``create_tables(url)`` looks like it targets ``url``. It does not: the
    embedded branch takes its engine from the process-wide singleton, and
    ``alembic/env.py`` overwrites the URL from settings. Calling it with the
    migration target created tables and stamped Alembic on the *configured*
    database instead — silently, because both are idempotent against a store
    already at head.
    """
    from hafiz.core.database import create_tables

    with pytest.raises(RuntimeError, match="cannot target a database other than"):
        await create_tables(target_url)
