"""Copy a whole store from one backend to the other.

``hafiz export`` dumps the wisdom layer to plain files and is deliberately
one-way — a sovereignty eject, not a transport. This is the transport: every
table, both layers, into a database on the other backend.

It exists because "re-ingest is the migration" (2026-04-25) was written when
the embedded backend was hypothetical, and it is only half true. Code and
docs can be re-derived — git is their source. Annotations cannot. Someone who
starts on SQLite, accumulates two years of decisions and then wants Postgres
for a second machine has nothing to re-ingest from.

Three properties, in the order they matter:

**Nothing is recomputed.** Every column copies verbatim, including
``retention_until``, ``valid_until`` and ``superseded_at``. Recomputing
retention windows from ``started_at`` would silently extend them for every
migrated communication — a regulator-visible change coming out of a command
that looks like plumbing. Tombstoned rows copy *as tombstoned*: not
resurrected, not dropped. They are the audit trail.

**It refuses rather than resumes.** The target must be empty. A run that dies
halfway therefore leaves a database the next run declines to touch — delete
it and start again. That is simpler than upsert logic and, more importantly,
has no half-merged state to reason about.

**It verifies itself.** Row counts are compared per table afterwards and a
vector is round-tripped, because "the command exited zero" is not evidence
that 290,000 rows arrived. A mismatch is a failure, not a warning.

The source is opened read-only and never written to.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from hafiz.core.database import (
    Annotation,
    AnnotationTarget,
    Base,
    Commit,
    Communication,
    CommunicationMessage,
    Edge,
    Embedding,
    File,
    Retrieval,
    Unit,
    UnitRevision,
)
from hafiz.core.database import (
    Session as SessionRow,
)
from hafiz.core.dialect import (
    backend_of_url,
    engine_options,
    is_embedded,
    normalize_url,
    prepare_engine,
)

#: Tables in dependency order — a table's referents always precede it.
#: ``sessions`` comes early because annotations, communications and retrievals
#: all point at it; ``units`` before ``unit_revisions`` before ``embeddings``.
COPY_ORDER: tuple[type[Base], ...] = (
    Commit,
    File,
    SessionRow,
    Unit,
    UnitRevision,
    Embedding,
    Edge,
    Annotation,
    Communication,
    CommunicationMessage,
    AnnotationTarget,
    Retrieval,
)

#: Columns that point at another row of the *same* table. Dependency order
#: cannot satisfy these — within one table a row may reference a row that has
#: not been inserted yet — so they are nulled on insert and filled by a second
#: pass.
#:
#: Deferring the constraint instead would work on SQLite
#: (``PRAGMA defer_foreign_keys``) and not on Postgres, where a constraint has
#: to be declared ``DEFERRABLE`` to be deferred, and these are not. A two-pass
#: copy needs no such asymmetry.
SELF_REFERENCES: dict[type[Base], str] = {
    UnitRevision: "superseded_by",
    Annotation: "supersedes_id",
    CommunicationMessage: "parent_message_id",
}

#: Rows per round trip. Vectors dominate the payload — 768 floats each, and a
#: real store holds ~150k of them — so this is a memory ceiling, not a
#: throughput knob. 500 embedding rows is roughly 1.5 MB of floats.
BATCH = 500


class MigrationError(RuntimeError):
    """Raised when the migration cannot start, or did not arrive intact."""


@dataclasses.dataclass
class TableResult:
    name: str
    source_rows: int
    copied: int
    back_references: int = 0

    @property
    def ok(self) -> bool:
        return self.source_rows == self.copied


@dataclasses.dataclass
class MigrationResult:
    source_backend: str
    target_backend: str
    target_url: str
    tables: list[TableResult]
    dry_run: bool
    vector_check: str | None = None

    @property
    def ok(self) -> bool:
        return all(t.ok for t in self.tables)

    @property
    def total_rows(self) -> int:
        return sum(t.copied for t in self.tables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "dry_run": self.dry_run,
            "source_backend": self.source_backend,
            "target_backend": self.target_backend,
            "target": self.target_url,
            "total_rows": self.total_rows,
            "vector_check": self.vector_check,
            "tables": [
                {
                    "table": t.name,
                    "source_rows": t.source_rows,
                    "copied": t.copied,
                    "back_references": t.back_references,
                    "ok": t.ok,
                }
                for t in self.tables
            ],
        }


def _open(url: str):
    """Build an engine for ``url`` without touching the process-wide singleton.

    ``get_engine`` caches one engine for the configured database. A migration
    needs two at once, on different backends, so it constructs them directly —
    through the same ``prepare_engine`` the singleton uses, so the embedded
    side still gets sqlite-vec loaded and its PRAGMAs applied.
    """
    resolved = normalize_url(url)
    if is_embedded(resolved):
        from hafiz.core.dialect import db_file_path

        path = db_file_path(resolved)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    engine = prepare_engine(create_async_engine(resolved, **engine_options(resolved)))
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _count(session: AsyncSession, model: type[Base]) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar() or 0


async def _assert_target_empty(session: AsyncSession) -> None:
    """Refuse a target that already holds anything.

    Merging two stores is a different problem with different answers — which
    side wins on a conflicting annotation? — and guessing at it inside a
    transport command would produce a store nobody could reason about.
    """
    occupied = []
    for model in COPY_ORDER:
        rows = await _count(session, model)
        if rows:
            occupied.append(f"{model.__tablename__} ({rows:,} rows)")
    if occupied:
        raise MigrationError(
            "The target database is not empty:\n  "
            + "\n  ".join(occupied)
            + "\n\nMigration copies into a fresh store; it does not merge, because "
            "merging needs a rule for which side wins and there isn't an obvious "
            "one. Point --to at a new database, or delete this one and retry."
        )


def _row_values(row: Base, model: type[Base]) -> dict[str, Any]:
    """Every column of ``row``, keyed by **column** name, read by attribute name.

    The two differ, and assuming they match cost a crash on the first table.
    Eight tables have a column called ``metadata``, mapped to the attribute
    ``metadata_`` because ``Model.metadata`` is already SQLAlchemy's own
    ``MetaData`` object. ``getattr(row, "metadata")`` therefore returns the
    schema, not the value — and the failure surfaces far away, as "Object of
    type MetaData is not JSON serializable" during the insert.
    """
    mapper = sa_inspect(model)
    return {attr.columns[0].name: getattr(row, attr.key) for attr in mapper.mapper.column_attrs}


async def _prepare_target(engine, url: str) -> None:
    """Create the schema on the target, and stamp it at the current head.

    Deliberately *not* :func:`hafiz.core.database.create_tables`. That takes a
    URL but resolves the engine from the process-wide singleton, and
    ``alembic/env.py`` overwrites ``sqlalchemy.url`` with ``load_settings()``
    — so calling it here created tables and stamped Alembic on the *currently
    configured* database rather than the target. Silent, because both
    operations are idempotent against a store already at head.

    The stamp is written directly instead of going through ``alembic stamp``
    for the same reason: env.py would send it to the wrong database. The head
    revision still comes from the migration scripts, so it cannot drift.
    """
    from alembic.script import ScriptDirectory
    from sqlalchemy import text

    from hafiz.core.database import _alembic_config

    if not is_embedded(url):
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    head = ScriptDirectory.from_config(_alembic_config(url)).get_current_head()
    async with engine.begin() as conn:
        await conn.execute(
            text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) NOT NULL)")
        )
        existing = (await conn.execute(text("SELECT count(*) FROM alembic_version"))).scalar()
        if not existing:
            await conn.execute(
                text("INSERT INTO alembic_version (version_num) VALUES (:v)"), {"v": head}
            )


async def _copy_table(
    source: async_sessionmaker[AsyncSession],
    target: async_sessionmaker[AsyncSession],
    model: type[Base],
) -> TableResult:
    """Stream one table across, nulling any self-reference for a later pass."""
    table = model.__table__
    self_ref = SELF_REFERENCES.get(model)
    pk = list(table.primary_key.columns)[0]

    async with source() as read:
        total = await _count(read, model)

    copied = 0
    offset = 0
    while True:
        async with source() as read:
            batch = (
                (await read.execute(select(model).order_by(pk).offset(offset).limit(BATCH)))
                .scalars()
                .all()
            )
        if not batch:
            break

        payload = []
        for row in batch:
            values = _row_values(row, model)
            if self_ref:
                values[self_ref] = None
            payload.append(values)

        async with target() as write:
            await write.execute(insert(table), payload)
            await write.commit()

        copied += len(batch)
        offset += BATCH

    return TableResult(name=table.name, source_rows=total, copied=copied)


async def _link_self_references(
    source: async_sessionmaker[AsyncSession],
    target: async_sessionmaker[AsyncSession],
    model: type[Base],
) -> int:
    """Second pass: restore the self-referencing column now every row exists."""
    column = SELF_REFERENCES[model]
    table = model.__table__
    pk = list(table.primary_key.columns)[0]
    attribute = getattr(model, column)

    linked = 0
    offset = 0
    while True:
        async with source() as read:
            rows = (
                await read.execute(
                    select(pk, attribute)
                    .where(attribute.is_not(None))
                    .order_by(pk)
                    .offset(offset)
                    .limit(BATCH)
                )
            ).all()
        if not rows:
            break

        async with target() as write:
            for row_id, points_to in rows:
                await write.execute(update(table).where(pk == row_id).values({column: points_to}))
            await write.commit()

        linked += len(rows)
        offset += BATCH

    return linked


async def _verify_vector(
    source: async_sessionmaker[AsyncSession], target: async_sessionmaker[AsyncSession]
) -> str | None:
    """Round-trip one embedding and compare it element-wise.

    Vectors are the one column whose *storage* differs by backend — pgvector
    on one side, a packed little-endian float32 blob on the other — so they
    are the one column a row-count check cannot vouch for. A silently
    corrupted vector would not fail anything; it would just quietly return the
    wrong search results forever.
    """
    async with source() as read:
        row = (
            await read.execute(
                select(Embedding)
                .where(Embedding.embedding.is_not(None))
                .order_by(Embedding.id)
                .limit(1)
            )
        ).scalar_one_or_none()
    if row is None:
        return "skipped: no embeddings to check"

    async with target() as write:
        landed = (
            await write.execute(select(Embedding).where(Embedding.id == row.id))
        ).scalar_one_or_none()
    if landed is None:
        raise MigrationError(f"embedding {row.id} did not arrive in the target")

    before = [float(x) for x in row.embedding]
    after = [float(x) for x in landed.embedding]
    if len(before) != len(after):
        raise MigrationError(
            f"embedding {row.id} changed length in transit: {len(before)} -> {len(after)}"
        )
    worst = max((abs(a - b) for a, b in zip(before, after, strict=True)), default=0.0)
    # float32 on both sides; anything beyond rounding means the pack/unpack
    # disagreed, which would corrupt every subsequent similarity score.
    if worst > 1e-5:
        raise MigrationError(
            f"embedding {row.id} changed value in transit (max element delta {worst:g}). "
            f"The vector encoding did not survive the backend change."
        )
    return f"{len(before)} dims, max element delta {worst:g}"


async def migrate_backend(
    *, source_url: str, target_url: str, dry_run: bool = False
) -> MigrationResult:
    """Copy every table from ``source_url`` into ``target_url``.

    The source is only ever read. The target must be empty, and its schema is
    created if absent.
    """
    source_url = normalize_url(source_url)
    target_url = normalize_url(target_url)
    if source_url == target_url:
        raise MigrationError("Source and target are the same database; nothing to migrate.")

    source_engine, source_factory = _open(source_url)
    target_engine, target_factory = _open(target_url)

    result = MigrationResult(
        source_backend=backend_of_url(source_url),
        target_backend=backend_of_url(target_url),
        target_url=target_url,
        tables=[],
        dry_run=dry_run,
    )

    try:
        if dry_run:
            async with source_factory() as read:
                for model in COPY_ORDER:
                    rows = await _count(read, model)
                    result.tables.append(
                        TableResult(name=model.__tablename__, source_rows=rows, copied=rows)
                    )
            return result

        await _prepare_target(target_engine, target_url)

        async with target_factory() as write:
            await _assert_target_empty(write)

        for model in COPY_ORDER:
            table_result = await _copy_table(source_factory, target_factory, model)
            if model in SELF_REFERENCES:
                table_result.back_references = await _link_self_references(
                    source_factory, target_factory, model
                )
            result.tables.append(table_result)

        # Verify, rather than trusting the absence of an exception.
        async with source_factory() as read, target_factory() as write:
            for table_result in result.tables:
                model = next(m for m in COPY_ORDER if m.__tablename__ == table_result.name)
                landed = await _count(write, model)
                source_rows = await _count(read, model)
                if landed != source_rows:
                    raise MigrationError(
                        f"{table_result.name}: {source_rows:,} rows in the source but "
                        f"{landed:,} in the target. The migration did not arrive intact; "
                        f"delete the target and retry."
                    )
                table_result.copied = landed

        result.vector_check = await _verify_vector(source_factory, target_factory)
        return result
    finally:
        await source_engine.dispose()
        await target_engine.dispose()
