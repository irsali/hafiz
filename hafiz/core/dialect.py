"""The one place where Hafiz knows which database it is talking to.

Hafiz targets Postgres (team/shared installs) and, from Phase 2 of
workitems/active/embedded-backend.md, an embedded SQLite + sqlite-vec
backend (solo installs). Two backends behind one ``--json`` contract is a
permanent maintenance cost, and the way that cost gets paid is by keeping
every dialect-sensitive construct **in this module**, so "did you go
through the dialect layer?" is a question a reviewer can actually ask.

Three kinds of thing live here:

1. **Column factories** — ``uuid_col()``, ``json_col()``, … Each returns a
   SQLAlchemy type whose Postgres rendering is *byte-identical to what
   Hafiz shipped before this module existed*, with a ``with_variant()``
   branch for SQLite. That is what makes Phase 1 a no-op migration:
   ``alembic revision --autogenerate`` must produce an empty diff.

2. **Expression constructs** — ``cosine_distance()``, ``similarity()``,
   ``tags_overlap()``. These are dialect-neutral at *construction* time
   and compile differently per dialect via ``@compiles``. That matters
   more than it looks: dispatching on a global "which backend is
   configured?" lookup would generate Postgres SQL for a SQLite engine
   whenever the two disagree — which is precisely what a dual-backend
   test matrix does on every run.

3. **Runtime dispatch** — ``backend_of()`` plus the SQL that cannot be
   expressed as an ORM construct at all (see ``most_recalled_sql``).

**The failure mode this module exists to prevent is silent divergence.**
A backend that raises is debuggable; a backend that quietly returns
differently-ordered results is not. So every SQLite branch that is not
yet implemented raises ``UnsupportedOnBackendError`` naming the phase that
owns it, and none of them fall back to "close enough".
"""

from __future__ import annotations

import struct
import uuid as _uuid
from datetime import UTC as _UTC
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Index, LargeBinary, Uuid
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import ColumnElement
from sqlalchemy.types import Boolean, Float, Text, TypeDecorator

#: Dimensionality of ``nomic-embed-text-v1.5``. Declared once here because
#: the embedded backend needs it at column-create time exactly as pgvector
#: does — see Open Question 2 in the work item (per-row dims are possible
#: on sqlite-vec but not on pgvector, so the schema stays fixed-dim).
EMBEDDING_DIM = 768

#: Dialect names as SQLAlchemy reports them.
POSTGRESQL = "postgresql"
SQLITE = "sqlite"


class UnsupportedOnBackendError(NotImplementedError):
    """A construct has no implementation on the dialect in play.

    Raised at *compile* time, so it surfaces when the statement is built
    rather than as a mangled result set.
    """


def _unsupported(construct: str, dialect: str, phase: str) -> UnsupportedOnBackendError:
    return UnsupportedOnBackendError(
        f"{construct} is not implemented for the '{dialect}' backend yet "
        f"({phase} owns it — see workitems/active/embedded-backend.md). "
        "Refusing to emit approximate SQL: a wrong ranking is harder to "
        "detect than a hard failure."
    )


# ---------------------------------------------------------------------------
# 0. Type decorators — the Python<->SQLite value conversions
# ---------------------------------------------------------------------------
# Postgres has native types for everything Hafiz stores. SQLite has none of
# them, so the conversion has to happen in Python on the way in and out.
# These are the only places where a value changes shape between backends.


class SqliteVector(TypeDecorator):
    """A 768-d embedding as sqlite-vec's on-disk representation.

    sqlite-vec reads vectors as a ``BLOB`` of **little-endian float32**,
    contiguous, no header. ``vec_distance_cosine(blob, blob)`` operates on
    exactly this, with no ``vec0`` virtual table required — which is what the
    Phase 0 benchmark measured.

    The byte order is explicit (``<``) rather than native. ``struct.pack("f")``
    without a prefix uses native order *and* native alignment; on a big-endian
    host that silently produces vectors sqlite-vec reads as garbage. Distances
    would still come back — plausible, ordered, and wrong. Hafiz has no
    big-endian users today, which is exactly why this would go unnoticed.
    """

    impl = LargeBinary
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> bytes | None:
        if value is None or isinstance(value, bytes):
            return value
        vec = list(value)
        return struct.pack(f"<{len(vec)}f", *(float(x) for x in vec))

    def process_result_value(self, value: Any, dialect: Dialect) -> list[float] | None:
        if value is None:
            return None
        return list(struct.unpack(f"<{len(value) // 4}f", value))


class SqliteUtcDateTime(TypeDecorator):
    """A timezone-aware timestamp, stored as naive UTC and read back aware.

    SQLite has no timestamp type and SQLAlchemy's SQLite ``DATETIME``
    **silently ignores** ``timezone=True`` — values go in aware and come back
    naive. That is a behavioural divergence, not a storage detail: Postgres
    returns ``tzinfo=UTC``, so without this the same row compares differently,
    serialises differently, and raises "can't subtract offset-naive and
    offset-aware datetimes" on one backend and not the other. Hafiz promises
    one ``--json`` contract from both.

    Hafiz writes ``datetime.now(UTC)`` everywhere, so normalising to UTC on
    the way in loses nothing. A naive value is assumed to be UTC already —
    the alternative, guessing local time, would shift timestamps by the
    developer's offset.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None or getattr(value, "tzinfo", None) is None:
            return value
        return value.astimezone(_UTC).replace(tzinfo=None)

    def process_result_value(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=_UTC)
        return value.astimezone(_UTC)


class SqliteUuidArray(TypeDecorator):
    """A list of uuids, stored as a JSON array of strings.

    ``retrievals.result_ids`` is ``uuid[]`` on Postgres. JSON has no uuid, so
    the conversion is explicit in both directions — without the result half,
    callers get ``str`` where they had ``uuid.UUID`` and comparisons silently
    stop matching.
    """

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> list[str] | None:
        if value is None:
            return None
        return [str(v) for v in value]

    def process_result_value(self, value: Any, dialect: Dialect) -> list[_uuid.UUID] | None:
        if value is None:
            return None
        out: list[_uuid.UUID] = []
        for raw in value:
            try:
                out.append(raw if isinstance(raw, _uuid.UUID) else _uuid.UUID(str(raw)))
            except (ValueError, AttributeError, TypeError):
                continue  # mirrors telemetry's own tolerance for a junk id
        return out


# ---------------------------------------------------------------------------
# 1. Column factories
# ---------------------------------------------------------------------------
# Every factory keeps the *Postgres* type exactly as it was before this
# module existed. The SQLite variant is additive. Do not "simplify" these
# to their generic SQLAlchemy equivalents (e.g. sa.Uuid for UUID): the
# generic types render different DDL, which shows up as a spurious
# autogenerate diff against five shipped migrations.


def uuid_col() -> Any:
    """Primary/foreign key type. ``UUID`` on PG, ``CHAR(32)`` on SQLite."""
    return UUID(as_uuid=True).with_variant(Uuid(as_uuid=True), SQLITE)


def json_col() -> Any:
    """Free-form object column. ``JSONB`` on PG, ``JSON`` (text) on SQLite."""
    return JSONB().with_variant(JSON(), SQLITE)


def ts_col() -> Any:
    """Timezone-aware timestamp.

    SQLite has no native timestamp type; SQLAlchemy's ``DateTime`` stores
    ISO-8601 strings and reattaches tzinfo on the way out.
    """
    return TIMESTAMP(timezone=True).with_variant(SqliteUtcDateTime(), SQLITE)


def string_array_col() -> Any:
    """A list of strings — ``annotations.tags``.

    ``ARRAY(Text)`` on PG, JSON array on SQLite. Anything that *queries*
    this column must go through :func:`tags_overlap`; the two storage
    shapes have no operator in common.
    """
    return ARRAY(Text()).with_variant(JSON(), SQLITE)


def uuid_array_col() -> Any:
    """A list of uuids — ``retrievals.result_ids``.

    ``uuid[]`` on PG; a JSON array of strings on SQLite, round-tripped by
    :class:`SqliteUuidArray`.
    """
    return ARRAY(UUID(as_uuid=True)).with_variant(SqliteUuidArray(), SQLITE)


def vector_col(dim: int = EMBEDDING_DIM) -> Any:
    """An embedding.

    ``vector(n)`` on PG; a little-endian float32 ``BLOB`` on SQLite, packed
    by :class:`SqliteVector`. Phase 0 measured 87.5 ms p50 at 152,247 x 768d
    against exactly this representation.
    """
    return Vector(dim).with_variant(SqliteVector(), SQLITE)


def partial_index(name: str, *columns: str, where: Any, unique: bool = False) -> Index:
    """A partial index, declared for both dialects.

    Both Postgres and SQLite support ``CREATE INDEX ... WHERE``, but each
    dialect reads its *own* namespaced kwarg — pass only
    ``postgresql_where`` and SQLite silently creates a **full** index
    instead. For the unique ones here that is not a performance nit: a
    full unique index over a nullable column enforces a different
    constraint than the partial one, so the schema would quietly disagree
    with itself across backends.
    """
    return Index(
        name,
        *columns,
        unique=unique,
        postgresql_where=where,
        sqlite_where=where,
    )


# ---------------------------------------------------------------------------
# 2. Expression constructs (compile-time dispatch)
# ---------------------------------------------------------------------------


class _CosineDistance(ColumnElement):
    """``col <=> vec`` — smaller is closer. Use :func:`cosine_distance`."""

    type = Float()
    inherit_cache = True

    def __init__(self, column: Any, vector: Any) -> None:
        self.column = column
        self.vector = vector


@compiles(_CosineDistance, POSTGRESQL)
def _cosine_distance_pg(element: _CosineDistance, compiler: Any, **kw: Any) -> str:
    # pgvector's own comparator, so this stays byte-identical to the eight
    # call sites that previously wrote `.cosine_distance(...)` inline.
    #
    # Parenthesised because a custom construct declares no operator
    # precedence, so SQLAlchemy will not group it when it is nested inside
    # another expression. Without this, `similarity()` renders as
    # `1 - embedding <=> $1`, which Postgres reads as `(1 - embedding) <=> $1`
    # and rejects with "operator does not exist: integer - vector".
    inner = compiler.process(element.column.cosine_distance(element.vector), **kw)
    return f"({inner})"


@compiles(_CosineDistance, SQLITE)
def _cosine_distance_sqlite(element: _CosineDistance, compiler: Any, **kw: Any) -> str:
    """sqlite-vec's ``vec_distance_cosine``, over two float32 blobs.

    Returns cosine *distance* on the same 0..2 scale as pgvector's ``<=>``,
    so ``similarity()`` composes identically and callers need no branch.

    The query vector is packed here rather than relying on the column's
    :class:`SqliteVector` decorator: that decorator applies to *column*
    binds, and this side of the comparison is a free-standing parameter
    that never passes through it.
    """
    from sqlalchemy import literal

    column_sql = compiler.process(element.column, **kw)
    vector = element.vector
    packed = (
        vector
        if isinstance(vector, bytes)
        else struct.pack(f"<{len(vector)}f", *(float(x) for x in vector))
    )
    param_sql = compiler.process(literal(packed, LargeBinary()), **kw)
    return f"vec_distance_cosine({column_sql}, {param_sql})"


@compiles(_CosineDistance)
def _cosine_distance_default(element: _CosineDistance, compiler: Any, **kw: Any) -> str:
    raise _unsupported("cosine_distance", compiler.dialect.name, "no phase")


def cosine_distance(column: Any, vector: Any) -> ColumnElement:
    """Distance between an embedding column and a query vector.

    This is the ``ORDER BY`` expression. Ascending order is best-first.
    """
    return _CosineDistance(column, vector)


def similarity(column: Any, vector: Any) -> ColumnElement:
    """``1 - cosine_distance`` — the score callers actually report.

    Kept next to :func:`cosine_distance` deliberately. The two are trivial
    to transpose, and transposing them inverts every ranking in the
    product while every test that only checks "results came back" still
    passes.
    """
    return 1 - _CosineDistance(column, vector)


class _TagsOverlap(ColumnElement):
    """``col && ARRAY[...]`` — true if any tag matches. See :func:`tags_overlap`."""

    type = Boolean()
    inherit_cache = True

    def __init__(self, column: Any, tags: list[str]) -> None:
        self.column = column
        self.tags = list(tags)


@compiles(_TagsOverlap, POSTGRESQL)
def _tags_overlap_pg(element: _TagsOverlap, compiler: Any, **kw: Any) -> str:
    # Parenthesised for the same precedence reason as _CosineDistance above.
    inner = compiler.process(element.column.overlap(element.tags), **kw)
    return f"({inner})"


@compiles(_TagsOverlap, SQLITE)
def _tags_overlap_sqlite(element: _TagsOverlap, compiler: Any, **kw: Any) -> str:
    # SQLite has no array type and so no `&&`. The JSON1 extension's
    # json_each() unrolls the stored array into rows, which is the closest
    # equivalent: EXISTS over the intersection.
    from sqlalchemy import literal

    if not element.tags:
        return "0"  # `&&` with an empty array is false on PG too
    column_sql = compiler.process(element.column, **kw)
    values = ", ".join(compiler.process(literal(t), **kw) for t in element.tags)
    return f"(EXISTS (SELECT 1 FROM json_each({column_sql}) WHERE json_each.value IN ({values})))"


@compiles(_TagsOverlap)
def _tags_overlap_default(element: _TagsOverlap, compiler: Any, **kw: Any) -> str:
    raise _unsupported("tags_overlap", compiler.dialect.name, "no phase")


def tags_overlap(column: Any, tags: list[str]) -> ColumnElement:
    """True where the row shares at least one tag with ``tags``.

    Overlap, not containment: any single matching tag qualifies.
    """
    return _TagsOverlap(column, tags)


# ---------------------------------------------------------------------------
# 3. Runtime dispatch
# ---------------------------------------------------------------------------


def backend_of_url(url: str) -> str:
    """The backend a URL selects, named the way :func:`backend_of` names it.

    For the case where there is no bind to ask — reporting on a database
    before connecting to it, or naming both sides of a migration. Kept beside
    ``backend_of`` so the two cannot start disagreeing about what to call a
    backend.
    """
    from sqlalchemy.engine import make_url

    return "sqlite" if is_embedded(url) else make_url(normalize_url(url)).get_dialect().name


def backend_of(bindable: Any) -> str:
    """The dialect name behind a session, connection, or engine.

    Read from the actual bind rather than from configuration, so a test
    session pointed at a different backend than ``hafiz.toml`` gets the
    right SQL.
    """
    dialect: Dialect | None = getattr(bindable, "dialect", None)
    if dialect is None:
        bind = getattr(bindable, "bind", None) or getattr(bindable, "get_bind", lambda: None)()
        dialect = getattr(bind, "dialect", None)
    if dialect is None:
        raise UnsupportedOnBackendError(f"cannot determine the backend behind {bindable!r}")
    return dialect.name


def unnest_ids(column: Any, entity: Any, backend: str) -> Any:
    """A one-column subquery (``id``) of every uuid inside an array column.

    Returns the whole subquery rather than an expression to splice into one,
    because the backends need different *shapes*, not different spellings:

    * Postgres — ``unnest()`` is a set-returning function usable directly in
      the select list: ``SELECT unnest(result_ids) FROM retrievals``.
    * SQLite — ``json_each()`` is table-valued and must sit in the FROM
      clause beside its source row:
      ``SELECT je.value FROM retrievals, json_each(retrievals.result_ids) je``.

    An earlier version returned only the column expression. On SQLite that
    compiled to a select over ``json_each(...)`` with no ``retrievals`` in
    the FROM clause — ``no such column: retrievals.result_ids`` at runtime,
    in five tests the SQLite-only module never reached.
    """
    from sqlalchemy import func, select, true

    if backend == POSTGRESQL:
        return select(func.unnest(column).label("id")).subquery()
    if backend == SQLITE:
        each = func.json_each(column).table_valued("value")
        # An explicit ``JOIN ... ON 1=1`` rather than a comma cross-join. The
        # two compile to the same plan, but SQLAlchemy's from-linter cannot
        # tell that a table-valued function is correlated to the table it
        # reads, so the comma form emits a cartesian-product SAWarning on
        # every execution. Nineteen spurious warnings per run is how a suite
        # teaches people to stop reading warnings.
        return select(each.c.value.label("id")).select_from(entity).join(each, true()).subquery()
    raise _unsupported("unnest_ids", backend, "no phase")


# ---------------------------------------------------------------------------
# 4. Connection setup — everything the embedded backend needs at connect time
# ---------------------------------------------------------------------------


def is_embedded(url: str) -> bool:
    """True when this URL selects the embedded (SQLite) backend."""
    return url.startswith("sqlite")


def default_db_path() -> Any:
    """Where a fresh embedded install puts its single file.

    XDG data dir, so the brain sits with other application state rather
    than in the user's config or cwd.
    """
    import os
    from pathlib import Path

    root = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(root).expanduser() / "hafiz" / "hafiz.db"


def normalize_url(url: str) -> str:
    """Expand and canonicalise a database URL.

    ``sqlite:///path`` is the user-facing spelling; it is rewritten to the
    async driver here because every DB call in Hafiz is ``async``. A sync
    SQLite URL reaching ``create_async_engine`` fails with "the asyncio
    extension requires an async driver", which reads as a bug in Hafiz
    rather than as a URL the user can fix.

    Parsed with SQLAlchemy's own URL parser rather than string surgery.
    SQLite's slash convention is genuinely tricky — ``sqlite:///rel.db`` is
    relative and ``sqlite:////abs.db`` is absolute — and a hand-rolled
    version of this silently turned absolute paths into relative ones,
    which then defeated the file-permission hardening downstream.
    """
    if not is_embedded(url):
        return url
    from pathlib import Path

    from sqlalchemy.engine import make_url

    parsed = make_url(url)
    if parsed.database:
        parsed = parsed.set(database=str(Path(parsed.database).expanduser()))
    if "+" not in parsed.drivername:
        parsed = parsed.set(drivername="sqlite+aiosqlite")
    return parsed.render_as_string(hide_password=False)


def db_file_path(url: str) -> Any:
    """The filesystem path behind an embedded URL, or None for in-memory."""
    from pathlib import Path

    from sqlalchemy.engine import make_url

    if not is_embedded(url):
        return None
    database = make_url(url).database
    if not database or database.startswith(":memory:"):
        return None
    return Path(database).expanduser()


def engine_options(url: str) -> dict:
    """Engine kwargs appropriate to the backend.

    Postgres keeps its explicit pool. SQLite gets SQLAlchemy's own default
    pool for the driver — passing ``pool_size``/``max_overflow`` at a
    single-writer file database is at best noise and at worst an error,
    depending on which pool class the dialect picks.
    """
    if is_embedded(url):
        return {"echo": False}
    return {"echo": False, "pool_size": 5, "max_overflow": 10}


#: PRAGMAs applied to every embedded connection.
#:
#: ``foreign_keys`` is the load-bearing one: SQLite ships it **OFF**, so
#: without it every ``ondelete="CASCADE"`` and ``SET NULL`` in the schema
#: silently does nothing. Deleting a communication would orphan its messages
#: rather than remove them — a retention guarantee quietly becoming a lie.
#:
#: ``busy_timeout`` covers the contention Open Question 5 raises: Hafiz now
#: writes from background hooks, so a capture can fire mid-ingest. WAL gives
#: one writer plus many readers; the timeout makes the writer wait instead of
#: raising "database is locked".
#:
#: ``secure_delete`` zeroes freed content instead of merely unlinking it from
#: the b-tree, which is what makes ``forget --hard`` mean what it says. It is
#: set explicitly because its default is decided at *compile time*
#: (``SQLITE_SECURE_DELETE``) and therefore varies by platform: Debian's
#: system SQLite enables it, the stock upstream build does not. Relying on the
#: build meant the redaction guarantee silently held on some users' machines
#: and not others — and testing on a machine where it happened to be on made
#: the difference invisible.
_SQLITE_PRAGMAS = (
    "journal_mode=WAL",
    "foreign_keys=ON",
    "busy_timeout=5000",
    "synchronous=NORMAL",
    "secure_delete=ON",
)


def prepare_engine(engine: Any) -> Any:
    """Attach backend-specific connection setup. Returns the engine.

    For SQLite this loads sqlite-vec and applies the PRAGMAs on **every**
    connection. Per-connection, not once: a pooled connection that missed
    the extension fails only when it happens to serve a vector query, which
    makes the bug look intermittent and unrelated to pooling.
    """
    from sqlalchemy import event

    sync_engine = getattr(engine, "sync_engine", engine)
    url = str(sync_engine.url)
    if not is_embedded(url):
        return engine

    @event.listens_for(sync_engine, "connect")
    def _configure_sqlite(dbapi_connection: Any, _record: Any) -> None:  # pragma: no cover
        import sqlite_vec
        from sqlalchemy.util import await_only

        raw = getattr(dbapi_connection, "driver_connection", dbapi_connection)
        # aiosqlite owns a worker thread and its sqlite3 handle is bound to
        # it, so the load cannot run on the caller's thread — `await_only`
        # is how a sync event handler reaches an async driver's thread.
        await_only(raw.enable_load_extension(True))
        await_only(raw._execute(sqlite_vec.load, raw._conn))
        await_only(raw.enable_load_extension(False))
        for pragma in _SQLITE_PRAGMAS:
            await_only(raw.execute(f"PRAGMA {pragma}"))
        # Hardened here rather than at init, because *here* the file provably
        # exists — SQLite creates it on first connect. Doing it only at
        # `hafiz init` left every database created by any other path
        # world-readable, and the WAL sibling is created later still.
        secure_db_file(url)

    return engine


async def reclaim_free_pages(session: Any) -> bool:
    """VACUUM after a destructive delete. True if it ran.

    SQLite keeps deleted content in the file's free pages until a VACUUM
    rewrites the database, so a "hard" delete leaves the bytes recoverable
    with `strings`. Postgres reclaims via autovacuum and needs nothing here,
    so this is a no-op there rather than a cross-backend cost.

    Best-effort: a redaction that already committed must not be reported as
    failed because the reclaim could not run (VACUUM cannot execute inside a
    transaction, and a concurrent reader can block it).
    """
    from sqlalchemy import text as sa_text

    try:
        if backend_of(session) != SQLITE:
            return False
        await session.commit()  # VACUUM cannot run inside a transaction
        await session.execute(sa_text("VACUUM"))
        return True
    except Exception:  # noqa: BLE001 — the delete already succeeded
        return False


def secure_db_file(url: str) -> list[Any]:
    """Restrict the embedded DB file to the owner. Returns paths changed.

    Postgres gated the brain behind database auth. A single file inherits
    the process umask instead — commonly ``0644``, i.e. world-readable on
    any shared machine. The ``-wal`` and ``-shm`` siblings carry recently
    written content and need the same treatment; chmod'ing only the ``.db``
    leaves the most recent turns readable.
    """
    path = db_file_path(url)
    if path is None:
        return []
    changed = []
    for candidate in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        try:
            if candidate.exists():
                candidate.chmod(0o600)
                changed.append(candidate)
        except OSError:
            continue  # a perms failure must not stop the CLI from working
    return changed


#: ``most_recalled`` in the retrieval report. Raw SQL because it joins a
#: set-returning function to a table — three Postgres-isms in six lines
#: (``unnest``, ``::text``, ``left()``), none of which survive a dialect
#: swap. Kept here rather than inline in telemetry.py so the SQLite
#: rewrite lands in the module a reviewer already checks.
_MOST_RECALLED_SQL = {
    POSTGRESQL: (
        "SELECT a.id::text, a.kind, left(a.content, 120) AS preview, count(*) AS hits "
        "FROM retrievals r, unnest(r.result_ids) AS rid "
        "JOIN annotations a ON a.id = rid "
        "WHERE r.at >= :since GROUP BY 1,2,3 ORDER BY hits DESC LIMIT :limit"
    ),
    SQLITE: (
        "SELECT a.id, a.kind, substr(a.content, 1, 120) AS preview, count(*) AS hits "
        "FROM retrievals r, json_each(r.result_ids) AS rid "
        "JOIN annotations a ON a.id = rid.value "
        "WHERE r.at >= :since GROUP BY 1,2,3 ORDER BY hits DESC LIMIT :limit"
    ),
}


#: "How many live annotations have a near-duplicate sibling?" — a self-join
#: over every pair, so it is raw SQL to keep the vectors in the database.
#: Two dialect-specific pieces: the distance operator, and null-safe
#: equality on ``project`` (``IS NOT DISTINCT FROM`` on PG; plain ``IS`` on
#: SQLite, where ``IS`` is already null-safe).
_CLUSTERED_SQL_TEMPLATE = (
    "SELECT count(*) FROM annotations a "
    "WHERE a.valid_until IS NULL AND a.embedding IS NOT NULL "
    "AND EXISTS (SELECT 1 FROM annotations b "
    "  WHERE b.valid_until IS NULL AND b.embedding IS NOT NULL "
    "    AND b.id <> a.id AND b.kind = a.kind "
    "    AND b.project {null_safe_eq} a.project "
    "    AND (1 - {distance}) >= :thr)"
)

_CLUSTERED_SQL = {
    POSTGRESQL: _CLUSTERED_SQL_TEMPLATE.format(
        null_safe_eq="IS NOT DISTINCT FROM",
        distance="(a.embedding <=> b.embedding)",
    ),
    SQLITE: _CLUSTERED_SQL_TEMPLATE.format(
        null_safe_eq="IS",
        distance="vec_distance_cosine(a.embedding, b.embedding)",
    ),
}


def clustered_annotations_sql(backend: str) -> str:
    """SQL counting live annotations that have a near-duplicate sibling."""
    try:
        return _CLUSTERED_SQL[backend]
    except KeyError:
        raise _unsupported("clustered_annotations", backend, "no phase") from None


#: "Which tables exist?" — a catalogue query, and every engine keeps its
#: catalogue somewhere different. Returns one ``name`` column either way.
_TABLE_LIST_SQL = {
    POSTGRESQL: "SELECT tablename AS name FROM pg_tables WHERE schemaname = 'public'",
    SQLITE: "SELECT name FROM sqlite_master WHERE type = 'table'",
}


def table_list_sql(backend: str) -> str:
    """SQL listing the user tables in the current database."""
    try:
        return _TABLE_LIST_SQL[backend]
    except KeyError:
        raise _unsupported("table_list", backend, "no phase") from None


def most_recalled_sql(backend: str) -> str:
    """The ``most_recalled`` query for ``backend``."""
    try:
        return _MOST_RECALLED_SQL[backend]
    except KeyError:
        raise _unsupported("most_recalled", backend, "no phase") from None
