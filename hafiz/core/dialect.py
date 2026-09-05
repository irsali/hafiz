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

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Index, LargeBinary, Uuid
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
from sqlalchemy.engine import Dialect
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import ColumnElement
from sqlalchemy.types import Boolean, Float, Text

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
    return TIMESTAMP(timezone=True).with_variant(DateTime(timezone=True), SQLITE)


def string_array_col() -> Any:
    """A list of strings — ``annotations.tags``.

    ``ARRAY(Text)`` on PG, JSON array on SQLite. Anything that *queries*
    this column must go through :func:`tags_overlap`; the two storage
    shapes have no operator in common.
    """
    return ARRAY(Text()).with_variant(JSON(), SQLITE)


def uuid_array_col() -> Any:
    """A list of uuids — ``retrievals.result_ids``.

    .. warning::
       On SQLite this stores a JSON array, and JSON cannot carry a
       ``uuid.UUID`` natively. Phase 2 owns the bind/result processing
       that converts to and from ``str``; until then this variant is
       declared (so the DDL is right) but unexercised, because Phase 1
       runs on Postgres only. Do not assume round-tripping works.
    """
    return ARRAY(UUID(as_uuid=True)).with_variant(JSON(), SQLITE)


def vector_col(dim: int = EMBEDDING_DIM) -> Any:
    """An embedding.

    ``vector(n)`` on PG. On SQLite, sqlite-vec reads float32 vectors from
    ordinary ``BLOB`` columns — ``vec_distance_cosine(blob, blob)`` works
    without a ``vec0`` virtual table, which is what the Phase 0 benchmark
    measured (87.5 ms p50 at 152,247 x 768d). Phase 2 confirms the
    packing.
    """
    return Vector(dim).with_variant(LargeBinary(), SQLITE)


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
    raise _unsupported("cosine_distance", SQLITE, "Phase 3 (backend-aware vector search)")


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


def unnest_ids(column: Any) -> Any:
    """A one-column selectable of the uuids inside an array column.

    Postgres-only for now: ``unnest()`` is a set-returning function with
    no SQLAlchemy-portable equivalent, and the SQLite shape
    (``json_each``) is a different FROM-clause construct rather than a
    different spelling of the same one. Phase 2 replaces this with a
    dispatching helper once there is a SQLite engine to test against.
    """
    from sqlalchemy import func

    return func.unnest(column)


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


def most_recalled_sql(backend: str) -> str:
    """The ``most_recalled`` query for ``backend``."""
    try:
        return _MOST_RECALLED_SQL[backend]
    except KeyError:
        raise _unsupported("most_recalled", backend, "no phase") from None
