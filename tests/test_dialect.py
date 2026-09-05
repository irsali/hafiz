"""The dialect seam: one module, two backends, no silent divergence.

These tests are deliberately compile-only where they can be. The whole
point of ``hafiz/core/dialect.py`` is that SQL for a backend Hafiz is not
currently connected to can still be inspected — otherwise the SQLite
branch stays unverified until Phase 2 wires an engine, which is exactly
how a "just swap the column type" refactor ships a wrong ranking.
"""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateIndex, CreateTable

from hafiz.core.database import Annotation, Base, CommunicationMessage, Embedding, Retrieval
from hafiz.core.dialect import (
    POSTGRESQL,
    SQLITE,
    UnsupportedOnBackendError,
    backend_of,
    cosine_distance,
    most_recalled_sql,
    similarity,
    tags_overlap,
)

PG = postgresql.dialect()
LITE = sqlite.dialect()

VECTOR = [0.1] * 768


def _sql(expr, dialect) -> str:
    return str(expr.compile(dialect=dialect, compile_kwargs={"literal_binds": False}))


def _table_ddl(model, dialect) -> str:
    return str(CreateTable(model.__table__).compile(dialect=dialect))


# ---------------------------------------------------------------------------
# Column factories — Postgres rendering must not have moved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "fragment"),
    [
        (Annotation, "id UUID NOT NULL"),
        (Annotation, "metadata JSONB"),
        (Annotation, "tags TEXT[]"),
        (Annotation, "embedding VECTOR(768)"),
        (Annotation, "valid_from TIMESTAMP WITH TIME ZONE"),
        (Retrieval, "result_ids UUID[]"),
    ],
)
def test_postgres_types_are_unchanged(model, fragment):
    """Phase 1 is a no-op migration or it is a bug.

    Hafiz has five shipped Alembic revisions written against these exact
    types. If a factory renders anything else, existing installs get a
    spurious autogenerate diff and a migration they do not need.
    """
    assert fragment in _table_ddl(model, PG)


@pytest.mark.parametrize(
    ("model", "fragment"),
    [
        (Annotation, "id CHAR(32) NOT NULL"),
        (Annotation, "metadata JSON"),
        (Annotation, "tags JSON"),
        (Annotation, "embedding BLOB"),
        (Annotation, "valid_from DATETIME"),
        (Retrieval, "result_ids JSON"),
    ],
)
def test_sqlite_variants_render(model, fragment):
    assert fragment in _table_ddl(model, LITE)


def test_every_table_renders_on_both_dialects():
    """No table may be Postgres-only by accident."""
    for table in Base.metadata.sorted_tables:
        assert str(CreateTable(table).compile(dialect=PG))
        assert str(CreateTable(table).compile(dialect=LITE))


# ---------------------------------------------------------------------------
# Partial indexes
# ---------------------------------------------------------------------------


PARTIAL_INDEXES = {
    "uq_unit_revisions_current": "superseded_at IS NULL",
    "uq_communications_agent_external": "external_id IS NOT NULL",
    "uq_messages_comm_source_id": "source_message_id IS NOT NULL",
    "idx_messages_salient": "marked_salient = true",
}


@pytest.mark.parametrize("dialect", [PG, LITE], ids=["postgresql", "sqlite"])
def test_partial_indexes_stay_partial_on_both_backends(dialect):
    """A partial unique index that loses its WHERE enforces a *different*
    constraint — so this is a correctness test, not a performance one.

    ``postgresql_where`` and ``sqlite_where`` are separate namespaced
    kwargs; passing only the first silently produces a full index on
    SQLite. ``uq_messages_comm_source_id`` is the one that would bite:
    full-unique over a nullable column would reject the second row whose
    ``source_message_id`` is NULL, which is every hand-built turn.
    """
    found = {}
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            if index.name in PARTIAL_INDEXES:
                found[index.name] = str(CreateIndex(index).compile(dialect=dialect))

    assert set(found) == set(PARTIAL_INDEXES), "a partial index went missing"
    for name, predicate in PARTIAL_INDEXES.items():
        assert "WHERE" in found[name], f"{name} lost its WHERE clause"
        assert predicate in found[name]


# ---------------------------------------------------------------------------
# Vector expressions
# ---------------------------------------------------------------------------


def test_similarity_is_parenthesised():
    """Regression: the custom construct declares no operator precedence.

    Unparenthesised, ``similarity()`` renders ``1 - embedding <=> $1``,
    which Postgres groups as ``(1 - embedding) <=> $1`` and rejects with
    "operator does not exist: integer - vector". Caught by 26 failing
    tests the first time; pinned here so it is caught by one.
    """
    rendered = _sql(similarity(Embedding.embedding, VECTOR), PG)
    # The distance is one bracketed unit, so the subtraction applies to the
    # whole of it rather than to the column alone.
    assert "(embeddings.embedding <=> " in rendered
    assert rendered.endswith(")")
    assert " - (" in rendered


def test_similarity_is_one_minus_distance_not_the_other_way_round():
    """Transposing these inverts every ranking in the product while every
    "results came back" assertion still passes."""
    dist = _sql(cosine_distance(Embedding.embedding, VECTOR), PG)
    sim = _sql(similarity(Embedding.embedding, VECTOR), PG)
    assert "<=>" in dist
    # The literal 1 is a bind param, hence the suffix match rather than ==.
    assert sim.endswith(f" - {dist}")
    assert sim.removesuffix(f" - {dist}").strip() != ""


@pytest.mark.parametrize(
    "column",
    [Embedding.embedding, Annotation.embedding, CommunicationMessage.embedding],
)
def test_all_three_embedding_columns_use_the_same_operator(column):
    assert "<=>" in _sql(cosine_distance(column, VECTOR), PG)


def test_vector_search_refuses_sqlite_rather_than_approximating():
    """Phase 3 owns this. Until then it must fail loudly.

    A backend that raises is debuggable. A backend that quietly returns
    differently-ordered results is not, and Hafiz serves one ``--json``
    contract from both.
    """
    with pytest.raises(UnsupportedOnBackendError, match="Phase 3"):
        _sql(cosine_distance(Embedding.embedding, VECTOR), LITE)


# ---------------------------------------------------------------------------
# tags_overlap — the operator with no SQLite equivalent
# ---------------------------------------------------------------------------


def test_tags_overlap_uses_the_array_operator_on_postgres():
    rendered = _sql(tags_overlap(Annotation.tags, ["auth", "db"]), PG)
    assert "&&" in rendered


def test_tags_overlap_falls_back_to_json_each_on_sqlite():
    rendered = _sql(tags_overlap(Annotation.tags, ["auth", "db"]), LITE)
    assert "json_each" in rendered
    assert "EXISTS" in rendered
    assert rendered.startswith("(") and rendered.endswith(")")


def test_tags_overlap_with_no_tags_matches_nothing_on_sqlite():
    """``&&`` against an empty array is false on Postgres, which carries the
    empty case as an ordinary bind param. SQLite builds an ``IN`` list by
    hand, so the empty case has to be spelled out — and it must be false,
    not a wildcard that quietly returns every annotation."""
    assert _sql(tags_overlap(Annotation.tags, []), LITE) == "0"


def test_tags_overlap_binds_its_values_rather_than_interpolating_them():
    """Tags reach this from ``--tags`` on the CLI. The SQLite branch builds
    SQL by string concatenation, so the values must still go through the
    compiler as bind params."""
    rendered = _sql(tags_overlap(Annotation.tags, ["'; DROP TABLE annotations--"]), LITE)
    assert "DROP TABLE" not in rendered
    assert "?" in rendered


# ---------------------------------------------------------------------------
# Runtime dispatch
# ---------------------------------------------------------------------------


def test_most_recalled_sql_differs_by_backend():
    pg = most_recalled_sql(POSTGRESQL)
    lite = most_recalled_sql(SQLITE)
    assert "unnest(" in pg and "::text" in pg and "left(" in pg
    # None of those three survive the dialect swap.
    assert "json_each(" in lite
    assert "unnest(" not in lite
    assert "::text" not in lite
    assert "substr(" in lite
    # Same contract: four columns, same names, same order.
    for sql in (pg, lite):
        assert "AS preview" in sql and "AS hits" in sql
        assert ":since" in sql and ":limit" in sql


def test_most_recalled_sql_refuses_an_unknown_backend():
    with pytest.raises(UnsupportedOnBackendError, match="mysql"):
        most_recalled_sql("mysql")


def test_backend_of_reads_the_bind_not_the_config():
    """Dispatching on configuration would emit Postgres SQL for a SQLite
    engine whenever the two disagree — i.e. on every run of a dual-backend
    test matrix."""
    # A sync engine, so this needs no driver Phase 2 has not shipped yet.
    from sqlalchemy import create_engine

    assert backend_of(create_engine("sqlite://")) == SQLITE


def test_backend_of_raises_on_something_unbindable():
    with pytest.raises(UnsupportedOnBackendError):
        backend_of(object())


# ---------------------------------------------------------------------------
# Runtime: the compile-time tests above cannot catch a transposed ranking
# ---------------------------------------------------------------------------


async def _db_available() -> bool:
    try:
        from sqlalchemy import text as sa_text

        from hafiz.core.database import get_session_factory

        async with get_session_factory()() as session:
            await session.execute(sa_text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


async def test_similarity_and_distance_agree_on_a_real_database():
    """The invariant the SQL-text tests cannot reach.

    Callers order by ``cosine_distance`` ascending and report
    ``similarity`` as the score. If those two ever disagree about
    direction, results still return, still fall in range, and still
    satisfy every "returns a list of SearchResult" assertion in the
    suite — they are just ranked backwards.

    Seeded with hand-built vectors rather than the ingested corpus, so it
    asserts a known order instead of skipping on an empty test database.
    """
    from sqlalchemy import select
    from sqlalchemy import text as sa_text

    from hafiz.core.database import Annotation, close_engine, get_session_factory

    if not await _db_available():
        pytest.skip("Postgres not reachable")

    mark = "dialect-ordering-probe"
    probe = [1.0] + [0.0] * 767
    # Angled progressively further from `probe`, so the expected order is
    # known without depending on anything already in the index.
    seeds = {
        "near": [1.0, 0.05] + [0.0] * 766,
        "mid": [1.0, 1.0] + [0.0] * 766,
        "far": [0.05, 1.0] + [0.0] * 766,
    }

    factory = get_session_factory()
    try:
        async with factory() as s:
            await s.execute(sa_text(f"DELETE FROM annotations WHERE content LIKE '{mark}%'"))
            for name, vec in seeds.items():
                s.add(Annotation(content=f"{mark}:{name}", kind="fact", embedding=vec))
            await s.commit()

        async with factory() as s:
            rows = (
                await s.execute(
                    select(Annotation.content, similarity(Annotation.embedding, probe))
                    .where(Annotation.content.like(f"{mark}%"))
                    .order_by(cosine_distance(Annotation.embedding, probe))
                )
            ).all()

        assert [c.split(":")[1] for c, _ in rows] == ["near", "mid", "far"]
        scores = [float(sc) for _, sc in rows]
        assert scores == sorted(scores, reverse=True), f"score disagrees with order: {scores}"
    finally:
        async with factory() as s:
            await s.execute(sa_text(f"DELETE FROM annotations WHERE content LIKE '{mark}%'"))
            await s.commit()
        await close_engine()
