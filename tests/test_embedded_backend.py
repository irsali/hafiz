"""The embedded SQLite + sqlite-vec backend.

These tests run against a real SQLite file rather than mocks. That is the
point: Phase 1 declared the SQLite column variants without ever executing
them, and three separate bugs (an absolute path parsed as relative, a
catalogue query against ``pg_tables``, a dedup query using pgvector's
``<=>``) survived a green suite because nothing opened an embedded
database.

Deliberately **not** skipped when something is missing. A dual-backend
suite that skips itself manufactures confidence about a backend nobody
exercised — this session watched ten Postgres tests silently skip while
the suite reported green.
"""

from __future__ import annotations

import sqlite3
import struct
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select, text

from hafiz.core.dialect import (
    SQLITE,
    SqliteUuidArray,
    SqliteVector,
    backend_of,
    clustered_annotations_sql,
    cosine_distance,
    db_file_path,
    is_embedded,
    normalize_url,
    similarity,
    table_list_sql,
)

# ---------------------------------------------------------------------------
# Fixtures — swap the process onto a real SQLite file, then put it back
# ---------------------------------------------------------------------------
# The engine is a module-level singleton and the shared conftest patches
# ``load_settings`` to pin the Postgres test URL, so both have to be
# redirected and both have to be restored. Leaving either in place would
# silently run the *rest* of the suite against SQLite.


@pytest.fixture
async def embedded_url(tmp_path):
    from hafiz.core import config as _config
    from hafiz.core.database import close_engine, create_tables

    url = f"sqlite:///{tmp_path / 'hafiz.db'}"
    real_load_settings = _config.load_settings

    def _patched():
        settings = real_load_settings()
        settings.database.url = url
        return settings

    _config.load_settings = _patched  # type: ignore[assignment]
    _config.reset_settings()
    await close_engine()
    try:
        await create_tables(url)
        yield url
    finally:
        await close_engine()
        _config.load_settings = real_load_settings  # type: ignore[assignment]
        _config.reset_settings()


@pytest.fixture
async def embedded_session(embedded_url):
    from hafiz.core.database import get_session_factory

    async with get_session_factory()() as session:
        yield session


# ---------------------------------------------------------------------------
# URL handling — where three of the four Phase 2 bugs lived
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected_driver"),
    [
        ("sqlite:///rel.db", "sqlite+aiosqlite"),
        ("sqlite:////abs/path.db", "sqlite+aiosqlite"),
        ("sqlite+aiosqlite:///already.db", "sqlite+aiosqlite"),
    ],
)
def test_sqlite_urls_get_an_async_driver(raw, expected_driver):
    """Users write the sync spelling; every DB call in Hafiz is async."""
    assert normalize_url(raw).startswith(expected_driver + "://")


def test_postgres_urls_are_left_alone():
    url = "postgresql+asyncpg://u:p@h:5432/db"
    assert normalize_url(url) == url


def test_absolute_paths_stay_absolute():
    """Regression: hand-rolled URL parsing turned ``sqlite:////abs/x.db`` into
    the *relative* path ``abs/x.db``. Nothing failed loudly — the database
    was simply created somewhere else, which then silently defeated the
    0600 hardening, because it chmod'd a file that did not exist."""
    path = db_file_path("sqlite:////tmp/hafiz-abs-test.db")
    assert path is not None
    assert path.is_absolute()
    assert str(path) == "/tmp/hafiz-abs-test.db"


def test_relative_and_memory_urls():
    assert db_file_path("sqlite:///rel.db").as_posix() == "rel.db"
    assert db_file_path("sqlite://") is None  # in-memory has no file
    assert db_file_path("postgresql+asyncpg://u:p@h/db") is None


def test_is_embedded_keys_off_the_scheme():
    assert is_embedded("sqlite:///x.db")
    assert is_embedded("sqlite+aiosqlite:///x.db")
    assert not is_embedded("postgresql+asyncpg://u:p@h/db")


# ---------------------------------------------------------------------------
# Value conversions
# ---------------------------------------------------------------------------


def test_vector_packs_little_endian_float32():
    """sqlite-vec reads exactly this layout. Native byte order would produce
    vectors it misreads as garbage on a big-endian host — and distances
    would still come back, plausible and wrong."""
    packed = SqliteVector().process_bind_param([1.0, -2.5, 0.0], None)
    assert packed == struct.pack("<3f", 1.0, -2.5, 0.0)
    assert len(packed) == 12  # 3 x float32, no header


def test_vector_round_trips():
    vec = [round(i * 0.001, 6) for i in range(768)]
    t = SqliteVector()
    out = t.process_result_value(t.process_bind_param(vec, None), None)
    assert len(out) == 768
    assert out == pytest.approx(vec, abs=1e-6)


def test_uuid_array_round_trips_as_uuids_not_strings():
    """Without the result half, callers get ``str`` where they had
    ``uuid.UUID`` and every comparison silently stops matching."""
    ids = [uuid.uuid4(), uuid.uuid4()]
    t = SqliteUuidArray()
    stored = t.process_bind_param(ids, None)
    assert stored == [str(i) for i in ids]
    assert t.process_result_value(stored, None) == ids


def test_uuid_array_tolerates_a_junk_id():
    assert SqliteUuidArray().process_result_value(["not-a-uuid"], None) == []


def test_timestamps_come_back_timezone_aware():
    """SQLAlchemy's SQLite ``DATETIME`` silently ignores ``timezone=True``.

    Postgres returns ``tzinfo=UTC``; unadjusted SQLite returns naive. That is
    a contract divergence, not a storage detail — it surfaced as "can't
    subtract offset-naive and offset-aware datetimes" on one backend only,
    and as a retention window that compared wrong.
    """
    from hafiz.core.dialect import SqliteUtcDateTime

    t = SqliteUtcDateTime()
    aware = datetime(2026, 9, 5, 12, 30, tzinfo=UTC)
    stored = t.process_bind_param(aware, None)
    assert stored.tzinfo is None, "stored naive, in UTC"
    assert t.process_result_value(stored, None) == aware


def test_a_naive_timestamp_is_read_as_utc_not_local():
    """Guessing local time would shift every pre-existing row by the
    developer's offset."""
    from hafiz.core.dialect import SqliteUtcDateTime

    naive = datetime(2026, 9, 5, 12, 30)
    out = SqliteUtcDateTime().process_result_value(naive, None)
    assert out.tzinfo is UTC
    assert out.replace(tzinfo=None) == naive


def test_unnest_ids_needs_a_different_shape_per_backend():
    """Not a different spelling — a different shape. ``unnest`` goes in the
    select list; ``json_each`` is table-valued and must sit in FROM beside
    its source row, or it compiles to ``no such column: retrievals.result_ids``."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.dialects import sqlite as sqlite_dialect

    from hafiz.core.database import Retrieval
    from hafiz.core.dialect import unnest_ids

    pg = str(
        unnest_ids(Retrieval.result_ids, Retrieval, "postgresql")
        .select()
        .compile(dialect=postgresql.dialect())
    )
    lite = str(
        unnest_ids(Retrieval.result_ids, Retrieval, SQLITE)
        .select()
        .compile(dialect=sqlite_dialect.dialect())
    )
    assert "unnest(" in pg
    assert "json_each(" in lite
    # The source table must be in the SQLite FROM clause, else the column
    # reference inside json_each() has nothing to resolve against.
    assert "retrievals" in lite.split("FROM", 1)[1]


# ---------------------------------------------------------------------------
# Dialect-specific SQL
# ---------------------------------------------------------------------------


def test_catalogue_and_dedup_sql_differ_by_backend():
    """Both of these shipped as Postgres-only raw SQL and failed at runtime
    on SQLite — one with ``no such table: pg_tables``, the other with a
    syntax error on ``<=>``."""
    assert "pg_tables" in table_list_sql("postgresql")
    assert "sqlite_master" in table_list_sql(SQLITE)

    pg, lite = clustered_annotations_sql("postgresql"), clustered_annotations_sql(SQLITE)
    assert "<=>" in pg and "IS NOT DISTINCT FROM" in pg
    assert "vec_distance_cosine" in lite and "IS NOT DISTINCT FROM" not in lite


# ---------------------------------------------------------------------------
# Live database
# ---------------------------------------------------------------------------


async def test_connection_pragmas_are_applied(embedded_session):
    """``foreign_keys`` and ``secure_delete`` are correctness, not tuning.

    SQLite ships foreign keys **off**, so without the pragma every
    ``ondelete=CASCADE`` in the schema silently does nothing. And
    ``secure_delete``'s default is set at compile time, so relying on it
    made ``forget --hard`` hold on some machines and not others.
    """
    expected = {
        "journal_mode": "wal",
        "foreign_keys": 1,
        "busy_timeout": 5000,
        "secure_delete": 1,
    }
    for pragma, want in expected.items():
        got = (await embedded_session.execute(text(f"PRAGMA {pragma}"))).scalar()
        assert got == want, f"PRAGMA {pragma} was {got!r}, expected {want!r}"


async def test_sqlite_vec_is_loaded(embedded_session):
    version = (await embedded_session.execute(text("SELECT vec_version()"))).scalar()
    assert version.startswith("v")


async def test_backend_of_reports_sqlite(embedded_session):
    assert backend_of(embedded_session) == SQLITE


async def test_schema_is_stamped_at_head(embedded_session):
    """Bootstrapping with ``create_all`` and *not* stamping would strand
    every future upgrade: ``create_all`` never ALTERs an existing table, so
    a user who updates Hafiz after a schema change gets a silently stale
    database."""
    stamped = (
        await embedded_session.execute(text("SELECT version_num FROM alembic_version"))
    ).scalar()
    assert stamped is not None and stamped != ""


async def test_uuid_and_vector_survive_a_write_and_read(embedded_session):
    from hafiz.core.database import Annotation

    vec = [0.5] * 768
    row = Annotation(content="round-trip probe", kind="fact", tags=["a", "b"], embedding=vec)
    embedded_session.add(row)
    await embedded_session.commit()

    got = (
        await embedded_session.execute(select(Annotation).where(Annotation.id == row.id))
    ).scalar_one()
    assert isinstance(got.id, uuid.UUID)
    assert got.tags == ["a", "b"]
    assert got.embedding == pytest.approx(vec, abs=1e-6)


async def test_foreign_key_cascade_actually_fires(embedded_session):
    """Guards the pragma from the other side: with foreign_keys off this
    leaves orphaned messages instead of deleting them, which would turn the
    retention guarantee into a lie without any error."""
    from hafiz.core.database import Communication, CommunicationMessage

    comm = Communication(agent="t", external_id="fk-probe", started_at=datetime.now(UTC))
    embedded_session.add(comm)
    await embedded_session.flush()
    embedded_session.add(
        CommunicationMessage(
            communication_id=comm.id, seq=0, role="user", content="x", ts=datetime.now(UTC)
        )
    )
    await embedded_session.commit()

    await embedded_session.delete(await embedded_session.get(Communication, comm.id))
    await embedded_session.commit()

    remaining = (
        await embedded_session.execute(
            select(CommunicationMessage).where(CommunicationMessage.communication_id == comm.id)
        )
    ).all()
    assert remaining == []


async def test_vector_search_ranks_correctly_on_sqlite_vec(embedded_session):
    """Same invariant as the Postgres test, on the other backend: ordered by
    distance ascending, scored as ``1 - distance``."""
    from hafiz.core.database import Annotation

    probe = [1.0] + [0.0] * 767
    seeds = {
        "near": [1.0, 0.05] + [0.0] * 766,
        "mid": [1.0, 1.0] + [0.0] * 766,
        "far": [0.05, 1.0] + [0.0] * 766,
    }
    for name, vec in seeds.items():
        embedded_session.add(Annotation(content=f"rank:{name}", kind="fact", embedding=vec))
    await embedded_session.commit()

    rows = (
        await embedded_session.execute(
            select(Annotation.content, similarity(Annotation.embedding, probe))
            .where(Annotation.content.like("rank:%"))
            .order_by(cosine_distance(Annotation.embedding, probe))
        )
    ).all()

    assert [c.split(":")[1] for c, _ in rows] == ["near", "mid", "far"]
    scores = [float(s) for _, s in rows]
    assert scores == sorted(scores, reverse=True)


async def test_timestamps_survive_a_real_round_trip_aware(embedded_session):
    """The end-to-end form of the timezone test: through the column type,
    the driver, and back."""
    from hafiz.core.database import Annotation

    row = Annotation(content="tz round-trip", kind="fact")
    embedded_session.add(row)
    await embedded_session.commit()
    embedded_session.expunge_all()

    got = (
        await embedded_session.execute(select(Annotation).where(Annotation.id == row.id))
    ).scalar_one()
    assert got.valid_from.tzinfo is not None, "read back naive — Postgres would be aware"
    # Must be comparable with an aware datetime without raising.
    assert (datetime.now(UTC) - got.valid_from).total_seconds() < 120


async def test_self_distance_is_zero(embedded_session):
    """The packing check that a smoke test cannot fake: a vector compared
    with itself must be exactly 0 distance. A wrong dtype or byte order
    still returns a number, just not this one."""
    from hafiz.core.database import Annotation

    vec = [0.3, -0.7, 0.1] + [0.0] * 765
    embedded_session.add(Annotation(content="self-distance", kind="fact", embedding=vec))
    await embedded_session.commit()

    distance = (
        await embedded_session.execute(
            select(cosine_distance(Annotation.embedding, vec)).where(
                Annotation.content == "self-distance"
            )
        )
    ).scalar()
    assert float(distance) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Redaction — the compliance guarantee
# ---------------------------------------------------------------------------


def test_vacuum_is_what_removes_deleted_bytes(tmp_path):
    """Proof that ``forget --hard``'s VACUUM is load-bearing.

    Written against raw sqlite3 with ``secure_delete`` forced **off**,
    because this machine's SQLite is compiled with ``SQLITE_SECURE_DELETE``
    and therefore zeroes freed pages by itself — which made an earlier
    version of this test pass with the VACUUM removed. Stock upstream
    SQLite does not, so the guarantee has to be verified under *those*
    conditions, not this host's.
    """
    db_path = tmp_path / "residue.db"
    secret = "MAGIC-SECRET-PHRASE-DO-NOT-LEAK"
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA secure_delete=OFF")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("CREATE TABLE msgs (id INTEGER PRIMARY KEY, comm INT, body TEXT)")
    for comm in range(10):
        for turn in range(60):
            body = (secret if comm == 4 else "ordinary") + f" {turn} " + "padding " * 60
            conn.execute("INSERT INTO msgs (comm, body) VALUES (?,?)", (comm, body))
    conn.commit()
    assert secret.encode() in db_path.read_bytes(), "precondition: content is on disk"

    conn.execute("PRAGMA secure_delete=OFF")
    conn.execute("DELETE FROM msgs WHERE comm = 4")
    conn.commit()
    assert secret.encode() in db_path.read_bytes(), (
        "DELETE alone leaves the bytes readable — this is exactly why forget --hard must VACUUM"
    )

    conn.execute("VACUUM")
    conn.commit()
    conn.close()
    assert secret.encode() not in db_path.read_bytes()


async def test_hard_forget_reports_that_it_vacuumed(embedded_session, embedded_url):
    from hafiz.core.communications import forget_communication
    from hafiz.core.database import Communication, CommunicationMessage

    comm = Communication(agent="t", external_id="redact", started_at=datetime.now(UTC))
    embedded_session.add(comm)
    await embedded_session.flush()
    embedded_session.add(
        CommunicationMessage(
            communication_id=comm.id, seq=0, role="user", content="secret", ts=datetime.now(UTC)
        )
    )
    await embedded_session.commit()
    comm_id = comm.id
    await embedded_session.close()

    result = await forget_communication(comm_id, hard=True)
    assert result["hard"] is True
    assert result["deleted_messages"] == 1
    assert result["vacuumed"] is True, "embedded hard-delete must reclaim freed pages"


async def test_db_file_is_owner_only(embedded_session, embedded_url):
    """Postgres gated the brain behind database auth; a single file inherits
    the umask instead, commonly 0644.

    Takes ``embedded_session`` so a connection has definitely been opened —
    the hardening runs on connect, which is the only moment the file (and
    its ``-wal`` sibling) provably exists.
    """
    import stat

    path = db_file_path(embedded_url)
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    wal = path.with_name(path.name + "-wal")
    if wal.exists():
        assert stat.S_IMODE(wal.stat().st_mode) == 0o600, "the WAL holds recent turns too"
