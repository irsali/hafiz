"""Tests for the retrieval log — the data hafiz needed to evaluate itself.

Hafiz could not say which annotations had ever been recalled, which surfaced and
were useful, or which had never come up once. Auditing a 3.5-week deployment for
"is this earning its keep?" required parsing 169 Claude Code transcripts, because
hafiz kept no record of its own reads.

Three properties are load-bearing, in this order:

1. **Recording must never fail a search.** A memory layer that can break the read
   path gets removed, so every failure mode here degrades to "no row written".
2. **The serving path must not be able to bypass it.** ``hafiz/core/daemon.py``
   calls ``search_annotations`` directly, so telemetry lives in core — wiring it
   at the command layer would have silently missed every warm request.
3. **It's opt-out-able and retention-bounded**, because the query text is a new
   category of data for this store: what somebody was *looking for*.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from hafiz.core import telemetry
from hafiz.core.config import reset_settings
from hafiz.core.database import Retrieval, close_engine, get_session_factory

MARK = "telemetry-test"


async def _db_available() -> bool:
    try:
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text("SELECT 1 FROM retrievals LIMIT 1"))
        return True
    except Exception:
        return False


async def _wipe() -> None:
    factory = get_session_factory()
    async with factory() as s:
        await s.execute(text(f"DELETE FROM retrievals WHERE query_text LIKE '{MARK}%'"))
        await s.commit()


@pytest.fixture
async def db():
    if not await _db_available():
        pytest.skip("Postgres not reachable (or migration 0007 not applied)")
    reset_settings()
    await _wipe()
    yield
    await _wipe()
    reset_settings()
    await close_engine()


async def _rows(query_prefix: str = MARK) -> list[Retrieval]:
    from sqlalchemy import select

    factory = get_session_factory()
    async with factory() as s:
        return list(
            (
                await s.execute(
                    select(Retrieval)
                    .where(Retrieval.query_text.startswith(query_prefix))
                    .order_by(Retrieval.at)
                )
            )
            .scalars()
            .all()
        )


# ── Recording ───────────────────────────────────────────────────────────


async def test_a_search_is_recorded_with_what_came_back(db):
    ids = [uuid.uuid4(), uuid.uuid4()]
    await telemetry.record_retrieval(
        command=telemetry.OBSERVATIONS,
        query=f"{MARK} why did we revert the Canada default",
        result_ids=ids,
        top_score=0.94,
        reranked=True,
        filters={"project": "p", "limit": 10},
    )
    (row,) = await _rows()
    assert row.command == "query --observations"
    assert row.n_results == 2
    assert row.result_ids == ids
    assert row.top_score == pytest.approx(0.94)
    assert row.reranked is True
    assert row.filters == {"project": "p", "limit": 10}


async def test_an_empty_result_set_is_recorded_not_dropped(db):
    """The whole point: a query that found nothing is the most useful row here,
    because it says what the store is missing."""
    await telemetry.record_retrieval(
        command=telemetry.QUERY, query=f"{MARK} something nobody wrote down"
    )
    (row,) = await _rows()
    assert row.n_results == 0
    assert row.result_ids == []
    assert row.top_score is None


async def test_null_filters_are_dropped_so_the_row_stays_readable(db):
    await telemetry.record_retrieval(
        command=telemetry.QUERY,
        query=f"{MARK} filters",
        filters={"project": None, "kind": "decision", "limit": None},
    )
    (row,) = await _rows()
    assert row.filters == {"kind": "decision"}


async def test_retention_is_bounded_on_write(db):
    await telemetry.record_retrieval(command=telemetry.QUERY, query=f"{MARK} retention")
    (row,) = await _rows()
    assert row.retention_until is not None
    expected = datetime.now(UTC) + timedelta(days=90)
    assert abs((row.retention_until - expected).total_seconds()) < 120


async def test_string_ids_are_accepted(db):
    """Callers hand over ``str(uuid)`` — SearchResult.id is a string."""
    an_id = uuid.uuid4()
    await telemetry.record_retrieval(
        command=telemetry.QUERY, query=f"{MARK} strings", result_ids=[str(an_id)]
    )
    (row,) = await _rows()
    assert row.result_ids == [an_id]


async def test_a_non_uuid_id_is_skipped_rather_than_losing_the_row(db):
    await telemetry.record_retrieval(
        command=telemetry.QUERY, query=f"{MARK} mixed", result_ids=["not-a-uuid"]
    )
    (row,) = await _rows()
    assert row.result_ids == []
    assert row.n_results == 0


# ── The three constraints ───────────────────────────────────────────────


async def test_recording_is_off_when_configured_off(db, monkeypatch):
    monkeypatch.setenv("HAFIZ_TELEMETRY__RETRIEVAL", "false")
    reset_settings()
    await telemetry.record_retrieval(command=telemetry.QUERY, query=f"{MARK} disabled")
    assert await _rows() == []


async def test_a_trivial_query_is_not_worth_recording(db):
    """ "yes" / "ok" say nothing about what the store was asked for."""
    await telemetry.record_retrieval(command=telemetry.QUERY, query="ok")
    await telemetry.record_retrieval(command=telemetry.QUERY, query="  ")
    assert await _rows("ok") == []


async def test_a_broken_store_does_not_raise(monkeypatch):
    """Constraint 1. This must hold even with no database at all."""

    def _boom():
        raise RuntimeError("db is gone")

    monkeypatch.setattr("hafiz.core.database.get_session_factory", _boom)
    await telemetry.record_retrieval(command=telemetry.QUERY, query="a query that cannot be stored")


async def test_a_broken_session_lookup_does_not_raise(monkeypatch):
    def _boom(_):
        raise RuntimeError("no cursor")

    monkeypatch.setattr("hafiz.core.session.resolve_session_tag", _boom)
    assert await telemetry._ambient_session_id() is None


def test_search_records_by_default_so_a_new_caller_cannot_forget():
    """Constraint 2, enforced by the signature rather than by discipline: the
    default is the label, and silence is the thing you have to ask for."""
    import inspect

    from hafiz.core.annotations import search_annotations
    from hafiz.core.search import vector_search

    assert (
        inspect.signature(vector_search).parameters["telemetry_command"].default == telemetry.QUERY
    )
    assert (
        inspect.signature(search_annotations).parameters["telemetry_command"].default
        == telemetry.OBSERVATIONS
    )


async def test_context_records_under_its_own_labels(monkeypatch):
    """`context` fans out to both layers. Labelling those calls `query` /
    `query --observations` made the flagship command invisible in its own
    telemetry — a 30-day audit could not distinguish "never used" from "never
    instrumented", and `telemetry.CONTEXT` sat defined-but-unreferenced.

    Asserted on the calls rather than on the source text: a `getsource`
    substring count passes even if `vector_search` ignored the argument, and
    breaks on a harmless reformat."""
    from hafiz.core import context as context_mod

    seen = {}

    async def fake_vector_search(query, **kw):
        seen["units"] = kw.get("telemetry_command")
        return []

    async def fake_search_annotations(query, **kw):
        seen["annotations"] = kw.get("telemetry_command")
        return []

    monkeypatch.setattr(context_mod, "vector_search", fake_vector_search)
    monkeypatch.setattr(context_mod, "search_annotations", fake_search_annotations)
    await context_mod.build_context("anything at all")

    assert seen["units"] == telemetry.CONTEXT
    assert seen["annotations"] == telemetry.CONTEXT_OBSERVATIONS
    # Distinct from the plain-query labels, or the whole point is lost.
    assert telemetry.CONTEXT != telemetry.QUERY
    assert telemetry.CONTEXT_OBSERVATIONS != telemetry.OBSERVATIONS


async def test_the_report_breaks_down_by_command(db):
    """A label with no reader is not instrumentation: the report has to say
    which entry points are in use, or wiring `context` changes nothing you can
    see."""
    await telemetry.record_retrieval(
        command=telemetry.CONTEXT, query=f"{MARK} bundle", result_ids=[], top_score=None
    )
    await telemetry.record_retrieval(
        command=telemetry.CONTEXT_OBSERVATIONS,
        query=f"{MARK} bundle",
        result_ids=[uuid.uuid4()],
        top_score=0.9,
    )
    report = await telemetry.retrieval_report(since_days=1)

    rows = {r["command"]: r for r in report["by_command"]}
    assert telemetry.CONTEXT in rows
    assert telemetry.CONTEXT_OBSERVATIONS in rows
    assert rows[telemetry.CONTEXT]["calls"] >= 1
    # Counts are per-command, and every row carries the same keys so a caller
    # can tabulate without probing for absent fields.
    for r in report["by_command"]:
        assert set(r) == {"command", "calls", "empty", "avg_top_score"}
        assert isinstance(r["empty"], int)
    # Not asserted equal to report["retrievals"]: `total` and `by_command` are
    # separate statements at READ COMMITTED, so each gets its own snapshot. On a
    # live store with a per-prompt recall hook writing rows, one concurrent
    # insert between them would fail an equality — a flaky test, not a defect.
    assert sum(r["calls"] for r in report["by_command"]) >= 2
    assert rows[telemetry.CONTEXT]["empty"] == 1  # the result_ids=[] call
    assert rows[telemetry.CONTEXT]["avg_top_score"] is None  # NULLs are skipped, not zeroed


# ── Retention ───────────────────────────────────────────────────────────


async def _seed(*, overdue_days: int | None, tombstoned: bool = False) -> uuid.UUID:
    row_id = uuid.uuid4()
    now = datetime.now(UTC)
    factory = get_session_factory()
    async with factory() as s:
        s.add(
            Retrieval(
                id=row_id,
                at=now - timedelta(days=200),
                command="query",
                query_text=f"{MARK} seeded",
                retention_until=(
                    None if overdue_days is None else now - timedelta(days=overdue_days)
                ),
                valid_until=now if tombstoned else None,
            )
        )
        await s.commit()
    return row_id


async def test_overdue_rows_are_counted(db):
    await _seed(overdue_days=3)
    assert await telemetry.count_overdue_retrievals() >= 1


async def test_a_row_inside_its_window_is_not_overdue(db):
    before = await telemetry.count_overdue_retrievals()
    await _seed(overdue_days=-30)
    assert await telemetry.count_overdue_retrievals() == before


async def test_the_sweep_is_a_soft_tombstone(db):
    row_id = await _seed(overdue_days=3)
    await telemetry.tombstone_expired_retrievals()
    factory = get_session_factory()
    async with factory() as s:
        row = await s.get(Retrieval, row_id)
        assert row is not None, "kept for audit"
        assert row.valid_until is not None


async def test_a_dry_run_sweep_changes_nothing(db):
    await _seed(overdue_days=3)
    result = await telemetry.tombstone_expired_retrievals(dry_run=True)
    assert result["matched"] >= 1
    assert result["tombstoned"] == 0
    assert await telemetry.count_overdue_retrievals() >= 1


# ── The report ──────────────────────────────────────────────────────────


async def test_unanswered_queries_are_surfaced_with_their_frequency(db):
    for _ in range(3):
        await telemetry.record_retrieval(command=telemetry.QUERY, query=f"{MARK} nothing here")
    await telemetry.record_retrieval(
        command=telemetry.QUERY, query=f"{MARK} found something", result_ids=[uuid.uuid4()]
    )

    report = await telemetry.retrieval_report(since_days=1)
    gaps = {row["query"]: row["times"] for row in report["unanswered"]}
    assert gaps.get(f"{MARK} nothing here") == 3
    assert f"{MARK} found something" not in gaps


async def test_the_empty_rate_is_reported(db):
    await telemetry.record_retrieval(command=telemetry.QUERY, query=f"{MARK} empty one")
    await telemetry.record_retrieval(
        command=telemetry.QUERY, query=f"{MARK} full one", result_ids=[uuid.uuid4()]
    )
    report = await telemetry.retrieval_report(since_days=1)
    assert report["empty_result_rate"] is not None
    assert 0.0 < report["empty_result_rate"] <= 1.0


async def test_the_report_says_how_much_it_is_blind_to(db):
    """A "never recalled" count is misleading on its own — everything written
    before telemetry existed has never been recalled by definition.

    ``blind_before`` must be a genuine *subset* of ``never_recalled``, or the
    "of which" framing in the report is a lie.
    """
    report = await telemetry.retrieval_report(since_days=1)
    assert isinstance(report["blind_before"], int)
    assert report["blind_before"] <= report["never_recalled"]


async def test_the_report_is_empty_not_broken_with_no_data(db):
    report = await telemetry.retrieval_report(since_days=1)
    assert isinstance(report["never_recalled"], int)
    assert isinstance(report["unanswered"], list)
    assert isinstance(report["most_recalled"], list)


# ── CLI ─────────────────────────────────────────────────────────────────


def test_the_command_reports_json():
    import json as _json

    from typer.testing import CliRunner

    from hafiz.cli import app

    result = CliRunner().invoke(app, ["retrievals", "--since-days", "1", "--json"])
    assert result.exit_code == 0, result.output
    payload = _json.loads(result.output)
    assert payload["ok"] is True
    assert "never_recalled" in payload
    assert "unanswered" in payload
    assert payload["enabled"] is True


def test_the_command_is_documented():
    from typer.testing import CliRunner

    from hafiz.cli import app

    out = CliRunner().invoke(app, ["retrievals", "--help"]).output
    assert "--since-days" in out
    assert "--json" in out
