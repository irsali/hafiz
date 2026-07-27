"""Tests for retention enforcement being *visible*, not just available.

Bounded 90-day retention is an outward-facing commitment, but nothing in hafiz
ever ran the sweep — it required the user to remember
``hafiz forget --all-expired``. Measured: 358 communications sat past
``retention_until`` indefinitely.

The sweep now runs on ``import``, which is the command that grows the source
layer. But an import-bound trigger provably isn't enough: in the observed case
imports stopped on 2026-06-30 and there were 358 overdue rows by 07-26 — the
trigger stops firing exactly when it's needed, because retention keeps ticking
after you stop importing. So the count is a first-class field on both ``status``
and ``doctor``. Visibility is the enforcement mechanism.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from hafiz.core.communications import (
    count_overdue_communications,
    tombstone_expired_communications,
)
from hafiz.core.database import Communication, close_engine, get_session_factory

AGENT = "retention-test"


async def _db_available() -> tuple[bool, str]:
    """(reachable, why-not). The reason is carried out rather than swallowed:
    a bare "Postgres not reachable" skip once hid a dirtied connection pool for
    an entire suite run, which is the same silent-failure shape this branch is
    about."""
    try:
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text("SELECT 1 FROM communications LIMIT 1"))
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


async def _wipe() -> None:
    factory = get_session_factory()
    async with factory() as s:
        await s.execute(text(f"DELETE FROM communications WHERE agent = '{AGENT}'"))
        await s.commit()


@pytest.fixture
async def db():
    """Requested explicitly, not autouse: the CLI-level test below drives its
    own ``asyncio.run`` and can't share this fixture's event loop."""
    ok, why = await _db_available()
    if not ok:
        pytest.skip(f"Postgres not reachable — {why}")
    await _wipe()
    yield
    await _wipe()
    await close_engine()


async def _seed(*, retention_offset_days: int | None, tombstoned: bool = False) -> uuid.UUID:
    """Insert one communication whose retention is N days in the past/future."""
    now = datetime.now(UTC)
    comm_id = uuid.uuid4()
    factory = get_session_factory()
    async with factory() as s:
        s.add(
            Communication(
                id=comm_id,
                agent=AGENT,
                external_id=str(comm_id),
                started_at=now - timedelta(days=120),
                retention_until=(
                    None
                    if retention_offset_days is None
                    else now - timedelta(days=retention_offset_days)
                ),
                valid_until=now if tombstoned else None,
            )
        )
        await s.commit()
    return comm_id


async def _overdue_for_agent() -> int:
    """Count only this test's rows, so a shared DB can't skew the assertion."""
    factory = get_session_factory()
    async with factory() as s:
        return (
            await s.execute(
                text(
                    "SELECT count(*) FROM communications WHERE agent = :a "
                    "AND retention_until IS NOT NULL AND retention_until <= now() "
                    "AND valid_until IS NULL"
                ),
                {"a": AGENT},
            )
        ).scalar()


# ── count_overdue_communications ────────────────────────────────────────


async def test_counts_a_row_past_retention(db):
    await _seed(retention_offset_days=5)
    assert await _overdue_for_agent() == 1
    assert await count_overdue_communications() >= 1


async def test_does_not_count_a_row_still_within_retention(db):
    await _seed(retention_offset_days=-30)  # 30 days in the future
    assert await _overdue_for_agent() == 0


async def test_does_not_count_an_already_tombstoned_row(db):
    """Already swept rows aren't outstanding work."""
    await _seed(retention_offset_days=5, tombstoned=True)
    assert await _overdue_for_agent() == 0


async def test_does_not_count_a_row_with_no_retention_set(db):
    await _seed(retention_offset_days=None)
    assert await _overdue_for_agent() == 0


async def test_count_drops_to_zero_after_a_sweep(db):
    await _seed(retention_offset_days=5)
    assert await _overdue_for_agent() == 1
    await tombstone_expired_communications()
    assert await _overdue_for_agent() == 0


async def test_dry_run_sweep_leaves_the_count_alone(db):
    await _seed(retention_offset_days=5)
    result = await tombstone_expired_communications(dry_run=True)
    assert result["matched"] >= 1
    assert result["tombstoned"] == 0
    assert await _overdue_for_agent() == 1


async def test_sweep_is_a_soft_tombstone_not_a_delete(db):
    """Retention prunes from recall but preserves the audit trail."""
    comm_id = await _seed(retention_offset_days=5)
    await tombstone_expired_communications()
    factory = get_session_factory()
    async with factory() as s:
        row = await s.get(Communication, comm_id)
        assert row is not None, "row must survive for audit"
        assert row.valid_until is not None


# ── status / doctor surfacing ───────────────────────────────────────────


def test_status_reports_the_overdue_count():
    """The field an operator reads. Absent it, 358 rows went unnoticed for
    four weeks.

    Synchronous on purpose: ``status`` drives its own ``asyncio.run``, which
    cannot nest inside a running event loop.
    """
    import asyncio
    import json

    from typer.testing import CliRunner

    from hafiz.cli import app

    def _run(coro):
        """Run one coroutine on a fresh loop, then drop the engine.

        The engine is a module-level singleton whose connection pool binds to
        whichever loop first used it. Reusing it across ``asyncio.run`` calls
        raises "attached to a different loop", so each step closes it.
        """

        async def _wrapped():
            try:
                return await coro
            finally:
                await close_engine()

        return asyncio.run(_wrapped())

    ok, why = _run(_db_available())
    if not ok:
        pytest.skip(f"Postgres not reachable — {why}")

    _run(_seed(retention_offset_days=5))
    try:
        result = CliRunner().invoke(app, ["status", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "retention" in payload
        assert payload["retention"]["overdue"] >= 1
    finally:
        _run(_wipe())
