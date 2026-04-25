"""Phase 6 — `hafiz forget` (targeted redaction + retention sweep).

The retention sweeper unit tests live in
``tests/test_communications_schema.py``; this file exercises the CLI
surface of the new ``hafiz forget`` command.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core.communications import (
    MessageInput,
    append_messages,
    get_communication,
    upsert_communication,
)
from hafiz.core.database import close_engine
from hafiz.core.sessions import create_session


@pytest.fixture(autouse=True)
async def _isolate_engine():
    await close_engine()
    yield
    await close_engine()


def _seed_comm_with_msg(*, agent="claude-code", project: str | None = None):
    """Synchronously create a communication + one short message. Returns
    the communication id (uuid). Closes the engine on exit so the next
    asyncio.run (CliRunner) starts fresh."""

    async def _go():
        try:
            sess = await create_session(
                slug=f"forget-test-{uuid.uuid4().hex[:6]}",
                name="forget test",
                scope_kind="project" if project else None,
                scope_value=project,
            )
            comm, _ = await upsert_communication(
                agent=agent,
                external_id=f"forget-{uuid.uuid4().hex[:8]}",
                session_id=sess.id,
            )
            await append_messages(
                comm.id,
                [
                    MessageInput(
                        seq=0,
                        role="user",
                        content="please forget me",
                        ts=datetime.now(timezone.utc),
                    )
                ],
                embed=False,
            )
            return sess.slug, comm.id
        finally:
            await close_engine()

    return asyncio.run(_go())


def test_forget_help():
    runner = CliRunner()
    result = runner.invoke(app, ["forget", "--help"])
    assert result.exit_code == 0
    assert "--hard" in result.output
    assert "--all-expired" in result.output


def test_forget_requires_target_or_sweep():
    runner = CliRunner()
    result = runner.invoke(app, ["forget"])
    assert result.exit_code == 1
    assert "target" in result.output.lower()


def test_forget_targeted_soft_then_hard():
    _, comm_id = _seed_comm_with_msg()

    runner = CliRunner()
    result = runner.invoke(app, ["forget", str(comm_id), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["communications_affected"] == 1
    assert payload["hard"] is False

    # Tombstoned but still readable with include_tombstoned.
    async def _check_tombstoned():
        try:
            return await get_communication(comm_id, include_tombstoned=True)
        finally:
            await close_engine()

    row = asyncio.run(_check_tombstoned())
    assert row is not None
    assert row.valid_until is not None

    result = runner.invoke(app, ["forget", str(comm_id), "--hard", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["hard"] is True

    async def _check_gone():
        try:
            return await get_communication(comm_id, include_tombstoned=True)
        finally:
            await close_engine()

    assert asyncio.run(_check_gone()) is None


def test_forget_unknown_target_returns_zero_affected():
    runner = CliRunner()
    bogus = f"unknown-{uuid.uuid4().hex[:6]}"
    result = runner.invoke(app, ["forget", bogus, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["communications_affected"] == 0


def test_forget_via_session_slug_affects_all_comms_in_session():
    """When the target is a session slug, every communication in that
    session is forgotten."""

    async def _seed_two():
        try:
            sess = await create_session(
                slug=f"forget-slug-{uuid.uuid4().hex[:6]}",
                name="forget slug test",
            )
            for _ in range(2):
                comm, _ = await upsert_communication(
                    agent="claude-code",
                    external_id=f"forget-{uuid.uuid4().hex[:8]}",
                    session_id=sess.id,
                )
            return sess.slug
        finally:
            await close_engine()

    slug = asyncio.run(_seed_two())
    runner = CliRunner()
    result = runner.invoke(app, ["forget", slug, "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["communications_affected"] == 2


def test_forget_all_expired_dry_run_reports_zero_when_clean():
    """With no expired communications, the sweep matches 0."""
    runner = CliRunner()
    result = runner.invoke(
        app, ["forget", "--all-expired", "--dry-run", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["matched"] >= 0


def test_forget_all_expired_tombstones_past_retention():
    """An expired-retention communication gets tombstoned by sweep."""

    async def _seed_expired():
        try:
            sess = await create_session(
                slug=f"forget-sweep-{uuid.uuid4().hex[:6]}",
                name="forget sweep test",
            )
            past_started = datetime.now(timezone.utc) - timedelta(days=120)
            expired = datetime.now(timezone.utc) - timedelta(days=1)
            comm, _ = await upsert_communication(
                agent="claude-code",
                external_id=f"forget-sweep-{uuid.uuid4().hex[:8]}",
                session_id=sess.id,
                started_at=past_started,
                retention_until=expired,
            )
            return comm.id
        finally:
            await close_engine()

    comm_id = asyncio.run(_seed_expired())
    runner = CliRunner()
    result = runner.invoke(app, ["forget", "--all-expired", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["matched"] >= 1
    assert payload["tombstoned"] >= 1

    async def _check():
        try:
            return await get_communication(comm_id, include_tombstoned=True)
        finally:
            await close_engine()

    row = asyncio.run(_check())
    assert row is not None
    assert row.valid_until is not None
