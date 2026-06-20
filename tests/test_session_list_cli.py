"""``hafiz session list`` — listing sessions without psql.

Small UX gap-fill from the Phase 2 follow-up. Verifies the command
exists, accepts the documented flags, and returns a stable JSON shape
agents can consume.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core.database import close_engine
from hafiz.core.sessions import create_session, end_session_db


@pytest.fixture(autouse=True)
async def _isolate_engine():
    await close_engine()
    yield
    await close_engine()


def _seed(*, agent="claude-code", project=None, ended=False) -> str:
    """Synchronously create one session and return its slug."""

    async def _go():
        try:
            slug = f"sl-test-{uuid.uuid4().hex[:6]}"
            sess = await create_session(
                slug=slug,
                name="session list smoke",
                agent=agent,
                scope_kind="project" if project else None,
                scope_value=project,
            )
            if ended:
                await end_session_db(sess.id)
            return slug
        finally:
            await close_engine()

    return asyncio.run(_go())


def test_session_list_help():
    runner = CliRunner()
    result = runner.invoke(app, ["session", "list", "--help"])
    assert result.exit_code == 0
    assert "--agent" in result.output
    assert "--project" in result.output
    assert "--active" in result.output
    assert "--limit" in result.output


def test_session_list_returns_seeded_session_in_json():
    slug = _seed(agent="claude-code", project="hafiz-list-test")
    runner = CliRunner()
    result = runner.invoke(app, ["session", "list", "--project", "hafiz-list-test", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["filters"]["project"] == "hafiz-list-test"
    slugs = [s["slug"] for s in payload["sessions"]]
    assert slug in slugs


def test_session_list_filters_by_agent():
    slug_claude = _seed(agent="claude-code", project="agent-filter-test")
    slug_cursor = _seed(agent="cursor", project="agent-filter-test")
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "session",
            "list",
            "--agent",
            "cursor",
            "--project",
            "agent-filter-test",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    slugs = [s["slug"] for s in payload["sessions"]]
    assert slug_cursor in slugs
    assert slug_claude not in slugs


def test_session_list_active_flag_excludes_ended():
    slug_active = _seed(agent="claude-code", project="active-flag-test", ended=False)
    slug_ended = _seed(agent="claude-code", project="active-flag-test", ended=True)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "session",
            "list",
            "--project",
            "active-flag-test",
            "--active",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    slugs = [s["slug"] for s in payload["sessions"]]
    assert slug_active in slugs
    assert slug_ended not in slugs


def test_session_list_empty_returns_zero_total():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "session",
            "list",
            "--project",
            f"definitely-not-real-{uuid.uuid4().hex[:6]}",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["total"] == 0
    assert payload["sessions"] == []
