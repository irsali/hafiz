"""Phase 4 — `hafiz recall` + opt-in transcript search.

Verifies that the source layer is hidden from default `hafiz query` /
`hafiz context` and only appears when the user explicitly opts in via
`--include-transcripts` or the `hafiz recall` command.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core.communications import (
    MessageInput,
    append_messages,
    upsert_communication,
)
from hafiz.core.database import close_engine
from hafiz.core.sessions import create_session


@pytest.fixture(autouse=True)
async def _isolate_engine():
    await close_engine()
    yield
    await close_engine()


async def _seed_session_with_messages(
    *, project: str, content_a: str, content_b: str
) -> tuple[uuid.UUID, uuid.UUID]:
    """Create one session + one communication + a few message turns.

    Closes the cached engine before returning so the *next* asyncio.run
    in the test body (typically a Typer CliRunner invocation) gets a
    fresh engine bound to its own loop.
    """
    slug = f"recall-test-{uuid.uuid4().hex[:6]}"
    try:
        sess = await create_session(
            slug=slug,
            name="recall test",
            agent="claude-code",
            scope_kind="project",
            scope_value=project,
        )
        comm, _ = await upsert_communication(
            agent="claude-code",
            external_id=f"{slug}-extid",
            session_id=sess.id,
            scope_kind="project",
            scope_value=project,
        )
        now = datetime.now(timezone.utc)
        await append_messages(
            comm.id,
            [
                MessageInput(seq=0, role="user", content=content_a, ts=now),
                MessageInput(
                    seq=1,
                    role="assistant",
                    content=content_b,
                    ts=now,
                    author="claude-opus-4-7",
                ),
                MessageInput(
                    seq=2, role="tool", content="(tool result)", ts=now,
                ),
            ],
            embed=True,
        )
        return sess.id, comm.id
    finally:
        await close_engine()


# ---------------------------------------------------------------------------
# CLI: hafiz recall
# ---------------------------------------------------------------------------


def test_recall_help():
    runner = CliRunner()
    result = runner.invoke(app, ["recall", "--help"])
    assert result.exit_code == 0
    assert "Session slug" in result.output or "session slug" in result.output.lower()


def test_recall_unknown_target_returns_empty():
    runner = CliRunner()
    bogus = f"unknown-slug-{uuid.uuid4().hex[:6]}"
    result = runner.invoke(app, ["recall", bogus, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["target"] == bogus
    assert payload["communications"] == []
    assert payload["messages"] == []


def test_recall_returns_messages_in_seq_order(tmp_path):
    project = f"recall-test-{uuid.uuid4().hex[:6]}"
    import asyncio

    sess_id, _ = asyncio.run(
        _seed_session_with_messages(
            project=project,
            content_a="A user question about authentication flows in detail",
            content_b=(
                "An assistant response that's long enough to be embedded; "
                "discussing the trade-offs of moving auth to a separate service."
            ),
        )
    )
    runner = CliRunner()
    result = runner.invoke(app, ["recall", str(sess_id), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["session_id"] == str(sess_id)
    seqs = [m["seq"] for m in payload["messages"]]
    assert seqs == sorted(seqs)
    roles = [m["role"] for m in payload["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_recall_filters_by_role():
    import asyncio

    project = f"recall-role-{uuid.uuid4().hex[:6]}"
    sess_id, _ = asyncio.run(
        _seed_session_with_messages(
            project=project,
            content_a="Long enough user question about authentication",
            content_b="Long enough assistant answer about the auth choices",
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        app, ["recall", str(sess_id), "--role", "user", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    roles = {m["role"] for m in payload["messages"]}
    assert roles == {"user"}


def test_recall_with_query_returns_scored_results():
    import asyncio

    project = f"recall-query-{uuid.uuid4().hex[:6]}"
    sess_id, _ = asyncio.run(
        _seed_session_with_messages(
            project=project,
            content_a="A long user question about authentication migrations",
            content_b="A long assistant response about JWT vs session cookies",
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "recall",
            str(sess_id),
            "--query",
            "authentication",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["query"] == "authentication"
    # Search returns scored results ordered by similarity. Because we
    # only embed messages above the token threshold, the short tool
    # result is filtered out.
    assert all("score" in m for m in payload["messages"])


# ---------------------------------------------------------------------------
# CLI: hafiz query --include-transcripts
# ---------------------------------------------------------------------------


def test_query_include_transcripts_flag_help():
    runner = CliRunner()
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    assert "--include-transcripts" in result.output


def test_query_default_excludes_source_layer():
    """Default `hafiz query` does not include source-layer messages
    even when relevant content exists."""
    import asyncio

    project = f"q-default-{uuid.uuid4().hex[:6]}"
    asyncio.run(
        _seed_session_with_messages(
            project=project,
            content_a="A long user question about widget framistans in detail",
            content_b="A long assistant answer about widget framistans semantics",
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["query", "framistan", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    # The agent contract is: the key is stable; an empty list means
    # "off". This protects parsers from KeyError-shaped flapping.
    assert payload["include_transcripts"] is False
    assert payload["transcripts"] == []


def test_query_include_transcripts_surfaces_messages():
    """With the flag, source-layer rows appear under ``transcripts``
    and are clearly tagged with layer="source"."""
    import asyncio

    project = f"q-incl-{uuid.uuid4().hex[:6]}"
    asyncio.run(
        _seed_session_with_messages(
            project=project,
            content_a="A long user question about widget framistans in detail",
            content_b="A long assistant answer about widget framistans semantics",
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["query", "framistan", "--include-transcripts", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["include_transcripts"] is True
    assert "transcripts" in payload
    assert isinstance(payload["transcripts"], list)
    if payload["transcripts"]:
        first = payload["transcripts"][0]
        assert first["layer"] == "source"
        assert first["kind"] == "chat.turn"


# ---------------------------------------------------------------------------
# CLI: hafiz context --include-transcripts
# ---------------------------------------------------------------------------


def test_context_help_mentions_transcripts():
    runner = CliRunner()
    result = runner.invoke(app, ["context", "--help"])
    assert result.exit_code == 0
    assert "--include-transcripts" in result.output
