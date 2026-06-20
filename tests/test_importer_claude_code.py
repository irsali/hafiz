"""Phase 3 — Claude Code JSONL importer.

Exercises the parser on a synthetic JSONL fixture (real shape from
``~/.claude/projects/.../sessions/*.jsonl``), then verifies idempotency
and the selective-embed policy via the live DB.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core.communications import list_messages
from hafiz.core.database import Communication, close_engine, get_session_factory
from hafiz.core.importers.claude_code import (
    discover_jsonl_files,
    import_claude_code,
    parse_jsonl_file,
)
from hafiz.core.sessions import get_session_by_slug


@pytest.fixture(autouse=True)
async def _isolate_engine():
    await close_engine()
    yield
    await close_engine()


def _record(
    *,
    record_type: str,
    session_id: str,
    rec_uuid: str,
    parent_uuid: str | None = None,
    role: str | None = None,
    text: str | None = None,
    tool_use: dict | None = None,
    tool_result: dict | None = None,
    timestamp: datetime | None = None,
    cwd: str = "/tmp/test",
    git_branch: str = "main",
    version: str = "2.1.116",
    extra_msg: dict | None = None,
) -> str:
    """Build a JSONL line in the Claude Code shape."""
    ts = (timestamp or datetime.now(UTC)).isoformat()
    rec: dict = {
        "type": record_type,
        "uuid": rec_uuid,
        "parentUuid": parent_uuid,
        "sessionId": session_id,
        "timestamp": ts,
        "cwd": cwd,
        "gitBranch": git_branch,
        "version": version,
        "isSidechain": False,
    }
    if record_type in ("user", "assistant"):
        content_blocks = []
        if text is not None:
            content_blocks.append({"type": "text", "text": text})
        if tool_use is not None:
            content_blocks.append({"type": "tool_use", **tool_use})
        if tool_result is not None:
            content_blocks.append({"type": "tool_result", **tool_result})
        msg = {
            "role": role or record_type,
            "content": content_blocks,
        }
        if extra_msg:
            msg.update(extra_msg)
        rec["message"] = msg
    return json.dumps(rec)


@pytest.fixture
def jsonl_session(tmp_path: Path) -> Path:
    """A small synthetic Claude Code session with all the shapes we
    care about: user, assistant, tool_use, tool_result, thinking,
    plus a queue-operation skipper and a short-message skipper."""
    sid = str(uuid.uuid4())
    base_ts = datetime(2026, 4, 1, 12, 0, 0, tzinfo=UTC)
    u1 = str(uuid.uuid4())
    a1 = str(uuid.uuid4())
    a2 = str(uuid.uuid4())
    u2 = str(uuid.uuid4())  # a tool_result (modeled as type="user")

    lines = [
        # Non-message control row — must be skipped.
        json.dumps(
            {
                "type": "queue-operation",
                "operation": "enqueue",
                "timestamp": base_ts.isoformat(),
                "sessionId": sid,
            }
        ),
        # User message
        _record(
            record_type="user",
            session_id=sid,
            rec_uuid=u1,
            role="user",
            text="Please reason about the auth migration in detail",
            timestamp=base_ts,
        ),
        # Assistant text reply (long enough to embed)
        _record(
            record_type="assistant",
            session_id=sid,
            rec_uuid=a1,
            parent_uuid=u1,
            role="assistant",
            text=(
                "Looking at the migration carefully — the foreign key "
                "ordering matters because we drop the old text column "
                "before adding the uuid column."
            ),
            timestamp=base_ts + timedelta(seconds=5),
            extra_msg={"model": "claude-opus-4-7"},
        ),
        # Assistant tool_use
        _record(
            record_type="assistant",
            session_id=sid,
            rec_uuid=a2,
            parent_uuid=a1,
            role="assistant",
            tool_use={
                "id": "tu_1",
                "name": "Read",
                "input": {"file_path": "/tmp/x"},
            },
            timestamp=base_ts + timedelta(seconds=8),
            extra_msg={"model": "claude-opus-4-7"},
        ),
        # Tool result — Claude Code emits these as type="user" with
        # content[0].type == "tool_result". Importer should re-tag
        # role to "tool".
        _record(
            record_type="user",
            session_id=sid,
            rec_uuid=u2,
            parent_uuid=a2,
            role="user",
            tool_result={
                "tool_use_id": "tu_1",
                "content": "file contents go here\n" * 100,
            },
            timestamp=base_ts + timedelta(seconds=10),
        ),
        # A tiny user message — should be written but NOT embedded
        # under the selective-embed policy.
        _record(
            record_type="user",
            session_id=sid,
            rec_uuid=str(uuid.uuid4()),
            role="user",
            text="ok",
            timestamp=base_ts + timedelta(seconds=15),
        ),
    ]

    target = tmp_path / f"{sid}.jsonl"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Pure-parse tests (no DB)
# ---------------------------------------------------------------------------


def test_discover_jsonl_files_handles_file_and_dir(tmp_path: Path):
    f = tmp_path / "sess.jsonl"
    f.write_text("{}", encoding="utf-8")
    assert discover_jsonl_files(f) == [f]
    found = discover_jsonl_files(tmp_path)
    assert found == [f]


def test_parse_jsonl_file_extracts_roles_and_tools(jsonl_session: Path):
    parsed = parse_jsonl_file(jsonl_session)
    assert parsed is not None
    assert parsed.git_branch == "main"
    assert parsed.cwd == "/tmp/test"
    # 5 message-shaped records (queue-operation skipped).
    assert len(parsed.messages) == 5
    roles = [m.role for m in parsed.messages]
    assert roles == ["user", "assistant", "assistant", "tool", "user"]
    # tool_use captured on the third message.
    assert parsed.messages[2].tool_calls is not None
    assert parsed.messages[2].tool_calls[0]["kind"] == "tool_use"
    assert parsed.messages[2].tool_calls[0]["name"] == "Read"
    # tool_result captured on the fourth message.
    assert parsed.messages[3].tool_calls is not None
    assert parsed.messages[3].tool_calls[0]["kind"] == "tool_result"


# ---------------------------------------------------------------------------
# DB round-trip + idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_claude_code_round_trip(jsonl_session: Path):
    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    summary = await import_claude_code(root=jsonl_session, project=project)
    assert summary.files_seen == 1
    assert summary.communications_created == 1
    assert summary.messages_written == 5

    # The short "ok" message should NOT have been embedded.
    assert summary.messages_embedded < summary.messages_written

    # A session row was created with a slug derived from the JSONL uuid.
    factory = get_session_factory()
    async with factory() as s:
        comm = (
            await s.execute(
                select(Communication).where(
                    Communication.agent == "claude-code",
                    Communication.scope_value == project,
                )
            )
        ).scalar_one()
        assert comm.external_id is not None
        slug = f"claude-code-{comm.external_id[:12]}"
    assert (await get_session_by_slug(slug)) is not None


@pytest.mark.asyncio
async def test_import_is_idempotent(jsonl_session: Path):
    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    s1 = await import_claude_code(root=jsonl_session, project=project)
    s2 = await import_claude_code(root=jsonl_session, project=project)

    # Second run sees the same file but writes no rows.
    assert s2.communications_created == 0
    assert s2.communications_existing == 1
    assert s2.messages_written == 0
    assert s1.messages_written > 0


@pytest.mark.asyncio
async def test_import_dry_run_writes_nothing(jsonl_session: Path):
    summary = await import_claude_code(
        root=jsonl_session,
        project=f"hafiz-test-{uuid.uuid4().hex[:6]}",
        dry_run=True,
    )
    assert summary.files_seen == 1
    assert summary.communications_created == 0
    assert summary.messages_written == 0


@pytest.mark.asyncio
async def test_import_no_embed_flag_skips_all_embeddings(jsonl_session: Path):
    summary = await import_claude_code(
        root=jsonl_session,
        project=f"hafiz-test-{uuid.uuid4().hex[:6]}",
        embed=False,
    )
    assert summary.messages_written == 5
    assert summary.messages_embedded == 0


@pytest.mark.asyncio
async def test_imported_message_content_is_canonical(jsonl_session: Path):
    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    summary = await import_claude_code(root=jsonl_session, project=project)
    assert summary.communications_created == 1
    factory = get_session_factory()
    async with factory() as s:
        comm = (
            await s.execute(
                select(Communication).where(
                    Communication.agent == "claude-code",
                    Communication.scope_value == project,
                )
            )
        ).scalar_one()
    rows = await list_messages(comm.id)
    # Raw content survived (raw is canonical).
    assert any("auth migration" in r.content for r in rows)
    # Tool use is preserved.
    tool_use_rows = [r for r in rows if r.tool_calls]
    assert tool_use_rows


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_import_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["import", "--help"])
    assert result.exit_code == 0
    assert "claude-code" in result.output.lower()


def test_import_claude_code_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["import", "claude-code", "--help"])
    assert result.exit_code == 0
    assert "--dry-run" in result.output
    assert "--no-embed" in result.output
    assert "--json" in result.output


def test_import_claude_code_cli_dry_run(jsonl_session: Path, tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "import",
            "claude-code",
            str(jsonl_session.parent),
            "--project",
            "hafiz-test-cli",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["summary"]["files_seen"] >= 1
    assert payload["summary"]["communications_created"] == 0
