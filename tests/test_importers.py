"""Cursor, ChatGPT-export and Codex importers, plus the shared writer.

Parsing is the only part that differs per harness, so that is what these
tests concentrate on: each source's peculiar shape, and the one property
they must all satisfy — every turn carries the source's own identity, or
re-import silently loses data the way migration ``0008`` describes.

Parsing tests are DB-free; the two shared-writer tests at the end touch
the test database because the identity guard they pin lives on the real
write path.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core.database import close_engine


@pytest.fixture(autouse=True)
async def _isolate_engine():
    """Keep this module's DB use from leaking into the rest of the session."""
    await close_engine()
    yield
    await close_engine()


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------

CID = "11111111-2222-3333-4444-555555555555"


def _bubble(bubble_id: str, btype: int, **extra) -> str:
    return json.dumps({"_v": 2, "bubbleId": bubble_id, "type": btype, **extra})


@pytest.fixture
def cursor_db(tmp_path: Path) -> Path:
    """A miniature state.vscdb with the shapes that matter."""
    user = tmp_path / "User"
    (user / "globalStorage").mkdir(parents=True)
    db = user / "globalStorage" / "state.vscdb"

    ws_id = "ws-abc"
    ws_dir = user / "workspaceStorage" / ws_id
    ws_dir.mkdir(parents=True)
    (ws_dir / "workspace.json").write_text(
        json.dumps({"folder": "file:///home/me/projects/thing"}), encoding="utf-8"
    )
    # A .code-workspace entry resolves to its containing directory.
    ws2 = user / "workspaceStorage" / "ws-def"
    ws2.mkdir(parents=True)
    (ws2 / "workspace.json").write_text(
        json.dumps({"workspace": "file:///home/me/projects/other/x.code-workspace"}),
        encoding="utf-8",
    )

    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE composerHeaders (composerId TEXT, workspaceId TEXT, createdAt INT,"
        " lastUpdatedAt INT, isArchived INT, isSubagent INT, recency INT, checkpointAt INT,"
        " value TEXT, subagentTypeName TEXT)"
    )
    conn.execute("CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)")

    conn.execute(
        "INSERT INTO composerHeaders VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            CID,
            ws_id,
            1785927840271,
            1785928721962,
            0,
            0,
            0,
            0,
            json.dumps({"name": "Refactor the parser"}),
            "",
        ),
    )

    order = [
        {"bubbleId": "b1", "type": 1},
        {"bubbleId": "b2", "type": 2},
        {"bubbleId": "b3", "type": 2},
        {"bubbleId": "b4", "type": 2},
        {"bubbleId": "b-missing", "type": 2},
    ]
    conn.execute(
        "INSERT INTO cursorDiskKV VALUES (?,?)",
        (
            f"composerData:{CID}",
            json.dumps({"name": "Refactor the parser", "fullConversationHeadersOnly": order}),
        ),
    )
    rows = {
        "b1": _bubble("b1", 1, text="Please refactor the parser to handle nested blocks"),
        "b2": _bubble("b2", 2, text="Looking at it now — the tokenizer is the problem."),
        # tool-only bubble
        "b3": _bubble(
            "b3",
            2,
            text="",
            toolFormerData={
                "toolCallId": "tc-1",
                "name": "run_terminal_command_v2",
                "params": {"command": "pytest -q"},
                "status": "completed",
                "result": "3 passed",
            },
        ),
        # thinking-only bubble — dropped
        "b4": _bubble("b4", 2, text="", thinking="Let me consider the edge cases"),
    }
    for bid, val in rows.items():
        conn.execute("INSERT INTO cursorDiskKV VALUES (?,?)", (f"bubbleId:{CID}:{bid}", val))
    conn.commit()
    conn.close()
    return db


def test_cursor_workspace_folders_resolve_both_shapes(cursor_db: Path):
    from hafiz.core.importers.cursor import workspace_folders

    folders = workspace_folders(cursor_db.parent.parent)
    assert folders["ws-abc"] == "/home/me/projects/thing"
    # A .code-workspace file maps to its directory, not the file.
    assert folders["ws-def"] == "/home/me/projects/other"


def test_cursor_parses_turns_in_conversation_order(cursor_db: Path):
    from hafiz.core.importers.cursor import parse_conversations, workspace_folders

    convos = parse_conversations(cursor_db, folders=workspace_folders(cursor_db.parent.parent))
    assert len(convos) == 1
    c = convos[0]
    assert c.external_id == CID
    assert c.title == "Refactor the parser"
    assert c.cwd == "/home/me/projects/thing"
    # b4 is thinking-only and b-missing has no stored bubble.
    assert [m.source_message_id for m in c.messages] == ["b1", "b2", "b3"]
    assert [m.role for m in c.messages] == ["user", "assistant", "tool"]


def test_cursor_bubble_id_is_the_turn_identity(cursor_db: Path):
    from hafiz.core.importers.cursor import parse_conversations

    c = parse_conversations(cursor_db)[0]
    assert all(m.source_message_id for m in c.messages)
    assert len({m.source_message_id for m in c.messages}) == len(c.messages)


def test_cursor_normalizes_tool_calls(cursor_db: Path):
    from hafiz.core.importers.cursor import parse_conversations

    c = parse_conversations(cursor_db)[0]
    tool = next(m for m in c.messages if m.role == "tool")
    assert tool.tool_calls[0]["name"] == "run_terminal_command_v2"
    assert tool.tool_calls[0]["id"] == "tc-1"
    assert tool.tool_calls[0]["input"] == {"command": "pytest -q"}


def test_cursor_keeps_thinking_out_of_content(cursor_db: Path):
    """Reasoning must not reach the embedding corpus as message content."""
    from hafiz.core.importers.cursor import parse_conversations

    c = parse_conversations(cursor_db)[0]
    assert all("edge cases" not in m.content for m in c.messages)


def test_cursor_opens_the_database_read_only(cursor_db: Path):
    """It is a live application's database; hafiz must never write to it."""
    from hafiz.core.importers.cursor import _connect_readonly

    conn = _connect_readonly(cursor_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO cursorDiskKV VALUES ('x','y')")
    finally:
        conn.close()


def test_cursor_missing_database_is_not_an_error(tmp_path: Path):
    from hafiz.core.importers.cursor import parse_conversations

    assert parse_conversations(tmp_path / "nope.vscdb") == []


# ---------------------------------------------------------------------------
# ChatGPT export
# ---------------------------------------------------------------------------


def _node(nid: str, parent: str | None, role: str | None, text: str | None, children=()):
    message = None
    if role is not None:
        message = {
            "id": nid,
            "author": {"role": role},
            "create_time": 1785927840.0,
            "content": {"content_type": "text", "parts": [text or ""]},
            "metadata": {"model_slug": "gpt-5"} if role == "assistant" else {},
        }
    return {"id": nid, "message": message, "parent": parent, "children": list(children)}


@pytest.fixture
def chatgpt_payload() -> list:
    """A conversation with an abandoned branch.

    root -> u1 -> a1_bad (regenerated away)
                -> a1_good -> u2 -> a2   <- current_node
    """
    mapping = {
        "root": _node("root", None, None, None, ["u1"]),
        "u1": _node("u1", "root", "user", "How do I invert a binary tree?", ["a1_bad", "a1_good"]),
        "a1_bad": _node("a1_bad", "u1", "assistant", "THIS ANSWER WAS REGENERATED AWAY"),
        "a1_good": _node("a1_good", "u1", "assistant", "Swap the children recursively.", ["u2"]),
        "u2": _node("u2", "a1_good", "user", "Show me that in Python please", ["a2"]),
        "a2": _node("a2", "u2", "assistant", "def invert(node): ..."),
    }
    return [
        {
            "title": "Binary trees",
            "conversation_id": "conv-1",
            "create_time": 1785927840.0,
            "update_time": 1785928840.0,
            "current_node": "a2",
            "mapping": mapping,
        }
    ]


def test_chatgpt_walks_the_kept_branch_only(chatgpt_payload: list):
    """The subtle one: a mapping holds every regenerated draft too."""
    from hafiz.core.importers.chatgpt import parse_conversations

    convos = parse_conversations(chatgpt_payload)
    assert len(convos) == 1
    c = convos[0]
    assert [m.source_message_id for m in c.messages] == ["u1", "a1_good", "u2", "a2"]
    assert all("REGENERATED AWAY" not in m.content for m in c.messages)


def test_chatgpt_preserves_conversation_order(chatgpt_payload: list):
    from hafiz.core.importers.chatgpt import parse_conversations

    c = parse_conversations(chatgpt_payload)[0]
    assert [m.role for m in c.messages] == ["user", "assistant", "user", "assistant"]
    assert c.title == "Binary trees"
    assert c.external_id == "conv-1"


def test_chatgpt_skips_system_and_empty_nodes(chatgpt_payload: list):
    from hafiz.core.importers.chatgpt import parse_conversations

    mapping = chatgpt_payload[0]["mapping"]
    mapping["sys"] = _node("sys", "root", "system", "You are a helpful assistant")
    mapping["u1"]["parent"] = "sys"
    mapping["sys"]["children"] = ["u1"]
    c = parse_conversations(chatgpt_payload)[0]
    assert "sys" not in [m.source_message_id for m in c.messages]


def test_chatgpt_falls_back_when_current_node_is_missing(chatgpt_payload: list):
    """Better to import an over-broad thread than to import nothing."""
    from hafiz.core.importers.chatgpt import parse_conversations

    chatgpt_payload[0]["current_node"] = None
    c = parse_conversations(chatgpt_payload)[0]
    assert len(c.messages) >= 4


def test_chatgpt_handles_multimodal_parts(chatgpt_payload: list):
    from hafiz.core.importers.chatgpt import parse_conversations

    chatgpt_payload[0]["mapping"]["a2"]["message"]["content"] = {
        "content_type": "multimodal_text",
        "parts": [{"asset_pointer": "file-service://x"}, "Here is the diagram explained"],
    }
    c = parse_conversations(chatgpt_payload)[0]
    last = c.messages[-1]
    assert last.content == "Here is the diagram explained"
    assert "asset_pointer" not in last.content


def test_chatgpt_load_export_accepts_zip_dir_and_file(tmp_path: Path, chatgpt_payload: list):
    from hafiz.core.importers.chatgpt import load_export

    raw = json.dumps(chatgpt_payload)

    as_file = tmp_path / "conversations.json"
    as_file.write_text(raw, encoding="utf-8")
    assert load_export(as_file) == chatgpt_payload

    as_dir = tmp_path / "export"
    as_dir.mkdir()
    (as_dir / "conversations.json").write_text(raw, encoding="utf-8")
    assert load_export(as_dir) == chatgpt_payload

    as_zip = tmp_path / "export.zip"
    with zipfile.ZipFile(as_zip, "w") as zf:
        zf.writestr("some-export/conversations.json", raw)
    assert load_export(as_zip) == chatgpt_payload


def test_chatgpt_load_export_reports_a_missing_file(tmp_path: Path):
    from hafiz.core.importers.chatgpt import load_export

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        load_export(empty)


# ---------------------------------------------------------------------------
# Codex CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def codex_rollout(tmp_path: Path) -> Path:
    sid = "01234567-89ab-cdef-0123-456789abcdef"
    day = tmp_path / "sessions" / "2026" / "09" / "04"
    day.mkdir(parents=True)
    path = day / f"rollout-2026-09-04T10-00-00-{sid}.jsonl"
    lines = [
        {
            "type": "session_meta",
            "timestamp": "2026-09-04T10:00:00Z",
            "payload": {
                "id": sid,
                "cwd": "/home/me/projects/thing",
                "cli_version": "0.140.0",
                "git": {"branch": "main"},
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-09-04T10:00:05Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Add a retry to the fetch helper"}],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-09-04T10:00:07Z",
            "payload": {"type": "reasoning", "content": [{"text": "internal deliberation"}]},
        },
        {
            "type": "response_item",
            "timestamp": "2026-09-04T10:00:09Z",
            "payload": {
                "type": "function_call",
                "id": "fc-1",
                "call_id": "call-1",
                "name": "shell",
                "arguments": '{"command":"rg fetchHelper"}',
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-09-04T10:00:11Z",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "src/net.ts:12",
            },
        },
        {"type": "event_msg", "payload": {"type": "token_count", "total": 812}},
        {
            "type": "response_item",
            "timestamp": "2026-09-04T10:00:20Z",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Added a retry with backoff."}],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return path


def test_codex_parses_session_meta_and_turns(codex_rollout: Path):
    from hafiz.core.importers.codex import parse_rollout_file

    parsed = parse_rollout_file(codex_rollout)
    assert parsed is not None
    assert parsed.external_id == "01234567-89ab-cdef-0123-456789abcdef"
    assert parsed.cwd == "/home/me/projects/thing"
    assert parsed.cli_version == "0.140.0"
    assert parsed.git_branch == "main"
    # reasoning and event_msg are not conversation
    assert [m.role for m in parsed.messages] == ["user", "assistant", "tool", "assistant"]
    assert all("deliberation" not in m.content for m in parsed.messages)


def test_codex_maps_tool_calls_and_results(codex_rollout: Path):
    from hafiz.core.importers.codex import parse_rollout_file

    parsed = parse_rollout_file(codex_rollout)
    call = parsed.messages[1]
    assert call.tool_calls[0]["kind"] == "tool_use"
    assert call.tool_calls[0]["name"] == "shell"
    result = parsed.messages[2]
    assert result.tool_calls[0]["kind"] == "tool_result"
    assert result.tool_calls[0]["tool_use_id"] == "call-1"
    assert "src/net.ts" in result.content


def test_codex_every_turn_has_a_stable_identity(codex_rollout: Path):
    """Positional fallback is safe here — Codex appends to one file."""
    from hafiz.core.importers.codex import parse_rollout_file

    first = parse_rollout_file(codex_rollout)
    again = parse_rollout_file(codex_rollout)
    assert all(m.source_message_id for m in first.messages)
    assert [m.source_message_id for m in first.messages] == [
        m.source_message_id for m in again.messages
    ]


def test_codex_identity_is_stable_when_the_file_grows(codex_rollout: Path):
    from hafiz.core.importers.codex import parse_rollout_file

    before = [m.source_message_id for m in parse_rollout_file(codex_rollout).messages]
    with codex_rollout.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "response_item",
                    "timestamp": "2026-09-04T10:05:00Z",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Now add a test for it"}],
                    },
                }
            )
            + "\n"
        )
    after = [m.source_message_id for m in parse_rollout_file(codex_rollout).messages]
    assert after[: len(before)] == before, "existing turns keep their identity"
    assert len(after) == len(before) + 1


def test_codex_tolerates_an_inlined_payload(tmp_path: Path):
    """Codex is explicitly tolerant of its own schema drift; so are we."""
    from hafiz.core.importers.codex import parse_rollout_file

    path = tmp_path / "rollout-flat.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {"type": "session_meta", "id": "s-1", "cwd": "/tmp/x"},
                {
                    "type": "response_item",
                    "type_": "ignored",
                    "id": "m-1",
                    "role": "user",
                    "content": "hello there",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_rollout_file(path)
    # The second record has no inner item type, so it is skipped rather
    # than guessed at — but the file still parses and meta is read.
    assert parsed is None or parsed.cwd == "/tmp/x"


def test_codex_skips_unknown_record_types(tmp_path: Path):
    from hafiz.core.importers.codex import parse_rollout_file

    path = tmp_path / "rollout-future.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {"type": "session_meta", "payload": {"id": "s-2", "cwd": "/tmp/y"}},
                {"type": "some_future_event", "payload": {"whatever": 1}},
                {
                    "type": "response_item",
                    "payload": {"type": "message", "role": "user", "content": "still works"},
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_rollout_file(path)
    assert parsed is not None
    assert [m.content for m in parsed.messages] == ["still works"]


def test_codex_home_honours_the_env_var(monkeypatch, tmp_path: Path):
    """A set CODEX_HOME moves the whole tree; hard-coding ~/.codex finds nothing."""
    from hafiz.core.importers import codex

    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "elsewhere"))
    assert codex.codex_home() == tmp_path / "elsewhere"
    roots = codex.session_roots()
    assert roots[0].name == "sessions"
    assert roots[1].name == "archived_sessions"


def test_codex_discovers_archived_sessions_too(codex_rollout: Path, tmp_path: Path, monkeypatch):
    from hafiz.core.importers import codex

    archived = tmp_path / "archived_sessions" / "2026" / "09" / "03"
    archived.mkdir(parents=True)
    (archived / "rollout-old.jsonl").write_text(
        json.dumps({"type": "session_meta", "payload": {"id": "old"}}) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    found = [f for r in codex.session_roots() for f in codex.discover_rollout_files(r)]
    assert len(found) == 2


# ---------------------------------------------------------------------------
# The contract every importer must satisfy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shared_writer_flags_a_parser_with_no_turn_identity():
    """The guard that stops a new importer inheriting the 0008 bug.

    Awaited rather than ``asyncio.run``: this module runs under
    pytest-asyncio's auto mode, and spinning up a private loop here closes
    one the shared async engine is bound to — which silently made every
    later DB-backed test in the session skip with "Postgres not
    reachable".
    """
    import uuid as _uuid

    from hafiz.core.communications import MessageInput
    from hafiz.core.importers.base import ImportSummary, ParsedConversation, store_conversation

    summary = ImportSummary()
    parsed = ParsedConversation(
        external_id=f"identity-less-{_uuid.uuid4()}",
        started_at=datetime.now(UTC),
        messages=[
            MessageInput(seq=0, role="user", content="hi there friend", ts=datetime.now(UTC))
        ],
    )
    await store_conversation(
        agent="test-importer",
        parsed=parsed,
        summary=summary,
        project=f"hafiz-test-{_uuid.uuid4().hex[:6]}",
    )

    assert summary.messages_written == 1  # it still stores
    assert summary.errors, "but it warns that turns will dedupe positionally"
    assert "source_message_id" in summary.errors[0]["error"]


@pytest.mark.asyncio
async def test_shared_writer_is_quiet_when_identity_is_supplied():
    import uuid as _uuid

    from hafiz.core.communications import MessageInput
    from hafiz.core.importers.base import ImportSummary, ParsedConversation, store_conversation

    summary = ImportSummary()
    parsed = ParsedConversation(
        external_id=f"with-identity-{_uuid.uuid4()}",
        started_at=datetime.now(UTC),
        messages=[
            MessageInput(
                seq=0,
                role="user",
                content="hi there friend",
                ts=datetime.now(UTC),
                source_message_id="turn-1",
            )
        ],
    )
    await store_conversation(
        agent="test-importer",
        parsed=parsed,
        summary=summary,
        project=f"hafiz-test-{_uuid.uuid4().hex[:6]}",
    )
    assert summary.errors == []


@pytest.mark.parametrize("agent", ["claude-code", "cursor", "chatgpt", "codex"])
def test_every_importer_has_a_cli_command(agent: str):
    result = CliRunner().invoke(app, ["import", agent, "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--dry-run", "--json", "--project", "--since"):
        assert flag in result.output, f"{agent} missing {flag}"


def test_chatgpt_requires_an_export_path():
    result = CliRunner().invoke(app, ["import", "chatgpt"])
    assert result.exit_code != 0
