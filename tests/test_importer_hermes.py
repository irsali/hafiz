"""Hermes ``state.db`` importer.

A synthetic ``state.db`` carries the shapes that matter: tool calls and
results, rewound turns, compaction's archived-plus-copied rows, and older
schemas missing optional columns. Then the cross-cutting behaviour the
importer owes the rest of hafiz — never duplicate what the Hermes memory
plugin already captured, never write a session already past retention, and
tell ``status`` truthfully what is still uncaptured.

Parsing tests are DB-free; the rest touch the test database.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core.capture import store_transcript
from hafiz.core.database import (
    Communication,
    CommunicationMessage,
    close_engine,
    get_session_factory,
)
from hafiz.core.importers import hermes

_token = ""


@pytest.fixture(autouse=True)
async def _isolate_engine():
    # Communications are unique on (agent, external_id) across the whole shared
    # test database, so every test mints its own session ids.
    global _token
    _token = uuid.uuid4().hex[:8]
    await close_engine()
    yield
    await close_engine()


def _sid(tag: str) -> str:
    """A Hermes-shaped session id, unique to the running test."""
    return f"20260901_{_token}_{tag}"


# ---------------------------------------------------------------------------
# Fixture: a miniature state.db
# ---------------------------------------------------------------------------

_SESSION_COLS = (
    "id TEXT PRIMARY KEY, source TEXT, model TEXT, parent_session_id TEXT, "
    "started_at REAL NOT NULL, ended_at REAL, end_reason TEXT, title TEXT, "
    "cwd TEXT, git_branch TEXT, git_repo_root TEXT"
)
_MESSAGE_COLS = (
    "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, "
    "content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL, "
    "finish_reason TEXT, reasoning TEXT, reasoning_content TEXT, "
    "active INTEGER NOT NULL DEFAULT 1, compacted INTEGER NOT NULL DEFAULT 0"
)


def _new_id() -> str:
    return f"{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def _epoch(days_ago: float = 1.0) -> float:
    return time.time() - days_ago * 86400


def make_db(path: Path, *, session_cols: str = _SESSION_COLS, message_cols: str = _MESSAGE_COLS):
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE sessions ({session_cols})")
    conn.execute(f"CREATE TABLE messages ({message_cols})")
    conn.commit()
    return conn


def add_session(conn, sid: str, *, days_ago: float = 1.0, ended: bool = True, **extra):
    started = _epoch(days_ago)
    row = {
        "id": sid,
        "source": "cli",
        "model": "m",
        "started_at": started,
        "ended_at": started + 600 if ended else None,
        "title": f"title {sid[-6:]}",
        **extra,
    }
    conn.execute(
        f"INSERT INTO sessions ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
        tuple(row.values()),
    )


def add_message(conn, sid: str, role: str, content: str | None, *, ts: float, **extra):
    row = {"session_id": sid, "role": role, "content": content, "timestamp": ts, **extra}
    conn.execute(
        f"INSERT INTO messages ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
        tuple(row.values()),
    )


def _tool_call(call_id: str, name: str, args: dict) -> str:
    return json.dumps(
        [
            {
                "id": call_id,
                "call_id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ]
    )


def add_simple_session(conn, sid: str, *, days_ago: float = 1.0, **extra) -> None:
    """user -> assistant(tool call) -> tool result -> assistant."""
    add_session(conn, sid, days_ago=days_ago, **extra)
    t = _epoch(days_ago)
    add_message(conn, sid, "user", "Find where the retry helper lives", ts=t + 1)
    add_message(
        conn,
        sid,
        "assistant",
        "",
        ts=t + 2,
        tool_calls=_tool_call("call_1", "search_files", {"pattern": "retry"}),
        reasoning_content="private deliberation",
    )
    add_message(
        conn,
        sid,
        "tool",
        '{"total_count": 1}',
        ts=t + 3,
        tool_call_id="call_1",
        tool_name="search_files",
    )
    add_message(conn, sid, "assistant", "It lives in net.py", ts=t + 4)


@pytest.fixture
def state_db(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    conn = make_db(path)
    add_simple_session(conn, _sid("aaaaaa"), days_ago=2)
    conn.commit()
    conn.close()
    return path


def _project() -> str:
    return f"hafiz-test-{uuid.uuid4().hex[:6]}"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _read(db: Path, sid: str):
    conn = hermes._connect_readonly(db)
    try:
        header = next(h for h in hermes.list_sessions(conn) if h.id == sid)
        return hermes.read_messages(conn, sid, header.started_at)
    finally:
        conn.close()


def test_parses_turns_in_order_with_the_row_id_as_identity(state_db: Path):
    msgs = _read(state_db, _sid("aaaaaa"))
    assert [m.role for m in msgs] == ["user", "assistant", "tool", "assistant"]
    assert [m.source_message_id for m in msgs] == ["1", "2", "3", "4"]
    assert msgs[0].content == "Find where the retry helper lives"


def test_maps_tool_calls_and_results(state_db: Path):
    msgs = _read(state_db, _sid("aaaaaa"))
    use = msgs[1].tool_calls[0]
    assert use["kind"] == "tool_use"
    assert use["id"] == "call_1"
    assert use["name"] == "search_files"
    assert use["input"] == {"pattern": "retry"}  # JSON arguments are parsed
    result = msgs[2].tool_calls[0]
    assert result["kind"] == "tool_result"
    assert result["tool_use_id"] == "call_1"
    assert result["name"] == "search_files"
    assert msgs[2].content == '{"total_count": 1}'


def test_reasoning_stays_out_of_content(state_db: Path):
    msgs = _read(state_db, _sid("aaaaaa"))
    assert all("deliberation" not in m.content for m in msgs)
    assert msgs[1].metadata["reasoning"] == "private deliberation"


def test_rewound_turns_are_skipped(tmp_path: Path):
    db = tmp_path / "state.db"
    conn = make_db(db)
    add_session(conn, "s1")
    t = _epoch()
    add_message(conn, "s1", "user", "keep me", ts=t)
    add_message(conn, "s1", "user", "taken back", ts=t + 1, active=0, compacted=0)
    add_message(conn, "s1", "assistant", "kept answer", ts=t + 2)
    conn.commit()
    conn.close()
    assert [m.content for m in _read(db, "s1")] == ["keep me", "kept answer"]


def test_compacted_history_is_kept_and_its_live_copies_are_not_doubled(tmp_path: Path):
    """Compaction archives the turns (compacted=1) and re-inserts the tail live."""
    db = tmp_path / "state.db"
    conn = make_db(db)
    add_session(conn, "s1")
    t = _epoch()
    add_message(conn, "s1", "user", "first question", ts=t, active=0, compacted=1)
    add_message(conn, "s1", "assistant", "first answer", ts=t + 1, active=0, compacted=1)
    add_message(conn, "s1", "user", "tail question", ts=t + 2, active=0, compacted=1)
    # what compaction re-inserted: a summary plus a copy of the tail
    add_message(conn, "s1", "assistant", "[CONTEXT COMPACTION] summary", ts=t + 3)
    add_message(conn, "s1", "user", "tail question", ts=t + 2)
    # a genuinely new turn after compaction, repeating an earlier wording
    add_message(conn, "s1", "user", "tail question", ts=t + 9)
    conn.commit()
    conn.close()

    msgs = _read(db, "s1")
    assert [m.content for m in msgs] == [
        "first question",
        "first answer",
        "tail question",
        "[CONTEXT COMPACTION] summary",
        "tail question",  # the post-compaction repeat survives; the copy did not
    ]
    assert [bool((m.metadata or {}).get("compacted")) for m in msgs] == [
        True,
        True,
        True,
        False,
        False,
    ]


def test_older_schema_without_optional_columns_is_tolerated(tmp_path: Path):
    db = tmp_path / "state.db"
    conn = make_db(
        db,
        session_cols="id TEXT PRIMARY KEY, started_at REAL NOT NULL",
        message_cols=(
            "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, "
            "role TEXT NOT NULL, content TEXT"
        ),
    )
    conn.execute("INSERT INTO sessions VALUES ('old1', ?)", (_epoch(),))
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES ('old1','user','hi there')"
    )
    conn.execute("INSERT INTO messages (session_id, role, content) VALUES ('old1','assistant','')")
    conn.commit()
    conn.close()

    conn = hermes._connect_readonly(db)
    try:
        header = hermes.list_sessions(conn)[0]
        assert header.title is None and header.cwd is None
        msgs = hermes.read_messages(conn, "old1", header.started_at)
    finally:
        conn.close()
    assert [m.content for m in msgs] == ["hi there"]  # empty turn dropped
    assert msgs[0].ts == header.started_at  # no timestamp column -> session start


def test_non_string_content_is_decoded(tmp_path: Path):
    db = tmp_path / "state.db"
    conn = make_db(db)
    add_session(conn, "s1")
    parts = [{"type": "text", "text": "look at this"}, {"type": "image_url", "image_url": "x"}]
    add_message(conn, "s1", "user", "\x00json:" + json.dumps(parts), ts=_epoch())
    conn.commit()
    conn.close()
    assert [m.content for m in _read(db, "s1")] == ["look at this"]


def test_database_is_opened_read_only(state_db: Path):
    conn = hermes._connect_readonly(state_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM messages")
    finally:
        conn.close()


def test_a_database_that_is_not_hermes_is_an_error_not_a_crash(tmp_path: Path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE sessions (name TEXT)")
    conn.commit()
    conn.close()
    conn = hermes._connect_readonly(db)
    try:
        with pytest.raises(sqlite3.DatabaseError):
            hermes.list_sessions(conn)
    finally:
        conn.close()


def test_hermes_home_honours_the_env_var(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "elsewhere"))
    assert hermes.default_db_path() == tmp_path / "elsewhere" / "state.db"
    monkeypatch.delenv("HERMES_HOME")
    assert hermes.default_db_path() == Path.home() / ".hermes" / "state.db"


# ---------------------------------------------------------------------------
# Import: idempotency, dry-run, project
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_import_round_trip_and_directory_argument(state_db: Path):
    project = _project()
    summary = await hermes.import_hermes(root=state_db.parent, project=project, embed=False)
    assert summary.files_seen == 1
    assert summary.communications_created == 1
    assert summary.messages_written == 4
    assert summary.skipped_expired == 0
    assert summary.errors == []

    factory = get_session_factory()
    async with factory() as s:
        comm = (
            await s.execute(select(Communication).where(Communication.scope_value == project))
        ).scalar_one()
        assert comm.agent == "hermes"
        assert comm.external_id == _sid("aaaaaa")
        # retention is the standard 90 days from the session's start
        assert comm.retention_until - comm.started_at == timedelta(days=90)
        tool_rows = (
            (
                await s.execute(
                    select(CommunicationMessage).where(
                        CommunicationMessage.communication_id == comm.id,
                        CommunicationMessage.role == "tool",
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(tool_rows) == 1


@pytest.mark.asyncio
async def test_rerun_is_idempotent_and_picks_up_new_turns(state_db: Path):
    project = _project()
    first = await hermes.import_hermes(root=state_db, project=project, embed=False)
    again = await hermes.import_hermes(root=state_db, project=project, embed=False)
    assert first.messages_written == 4
    assert again.communications_created == 0
    assert again.communications_existing == 1
    assert again.messages_written == 0

    conn = sqlite3.connect(state_db)
    add_message(conn, _sid("aaaaaa"), "user", "one more thing", ts=_epoch(0.5))
    conn.commit()
    conn.close()
    grown = await hermes.import_hermes(root=state_db, project=project, embed=False)
    assert grown.messages_written == 1


@pytest.mark.asyncio
async def test_dry_run_writes_nothing_and_matches_the_real_run(state_db: Path):
    project = _project()
    preview = await hermes.import_hermes(root=state_db, project=project, dry_run=True, embed=False)

    factory = get_session_factory()
    async with factory() as s:
        stored = (
            await s.execute(select(Communication).where(Communication.scope_value == project))
        ).all()
    assert stored == []

    real = await hermes.import_hermes(root=state_db, project=project, embed=False)
    for field in (
        "files_seen",
        "communications_created",
        "communications_existing",
        "messages_written",
        "sessions_created",
        "skipped_expired",
    ):
        assert getattr(preview, field) == getattr(real, field), field

    after = await hermes.import_hermes(root=state_db, project=project, dry_run=True, embed=False)
    assert after.communications_existing == 1
    assert after.messages_written == 0


@pytest.mark.asyncio
async def test_limit_since_and_missing_database(tmp_path: Path):
    db = tmp_path / "state.db"
    conn = make_db(db)
    add_simple_session(conn, _sid("aaaaaa"), days_ago=2)
    add_simple_session(conn, _sid("bbbbbb"), days_ago=1)
    add_simple_session(conn, _sid("cccccc"), days_ago=30)
    conn.commit()
    conn.close()

    project = _project()
    limited = await hermes.import_hermes(
        root=db, project=project, dry_run=True, embed=False, limit=1
    )
    assert limited.files_seen == 1 and limited.communications_created == 1

    recent = await hermes.import_hermes(
        root=db,
        project=project,
        dry_run=True,
        embed=False,
        since=datetime.now(UTC) - timedelta(days=10),
    )
    assert recent.communications_created == 2

    nothing = await hermes.import_hermes(root=tmp_path / "absent" / "state.db", embed=False)
    assert nothing.files_seen == 0 and nothing.errors == []


@pytest.mark.asyncio
async def test_a_session_still_in_progress_is_left_for_the_plugin(tmp_path: Path):
    """The plugin captures a live session when it ends, under its own random
    key; importing it first would store the conversation twice."""
    db = tmp_path / "state.db"
    conn = make_db(db)
    add_simple_session(conn, _sid("openxx"), days_ago=0.001, ended=False)  # ~90 s ago
    add_simple_session(conn, _sid("stalex"), days_ago=1, ended=False)  # abandoned
    conn.commit()
    conn.close()

    got = await hermes.import_hermes(root=db, project=_project(), dry_run=True, embed=False)
    assert got.communications_created == 1  # only the abandoned one


@pytest.mark.asyncio
async def test_project_resolves_from_git_repo_root_then_cwd(state_db: Path, monkeypatch):
    project = _project()
    conn = sqlite3.connect(state_db)
    add_simple_session(
        conn, _sid("dddddd"), days_ago=1, cwd="/work/app/sub", git_repo_root="/work/app"
    )
    add_simple_session(conn, _sid("eeeeee"), days_ago=1, cwd="/other/place")
    conn.commit()
    conn.close()

    async def fake_roots():
        return {project: "/work/app"}

    monkeypatch.setattr("hafiz.core.store.indexed_root_per_project", fake_roots)
    await hermes.import_hermes(root=state_db, embed=False)

    factory = get_session_factory()
    async with factory() as s:
        rows = {
            c.external_id: c.scope_value
            for c in (
                await s.execute(
                    select(Communication).where(
                        Communication.external_id.in_([_sid("dddddd"), _sid("eeeeee")])
                    )
                )
            )
            .scalars()
            .all()
        }
    assert rows == {_sid("dddddd"): project, _sid("eeeeee"): None}


# ---------------------------------------------------------------------------
# Plugin-captured sessions are not imported twice
# ---------------------------------------------------------------------------


async def _plugin_capture(session_id: str, project: str):
    """Exactly what the Hermes memory plugin's `hafiz capture` call stores."""
    return await store_transcript(
        "User: hello\n\nAssistant: hi",
        title=f"hermes:{session_id}",
        project=project,
        source="agent:hermes",
        tags=["hermes"],
        session_id=session_id,
    )


@pytest.mark.asyncio
async def test_a_plugin_captured_session_is_counted_not_imported(state_db: Path):
    project = _project()
    sid = _sid("aaaaaa")
    await _plugin_capture(sid, project)
    assert sid in await hermes.plugin_captured_ids()

    summary = await hermes.import_hermes(root=state_db, project=project, embed=False)
    assert summary.files_seen == 1
    assert summary.communications_existing == 1
    assert summary.communications_created == 0
    assert summary.messages_written == 0

    preview = await hermes.import_hermes(root=state_db, project=project, dry_run=True, embed=False)
    assert preview.communications_existing == 1 and preview.communications_created == 0

    factory = get_session_factory()
    async with factory() as s:
        # still exactly one communication for that session: the plugin's
        rows = (
            (
                await s.execute(
                    select(Communication).where(
                        Communication.agent == "hermes", Communication.scope_value == project
                    )
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 1 and rows[0].external_id != sid


@pytest.mark.asyncio
async def test_a_title_only_plugin_capture_is_recognised(state_db: Path):
    """If only the title survived (no session slug), it still identifies the session."""
    project = _project()
    sid = _sid("aaaaaa")
    await store_transcript(
        "User: hi",
        title=f"hermes:{sid}",
        project=project,
        source="agent:hermes",
    )
    summary = await hermes.import_hermes(root=state_db, project=project, embed=False)
    assert summary.communications_existing == 1 and summary.messages_written == 0


# ---------------------------------------------------------------------------
# Retention: expired sessions are never written
# ---------------------------------------------------------------------------


@pytest.fixture
def mixed_age_db(tmp_path: Path) -> Path:
    db = tmp_path / "state.db"
    conn = make_db(db)
    add_simple_session(conn, _sid("aaaaaa"), days_ago=5)
    add_simple_session(conn, _sid("bbbbbb"), days_ago=200)
    add_simple_session(conn, _sid("cccccc"), days_ago=91)
    conn.commit()
    conn.close()
    return db


@pytest.mark.asyncio
async def test_sessions_past_retention_are_skipped_and_reported(mixed_age_db: Path):
    project = _project()
    summary = await hermes.import_hermes(root=mixed_age_db, project=project, embed=False)
    assert summary.skipped_expired == 2
    assert summary.communications_created == 1
    assert summary.messages_written == 4
    assert summary.to_dict()["skipped_expired"] == 2

    factory = get_session_factory()
    async with factory() as s:
        ids = {
            e
            for (e,) in await s.execute(
                select(Communication.external_id).where(Communication.agent == "hermes")
            )
        }
    assert _sid("bbbbbb") not in ids and _sid("cccccc") not in ids
    assert _sid("aaaaaa") in ids

    preview = await hermes.import_hermes(
        root=mixed_age_db, project=project, dry_run=True, embed=False
    )
    assert preview.skipped_expired == 2


@pytest.mark.asyncio
async def test_expired_wins_over_plugin_captured(mixed_age_db: Path):
    """Neither path writes; an expired session is reported as expired."""
    project = _project()
    await _plugin_capture(_sid("bbbbbb"), project)
    summary = await hermes.import_hermes(root=mixed_age_db, project=project, embed=False)
    assert summary.skipped_expired == 2


# ---------------------------------------------------------------------------
# status: pending_on_disk
# ---------------------------------------------------------------------------


def _settle() -> datetime:
    return datetime.now(UTC) - timedelta(minutes=30)


def test_pending_counts_only_what_an_import_would_store(tmp_path: Path, monkeypatch):
    db = tmp_path / "state.db"
    conn = make_db(db)
    add_simple_session(conn, _sid("fresh_unseen"), days_ago=2)  # pending
    add_simple_session(conn, _sid("already_known"), days_ago=2)  # in the store
    add_simple_session(conn, _sid("ancient"), days_ago=200)  # past retention
    add_session(conn, _sid("empty"), days_ago=2)  # no turns
    add_simple_session(conn, _sid("just_now"), days_ago=0.0, ended=False)  # still running
    conn.commit()
    conn.close()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    assert hermes.pending_on_disk({_sid("already_known")}, _settle()) == (1, 1)


def test_pending_is_zero_without_a_database(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "none"))
    assert hermes.pending_on_disk(set(), _settle()) == (0, 0)


@pytest.mark.asyncio
async def test_status_capture_block_reports_hermes_pending(tmp_path: Path, monkeypatch):
    from hafiz.core.freshness import capture_freshness

    db = tmp_path / "state.db"
    conn = make_db(db)
    add_simple_session(conn, _sid("aaaaaa"), days_ago=2)  # plugin-captured
    add_simple_session(conn, _sid("bbbbbb"), days_ago=2)  # genuinely pending
    add_simple_session(conn, _sid("cccccc"), days_ago=400)  # expired: not pending
    conn.commit()
    conn.close()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Count against this test's own baseline: the shared test DB may hold other
    # hermes rows, but none of them can name these session ids.
    await _plugin_capture(_sid("aaaaaa"), _project())
    capture = await capture_freshness()
    assert capture["hermes"]["pending_on_disk"] == 1
    assert "active_on_disk" in capture["hermes"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_help_lists_the_shared_flags():
    result = CliRunner().invoke(app, ["import", "hermes", "--help"])
    assert result.exit_code == 0, result.output
    for flag in ("--dry-run", "--json", "--project", "--since", "--limit", "--no-embed"):
        assert flag in result.output


def test_cli_dry_run_json_shape(mixed_age_db: Path):
    result = CliRunner().invoke(
        app,
        [
            "import",
            "hermes",
            str(mixed_age_db),
            "--project",
            "hafiz-test-cli",
            "--dry-run",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["action"] == "import_hermes"
    assert payload["dry_run"] is True
    assert payload["summary"]["skipped_expired"] == 2
    assert payload["summary"]["communications_created"] == 1
    assert payload["summary"]["messages_written"] == 4
    assert payload["retention_sweep"]["dry_run"] is True
