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
from hafiz.core.database import (
    Communication,
    CommunicationMessage,
    close_engine,
    get_session_factory,
)
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
    """A dry run must persist nothing — but still *report* what it would do.

    This test previously asserted ``communications_created == 0``, which
    encoded a bug rather than the intent: the importer returned zeros for
    every dry run because it skipped the file before touching a counter.
    "Writes nothing" is a claim about the database, so assert that
    against the database, and assert the preview separately.
    """
    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    summary = await import_claude_code(root=jsonl_session, project=project, dry_run=True)

    assert summary.files_seen == 1
    # Nothing persisted.
    factory = get_session_factory()
    async with factory() as s:
        stored = (
            await s.execute(
                select(Communication).where(Communication.scope_value == project),
            )
        ).all()
    assert stored == []
    # But the preview is truthful.
    assert summary.communications_created == 1
    assert summary.messages_written == 5


@pytest.mark.asyncio
async def test_dry_run_preview_matches_the_real_import(jsonl_session: Path):
    """The preview's whole purpose is to predict the real run."""
    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    preview = await import_claude_code(root=jsonl_session, project=project, dry_run=True)
    real = await import_claude_code(root=jsonl_session, project=project)

    assert preview.communications_created == real.communications_created
    assert preview.messages_written == real.messages_written
    assert preview.messages_embedded == real.messages_embedded
    assert preview.sessions_created == real.sessions_created


@pytest.mark.asyncio
async def test_dry_run_reports_pending_turns_for_a_known_session(jsonl_session: Path):
    """A session imported mid-flight is 'existing' yet still has work.

    Hook-driven capture hits this constantly: compaction imports a live
    session, then more turns arrive. Reporting "already seen, nothing to
    do" would make the preview useless exactly where it is used most.
    """
    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    await import_claude_code(root=jsonl_session, project=project)

    again = await import_claude_code(root=jsonl_session, project=project, dry_run=True)
    assert again.communications_existing == 1
    assert again.communications_created == 0
    assert again.messages_written == 0  # fully caught up


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


@pytest.mark.asyncio
async def test_import_survives_null_bytes_in_tool_output(tmp_path: Path):
    """A single U+0000 must not cost the whole session.

    Postgres rejects null bytes in both text and jsonb, and the message
    batch shares one commit — so before sanitization, one binary-ish
    tool_result took every turn in the session down with it.
    """
    sid = str(uuid.uuid4())
    base = datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC)
    u1, a1 = str(uuid.uuid4()), str(uuid.uuid4())
    path = tmp_path / f"{sid}.jsonl"
    path.write_text(
        "\n".join(
            [
                _record(
                    record_type="user",
                    session_id=sid,
                    rec_uuid=u1,
                    role="user",
                    text="read the binary fixture and tell me what changed in it",
                    timestamp=base,
                ),
                _record(
                    record_type="assistant",
                    session_id=sid,
                    rec_uuid=a1,
                    parent_uuid=u1,
                    role="assistant",
                    text="Here is the payload I read back:\x00\x00 truncated binary \x00",
                    timestamp=base + timedelta(seconds=2),
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    summary = await import_claude_code(root=path, project=project)

    assert summary.errors == []
    assert summary.messages_written == 2

    factory = get_session_factory()
    async with factory() as s:
        comm = (
            await s.execute(
                select(Communication).where(Communication.scope_value == project),
            )
        ).scalar_one()
    rows = await list_messages(comm.id)
    assert len(rows) == 2
    assert all("\x00" not in r.content for r in rows)
    # The surrounding text is preserved — only the null byte is dropped.
    assert any("truncated binary" in r.content for r in rows)


@pytest.mark.asyncio
async def test_same_session_across_two_files_keeps_all_turns(tmp_path: Path):
    """Regression: sibling files must not cannibalise each other's turns.

    Claude Code reuses one sessionId across resumed/forked JSONL files, and
    each file restarts ``seq`` at 0. While idempotency keyed on
    ``(communication_id, seq)``, every file after the first collided with
    the first and had its turns dropped as "already present" — 11,214 of
    38,249 turns (29.3%) on a real store, one session spanning 25 files.
    Fixed by deduping on the source's own message id.
    """
    sid = str(uuid.uuid4())
    base = datetime(2026, 5, 2, 9, 0, 0, tzinfo=UTC)

    def _file(name: str, texts: list[str], offset: int) -> Path:
        lines = []
        prev = None
        for i, text in enumerate(texts):
            rid = str(uuid.uuid4())
            lines.append(
                _record(
                    record_type="user" if i % 2 == 0 else "assistant",
                    session_id=sid,
                    rec_uuid=rid,
                    parent_uuid=prev,
                    role="user" if i % 2 == 0 else "assistant",
                    text=text,
                    timestamp=base + timedelta(seconds=offset + i),
                )
            )
            prev = rid
        p = tmp_path / name
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    _file("a.jsonl", [f"first file turn number {i} about the migration" for i in range(4)], 0)
    _file("b.jsonl", [f"second file turn number {i} about the rollback" for i in range(4)], 100)

    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    summary = await import_claude_code(root=tmp_path, project=project)

    # Both files belong to one session, so one communication holding all 8.
    assert summary.communications_created == 1
    assert summary.messages_written == 8


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
    # Previews the session it found rather than reporting a flat zero.
    assert payload["summary"]["communications_created"] == 1
    assert payload["summary"]["messages_written"] > 0


# ---------------------------------------------------------------------------
# peek_session_id — cheap identity, for the capture-freshness probe
# ---------------------------------------------------------------------------


def test_peek_session_id_reads_the_id_from_the_head(tmp_path: Path):
    from hafiz.core.importers.claude_code import peek_session_id

    sid = str(uuid.uuid4())
    path = tmp_path / "whatever.jsonl"
    path.write_text(
        _record(record_type="user", session_id=sid, rec_uuid=str(uuid.uuid4()), text="hi") + "\n",
        encoding="utf-8",
    )
    assert peek_session_id(path) == sid


def test_peek_session_id_does_not_trust_the_filename(tmp_path: Path):
    """The property that invalidated using the stem as the id.

    Claude Code reuses one sessionId across resumed/forked files with
    different names — 124 of 200 files disagreed with their stem on a real
    store, so a freshness probe keyed on the stem reported 124 false
    "uncaptured" sessions.
    """
    from hafiz.core.importers.claude_code import peek_session_id

    real_sid = str(uuid.uuid4())
    path = tmp_path / f"{uuid.uuid4()}.jsonl"  # stem deliberately != sessionId
    path.write_text(
        _record(record_type="user", session_id=real_sid, rec_uuid=str(uuid.uuid4()), text="x")
        + "\n",
        encoding="utf-8",
    )
    assert peek_session_id(path) == real_sid
    assert peek_session_id(path) != path.stem


def test_peek_session_id_skips_leading_junk(tmp_path: Path):
    from hafiz.core.importers.claude_code import peek_session_id

    sid = str(uuid.uuid4())
    path = tmp_path / "s.jsonl"
    path.write_text(
        "\n".join(
            [
                "",
                "not json",
                json.dumps({"type": "queue-operation"}),  # valid JSON, no sessionId
                _record(record_type="user", session_id=sid, rec_uuid=str(uuid.uuid4()), text="x"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert peek_session_id(path) == sid


def test_peek_session_id_gives_up_rather_than_reading_the_whole_file(tmp_path: Path):
    """Bounded by design — this runs a few hundred times per `status`."""
    from hafiz.core.importers.claude_code import peek_session_id

    sid = str(uuid.uuid4())
    path = tmp_path / "s.jsonl"
    filler = [json.dumps({"type": "queue-operation"})] * 50
    tail = _record(record_type="user", session_id=sid, rec_uuid=str(uuid.uuid4()), text="x")
    path.write_text("\n".join(filler + [tail]) + "\n", encoding="utf-8")

    assert peek_session_id(path) is None  # past the default max_lines
    assert peek_session_id(path, max_lines=100) == sid


@pytest.mark.parametrize("content", ["", "   \n\n", "not json\nstill not json\n"])
def test_peek_session_id_returns_none_for_unusable_files(tmp_path: Path, content: str):
    from hafiz.core.importers.claude_code import peek_session_id

    path = tmp_path / "s.jsonl"
    path.write_text(content, encoding="utf-8")
    assert peek_session_id(path) is None


def test_peek_session_id_tolerates_a_missing_file(tmp_path: Path):
    from hafiz.core.importers.claude_code import peek_session_id

    assert peek_session_id(tmp_path / "absent.jsonl") is None


# ---------------------------------------------------------------------------
# Identity-based idempotency (source_message_id)
# ---------------------------------------------------------------------------


def _session_file(tmp_path: Path, name: str, sid: str, records: list[tuple[str, str]]) -> Path:
    """Write a JSONL file for session ``sid`` from (record_uuid, text) pairs."""
    base = datetime(2026, 6, 1, 9, 0, 0, tzinfo=UTC)
    lines = []
    prev = None
    for i, (rid, text) in enumerate(records):
        lines.append(
            _record(
                record_type="user" if i % 2 == 0 else "assistant",
                session_id=sid,
                rec_uuid=rid,
                parent_uuid=prev,
                role="user" if i % 2 == 0 else "assistant",
                text=text,
                timestamp=base + timedelta(seconds=i),
            )
        )
        prev = rid
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_resumed_session_replay_is_collapsed_not_duplicated(tmp_path: Path):
    """A resumed file re-emits earlier turns under their original ids.

    Those must dedupe, or every resume inflates the transcript. This is the
    same mechanism as the cross-file fix, seen from the other side: identity
    both *keeps* genuinely new turns and *collapses* genuine repeats, which
    positional seq could do neither of.
    """
    sid = str(uuid.uuid4())
    shared = [(str(uuid.uuid4()), f"shared turn {i} about the schema change") for i in range(3)]
    fresh = [(str(uuid.uuid4()), f"new turn {i} about the rollback plan") for i in range(2)]

    _session_file(tmp_path, "a.jsonl", sid, shared)
    _session_file(tmp_path, "b.jsonl", sid, shared + fresh)  # replays a.jsonl

    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    summary = await import_claude_code(root=tmp_path, project=project)

    assert summary.communications_created == 1
    assert summary.messages_written == 5  # 3 shared + 2 new, not 8

    factory = get_session_factory()
    async with factory() as s:
        comm = (
            await s.execute(
                select(Communication).where(Communication.scope_value == project),
            )
        ).scalar_one()
    rows = await list_messages(comm.id)
    assert len(rows) == 5
    assert len({r.seq for r in rows}) == 5  # seq stayed unique per communication


@pytest.mark.asyncio
async def test_reimport_after_growth_appends_only_the_new_turns(tmp_path: Path):
    """The hook case: capture mid-flight, session continues, capture again."""
    sid = str(uuid.uuid4())
    first = [(str(uuid.uuid4()), f"turn {i} of the first pass over the code") for i in range(3)]

    path = _session_file(tmp_path, "s.jsonl", sid, first)
    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    one = await import_claude_code(root=path, project=project)
    assert one.messages_written == 3

    grown = first + [(str(uuid.uuid4()), f"later turn {i} once tests went green") for i in range(2)]
    _session_file(tmp_path, "s.jsonl", sid, grown)

    two = await import_claude_code(root=path, project=project)
    assert two.communications_created == 0
    assert two.communications_existing == 1
    assert two.messages_written == 2

    three = await import_claude_code(root=path, project=project)
    assert three.messages_written == 0  # fully idempotent


@pytest.mark.asyncio
async def test_dry_run_matches_the_real_run_for_a_multi_file_session(tmp_path: Path):
    """The preview must model identity dedup, not positional dedup."""
    sid = str(uuid.uuid4())
    shared = [(str(uuid.uuid4()), f"shared turn {i} about indexes") for i in range(3)]
    fresh = [(str(uuid.uuid4()), f"new turn {i} about the backfill") for i in range(4)]
    _session_file(tmp_path, "a.jsonl", sid, shared)
    _session_file(tmp_path, "b.jsonl", sid, shared + fresh)

    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    preview = await import_claude_code(root=tmp_path, project=project, dry_run=True)
    real = await import_claude_code(root=tmp_path, project=project)

    assert preview.messages_written == real.messages_written == 7
    assert preview.communications_created == real.communications_created == 1


@pytest.mark.asyncio
async def test_source_message_id_is_stored_for_every_imported_turn(jsonl_session: Path):
    project = f"hafiz-test-{uuid.uuid4().hex[:6]}"
    await import_claude_code(root=jsonl_session, project=project)

    factory = get_session_factory()
    async with factory() as s:
        comm = (
            await s.execute(
                select(Communication).where(Communication.scope_value == project),
            )
        ).scalar_one()
        ids = (
            await s.execute(
                select(CommunicationMessage.source_message_id).where(
                    CommunicationMessage.communication_id == comm.id
                )
            )
        ).all()

    values = [v for (v,) in ids]
    assert all(v for v in values), "every claude-code turn carries its source uuid"
    assert len(set(values)) == len(values), "and they are distinct"
