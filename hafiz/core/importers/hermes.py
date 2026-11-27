"""Hermes agent session importer.

Hermes keeps every session in one SQLite database, ``state.db`` under
``$HERMES_HOME`` (default ``~/.hermes``; ``%LOCALAPPDATA%\\hermes`` on
Windows). Two tables matter:

``sessions``
    One row per session. ``id`` looks like ``20260628_184250_60c07a``;
    ``started_at`` / ``ended_at`` are epoch seconds; ``cwd`` and
    ``git_repo_root`` say where it ran.

``messages``
    One row per turn, ordered by the autoincrement ``id``. ``role`` is
    ``user`` / ``assistant`` / ``tool``. An assistant turn that calls tools
    carries them as an OpenAI-style JSON list in ``tool_calls``; the
    matching result is a separate ``tool`` row whose ``tool_call_id`` points
    back at the call.

The row ``id`` is a real per-turn identity, so it becomes
``source_message_id`` and re-import is idempotent at turn granularity.

**The database is opened read-only.** It belongs to a running agent and is
in WAL mode; hafiz reads it and must never write, checkpoint or migrate it.

Which rows count as the conversation
------------------------------------

Hermes soft-deletes instead of deleting, with two flags on ``messages``:

``active = 0, compacted = 0``
    The user rewound or undid the turn. It was taken back, so it is not
    part of the conversation and is **skipped**.

``active = 0, compacted = 1``
    Context compaction archived the turn: the model no longer sees it, but
    it really happened and is the only full record of the session. It is
    **imported**.

Compaction also re-inserts the surviving tail of the conversation as fresh
``active = 1`` rows, so those turns exist twice. On a real session 202 of
204 live rows were exact copies of archived ones. Importing both would
double the transcript, so a live row that exactly matches a not-yet-matched
archived row (same role, content, tool call id and tool calls) is
**skipped**; each archived row cancels at most one live row, so a genuinely
repeated message ("yes") typed after compaction survives.

Older Hermes versions lack some of these columns. Every optional column is
probed with ``PRAGMA table_info`` and a missing one is simply not used.

Two interactions with the rest of hafiz
---------------------------------------

* The Hermes memory plugin already captures each session at its end via
  ``hafiz capture``, under a random ``external_id`` but tagged with the
  Hermes session id (``metadata.session_slug`` and the title
  ``hermes:<id>``). A session found there is counted as already present and
  not imported again — see :func:`plugin_captured_ids`.
* Source-layer rows are retention-bounded (``started_at`` + 90 days). A
  session already past that is **not imported** and is reported as
  ``skipped_expired``, rather than written and tombstoned by the next sweep.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select

from hafiz.core.communications import MessageInput, is_past_retention
from hafiz.core.importers.base import (
    ImportSummary,
    ParsedConversation,
    store_conversation,
)

HERMES_AGENT = "hermes"

_JSON_CONTENT_PREFIX = "\x00json:"
_ROLES = {"user", "assistant", "tool", "system"}
# The plugin titles its capture "hermes:<session id>".
_PLUGIN_TITLE_PREFIX = f"{HERMES_AGENT}:"
#: A session whose last activity (its end, else its newest turn) is this recent
#: is left to the plugin — the same settle window ``status`` uses.
OPEN_SESSION_MINUTES = 30


def hermes_home() -> Path:
    """``$HERMES_HOME``, else this platform's default Hermes directory."""
    override = os.environ.get("HERMES_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def default_db_path() -> Path:
    return hermes_home() / "state.db"


def _resolve_db_path(root: Path) -> Path:
    """Accept ``state.db`` itself or the Hermes home directory holding it."""
    return root / "state.db" if root.is_dir() else root


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open Hermes's database read-only. It is not ours to write."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    # PRAGMA does not take bound parameters; ``table`` is always a literal here.
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _coerce_ts(value: Any) -> datetime | None:
    """Epoch seconds (Hermes) — tolerate a milliseconds-scale value."""
    if value is None or value == "":
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds > 1e11:
        seconds /= 1000
    try:
        return datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


@dataclass
class _SessionHeader:
    id: str
    started_at: datetime
    ended_at: datetime | None
    title: str | None
    cwd: str | None
    git_repo_root: str | None
    git_branch: str | None
    source: str | None
    model: str | None
    parent_session_id: str | None
    end_reason: str | None

    def project_paths(self) -> list[str]:
        """Where this session ran, most specific signal first."""
        return [p for p in (self.git_repo_root, self.cwd) if p]


def _optional(row: sqlite3.Row, name: str) -> Any:
    return row[name] if name in row.keys() else None  # noqa: SIM118 - sqlite3.Row


def list_sessions(conn: sqlite3.Connection) -> list[_SessionHeader]:
    """Every session header, newest first.

    ``id`` and ``started_at`` are required; everything else is read only if
    this Hermes version has the column.
    """
    cols = _columns(conn, "sessions")
    if not {"id", "started_at"} <= cols:
        raise sqlite3.DatabaseError("sessions table lacks id/started_at — not a Hermes state.db")
    wanted = [
        c
        for c in (
            "id",
            "started_at",
            "ended_at",
            "title",
            "cwd",
            "git_repo_root",
            "git_branch",
            "source",
            "model",
            "parent_session_id",
            "end_reason",
        )
        if c in cols
    ]
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT {', '.join(wanted)} FROM sessions ORDER BY started_at DESC")
    headers: list[_SessionHeader] = []
    for row in rows:
        started = _coerce_ts(row["started_at"])
        if not row["id"] or started is None:
            continue
        headers.append(
            _SessionHeader(
                id=str(row["id"]),
                started_at=started,
                ended_at=_coerce_ts(_optional(row, "ended_at")),
                title=_optional(row, "title") or None,
                cwd=_optional(row, "cwd") or None,
                git_repo_root=_optional(row, "git_repo_root") or None,
                git_branch=_optional(row, "git_branch") or None,
                source=_optional(row, "source") or None,
                model=_optional(row, "model") or None,
                parent_session_id=_optional(row, "parent_session_id") or None,
                end_reason=_optional(row, "end_reason") or None,
            )
        )
    return headers


def _decode_content(raw: Any) -> str:
    """Message text. Hermes stores non-string content as ``\\x00json:<json>``."""
    if raw is None:
        return ""
    if isinstance(raw, str) and raw.startswith(_JSON_CONTENT_PREFIX):
        try:
            raw = json.loads(raw[len(_JSON_CONTENT_PREFIX) :])
        except (json.JSONDecodeError, TypeError):
            raw = raw[len(_JSON_CONTENT_PREFIX) :]
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        # Multimodal parts: keep the text, drop images and other payloads.
        parts = []
        for part in raw:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(p.strip() for p in parts if p and p.strip())
    if isinstance(raw, dict) and isinstance(raw.get("text"), str):
        return raw["text"].strip()
    return ""


def _parse_tool_calls(raw: Any) -> list[dict]:
    """Map Hermes's OpenAI-style ``tool_calls`` onto the source layer's shape."""
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for call in raw:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        args = fn.get("arguments", call.get("arguments"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                pass  # keep the raw string rather than lose it
        out.append(
            {
                "kind": "tool_use",
                "id": call.get("id") or call.get("call_id"),
                "name": fn.get("name") or call.get("name"),
                "input": args,
            }
        )
    return out


def _reasoning(row: sqlite3.Row) -> str | None:
    for name in ("reasoning_content", "reasoning"):
        text = _optional(row, name)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def read_messages(
    conn: sqlite3.Connection, session_id: str, started_at: datetime
) -> list[MessageInput]:
    """One session's turns, in Hermes's insertion order.

    Applies the ``active`` / ``compacted`` rules described in the module
    docstring. Reasoning rides in metadata, never in ``content``, so it does
    not pollute the embedding corpus (same as the cursor importer).
    """
    cols = _columns(conn, "messages")
    if not {"id", "session_id", "role"} <= cols:
        raise sqlite3.DatabaseError("messages table lacks id/session_id/role")
    optional = [
        c
        for c in (
            "content",
            "tool_call_id",
            "tool_calls",
            "tool_name",
            "timestamp",
            "finish_reason",
            "reasoning",
            "reasoning_content",
            "active",
            "compacted",
        )
        if c in cols
    ]
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT id, role, {', '.join(optional)} FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()

    def _flag(row: sqlite3.Row, name: str, default: int) -> int:
        value = _optional(row, name)
        return default if value is None else int(value)

    # Rewound turns (taken back by the user) are not part of the conversation.
    kept = [r for r in rows if _flag(r, "active", 1) or _flag(r, "compacted", 0)]

    def _key(r: sqlite3.Row) -> tuple:
        return (
            r["role"],
            _optional(r, "content"),
            _optional(r, "tool_call_id"),
            _optional(r, "tool_calls"),
        )

    # Compaction re-inserts the surviving tail as copies, and a later
    # compaction archives those copies in turn, so one turn can exist many
    # times. A copy keeps the original's timestamp; a turn genuinely repeated
    # later does not. So with timestamps, the first row of each (turn,
    # timestamp) wins. Without them (older Hermes), only a live row can be a
    # copy, and each archived row cancels at most one.
    has_ts = "timestamp" in cols
    seen: set[tuple] = set()
    archived = Counter(_key(r) for r in kept if _flag(r, "compacted", 0))

    messages: list[MessageInput] = []
    for row in kept:
        compacted = bool(_flag(row, "compacted", 0))
        if has_ts:
            ident = (*_key(row), _optional(row, "timestamp"))
            if ident in seen:
                continue
            seen.add(ident)
        elif not compacted and archived[_key(row)] > 0:
            archived[_key(row)] -= 1
            continue

        role = row["role"]
        if role not in _ROLES:
            continue
        text = _decode_content(_optional(row, "content"))
        tool_calls = _parse_tool_calls(_optional(row, "tool_calls"))
        if role == "tool":
            tool_calls = [
                {
                    "kind": "tool_result",
                    "tool_use_id": _optional(row, "tool_call_id"),
                    "name": _optional(row, "tool_name"),
                    "content_preview": text[:2000],
                }
            ]
        # Require prose or a tool call; an empty turn is noise in `recall`.
        if not text and not tool_calls:
            continue

        metadata: dict = {}
        if compacted:
            metadata["compacted"] = True
        if _optional(row, "tool_name"):
            metadata["tool_name"] = _optional(row, "tool_name")
        if _optional(row, "finish_reason"):
            metadata["finish_reason"] = _optional(row, "finish_reason")
        reasoning = _reasoning(row)
        if reasoning:
            metadata["reasoning"] = reasoning

        messages.append(
            MessageInput(
                id=uuid.uuid4(),
                seq=len(messages),
                role=role,
                content=text,
                ts=_coerce_ts(_optional(row, "timestamp")) or started_at,
                tool_calls=tool_calls or None,
                source_message_id=str(row["id"]),
                metadata=metadata,
            )
        )
    return messages


async def plugin_captured_ids() -> set[str]:
    """Hermes session ids the memory plugin already captured.

    The plugin runs ``hafiz capture --session <hermes id> --title
    hermes:<hermes id> --source agent:hermes``. ``store_transcript`` gives
    that communication ``agent='hermes'``, a random ``external_id`` (so the
    importer's own idempotency key never matches it), and records the id in
    ``metadata.session_slug`` and ``metadata.title``. Those two fields are
    the only place the Hermes session id lands, so they are what is read.

    Tombstoned rows count too: a capture that was forgotten or expired must
    not be resurrected by an import.

    The filter runs in Python rather than SQL because JSON-path syntax
    differs between Postgres and SQLite, and this agent's rows are few.
    """
    from hafiz.core.database import Communication, get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        rows = await session.execute(
            select(Communication.metadata_).where(Communication.agent == HERMES_AGENT)
        )
        found: set[str] = set()
        for (meta,) in rows.all():
            if not isinstance(meta, dict):
                continue
            slug = meta.get("session_slug")
            if isinstance(slug, str) and slug:
                found.add(slug)
            title = meta.get("title")
            if isinstance(title, str) and title.startswith(_PLUGIN_TITLE_PREFIX):
                found.add(title[len(_PLUGIN_TITLE_PREFIX) :])
        return found


async def import_hermes(
    *,
    root: Path | None = None,
    project: str | None = None,
    limit: int | None = None,
    since: datetime | None = None,
    dry_run: bool = False,
    embed: bool = True,
    resolve_project: bool = True,
) -> ImportSummary:
    """Import Hermes sessions into ``communications`` + messages.

    ``root`` may be ``state.db`` or the Hermes home directory containing it;
    it defaults to ``$HERMES_HOME`` / ``~/.hermes``. Sessions are visited
    newest first, and ``limit`` stops after that many have been examined.

    Per session, in order:

    1. past its retention → not imported, ``skipped_expired``;
    2. already captured by the Hermes memory plugin → counted in
       ``communications_existing``, not imported;
    3. otherwise stored (or previewed) through the shared writer.

    When ``project`` is not given and ``resolve_project`` is set, each
    session is scoped to the indexed project containing its
    ``git_repo_root`` (else its ``cwd``). No match yields no project.
    """
    db_path = _resolve_db_path(root or default_db_path())
    summary = ImportSummary()
    if not db_path.is_file():
        return summary

    try:
        conn = _connect_readonly(db_path)
    except sqlite3.Error as exc:
        summary.errors.append({"path": str(db_path), "error": f"sqlite: {exc}"})
        return summary

    roots = None
    if project is None and resolve_project:
        from hafiz.core.store import indexed_root_per_project

        roots = await indexed_root_per_project()

    try:
        try:
            headers = list_sessions(conn)
        except sqlite3.Error as exc:
            summary.errors.append({"path": str(db_path), "error": f"sqlite: {exc}"})
            return summary

        captured = await plugin_captured_ids()
        open_cutoff = datetime.now(UTC) - timedelta(minutes=OPEN_SESSION_MINUTES)
        previewed: dict[str, set[str]] = {}

        for header in headers:
            if since is not None and (header.ended_at or header.started_at) < since:
                continue
            if limit is not None and summary.files_seen >= limit:
                break

            if is_past_retention(header.started_at):
                summary.files_seen += 1
                summary.skipped_expired += 1
                continue
            if header.id in captured:
                summary.files_seen += 1
                summary.communications_existing += 1
                continue
            # A session that is still running, or ended moments ago, is left
            # for the plugin, which captures it at session end; importing it
            # first would store it twice (the plugin's capture has its own
            # random key). Same settle window as `status`.
            if _last_activity(conn, header) > open_cutoff:
                continue

            try:
                messages = read_messages(conn, header.id, header.started_at)
                if not messages:
                    continue  # a session with no turns is not worth a row
                summary.files_seen += 1

                scope = project
                if scope is None and roots is not None:
                    from hafiz.core.store import project_for_path

                    for path in header.project_paths():
                        scope = await project_for_path(path, roots=roots)
                        if scope:
                            break

                await store_conversation(
                    agent=HERMES_AGENT,
                    parsed=ParsedConversation(
                        external_id=header.id,
                        title=header.title,
                        started_at=header.started_at,
                        ended_at=header.ended_at,
                        cwd=header.cwd or header.git_repo_root,
                        source_path=str(db_path),
                        messages=messages,
                        metadata={
                            "hermes_source": header.source,
                            "model": header.model,
                            "git_branch": header.git_branch,
                            "git_repo_root": header.git_repo_root,
                            "parent_session_id": header.parent_session_id,
                            "end_reason": header.end_reason,
                        },
                    ),
                    summary=summary,
                    project=scope,
                    embed=embed,
                    dry_run=dry_run,
                    previewed=previewed,
                )
            except Exception as exc:  # noqa: BLE001 - capture-and-continue
                summary.errors.append({"path": header.id, "error": str(exc)})
                summary.files_skipped += 1
    finally:
        conn.close()

    return summary


def _last_activity(conn: sqlite3.Connection, header: _SessionHeader) -> datetime:
    """When a session last did anything: its end, else its newest turn."""
    if header.ended_at is not None:
        return header.ended_at
    if "timestamp" in _columns(conn, "messages"):
        row = conn.execute(
            "SELECT MAX(timestamp) FROM messages WHERE session_id = ?", (header.id,)
        ).fetchone()
        newest = _coerce_ts(row[0]) if row else None
        if newest is not None:
            return max(newest, header.started_at)
    return header.started_at


def pending_on_disk(known: set[str], settle_before: datetime) -> tuple[int, int]:
    """(uncaptured, still-open) session counts in Hermes's database.

    ``known`` must already include the ids the memory plugin captured (see
    :func:`plugin_captured_ids`); ``status`` adds them. A session is
    *uncaptured work* only if an import would actually store it, so sessions
    past retention and sessions with no turns are not counted — otherwise
    the warning would be permanent and unfixable. Settledness is decided by
    last activity, not a file mtime, because sessions are rows.
    """
    db_path = default_db_path()
    if not db_path.is_file():
        return 0, 0
    try:
        conn = _connect_readonly(db_path)
    except sqlite3.Error:
        return 0, 0
    pending = 0
    active = 0
    try:
        for header in list_sessions(conn):
            if header.id in known or is_past_retention(header.started_at):
                continue
            # The same parse an import runs, so a session an import would skip
            # (only blank turns) is not reported as pending forever.
            if not read_messages(conn, header.id, header.started_at):
                continue
            if _last_activity(conn, header) > settle_before:
                active += 1
            else:
                pending += 1
    except sqlite3.Error:
        return 0, 0
    finally:
        conn.close()
    return pending, active


__all__ = [
    "HERMES_AGENT",
    "default_db_path",
    "hermes_home",
    "import_hermes",
    "list_sessions",
    "pending_on_disk",
    "plugin_captured_ids",
    "read_messages",
]
