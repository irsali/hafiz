"""Cursor chat importer.

Cursor stores its agent conversations in a SQLite database under
``~/.config/Cursor/User/globalStorage/state.vscdb`` (``~/Library/…`` on
macOS, ``%APPDATA%`` on Windows). Three pieces matter:

``composerHeaders`` (table)
    One row per conversation — Cursor calls them *composers*. Carries
    ``composerId``, ``workspaceId``, created/updated timestamps, and a
    JSON ``value`` holding the conversation's display name.

``cursorDiskKV['composerData:<composerId>']``
    The conversation body. ``fullConversationHeadersOnly`` is the ordered
    list of turn references — this is the only reliable ordering, since
    the turns themselves are stored as independent key/value rows.

``cursorDiskKV['bubbleId:<composerId>:<bubbleId>']``
    One turn, Cursor's *bubble*. ``type`` is 1 for user and 2 for
    assistant; ``text`` is the prose, ``thinking`` the reasoning, and
    ``toolFormerData`` a tool invocation.

``bubbleId`` is a real per-turn identity, so it becomes
``source_message_id`` and re-import is idempotent at turn granularity —
see migration ``0008`` for why that matters.

**The database is opened read-only.** It belongs to a running
application; hafiz reads it and must never write, lock, or migrate it.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from hafiz.core.communications import MessageInput
from hafiz.core.importers.base import (
    ImportSummary,
    ParsedConversation,
    store_conversation,
)

CURSOR_AGENT = "cursor"

# Cursor's bubble `type` enum.
_ROLE_USER = 1
_ROLE_ASSISTANT = 2


def default_storage_dir() -> Path:
    """Where Cursor keeps ``User/`` on this platform."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Cursor" / "User"
    if sys.platform == "win32":
        import os

        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
        return base / "Cursor" / "User"
    return Path.home() / ".config" / "Cursor" / "User"


def default_db_path() -> Path:
    return default_storage_dir() / "globalStorage" / "state.vscdb"


@dataclass
class _Bubble:
    bubble_id: str
    role: str
    text: str
    tool_calls: list | None
    thinking: str | None
    created_at: datetime | None


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open Cursor's database read-only.

    ``immutable=0`` (the default) still takes read locks, which is the
    correct, safe behaviour against a database another process may be
    writing. We never open it read-write: it is not ours.
    """
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def _coerce_ts(value: Any) -> datetime | None:
    """Cursor mixes epoch-milliseconds ints and ISO-8601 strings."""
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        # Values are milliseconds; guard against a seconds-scale value.
        seconds = value / 1000 if value > 1e11 else value
        try:
            return datetime.fromtimestamp(seconds, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(str(value).rstrip("Z"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _extract_tool_calls(bubble: dict) -> list | None:
    """Normalize ``toolFormerData`` onto the shape the source layer uses."""
    tf = bubble.get("toolFormerData")
    if not isinstance(tf, dict):
        return None
    raw_result = tf.get("result")
    if raw_result is not None and not isinstance(raw_result, str):
        raw_result = json.dumps(raw_result)
    return [
        {
            "kind": "tool_use",
            "id": tf.get("toolCallId") or tf.get("modelCallId"),
            "name": tf.get("name") or tf.get("tool"),
            "input": tf.get("params") if tf.get("params") is not None else tf.get("rawArgs"),
            "status": tf.get("status"),
            "content_preview": (raw_result or "")[:2000] or None,
        }
    ]


def _parse_bubble(raw: str) -> _Bubble | None:
    try:
        bubble = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(bubble, dict):
        return None

    bubble_id = bubble.get("bubbleId")
    if not bubble_id:
        return None

    text = (bubble.get("text") or "").strip()
    thinking = bubble.get("thinking")
    if isinstance(thinking, dict):
        thinking = thinking.get("text") or thinking.get("thinking")
    thinking = (thinking or "").strip() or None
    tool_calls = _extract_tool_calls(bubble)

    # Require prose or a tool call. A bubble carrying only `thinking`
    # would store as an empty-content row — noise in `recall`, never
    # embedded — and the claude-code importer drops the same shape.
    # Reasoning rides along in metadata when a turn has other substance.
    if not text and not tool_calls:
        return None

    btype = bubble.get("type")
    if btype == _ROLE_USER:
        role = "user"
    elif tool_calls and not text:
        # A bubble that is only a tool invocation is re-tagged `tool`, so
        # the schema's role CHECK and recall's --has-tool-call filter line
        # up with the claude-code importer's conventions.
        role = "tool"
    else:
        role = "assistant"

    return _Bubble(
        bubble_id=str(bubble_id),
        role=role,
        text=text,
        tool_calls=tool_calls,
        thinking=thinking,
        created_at=_coerce_ts(bubble.get("createdAt")),
    )


def workspace_folders(storage_dir: Path | None = None) -> dict[str, str]:
    """Map Cursor ``workspaceId`` → the filesystem folder it was opened on.

    Read from ``workspaceStorage/<id>/workspace.json``, which holds a
    ``file://`` URI under either ``folder`` (a directory) or ``workspace``
    (a ``.code-workspace`` file, whose parent is the directory). Lets a
    Cursor conversation land project-scoped rather than untagged.
    """
    storage_dir = storage_dir or default_storage_dir()
    root = storage_dir / "workspaceStorage"
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for entry in root.iterdir():
        meta = entry / "workspace.json"
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        uri = data.get("folder") or data.get("workspace")
        if not uri:
            continue
        try:
            path = Path(unquote(urlparse(str(uri)).path))
        except ValueError:
            continue
        if data.get("folder") is None:
            path = path.parent  # `.code-workspace` file → its directory
        out[entry.name] = str(path)
    return out


def parse_conversations(
    db_path: Path,
    *,
    limit: int | None = None,
    since: datetime | None = None,
    folders: dict[str, str] | None = None,
) -> list[ParsedConversation]:
    """Read every composer out of Cursor's database.

    Conversations with no usable turns are dropped rather than stored as
    empty shells — Cursor keeps drafts and abandoned composers around.
    """
    if not db_path.is_file():
        return []
    folders = folders if folders is not None else {}

    conn = _connect_readonly(db_path)
    try:
        conn.row_factory = sqlite3.Row
        headers = conn.execute(
            "SELECT composerId, workspaceId, createdAt, lastUpdatedAt, isArchived,"
            " isSubagent, value FROM composerHeaders ORDER BY createdAt"
        ).fetchall()

        out: list[ParsedConversation] = []
        for head in headers:
            composer_id = head["composerId"]
            if not composer_id:
                continue
            started = _coerce_ts(head["createdAt"]) or datetime.now(UTC)
            ended = _coerce_ts(head["lastUpdatedAt"])
            if since is not None and (ended or started) < since:
                continue

            row = conn.execute(
                "SELECT value FROM cursorDiskKV WHERE key = ?",
                (f"composerData:{composer_id}",),
            ).fetchone()
            if row is None:
                continue
            try:
                data = json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                continue

            title = data.get("name")
            if not title:
                try:
                    title = json.loads(head["value"]).get("name")
                except (json.JSONDecodeError, TypeError):
                    title = None

            order = data.get("fullConversationHeadersOnly") or []
            messages: list[MessageInput] = []
            for seq, entry in enumerate(order):
                if not isinstance(entry, dict):
                    continue
                bubble_id = entry.get("bubbleId")
                if not bubble_id:
                    continue
                brow = conn.execute(
                    "SELECT value FROM cursorDiskKV WHERE key = ?",
                    (f"bubbleId:{composer_id}:{bubble_id}",),
                ).fetchone()
                if brow is None:
                    continue
                bubble = _parse_bubble(brow["value"])
                if bubble is None:
                    continue

                metadata: dict = {"bubble_type": entry.get("type")}
                if bubble.thinking:
                    # Kept off `content` for the same reason as the
                    # claude-code importer: high value for debugging, but
                    # it would pollute the embedding corpus.
                    metadata["thinking"] = bubble.thinking

                messages.append(
                    MessageInput(
                        id=uuid.uuid4(),
                        seq=seq,
                        role=bubble.role,
                        content=bubble.text,
                        ts=bubble.created_at or _coerce_ts(entry.get("createdAt")) or started,
                        tool_calls=bubble.tool_calls,
                        source_message_id=bubble.bubble_id,
                        metadata=metadata,
                    )
                )

            if not messages:
                continue

            workspace_id = head["workspaceId"]
            out.append(
                ParsedConversation(
                    external_id=str(composer_id),
                    title=title,
                    started_at=started,
                    ended_at=ended,
                    cwd=folders.get(workspace_id),
                    source_path=str(db_path),
                    messages=messages,
                    metadata={
                        "workspace_id": workspace_id,
                        "is_archived": bool(head["isArchived"]),
                        "is_subagent": bool(head["isSubagent"]),
                    },
                )
            )
            if limit is not None and len(out) >= limit:
                break
        return out
    finally:
        conn.close()


async def import_cursor(
    *,
    root: Path | None = None,
    project: str | None = None,
    limit: int | None = None,
    since: datetime | None = None,
    dry_run: bool = False,
    embed: bool = True,
    resolve_project: bool = True,
) -> ImportSummary:
    """Import Cursor conversations into ``communications`` + messages.

    ``root`` may be the ``state.vscdb`` file or the ``User`` directory
    containing it; it defaults to this platform's Cursor storage.

    When ``project`` is not given and ``resolve_project`` is set, each
    conversation is scoped to the indexed project containing the folder
    its workspace was opened on. An unmatched workspace yields no
    project rather than a guess.
    """
    root = root or default_db_path()
    storage_dir = None
    if root.is_dir():
        storage_dir = root
        root = root / "globalStorage" / "state.vscdb"
    else:
        # …/User/globalStorage/state.vscdb → …/User
        storage_dir = root.parent.parent

    summary = ImportSummary()
    if not root.is_file():
        return summary

    folders = workspace_folders(storage_dir)
    try:
        conversations = parse_conversations(root, limit=limit, since=since, folders=folders)
    except sqlite3.Error as exc:
        summary.errors.append({"path": str(root), "error": f"sqlite: {exc}"})
        return summary

    summary.files_seen = len(conversations)
    previewed: dict[str, set[str]] = {}
    # Hoisted once rather than re-queried per conversation.
    roots = None
    if project is None and resolve_project:
        from hafiz.core.store import indexed_root_per_project

        roots = await indexed_root_per_project()

    for parsed in conversations:
        scope = project
        if scope is None and roots is not None and parsed.cwd:
            from hafiz.core.store import project_for_path

            scope = await project_for_path(parsed.cwd, roots=roots)
        try:
            await store_conversation(
                agent=CURSOR_AGENT,
                parsed=parsed,
                summary=summary,
                project=scope,
                embed=embed,
                dry_run=dry_run,
                previewed=previewed,
            )
        except Exception as exc:  # noqa: BLE001 - capture-and-continue
            summary.errors.append({"path": parsed.external_id, "error": str(exc)})
            summary.files_skipped += 1

    return summary


__all__ = [
    "CURSOR_AGENT",
    "default_db_path",
    "default_storage_dir",
    "import_cursor",
    "parse_conversations",
    "workspace_folders",
]


def pending_on_disk(known: set[str], settle_before: datetime) -> tuple[int, int]:
    """(uncaptured, still-open) conversation counts in Cursor's database.

    Composers are rows, not files, so "settled" is decided by
    ``lastUpdatedAt`` rather than a file mtime — a conversation you are
    still typing in is not uncaptured work.
    """
    db_path = default_db_path()
    if not db_path.is_file():
        return 0, 0
    try:
        conn = _connect_readonly(db_path)
    except sqlite3.Error:
        return 0, 0
    try:
        rows = conn.execute("SELECT composerId, lastUpdatedAt FROM composerHeaders").fetchall()
        # Cursor keeps drafts and abandoned composers: on a real store, 70
        # headers yielded 26 conversations with any turns. Counting the
        # empty ones as "uncaptured" would make the warning permanent and
        # unfixable — importing them changes nothing. So restrict to
        # composers that actually have bubbles, which one indexed LIKE
        # over the key column answers without reading any values.
        with_turns = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT substr(key, 10, instr(substr(key, 10), ':') - 1)"
                " FROM cursorDiskKV WHERE key LIKE 'bubbleId:%'"
            )
            if r[0]
        }
    except sqlite3.Error:
        return 0, 0
    finally:
        conn.close()

    pending = 0
    active = 0
    for composer_id, last_updated in rows:
        if not composer_id or str(composer_id) in known:
            continue
        if str(composer_id) not in with_turns:
            continue
        ts = _coerce_ts(last_updated)
        if ts is not None and ts > settle_before:
            active += 1
        else:
            pending += 1
    return pending, active
