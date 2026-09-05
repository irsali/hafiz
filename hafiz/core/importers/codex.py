"""Codex CLI rollout importer.

Codex records each session as an append-only JSONL "rollout" under
``$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<timestamp>-<uuid>.jsonl``
(``CODEX_HOME`` defaults to ``~/.codex``). Archived sessions move to a
parallel ``archived_sessions/`` tree unchanged, so both are read.

Record shapes:

``{"type": "session_meta", "payload": {...}}``
    First line. Carries ``id``, ``cwd``, ``cli_version``, ``git``.
    ``cwd`` is what lets a session land project-scoped.

``{"type": "response_item", "payload": {"type": "message"|"reasoning"|
"function_call"|"function_call_output", ...}}``
    The conversation itself.

``{"type": "event_msg", ...}``
    Telemetry (token counts, approvals). Not conversational; skipped.

Two deliberate accommodations, both because Codex is explicitly tolerant
of its own schema drift:

* Field lookups fall back through several spellings and tolerate a
  ``payload`` wrapper being present or absent, rather than assuming one
  client version's layout.
* Unrecognised record types are skipped silently rather than treated as
  errors, so a newer Codex adding an event type does not fail an import.

**Turn identity.** Codex *appends to the same rollout file* when a
session resumes, so one session is one file and a record's position in it
is stable — unlike Claude Code, which spreads a session over many files
that each restart at zero. Where a record carries no native id we
therefore synthesise ``<session-id>:<line-number>``, which is stable
across re-imports of a growing file. A native ``id`` is preferred
whenever present.

**Unverified against a real install.** This parser was written from the
documented rollout format and is exercised against synthetic fixtures;
no Codex session data was available on the machine where it was built.
The shape is deliberately forgiving for that reason, and
``hafiz import codex --dry-run`` is the safe way to check it against a
real ``~/.codex`` before writing anything.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hafiz.core.communications import MessageInput
from hafiz.core.importers.base import (
    ImportSummary,
    ParsedConversation,
    store_conversation,
)

CODEX_AGENT = "codex"

# Record types that carry conversation. Everything else (event_msg,
# telemetry, future additions) is skipped without complaint.
_CONVERSATIONAL = {"response_item"}
_META = {"session_meta"}


def codex_home() -> Path:
    """``$CODEX_HOME`` if set, else ``~/.codex``.

    Honouring the env var matters: when it is set, the whole tree moves,
    and anything hard-coding ``~/.codex/sessions`` silently finds nothing.
    """
    env = os.environ.get("CODEX_HOME")
    return Path(env).expanduser() if env else Path.home() / ".codex"


def session_roots(home: Path | None = None) -> list[Path]:
    """The directories holding rollout files: live and archived."""
    home = home or codex_home()
    return [home / "sessions", home / "archived_sessions"]


def discover_rollout_files(root: Path) -> list[Path]:
    """Find rollout JSONL under ``root`` (a file, or a tree to walk)."""
    if root.is_file() and root.suffix == ".jsonl":
        return [root]
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.jsonl") if p.is_file())


def _coerce_ts(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        seconds = value / 1000 if value > 1e11 else value
        try:
            return datetime.fromtimestamp(float(seconds), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(str(value).rstrip("Z"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _unwrap(record: dict) -> dict:
    """Codex nests details under ``payload`` in some versions, inlines them
    in others. Merge so field lookups don't have to care."""
    payload = record.get("payload")
    if isinstance(payload, dict):
        return {**record, **payload}
    return record


def _collapse_content(value: Any) -> str:
    """Flatten Codex ``content`` onto text.

    It may be a plain string, or a list of parts each of which is a string
    or a dict with ``text`` (``input_text`` / ``output_text`` variants).
    Non-text parts are dropped rather than stringified.
    """
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    out: list[str] = []
    for part in value:
        if isinstance(part, str):
            if part.strip():
                out.append(part)
        elif isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                out.append(text)
    return "\n".join(out).strip()


@dataclass
class ParsedRollout:
    external_id: str
    started_at: datetime
    ended_at: datetime | None
    cwd: str | None
    cli_version: str | None
    git_branch: str | None
    messages: list[MessageInput]


def _iter_records(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield lineno, record


def parse_rollout_file(path: Path) -> ParsedRollout | None:
    """Parse one rollout JSONL. Returns None when it holds no turns."""
    if not path.is_file():
        return None

    external_id: str | None = None
    cwd: str | None = None
    cli_version: str | None = None
    git_branch: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    messages: list[MessageInput] = []

    for lineno, record in _iter_records(path):
        rtype = record.get("type")
        flat = _unwrap(record)

        ts = _coerce_ts(record.get("timestamp") or flat.get("timestamp"))
        if ts is not None:
            started_at = started_at or ts
            ended_at = ts

        if rtype in _META:
            external_id = external_id or (flat.get("id") or flat.get("session_id"))
            cwd = flat.get("cwd") or cwd
            cli_version = flat.get("cli_version") or cli_version
            git = flat.get("git")
            if isinstance(git, dict):
                git_branch = git.get("branch") or git_branch
            continue

        if rtype not in _CONVERSATIONAL:
            continue

        item_type = flat.get("type")
        tool_calls = None
        role = flat.get("role")

        if item_type == "message":
            role = role or "assistant"
            text = _collapse_content(flat.get("content"))
        elif item_type == "reasoning":
            # Reasoning is not conversation; the claude-code and cursor
            # importers keep it out of the embedding corpus too.
            continue
        elif item_type in ("function_call", "local_shell_call", "custom_tool_call"):
            role = "assistant"
            text = _collapse_content(flat.get("content"))
            tool_calls = [
                {
                    "kind": "tool_use",
                    "id": flat.get("call_id") or flat.get("id"),
                    "name": flat.get("name"),
                    "input": flat.get("arguments") or flat.get("action"),
                }
            ]
        elif item_type in ("function_call_output", "custom_tool_call_output"):
            role = "tool"
            output = flat.get("output")
            text = _collapse_content(output) if not isinstance(output, str) else output.strip()
            tool_calls = [
                {
                    "kind": "tool_result",
                    "tool_use_id": flat.get("call_id"),
                    "content_preview": text[:2000],
                }
            ]
        else:
            continue

        if not text and not tool_calls:
            continue
        if role not in ("user", "assistant", "tool", "system"):
            role = "assistant"

        native_id = flat.get("id") or flat.get("call_id")
        messages.append(
            MessageInput(
                id=uuid.uuid4(),
                seq=len(messages),
                role=role,
                content=text,
                ts=ts or started_at or datetime.now(UTC),
                tool_calls=tool_calls,
                # Positional fallback is safe here in a way it is not for
                # claude-code: Codex appends to this same file on resume,
                # so one session is one file and line numbers are stable.
                source_message_id=str(native_id)
                if native_id
                else f"{external_id or path.stem}:{lineno}",
                metadata={"item_type": item_type},
            )
        )

    if not messages:
        return None
    # A rollout without session_meta still has a filename containing its
    # uuid; falling back to the stem beats dropping the session.
    external_id = external_id or path.stem
    return ParsedRollout(
        external_id=str(external_id),
        started_at=started_at or datetime.now(UTC),
        ended_at=ended_at,
        cwd=cwd,
        cli_version=cli_version,
        git_branch=git_branch,
        messages=messages,
    )


async def import_codex(
    *,
    root: Path | None = None,
    project: str | None = None,
    limit: int | None = None,
    since: datetime | None = None,
    dry_run: bool = False,
    embed: bool = True,
    resolve_project: bool = True,
) -> ImportSummary:
    """Import Codex CLI rollouts into ``communications`` + messages.

    ``root`` defaults to both ``$CODEX_HOME/sessions`` and
    ``archived_sessions``; pass a directory or a single ``.jsonl`` to
    narrow it.
    """
    if root is not None:
        files = discover_rollout_files(Path(root))
    else:
        files = [f for r in session_roots() for f in discover_rollout_files(r)]

    summary = ImportSummary(files_seen=len(files))
    if limit is not None:
        files = files[:limit]

    previewed: dict[str, set[str]] = {}
    # Hoisted once rather than re-queried per conversation.
    roots = None
    if project is None and resolve_project:
        from hafiz.core.store import indexed_root_per_project

        roots = await indexed_root_per_project()

    for path in files:
        try:
            parsed = parse_rollout_file(path)
        except Exception as exc:  # noqa: BLE001 - capture-and-continue
            summary.errors.append({"path": str(path), "error": str(exc)})
            summary.files_skipped += 1
            continue
        if parsed is None:
            summary.files_skipped += 1
            continue
        if since is not None and (parsed.ended_at or parsed.started_at) < since:
            summary.files_skipped += 1
            continue

        scope = project
        if scope is None and roots is not None and parsed.cwd:
            from hafiz.core.store import project_for_path

            scope = await project_for_path(parsed.cwd, roots=roots)

        await store_conversation(
            agent=CODEX_AGENT,
            parsed=ParsedConversation(
                external_id=parsed.external_id,
                title=f"Codex session {parsed.external_id[:8]}",
                started_at=parsed.started_at,
                ended_at=parsed.ended_at,
                cwd=parsed.cwd,
                source_path=str(path),
                messages=parsed.messages,
                metadata={
                    "git_branch": parsed.git_branch,
                    "codex_cli_version": parsed.cli_version,
                },
            ),
            summary=summary,
            project=scope,
            embed=embed,
            dry_run=dry_run,
            previewed=previewed,
        )

    return summary


__all__ = [
    "CODEX_AGENT",
    "codex_home",
    "discover_rollout_files",
    "import_codex",
    "parse_rollout_file",
    "session_roots",
]


def peek_session_id(path: Path, *, max_lines: int = 8) -> str | None:
    """A rollout's session id, read from its head.

    Cheap counterpart to :func:`parse_rollout_file` for the
    capture-freshness probe, which must answer "is this session stored?"
    without reading whole rollouts.
    """
    try:
        for _lineno, record in _iter_records(path):
            flat = _unwrap(record)
            if record.get("type") in _META:
                sid = flat.get("id") or flat.get("session_id")
                if sid:
                    return str(sid)
            max_lines -= 1
            if max_lines <= 0:
                break
    except OSError:
        return None
    return None


def pending_on_disk(known: set[str], settle_before: datetime) -> tuple[int, int]:
    """(uncaptured, still-being-written) rollout counts, live and archived."""
    pending = 0
    active = 0
    for root in session_roots():
        for path in discover_rollout_files(root):
            session_id = peek_session_id(path) or path.stem
            if session_id in known:
                continue
            try:
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            except OSError:
                continue
            if mtime > settle_before:
                active += 1
            else:
                pending += 1
    return pending, active
