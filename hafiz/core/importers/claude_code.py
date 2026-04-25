"""Claude Code session JSONL importer.

Reads the per-session JSONL files Claude Code stores under
``~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl``, extracts
``user`` / ``assistant`` / ``tool`` turns, and stores them in the
``communications`` + ``communication_messages`` tables.

Idempotent by ``(agent='claude-code', external_id=<jsonl session uuid>)``
— re-importing the same file is a no-op at the communication level,
and append_messages also dedupes per-(communication_id, seq).

Selective embedding policy (from
:mod:`hafiz.core.communications`) is enforced at write time: short
turns and pure tool-result echoes don't get embedded. Non-message
records (queue-operation, attachment, file-history-snapshot) are
skipped.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hafiz.core.communications import (
    MessageInput,
    append_messages,
    upsert_communication,
)
from hafiz.core.sessions import (
    create_session,
    get_session_by_slug,
)


CLAUDE_CODE_AGENT = "claude-code"
DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Records with these top-level "type" values are not turn-shaped and
# get dropped during parse. The remaining rows are user / assistant
# (and sometimes tool, modeled below by inspecting content blocks).
_NON_MESSAGE_TYPES = {"queue-operation", "attachment", "file-history-snapshot"}


@dataclass
class ImportSummary:
    """High-level outcome of one ``hafiz import claude-code`` run."""

    files_seen: int = 0
    files_skipped: int = 0
    communications_created: int = 0
    communications_existing: int = 0
    messages_written: int = 0
    messages_embedded: int = 0
    sessions_created: int = 0
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files_seen": self.files_seen,
            "files_skipped": self.files_skipped,
            "communications_created": self.communications_created,
            "communications_existing": self.communications_existing,
            "messages_written": self.messages_written,
            "messages_embedded": self.messages_embedded,
            "sessions_created": self.sessions_created,
            "errors": self.errors,
        }


@dataclass
class ParsedFile:
    """Outcome of parsing one JSONL file (still in-memory)."""

    path: Path
    external_id: str
    started_at: datetime
    ended_at: datetime | None
    cwd: str | None
    git_branch: str | None
    version: str | None
    messages: list[MessageInput]


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------


def _coerce_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not value:
        return datetime.now(timezone.utc)
    s = str(value).rstrip("Z")
    try:
        parsed = datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _extract_text_and_tools(
    content_blocks: Iterable[dict],
) -> tuple[str, list[dict] | None, str | None]:
    """Walk the message ``content`` list. Returns:

    * concatenated text for embedding/display,
    * ``tool_calls`` JSON or None (tool_use + tool_result blocks),
    * ``thinking`` text if present (stashed in metadata, not content).
    """
    text_parts: list[str] = []
    tools: list[dict] = []
    thinking_parts: list[str] = []

    for block in content_blocks or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text") or ""
            if text.strip():
                text_parts.append(text)
        elif btype == "tool_use":
            tools.append(
                {
                    "kind": "tool_use",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input"),
                }
            )
        elif btype == "tool_result":
            # tool_result content is sometimes a string, sometimes a
            # list of block dicts (text, image…). Normalize to a
            # text preview so downstream display is uniform.
            raw_content = block.get("content")
            if isinstance(raw_content, list):
                preview = "\n".join(
                    part.get("text", "")
                    for part in raw_content
                    if isinstance(part, dict)
                )
            else:
                preview = str(raw_content) if raw_content is not None else ""
            tools.append(
                {
                    "kind": "tool_result",
                    "tool_use_id": block.get("tool_use_id"),
                    "is_error": block.get("is_error", False),
                    "content_preview": preview[:2000],
                }
            )
        elif btype == "thinking":
            thinking = block.get("thinking") or ""
            if thinking.strip():
                thinking_parts.append(thinking)
        # else: image, document, etc. — skip silently for v1.

    text = "\n".join(text_parts).strip()
    thinking = "\n".join(thinking_parts).strip() or None
    return text, (tools if tools else None), thinking


def _classify_role(record: dict, tools: list[dict] | None) -> str:
    """Map a Claude Code JSONL record onto our role enum.

    Tool results arrive as records with ``type='user'`` whose content
    blocks are tool_result kinds; we re-tag those as ``role='tool'`` so
    the schema's CHECK constraint (and recall filters) work cleanly.
    """
    rtype = record.get("type")
    if rtype == "assistant":
        return "assistant"
    if rtype == "user":
        if tools and all(t.get("kind") == "tool_result" for t in tools):
            return "tool"
        return "user"
    if rtype == "system":
        return "system"
    return rtype or "user"


def parse_jsonl_file(path: Path) -> ParsedFile | None:
    """Parse one JSONL file into a :class:`ParsedFile`. Returns None
    when the file has no message-shaped records.

    The first valid line provides the canonical session uuid (from
    ``sessionId``); subsequent records contribute messages in file
    order. Parent linkage is preserved within the file via a
    ``{claude_uuid → our_uuid}`` map.
    """
    if not path.is_file():
        return None

    external_id: str | None = None
    cwd: str | None = None
    git_branch: str | None = None
    version: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None

    seq = 0
    messages: list[MessageInput] = []
    claude_to_ours: dict[str, uuid.UUID] = {}

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            rtype = rec.get("type")
            if rtype in _NON_MESSAGE_TYPES:
                continue
            sid = rec.get("sessionId")
            if sid and external_id is None:
                external_id = sid
            cwd = rec.get("cwd") or cwd
            git_branch = rec.get("gitBranch") or git_branch
            version = rec.get("version") or version
            ts = _coerce_ts(rec.get("timestamp"))
            if started_at is None:
                started_at = ts
            ended_at = ts

            message = rec.get("message") or {}
            content_blocks = message.get("content") or []
            text, tools, thinking = _extract_text_and_tools(content_blocks)
            role = _classify_role(rec, tools)

            # Drop empty rows: no text, no tool calls. They're usually
            # control / metadata events that slipped through.
            if not text and not tools:
                continue

            our_uuid = uuid.uuid4()
            claude_uuid = rec.get("uuid")
            parent_claude_uuid = rec.get("parentUuid")
            parent_msg_id = (
                claude_to_ours.get(parent_claude_uuid)
                if parent_claude_uuid
                else None
            )

            metadata = {
                "claude_uuid": claude_uuid,
                "is_sidechain": rec.get("isSidechain"),
                "request_id": rec.get("requestId"),
                "prompt_id": rec.get("promptId"),
                "git_branch": git_branch,
                "cwd": cwd,
                "version": version,
                "model": message.get("model"),
            }
            if thinking:
                # Stash thinking in metadata. It's high-value for
                # debugging but should not pollute the embedding
                # corpus, so it's kept off the content column.
                metadata["thinking"] = thinking

            messages.append(
                MessageInput(
                    id=our_uuid,
                    seq=seq,
                    role=role,
                    content=text or "",
                    ts=ts,
                    author=message.get("model") if role == "assistant" else None,
                    tool_calls=tools,
                    parent_message_id=parent_msg_id,
                    metadata={k: v for k, v in metadata.items() if v is not None},
                )
            )
            # ``our_uuid`` is the prescribed row id (see MessageInput.id).
            # Tracking it here lets later rows in this file resolve
            # parent_message_id correctly, since FKs reference these
            # uuids at insert time.
            if claude_uuid:
                claude_to_ours[claude_uuid] = our_uuid
            seq += 1

    if not messages or external_id is None or started_at is None:
        return None

    return ParsedFile(
        path=path,
        external_id=external_id,
        started_at=started_at,
        ended_at=ended_at,
        cwd=cwd,
        git_branch=git_branch,
        version=version,
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Importer entry point
# ---------------------------------------------------------------------------


def discover_jsonl_files(root: Path) -> list[Path]:
    """Find all session JSONL files under ``root`` (the projects dir)."""
    if not root.exists():
        return []
    if root.is_file() and root.suffix == ".jsonl":
        return [root]
    return sorted(p for p in root.rglob("*.jsonl") if p.is_file())


async def import_claude_code(
    *,
    root: Path | None = None,
    project: str | None = None,
    limit: int | None = None,
    since: datetime | None = None,
    dry_run: bool = False,
    embed: bool = True,
) -> ImportSummary:
    """Import every session JSONL under ``root`` (defaults to
    ``~/.claude/projects``) into ``communications`` + messages.

    Returns an :class:`ImportSummary`. Idempotent — running twice
    over the same files writes nothing the second time.
    """
    root = root or DEFAULT_PROJECTS_DIR
    files = discover_jsonl_files(root)
    summary = ImportSummary(files_seen=len(files))

    if limit is not None:
        files = files[:limit]

    for path in files:
        try:
            parsed = parse_jsonl_file(path)
        except Exception as exc:  # noqa: BLE001 - capture-and-continue
            summary.errors.append({"path": str(path), "error": str(exc)})
            summary.files_skipped += 1
            continue
        if parsed is None:
            summary.files_skipped += 1
            continue
        if since is not None and parsed.ended_at and parsed.ended_at < since:
            summary.files_skipped += 1
            continue
        if dry_run:
            continue

        # Wire a session row keyed by the JSONL session uuid so
        # subsequent annotations / recalls bind to the same session.
        session_slug = f"claude-code-{parsed.external_id[:12]}"
        session_row = await get_session_by_slug(session_slug)
        if session_row is None:
            stored = await create_session(
                slug=session_slug,
                name=f"Claude Code session {parsed.external_id[:8]}",
                agent=CLAUDE_CODE_AGENT,
                scope_kind="project" if project else None,
                scope_value=project,
                started_at=parsed.started_at,
                metadata={
                    "external_id": parsed.external_id,
                    "cwd": parsed.cwd,
                    "git_branch": parsed.git_branch,
                    "claude_code_version": parsed.version,
                },
            )
            session_id = stored.id
            summary.sessions_created += 1
        else:
            session_id = session_row.id

        comm, created = await upsert_communication(
            agent=CLAUDE_CODE_AGENT,
            external_id=parsed.external_id,
            session_id=session_id,
            channel="cli",
            participants=[
                {"role": "user", "identity": "user:host"},
                {"role": "assistant", "identity": "agent:claude-code"},
            ],
            scope_kind="project" if project else None,
            scope_value=project,
            started_at=parsed.started_at,
            ended_at=parsed.ended_at,
            metadata={
                "source_file": str(path),
                "cwd": parsed.cwd,
                "git_branch": parsed.git_branch,
                "claude_code_version": parsed.version,
            },
        )
        if created:
            summary.communications_created += 1
        else:
            summary.communications_existing += 1

        written, embedded = await append_messages(
            comm.id, parsed.messages, embed=embed
        )
        summary.messages_written += written
        summary.messages_embedded += embedded

    return summary


__all__ = [
    "CLAUDE_CODE_AGENT",
    "DEFAULT_PROJECTS_DIR",
    "ImportSummary",
    "ParsedFile",
    "discover_jsonl_files",
    "parse_jsonl_file",
    "import_claude_code",
]
