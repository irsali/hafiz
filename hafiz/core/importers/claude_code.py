"""Claude Code session JSONL importer.

Reads the per-session JSONL files Claude Code stores under
``~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl``, extracts
``user`` / ``assistant`` / ``tool`` turns, and stores them in the
``communications`` + ``communication_messages`` tables.

Idempotent by ``(agent='claude-code', external_id=<jsonl session uuid>)``
— re-importing the same file is a no-op at the communication level, and
``append_messages`` dedupes turns on Claude Code's own per-record
``uuid`` (stored as ``source_message_id``).

That identity, not position, is what makes this safe. One ``sessionId``
spans many files — resumed sessions and sidechains — and each file
restarts its positional ``seq`` at 0, so deduping on ``(communication_id,
seq)`` made every file after the first collide with the first and lose
its turns (29.3% of all turns on a real store). Identity also collapses
the reverse case: a resumed file replays earlier turns under their
original ids, which must not duplicate.

Selective embedding policy (from
:mod:`hafiz.core.communications`) is enforced at write time: short
turns and pure tool-result echoes don't get embedded. Non-message
records (queue-operation, attachment, file-history-snapshot) are
skipped.
"""

from __future__ import annotations

import json
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

CLAUDE_CODE_AGENT = "claude-code"
DEFAULT_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Records with these top-level "type" values are not turn-shaped and
# get dropped during parse. The remaining rows are user / assistant
# (and sometimes tool, modeled below by inspecting content blocks).
_NON_MESSAGE_TYPES = {"queue-operation", "attachment", "file-history-snapshot"}


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
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if not value:
        return datetime.now(UTC)
    s = str(value).rstrip("Z")
    try:
        parsed = datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(UTC)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


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
                    part.get("text", "") for part in raw_content if isinstance(part, dict)
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
            parent_msg_id = claude_to_ours.get(parent_claude_uuid) if parent_claude_uuid else None

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
                    # Claude Code's own per-record uuid is this turn's
                    # stable identity: unique per turn and preserved when a
                    # session resumes into a new file. Positional `seq` is
                    # not — it restarts at 0 in every file.
                    source_message_id=claude_uuid,
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


def peek_session_id(path: Path, *, max_lines: int = 8) -> str | None:
    """The session uuid a JSONL file belongs to, read from its head.

    Cheap counterpart to :func:`parse_jsonl_file` for callers that only
    need the identity — notably the capture-freshness probe on
    ``hafiz status``, which must answer "is this session in the store?"
    for a few hundred files without reading tens of thousands of turns.

    Do **not** substitute the filename stem for this. The stem matches the
    session uuid only for a session's *first* file; Claude Code reuses one
    ``sessionId`` across resumed/forked files with different names, and on
    a real store 124 of 200 files disagreed with their stem.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    return None
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and rec.get("sessionId"):
                    return str(rec["sessionId"])
    except OSError:
        return None
    return None


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
    resolve_project: bool = True,
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

    # Claude Code reuses one sessionId across several JSONL files (resumed
    # sessions, sidechains) — measured at 200 files over 77 session ids.
    # They all resolve to the *same* communication, so the run-level
    # accumulator below is what keeps a `--dry-run` from counting one new
    # communication per file, and from counting a replayed turn twice.
    previewed: dict[str, set[str]] = {}
    # Hoisted once: resolving per file would re-query every project's
    # indexed root for each of ~200 files.
    roots = None
    if project is None and resolve_project:
        from hafiz.core.store import indexed_root_per_project

        roots = await indexed_root_per_project()

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

        # Scope to the project the session actually ran in. `--from-hook`
        # has always done this; bulk import did not, which left every
        # backfilled session untagged and invisible to project-scoped
        # recall. An unindexed cwd yields None rather than a guess.
        scope = project
        if scope is None and roots is not None and parsed.cwd:
            from hafiz.core.store import project_for_path

            scope = await project_for_path(parsed.cwd, roots=roots)

        await store_conversation(
            agent=CLAUDE_CODE_AGENT,
            parsed=ParsedConversation(
                external_id=parsed.external_id,
                title=f"Claude Code session {parsed.external_id[:8]}",
                started_at=parsed.started_at,
                ended_at=parsed.ended_at,
                cwd=parsed.cwd,
                source_path=str(path),
                messages=parsed.messages,
                metadata={
                    "git_branch": parsed.git_branch,
                    "claude_code_version": parsed.version,
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
    "CLAUDE_CODE_AGENT",
    "DEFAULT_PROJECTS_DIR",
    "ImportSummary",
    "ParsedFile",
    "discover_jsonl_files",
    "parse_jsonl_file",
    "import_claude_code",
]


def pending_on_disk(known: set[str], settle_before: datetime) -> tuple[int, int]:
    """(uncaptured, still-being-written) session counts under the default root.

    Reads each file's head for its ``sessionId`` rather than parsing it —
    see :func:`peek_session_id` for why the filename stem will not do.
    """
    pending = 0
    active = 0
    if not DEFAULT_PROJECTS_DIR.exists():
        return 0, 0
    for path in DEFAULT_PROJECTS_DIR.rglob("*.jsonl"):
        session_id = peek_session_id(path)
        if session_id is None or session_id in known:
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
