"""ChatGPT data-export importer.

Reads the ``conversations.json`` from an OpenAI data export (Settings →
Data controls → Export data). The importer accepts the export ``.zip``
directly, the unzipped directory, or the ``conversations.json`` file, so
a user does not have to know which of those they have.

The export's shape is unusual in one important way: a conversation is not
a list of turns but a **message tree**. Each conversation has a ``mapping``
of ``{node_id: {id, message, parent, children}}`` plus a
``current_node`` pointer. Regenerated answers and edited prompts create
branches, so the linear conversation a person actually had is the path
from ``current_node`` back to the root — not the mapping's iteration
order, which interleaves abandoned branches with kept ones.

Walking that path (rather than flattening the mapping) is the difference
between importing the conversation and importing every draft of it.

Each message node carries a stable ``id``, which becomes
``source_message_id`` — so re-importing a later export that includes the
same conversations is idempotent, and only genuinely new turns land.
"""

from __future__ import annotations

import json
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hafiz.core.communications import MessageInput
from hafiz.core.importers.base import (
    ImportSummary,
    ParsedConversation,
    store_conversation,
)

CHATGPT_AGENT = "chatgpt"

CONVERSATIONS_FILENAME = "conversations.json"

# Roles the export uses that carry no conversational content.
_SKIPPED_ROLES = {"system"}


def _coerce_ts(value: Any) -> datetime | None:
    """Export timestamps are float epoch-seconds; be tolerant anyway."""
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    try:
        parsed = datetime.fromisoformat(str(value).rstrip("Z"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _message_text(message: dict) -> str:
    """Flatten a node's ``content`` onto text.

    The export uses several content types: ``text`` (parts of strings),
    ``code`` (a ``text`` field), ``multimodal_text`` (parts that mix
    strings and image dicts), and ``execution_output``. Anything that
    isn't a string is skipped rather than stringified, so an image
    reference doesn't land in the embedding corpus as JSON noise.
    """
    content = message.get("content")
    if not isinstance(content, dict):
        return ""
    if isinstance(content.get("text"), str):
        return content["text"].strip()
    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""
    out: list[str] = []
    for part in parts:
        if isinstance(part, str) and part.strip():
            out.append(part)
    return "\n".join(out).strip()


def _linear_path(mapping: dict, current_node: str | None) -> list[dict]:
    """The conversation as it stood, newest-last.

    Walks parent pointers from ``current_node`` to the root, then
    reverses. Falls back to the mapping's own order only when there is no
    usable ``current_node`` — that fallback includes abandoned branches,
    which is wrong but better than importing nothing.
    """
    if current_node and current_node in mapping:
        chain: list[dict] = []
        seen: set[str] = set()
        node_id: str | None = current_node
        while node_id and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            node = mapping[node_id]
            chain.append(node)
            parent = node.get("parent")
            node_id = parent if isinstance(parent, str) else None
        chain.reverse()
        return chain
    return [n for n in mapping.values() if isinstance(n, dict)]


def parse_conversations(payload: Any) -> list[ParsedConversation]:
    """Parse a decoded ``conversations.json`` into conversations.

    The top level is a list of conversation objects; a single-conversation
    export (or a hand-trimmed file) may be one object, which is accepted.
    """
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []

    out: list[ParsedConversation] = []
    for convo in payload:
        if not isinstance(convo, dict):
            continue
        mapping = convo.get("mapping")
        if not isinstance(mapping, dict):
            continue

        external_id = convo.get("conversation_id") or convo.get("id")
        if not external_id:
            continue

        created = _coerce_ts(convo.get("create_time"))
        updated = _coerce_ts(convo.get("update_time"))

        messages: list[MessageInput] = []
        for node in _linear_path(mapping, convo.get("current_node")):
            message = node.get("message")
            if not isinstance(message, dict):
                continue  # root/anchor nodes carry no message
            author = message.get("author") or {}
            role = author.get("role") or "user"
            if role in _SKIPPED_ROLES:
                continue
            if role not in ("user", "assistant", "tool", "system"):
                # e.g. "browser" / plugin authors — model them as tool
                # turns so the schema's role CHECK holds.
                role = "tool"

            text = _message_text(message)
            if not text:
                continue

            node_id = message.get("id") or node.get("id")
            if not node_id:
                continue

            metadata = {"model": (message.get("metadata") or {}).get("model_slug")}
            messages.append(
                MessageInput(
                    id=uuid.uuid4(),
                    seq=len(messages),
                    role=role,
                    content=text,
                    ts=_coerce_ts(message.get("create_time")) or created or datetime.now(UTC),
                    author=(message.get("metadata") or {}).get("model_slug")
                    if role == "assistant"
                    else None,
                    source_message_id=str(node_id),
                    metadata={k: v for k, v in metadata.items() if v is not None},
                )
            )

        if not messages:
            continue

        out.append(
            ParsedConversation(
                external_id=str(external_id),
                title=convo.get("title"),
                started_at=created or messages[0].ts,
                ended_at=updated or messages[-1].ts,
                messages=messages,
                metadata={
                    "is_archived": convo.get("is_archived"),
                    "default_model_slug": convo.get("default_model_slug"),
                },
            )
        )
    return out


def load_export(path: Path) -> Any:
    """Read ``conversations.json`` from a zip, a directory, or the file.

    Accepting all three is deliberate: the user has whatever OpenAI
    emailed them, and making them find the right inner file is friction
    for no benefit.
    """
    path = path.expanduser()
    if path.is_dir():
        candidate = path / CONVERSATIONS_FILENAME
        if not candidate.is_file():
            matches = sorted(path.rglob(CONVERSATIONS_FILENAME))
            if not matches:
                raise FileNotFoundError(f"no {CONVERSATIONS_FILENAME} under {path}")
            candidate = matches[0]
        return json.loads(candidate.read_text(encoding="utf-8"))

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.rsplit("/", 1)[-1] == CONVERSATIONS_FILENAME]
            if not names:
                raise FileNotFoundError(f"no {CONVERSATIONS_FILENAME} inside {path}")
            with zf.open(names[0]) as fh:
                return json.loads(fh.read().decode("utf-8"))

    return json.loads(path.read_text(encoding="utf-8"))


async def import_chatgpt(
    *,
    root: Path,
    project: str | None = None,
    limit: int | None = None,
    since: datetime | None = None,
    dry_run: bool = False,
    embed: bool = True,
) -> ImportSummary:
    """Import a ChatGPT data export into ``communications`` + messages.

    ``root`` is required — unlike the agent harnesses there is no
    well-known location, because an export is a file the user downloads.
    """
    summary = ImportSummary()
    try:
        payload = load_export(Path(root))
    except (OSError, ValueError, zipfile.BadZipFile, FileNotFoundError) as exc:
        summary.errors.append({"path": str(root), "error": str(exc)})
        return summary

    conversations = parse_conversations(payload)
    summary.files_seen = len(conversations)

    if since is not None:
        kept = []
        for convo in conversations:
            if (convo.ended_at or convo.started_at) < since:
                summary.files_skipped += 1
            else:
                kept.append(convo)
        conversations = kept
    if limit is not None:
        conversations = conversations[:limit]

    previewed: dict[str, set[str]] = {}
    for parsed in conversations:
        try:
            await store_conversation(
                agent=CHATGPT_AGENT,
                parsed=parsed,
                summary=summary,
                project=project,
                embed=embed,
                dry_run=dry_run,
                previewed=previewed,
            )
        except Exception as exc:  # noqa: BLE001 - capture-and-continue
            summary.errors.append({"path": parsed.external_id, "error": str(exc)})
            summary.files_skipped += 1

    return summary


__all__ = [
    "CHATGPT_AGENT",
    "CONVERSATIONS_FILENAME",
    "import_chatgpt",
    "load_export",
    "parse_conversations",
]
