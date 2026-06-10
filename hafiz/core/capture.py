"""Transcript capture — ingest multi-page text into the **source layer**.

A capture becomes one ``communications`` row with its turns stored as
``communication_messages``, reusing the selective-embed policy in
:mod:`hafiz.core.communications`. This keeps transcripts in the
firehose layer: hidden from default ``hafiz query`` / ``hafiz context``,
retention-bounded, and surfaced only via ``hafiz recall`` or the
``--include-transcripts`` opt-in — exactly like ``import claude-code``.

No file is written to disk; the synthetic title/slug is for display only.
"""

from __future__ import annotations

import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from hafiz.core.communications import (
    MessageInput,
    append_messages,
    upsert_communication,
)
from hafiz.core.sessions import resolve_session_uuid

_TURN_SPLITTER = re.compile(r"\n\s*\n+")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


@dataclass
class TranscriptStored:
    """Summary returned by :func:`store_transcript`."""

    communication_id: str
    title: str | None
    turn_count: int
    messages_embedded: int


def _agent_from_source(source: str | None) -> str:
    """Derive the communication ``agent`` from a ``--source`` value.

    ``agent:hermes`` → ``hermes``; ``user:anjum`` stays ``user:anjum``;
    a bare value passes through; missing/empty → ``capture``. This lets
    ``hafiz recall --agent hermes`` filter to a tool's own captures.
    """
    if not source:
        return "capture"
    if source.startswith("agent:"):
        return source.split(":", 1)[1] or "capture"
    return source


def split_transcript(text: str) -> list[str]:
    """Split raw transcript text into turn-sized chunks.

    Splits on blank lines (one or more newlines with only whitespace),
    which handles paragraph-based notes, speaker-delimited dialogues
    ("Q: ...\\n\\nA: ..."), and chat logs equally well. Empty turns
    are skipped.
    """
    turns = [t.strip() for t in _TURN_SPLITTER.split(text or "")]
    return [t for t in turns if t]


def _slugify(title: str | None) -> str:
    """Produce a URL-safe short slug from a title, with random suffix."""
    suffix = secrets.token_hex(3)
    if not title:
        return suffix
    base = _SLUG_STRIP.sub("-", title.strip().lower()).strip("-")
    base = base[:40] if base else "capture"
    return f"{base}-{suffix}"


async def store_transcript(
    text: str,
    *,
    title: str | None = None,
    project: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    session_id: str | None = None,
    task: str | None = None,
) -> TranscriptStored:
    """Split, embed, and store a transcript in the source layer.

    The transcript becomes one ``communications`` row (agent derived from
    ``source``; ``external_id`` a fresh uuid so re-running is never a false
    no-op) with its turns appended as ``communication_messages``. The
    selective-embed policy from :mod:`hafiz.core.communications` applies —
    short turns and pure tool-result echoes are stored but not embedded.
    """
    turns = split_transcript(text)
    if not turns:
        raise ValueError("Transcript is empty after splitting — nothing to store.")

    total = len(turns)
    now = datetime.now(timezone.utc)
    resolved_session_uuid = await resolve_session_uuid(session_id)

    metadata = {
        "title": title,
        "task": task,
        "session_slug": session_id,
        "tags": tags or [],
        "kind": "capture",
    }

    comm, _created = await upsert_communication(
        agent=_agent_from_source(source),
        external_id=str(uuid.uuid4()),
        session_id=resolved_session_uuid,
        scope_kind="project" if project else None,
        scope_value=project,
        started_at=now,
        ended_at=now,
        metadata=metadata,
    )

    messages = [
        MessageInput(
            seq=idx,
            role="user",
            content=turn,
            ts=now,
            author=source,
            metadata={"turn_index": idx, "total_turns": total},
        )
        for idx, turn in enumerate(turns)
    ]

    _written, embedded = await append_messages(comm.id, messages, embed=True)

    return TranscriptStored(
        communication_id=str(comm.id),
        title=title,
        turn_count=total,
        messages_embedded=embedded,
    )
