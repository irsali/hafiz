"""Shared machinery every source-layer importer uses.

An importer's only real job is *parsing* — turning one harness's storage
format into turns. Everything after that is identical across harnesses:
resolve a session row, upsert the communication, append messages
idempotently, and account for what happened. That common half lives here
so a new importer is a parser plus a few lines, not a copy of the whole
loop.

This is not only DRY. Turn-level idempotency depends on each importer
supplying ``MessageInput.source_message_id`` — the id the *source* gives
a turn — and an importer that quietly omits it silently reintroduces the
positional-dedup bug that cost 29.3% of turns (see migration ``0008``).
Routing every importer through one writer makes that a property of the
shared path rather than a discipline each importer has to remember, and
:func:`store_conversation` warns when a parser hands over turns with no
identity at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from hafiz.core.communications import (
    MessageInput,
    append_messages,
    communication_state,
    should_embed_message,
    upsert_communication,
)
from hafiz.core.sessions import create_session, get_session_by_slug


@dataclass
class ImportSummary:
    """High-level outcome of one ``hafiz import <agent>`` run."""

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
class ParsedConversation:
    """One conversation, parsed out of a harness's storage and ready to store.

    ``external_id`` is the harness's own conversation identifier and is
    what makes re-import idempotent at the communication level. Several
    source files may share one — Claude Code does this for resumed
    sessions — so importers must expect to hand over the same
    ``external_id`` more than once in a run.

    ``source_path`` is only provenance (recorded in metadata); it is never
    part of identity, because a harness may rename or rotate its files.
    """

    external_id: str
    started_at: datetime
    messages: list[MessageInput]
    title: str | None = None
    ended_at: datetime | None = None
    cwd: str | None = None
    source_path: str | None = None
    metadata: dict = field(default_factory=dict)


def _session_slug(agent: str, external_id: str) -> str:
    """The session slug a conversation maps to.

    One definition shared by the real store and the ``--dry-run``
    preview — a slug that disagreed between the two would make the
    preview's ``sessions_created`` count quietly wrong.
    """
    return f"{agent}-{external_id[:12]}"


async def store_conversation(
    *,
    agent: str,
    parsed: ParsedConversation,
    summary: ImportSummary,
    project: str | None = None,
    embed: bool = True,
    dry_run: bool = False,
    previewed: dict[str, set[str]] | None = None,
) -> None:
    """Store (or preview) one parsed conversation, updating ``summary``.

    ``previewed`` is the run's accumulator of turn identities already
    accounted for, keyed by ``external_id``. It must be passed for
    ``dry_run`` and is seeded from the database the first time a
    conversation is seen, so that:

    * several source files sharing one ``external_id`` count as one
      communication rather than one per file, and
    * a turn appearing in two files (a resumed session replays earlier
      turns) counts once — matching what the real write does.
    """
    if dry_run:
        if previewed is None:
            previewed = {}
        seen = previewed.get(parsed.external_id)
        if seen is None:
            exists, seen = await communication_state(agent=agent, external_id=parsed.external_id)
            previewed[parsed.external_id] = seen
            if exists:
                summary.communications_existing += 1
            else:
                summary.communications_created += 1
                if await get_session_by_slug(_session_slug(agent, parsed.external_id)) is None:
                    summary.sessions_created += 1

        pending_msgs = []
        for msg in parsed.messages:
            sid = msg.source_message_id
            if not sid or sid in seen:
                continue
            seen.add(sid)
            pending_msgs.append(msg)

        summary.messages_written += len(pending_msgs)
        if embed:
            summary.messages_embedded += sum(
                1
                for m in pending_msgs
                if should_embed_message(
                    role=m.role, content=m.content, marked_salient=m.marked_salient
                )
            )
        return

    slug = _session_slug(agent, parsed.external_id)
    session_row = await get_session_by_slug(slug)
    if session_row is None:
        stored = await create_session(
            slug=slug,
            name=parsed.title or f"{agent} session {parsed.external_id[:8]}",
            agent=agent,
            scope_kind="project" if project else None,
            scope_value=project,
            started_at=parsed.started_at,
            metadata={
                "external_id": parsed.external_id,
                "cwd": parsed.cwd,
                **parsed.metadata,
            },
        )
        session_id = stored.id
        summary.sessions_created += 1
    else:
        session_id = session_row.id

    comm, created = await upsert_communication(
        agent=agent,
        external_id=parsed.external_id,
        session_id=session_id,
        channel="cli",
        participants=[
            {"role": "user", "identity": "user:host"},
            {"role": "assistant", "identity": f"agent:{agent}"},
        ],
        scope_kind="project" if project else None,
        scope_value=project,
        started_at=parsed.started_at,
        ended_at=parsed.ended_at,
        metadata={
            "source_file": parsed.source_path,
            "cwd": parsed.cwd,
            "title": parsed.title,
            **parsed.metadata,
        },
    )
    if created:
        summary.communications_created += 1
    else:
        summary.communications_existing += 1

    written, embedded = await append_messages(comm.id, parsed.messages, embed=embed)
    summary.messages_written += written
    summary.messages_embedded += embedded

    # A parser that supplies no turn identity falls back to positional
    # dedup, which is wrong for any harness that can split one
    # conversation across files. Surface it rather than let a new importer
    # silently inherit the bug 0008 fixed.
    if parsed.messages and not any(m.source_message_id for m in parsed.messages):
        summary.errors.append(
            {
                "path": parsed.source_path or parsed.external_id,
                "error": (
                    "parser supplied no source_message_id; turns dedupe positionally, "
                    "which loses data if this harness splits a conversation across files"
                ),
            }
        )


__all__ = [
    "ImportSummary",
    "ParsedConversation",
    "store_conversation",
]
