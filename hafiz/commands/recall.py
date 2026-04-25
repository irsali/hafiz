"""hafiz recall — opt-in source-layer access.

Default ``hafiz query`` and ``hafiz context`` keep returning curated
knowledge-layer rows (code units, annotations). ``hafiz recall``
surfaces the source layer (communications, messages) deliberately —
ordered turns from a single session or communication, with role /
seq / time filters.

Inputs accepted (positional):

* a communication uuid (returns that communication's messages),
* a session uuid (returns every message from every communication in
  that session, ordered),
* a session slug (resolved via :func:`hafiz.core.sessions.get_session_by_slug`).

The flag ``--query "<text>"`` switches to vector search across the
session's messages, returning the best-matching turns (still
opt-in source-layer access).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime

from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from hafiz.core.communications import (
    MessageRow,
    list_messages,
    search_messages,
)
from hafiz.core.database import (
    Communication,
    close_engine,
    get_session_factory,
)
from hafiz.core.sessions import get_session_by_id, get_session_by_slug

console = Console()


async def _resolve_target(
    target: str,
) -> tuple[uuid.UUID | None, list[uuid.UUID]]:
    """Return ``(session_id_or_None, communication_ids)``.

    The target can be a uuid (session or communication) or a session
    slug. ``communication_ids`` is the ordered list of comms to read
    from; empty when no match.
    """
    raw = (target or "").strip()
    if not raw:
        return None, []
    factory = get_session_factory()
    try:
        as_uuid = uuid.UUID(raw)
    except ValueError:
        as_uuid = None

    if as_uuid is not None:
        # Could be a session id…
        sess = await get_session_by_id(as_uuid)
        if sess is not None:
            async with factory() as s:
                comms = (
                    await s.execute(
                        select(Communication.id)
                        .where(Communication.session_id == sess.id)
                        .order_by(Communication.started_at.asc())
                    )
                ).all()
            return sess.id, [c[0] for c in comms]
        # …or a communication id directly.
        async with factory() as s:
            row = await s.get(Communication, as_uuid)
            if row is not None:
                return row.session_id, [row.id]
        return None, []

    # String — look up as a session slug.
    sess = await get_session_by_slug(raw)
    if sess is None:
        return None, []
    async with factory() as s:
        comms = (
            await s.execute(
                select(Communication.id)
                .where(Communication.session_id == sess.id)
                .order_by(Communication.started_at.asc())
            )
        ).all()
    return sess.id, [c[0] for c in comms]


def _row_to_dict(r: MessageRow, score: float | None = None) -> dict:
    out = {
        "id": r.id,
        "communication_id": r.communication_id,
        "seq": r.seq,
        "role": r.role,
        "author": r.author,
        "content": r.content,
        "ts": r.ts.isoformat(),
        "tool_calls": r.tool_calls,
        "marked_salient": r.marked_salient,
        "metadata": r.metadata,
    }
    if score is not None:
        out["score"] = score
    return out


def run_recall(
    target: str,
    *,
    role: str | None = None,
    has_tool_call: bool | None = None,
    seq_from: int | None = None,
    seq_to: int | None = None,
    limit: int = 1000,
    query_text: str | None = None,
    output_json: bool = False,
) -> None:
    async def _do() -> tuple[uuid.UUID | None, list[uuid.UUID], list[tuple[MessageRow, float | None]]]:
        try:
            session_id, comm_ids = await _resolve_target(target)
            if not comm_ids:
                return session_id, [], []
            if query_text:
                # Vector search across the session's messages.
                results: list[tuple[MessageRow, float | None]] = []
                if session_id is not None:
                    rows = await search_messages(
                        query_text, limit=limit, session_id=session_id
                    )
                    results.extend([(r, s) for r, s in rows])
                else:
                    # Single communication path.
                    for cid in comm_ids:
                        rows = await search_messages(
                            query_text,
                            limit=limit,
                            communication_id=cid,
                        )
                        results.extend([(r, s) for r, s in rows])
                results.sort(key=lambda pair: pair[1] or 0.0, reverse=True)
                return session_id, comm_ids, results[:limit]
            # Linear walk through every comm in seq order.
            ordered: list[tuple[MessageRow, float | None]] = []
            remaining = limit
            for cid in comm_ids:
                if remaining <= 0:
                    break
                rows = await list_messages(
                    cid,
                    role=role,
                    has_tool_call=has_tool_call,
                    seq_from=seq_from,
                    seq_to=seq_to,
                    limit=remaining,
                )
                ordered.extend([(r, None) for r in rows])
                remaining -= len(rows)
            return session_id, comm_ids, ordered
        finally:
            await close_engine()

    session_id, comm_ids, rows = asyncio.run(_do())

    if not comm_ids:
        if output_json:
            console.print_json(
                json.dumps(
                    {
                        "target": target,
                        "session_id": str(session_id) if session_id else None,
                        "communications": [],
                        "messages": [],
                    }
                )
            )
        else:
            console.print(
                f"[yellow]No communications found for target {target!r}.[/yellow]"
            )
        return

    if output_json:
        payload = {
            "target": target,
            "session_id": str(session_id) if session_id else None,
            "communications": [str(c) for c in comm_ids],
            "messages": [_row_to_dict(r, s) for r, s in rows],
            "query": query_text,
        }
        console.print_json(json.dumps(payload))
        return

    # Rich table — keep the agent contract identical, but humans get a
    # readable view.
    title_bits = [f"target={target}"]
    if session_id is not None:
        title_bits.append(f"session={session_id}")
    title_bits.append(f"comms={len(comm_ids)}")
    if query_text:
        title_bits.append(f'query="{query_text}"')
    table = Table(title=" · ".join(title_bits), border_style="cyan")
    if query_text:
        table.add_column("Score", justify="right", width=8)
    table.add_column("Seq", justify="right", width=5)
    table.add_column("Role", width=10)
    table.add_column("Author", style="dim", width=18)
    table.add_column("When", style="dim", width=16)
    table.add_column("Content", ratio=4)

    for r, score in rows:
        when = r.ts.strftime("%Y-%m-%d %H:%M")
        content = r.content[:160]
        if len(r.content) > 160:
            content += "…"
        if query_text:
            table.add_row(
                f"{score or 0:.2%}",
                str(r.seq),
                r.role,
                r.author or "—",
                when,
                content,
            )
        else:
            table.add_row(str(r.seq), r.role, r.author or "—", when, content)
    console.print(table)
    console.print(
        f"[dim]{len(rows)} message(s) — source layer is opt-in. "
        f"Use --json for stable agent output.[/dim]"
    )
