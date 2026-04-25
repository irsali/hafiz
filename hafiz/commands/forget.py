"""hafiz forget — explicit redaction + retention sweep for source-layer rows.

Two modes:

* ``hafiz forget <comm-id-or-slug>`` — targeted redaction. Soft by
  default (sets ``valid_until = now``); ``--hard`` deletes the
  communication and cascades messages.
* ``hafiz forget --all-expired`` — sweep mode. Tombstones every
  communication whose ``retention_until`` has passed.

The default retention is ``started_at + 90 days`` from the
communications-and-sessions work item (configurable per-communication
at insert time). Sweep is intended to run from cron / a periodic job
later; for now the user invokes it explicitly.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from rich.console import Console
from rich.table import Table

from hafiz.core.communications import (
    forget_communication,
    tombstone_expired_communications,
)
from hafiz.core.database import close_engine
from hafiz.core.sessions import get_session_by_slug

console = Console()


async def _resolve_target_comm_ids(
    target: str,
) -> list[uuid.UUID]:
    """Accept a communication uuid, a session uuid (returns all
    communications in that session), or a session slug. Returns the
    list of communication ids to operate on."""
    raw = (target or "").strip()
    if not raw:
        return []
    try:
        as_uuid = uuid.UUID(raw)
        # First try the communications table directly. If the uuid is
        # actually a session uuid, fall through to the session path.
        from hafiz.core.database import (
            Communication,
            get_session_factory,
        )

        factory = get_session_factory()
        async with factory() as s:
            row = await s.get(Communication, as_uuid)
            if row is not None:
                return [as_uuid]

            from hafiz.core.database import Session as SessionRow
            from sqlalchemy import select

            sess = await s.get(SessionRow, as_uuid)
            if sess is not None:
                rows = (
                    await s.execute(
                        select(Communication.id).where(
                            Communication.session_id == sess.id
                        )
                    )
                ).all()
                return [r[0] for r in rows]
        return []
    except ValueError:
        pass

    # Slug path.
    sess = await get_session_by_slug(raw)
    if sess is None:
        return []
    from hafiz.core.database import Communication, get_session_factory
    from sqlalchemy import select

    factory = get_session_factory()
    async with factory() as s:
        rows = (
            await s.execute(
                select(Communication.id).where(
                    Communication.session_id == sess.id
                )
            )
        ).all()
    return [r[0] for r in rows]


def run_forget_target(
    target: str,
    *,
    hard: bool = False,
    output_json: bool = False,
) -> None:
    async def _do() -> dict:
        try:
            comm_ids = await _resolve_target_comm_ids(target)
            results: list[dict] = []
            for cid in comm_ids:
                results.append(await forget_communication(cid, hard=hard))
            return {
                "action": "forget",
                "target": target,
                "hard": hard,
                "communications_affected": len(results),
                "results": results,
            }
        finally:
            await close_engine()

    summary = asyncio.run(_do())

    if output_json:
        console.print_json(json.dumps(summary))
        return

    if summary["communications_affected"] == 0:
        console.print(
            f"[yellow]No communications matched target {target!r}.[/yellow]"
        )
        return
    mode = "deleted (hard)" if hard else "tombstoned (soft)"
    console.print(
        f"[bold green]Forget complete:[/bold green] {summary['communications_affected']} "
        f"communication(s) {mode}."
    )


def run_forget_sweep(
    *,
    dry_run: bool = False,
    output_json: bool = False,
) -> None:
    """Tombstone every communication past its ``retention_until``."""

    async def _do() -> dict:
        try:
            return await tombstone_expired_communications(dry_run=dry_run)
        finally:
            await close_engine()

    result = asyncio.run(_do())
    payload = {"action": "forget_sweep", **result}

    if output_json:
        console.print_json(json.dumps(payload))
        return

    table = Table(title="Retention sweep", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("Past retention", str(result["matched"]))
    table.add_row(
        "Tombstoned", str(result["tombstoned"]) + (" (dry run)" if dry_run else "")
    )
    console.print(table)
