"""hafiz prune — clean up index rows nothing else can reach.

Ordinary stale-file cleanup is automatic: every ``hafiz ingest`` runs
:func:`hafiz.core.store.tombstone_vanished_files`, which tombstones any file
under the project whose path wasn't seen in the pass. So a bare ``hafiz prune``
stays a reporting no-op, and keeps working for installed hooks that call it.

``--untagged`` covers the one case that cleanup can never reach. An ingest with
no ``--project`` writes a *parallel untagged copy* of the tree rather than
updating the project's rows, because ``files`` is unique on ``(project, path)``.
Those rows are immortal: ingest skips vanished-file tombstoning outright when
``project is None``, and the paths are still on disk anyway, so no walk would
call them vanished. On the deployment that surfaced this, 1,958 untagged rows
had accumulated and search was returning them alongside the real ones.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console

console = Console()

_MESSAGE = (
    "Stale-file cleanup is automatic on ingest (tombstone_vanished_files). "
    "There is nothing to prune separately — re-run `hafiz ingest` to reconcile."
)


def _report_untagged(stats: dict) -> None:
    if not stats["untagged"]:
        console.print("[green]No untagged files.[/green] Every file row carries a project.")
        return

    verb = "Would tombstone" if stats["dry_run"] else "Tombstoned"
    console.print(
        f"\n  [bold]{stats['untagged']}[/bold] untagged file row(s):\n"
        f"    {stats['duplicated']} also indexed under a project [dim](redundant)[/dim]\n"
        f"    {stats['unindexed']} indexed nowhere else [dim](the only copy)[/dim]\n"
    )
    if stats["dry_run"]:
        target = stats["duplicated"] if not stats["_include"] else stats["untagged"]
        console.print(f"  [yellow]{verb} {target} file row(s).[/yellow]")
    else:
        console.print(
            f"  [green]{verb} {stats['files_tombstoned']} file row(s) "
            f"and {stats['units_tombstoned']} unit(s).[/green]"
        )

    if stats["unindexed"] and not stats["_include"]:
        console.print(
            f"  [dim]{stats['unindexed']} row(s) left alone — no project covers those paths,\n"
            f"  so dropping them would lose index coverage. Re-ingest the repos with\n"
            f"  --project first, or pass --include-unindexed to drop them anyway.[/dim]"
        )
    console.print(
        "  [dim]Soft tombstone: rows stay for audit and drop out of search.\n"
        "  Stop new ones appearing with: hafiz hooks install <repo> --project <name>[/dim]"
    )


def run_prune(
    project: str | None = None,
    dry_run: bool = False,
    output_json: bool = False,
    *,
    untagged: bool = False,
    include_unindexed: bool = False,
    under: str | None = None,
) -> None:
    """Report automatic cleanup, or tombstone the untagged shadow index."""
    if not untagged:
        if output_json:
            # print_json, not print: Rich wraps a plain string at the terminal
            # width, which inserts newlines mid-token and makes the payload
            # unparseable in a narrow terminal.
            console.print_json(
                json.dumps(
                    {
                        "action": "prune",
                        "noop": True,
                        "reason": "handled-on-ingest",
                        "project": project,
                        "dry_run": dry_run,
                        "message": _MESSAGE,
                    }
                )
            )
            return
        console.print(f"[green]Nothing to prune.[/green] {_MESSAGE}")
        return

    from hafiz.core.database import close_engine
    from hafiz.core.store import tombstone_untagged_files

    prefix = str(Path(under).resolve()) if under else None

    async def _go() -> dict:
        try:
            return await tombstone_untagged_files(
                include_unindexed=include_unindexed, dry_run=dry_run, path_prefix=prefix
            )
        finally:
            await close_engine()

    try:
        stats = asyncio.run(_go())
    except Exception as e:  # noqa: BLE001 — surface as data, not a traceback
        if output_json:
            console.print_json(json.dumps({"ok": False, "error": str(e)}))
        else:
            console.print(f"[red]Prune failed:[/red] {e}")
        raise SystemExit(1) from e

    if output_json:
        console.print_json(
            json.dumps({"ok": True, "action": "prune-untagged", "under": prefix, **stats})
        )
        return

    stats["_include"] = include_unindexed
    _report_untagged(stats)
