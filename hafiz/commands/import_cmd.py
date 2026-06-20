"""hafiz import — agent-transcript importers.

Currently only ``hafiz import claude-code`` is implemented; the
command group is shaped so additional harnesses (Cursor, Copilot)
can slot in additively.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hafiz.core.database import close_engine
from hafiz.core.durations import parse_duration
from hafiz.core.importers.claude_code import (
    DEFAULT_PROJECTS_DIR,
    import_claude_code,
)

console = Console()


def _resolve_since(since: str | None) -> datetime | None:
    if not since:
        return None
    try:
        delta = parse_duration(since)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    return datetime.now(UTC) - delta


def run_import_claude_code(
    path: str | None = None,
    *,
    project: str | None = None,
    limit: int | None = None,
    since: str | None = None,
    dry_run: bool = False,
    no_embed: bool = False,
    output_json: bool = False,
) -> None:
    root = Path(path).expanduser().resolve() if path else DEFAULT_PROJECTS_DIR
    cutoff = _resolve_since(since)

    async def _run():
        try:
            return await import_claude_code(
                root=root,
                project=project,
                limit=limit,
                since=cutoff,
                dry_run=dry_run,
                embed=not no_embed,
            )
        finally:
            await close_engine()

    summary = asyncio.run(_run())

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "action": "import_claude_code",
                    "root": str(root),
                    "project": project,
                    "since": cutoff.isoformat() if cutoff else None,
                    "dry_run": dry_run,
                    "embed": not no_embed,
                    "summary": summary.to_dict(),
                }
            )
        )
        return

    table = Table(title=f"Claude Code import — {root}", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")
    s = summary
    table.add_row("Files seen", str(s.files_seen))
    table.add_row("Files skipped", str(s.files_skipped))
    table.add_row("Communications created", str(s.communications_created))
    table.add_row("Communications already present", str(s.communications_existing))
    table.add_row("Sessions created", str(s.sessions_created))
    table.add_row("Messages written", str(s.messages_written))
    table.add_row("Messages embedded", str(s.messages_embedded))
    if s.errors:
        table.add_row("Errors", str(len(s.errors)))
    console.print(table)

    if dry_run:
        console.print("[yellow]Dry run — no rows written.[/yellow]")
    if s.errors:
        console.print()
        err_panel = Panel(
            "\n".join(f"[red]{e['path']}[/red]: {e['error']}" for e in s.errors[:10]),
            title="Errors (first 10)",
            border_style="red",
        )
        console.print(err_panel)
