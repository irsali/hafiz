"""hafiz export — sovereignty dump (presentation layer).

Thin wrapper over :func:`hafiz.core.export.export_brain`. Handles
``--json`` vs. rich output and exit codes; all logic lives in core.

Note the deliberate naming distinction from ``hafiz extract export``,
which dumps AST units as an agent-extraction payload. This command is a
whole-brain, human-facing dump and is a peer to ``hafiz forget``.
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

from hafiz.core.database import close_engine
from hafiz.core.export import VALID_FORMATS, export_brain

console = Console()


def run_export(
    *,
    out_dir: str,
    fmt: str = "md",
    project: str | None = None,
    include_transcripts: bool = False,
    output_json: bool = False,
) -> None:
    """Dump the wisdom layer to ``out_dir`` and report the result."""

    async def _do() -> dict:
        try:
            return await export_brain(
                out_dir=out_dir,
                fmt=fmt,
                project=project,
                include_transcripts=include_transcripts,
            )
        finally:
            await close_engine()

    summary = asyncio.run(_do())

    if output_json:
        console.print_json(json.dumps(summary))
        if not summary.get("ok"):
            raise typer.Exit(1)
        return

    if not summary.get("ok"):
        console.print(f"[red]Error:[/red] {summary['error']}")
        console.print(f"[dim]Valid formats: {', '.join(VALID_FORMATS)}[/dim]")
        raise typer.Exit(1)

    counts = summary["counts"]
    table = Table(title=f"Exported ({summary['format']})", border_style="cyan")
    table.add_column("Layer", style="bold")
    table.add_column("Rows", justify="right")
    table.add_row("observations", str(counts["annotations"]))
    if include_transcripts:
        table.add_row("transcripts", str(counts["communications"]))
        table.add_row("  └ turns", str(counts["communication_messages"]))
    console.print(table)
    console.print(f"[bold green]→[/bold green] {summary['path']}")
    console.print(f"[yellow]⚠[/yellow]  {summary['warning']}")
