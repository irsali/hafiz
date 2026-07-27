"""hafiz retrievals — what the store was asked for, and what it couldn't answer.

The read side of :mod:`hafiz.core.telemetry`. Three numbers matter here, and the
third is the one that changes behaviour:

* **never recalled** — knowledge that has never surfaced once. Candidates for
  retirement, or a sign that recall isn't reaching them.
* **most recalled** — what's earning its keep.
* **unanswered** — queries that returned nothing. This is the gap between what
  agents ask for and what the store holds, so it's a list of things to write
  down. Nothing else in hafiz produces it.
"""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.table import Table

console = Console()


def run_retrievals(
    *,
    since_days: int = 30,
    limit: int = 20,
    output_json: bool = False,
) -> None:
    """Report on recorded retrievals."""
    from hafiz.core.config import get_settings
    from hafiz.core.database import close_engine
    from hafiz.core.telemetry import retrieval_report

    async def _go() -> dict:
        try:
            return await retrieval_report(since_days=since_days, limit=limit)
        finally:
            await close_engine()

    enabled = get_settings().telemetry.retrieval
    try:
        report = asyncio.run(_go())
    except Exception as e:  # noqa: BLE001
        if output_json:
            console.print_json(json.dumps({"ok": False, "error": str(e)}))
        else:
            console.print(f"[red]Could not read retrievals:[/red] {e}")
        raise SystemExit(1) from e

    report["enabled"] = enabled

    if output_json:
        console.print_json(json.dumps({"ok": True, **report}))
        return

    if not enabled:
        console.print(
            "[yellow]Retrieval telemetry is off[/yellow] "
            "([dim]set [telemetry] retrieval = true to record[/dim])"
        )
    if not report["retrievals"]:
        console.print(
            f"[dim]No retrievals recorded in the last {since_days} days.[/dim]\n"
            f"[dim]Nothing to report until searches run.[/dim]"
        )
        return

    table = Table(title=f"Retrievals (last {since_days}d)", show_header=False, border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Searches recorded", str(report["retrievals"]))
    rate = report["empty_result_rate"]
    table.add_row(
        "Returned nothing",
        f"[yellow]{rate:.0%}[/yellow]" if rate else "0%",
    )
    table.add_row("Never recalled", str(report["never_recalled"]))
    if report["blind_before"]:
        # Without this the "never recalled" count reads as an indictment of rows
        # that simply predate the telemetry.
        table.add_row(
            "  [dim]of which predate telemetry[/dim]",
            f"[dim]{report['blind_before']}[/dim]",
        )
    console.print()
    console.print(table)

    if report["unanswered"]:
        console.print()
        gaps = Table(
            title="Asked for, not found — candidates to write down",
            border_style="yellow",
        )
        gaps.add_column("Query")
        gaps.add_column("Times", justify="right")
        for row in report["unanswered"]:
            gaps.add_row(row["query"][:80], str(row["times"]))
        console.print(gaps)

    if report["most_recalled"]:
        console.print()
        top = Table(title="Most recalled", border_style="green")
        top.add_column("Kind", style="yellow", width=10)
        top.add_column("Content")
        top.add_column("Hits", justify="right")
        for row in report["most_recalled"]:
            top.add_row(row["kind"], row["preview"].replace("\n", " ")[:90], str(row["hits"]))
        console.print(top)
