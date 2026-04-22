"""hafiz parsers list — show registered parsers and their coverage.

Layer 2 observability for "is AST active for my .go files?" without
poking at config or restarting. In-tree parsers self-register; third-
party parsers plug in via the ``hafiz.parsers`` Python entry-point
group, and this command reveals both.
"""

from __future__ import annotations

import json
import sys

from rich.console import Console
from rich.table import Table

from hafiz.core.parsers import get_registry

console = Console()


def run_parsers_list(*, output_json: bool = False) -> None:
    """List every registered parser with its language coverage."""
    registry = get_registry()
    parsers = registry.all_parsers()

    rows = [
        {
            "name": p.name,
            "languages": registry.extensions_for(p),
            "module": type(p).__module__,
            "class": type(p).__name__,
        }
        for p in parsers
    ]

    if output_json:
        json.dump({"parsers": rows}, sys.stdout)
        sys.stdout.write("\n")
        return

    if not rows:
        console.print(
            "[yellow]No parsers registered — this is a bug. "
            "Re-install hafiz.[/yellow]"
        )
        return

    table = Table(title="Registered parsers", border_style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Languages / extensions", style="yellow")
    table.add_column("Module", style="dim")

    for r in rows:
        langs = ", ".join(r["languages"]) or "—"
        table.add_row(r["name"], langs, f"{r['module']}.{r['class']}")

    console.print()
    console.print(table)
    console.print()
    console.print(
        "[dim]Install third-party parsers via pip; they register "
        "themselves under the `hafiz.parsers` entry-point group "
        "and show up here automatically.[/dim]"
    )
