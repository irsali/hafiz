"""``hafiz mcp`` — serve the knowledge surface over MCP on stdio.

Thin by design: everything real lives in :mod:`hafiz.core.mcp_server` and
:mod:`hafiz.core.mcp_registry`. This module only decides between "run the
server" and "describe the surface".

``--list`` exists because the alternative way to find out what an MCP server
exposes is to configure a client and look — which is a slow loop for a
surface that is generated. It prints to stdout, but only on the path that
never speaks the protocol.
"""

from __future__ import annotations

import asyncio
import json
import sys

import typer
from rich.console import Console
from rich.table import Table

from hafiz.core.mcp_registry import TOOLS, check_drift

console = Console()


def run_mcp(*, list_tools: bool = False, output_json: bool = False) -> None:
    if list_tools:
        _list(output_json=output_json)
        return

    from hafiz.core.mcp_server import MissingDependencyError, serve_stdio

    try:
        asyncio.run(serve_stdio())
    except MissingDependencyError as e:
        # To stderr: a client may already be reading stdout as JSON-RPC, and
        # an error message in that stream is worse than no message at all.
        console_err = Console(stderr=True)
        console_err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from e
    except KeyboardInterrupt:  # pragma: no cover — interactive only
        pass


def _list(*, output_json: bool) -> None:
    problems = check_drift()

    if output_json:
        print(
            json.dumps(
                {
                    "ok": not problems,
                    "transport": "stdio",
                    "count": len(TOOLS),
                    "drift": problems,
                    "tools": [
                        {
                            "name": spec.name,
                            "summary": spec.summary,
                            "writes": spec.writes,
                            "target": spec.target,
                            "input_schema": spec.input_schema(),
                        }
                        for spec in TOOLS
                    ],
                },
                indent=2,
            )
        )
        if problems:
            raise typer.Exit(code=1)
        return

    table = Table(title=f"MCP tools ({len(TOOLS)}) — transport: stdio")
    table.add_column("Tool", style="bold")
    table.add_column("W", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("Purpose")
    for spec in TOOLS:
        table.add_row(
            spec.name,
            "[yellow]w[/yellow]" if spec.writes else "",
            str(len(spec.input_schema()["properties"])),
            spec.summary.split(".")[0] + ".",
        )
    console.print(table)

    if problems:
        console.print("\n[red]Registry drift — the surface does not match the code:[/red]")
        for problem in problems:
            console.print(f"  [red]•[/red] {problem}")
        raise typer.Exit(code=1)

    console.print(
        "\n[dim]Add to a client's MCP config as: "
        f'{{"command": "{sys.argv[0].split("/")[-1] or "hafiz"}", "args": ["mcp"]}}[/dim]'
    )
