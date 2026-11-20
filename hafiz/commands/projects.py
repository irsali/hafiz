"""``hafiz projects`` — see every project name, declare synonyms, retire, merge.

Presentation only; the rules live in :mod:`hafiz.core.projects`.
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def _run(coro_fn, *args, **kwargs):
    async def go():
        from hafiz.core.database import close_engine

        try:
            return await coro_fn(*args, **kwargs)
        finally:
            await close_engine()

    return asyncio.run(go())


def _fail(message: str, output_json: bool) -> None:
    if output_json:
        typer.echo(json.dumps({"ok": False, "error": message}))
    else:
        typer.echo(f"Error: {message}")
    raise typer.Exit(2)


def run_list(*, output_json: bool) -> None:
    from hafiz.core.projects import list_projects

    report = _run(list_projects)
    if output_json:
        typer.echo(json.dumps(report))
        return
    table = Table(title="Projects", show_lines=False)
    table.add_column("Name")
    table.add_column("Rows", justify="right")
    table.add_column("Also known as")
    table.add_column("State")
    for p in report["projects"]:
        aka = ", ".join(p["aliases"])
        if p["unmerged_alias_rows"]:
            aka += f"  [yellow]({p['unmerged_alias_rows']} rows to merge)[/yellow]"
        table.add_row(p["name"], str(p["rows"]), aka, "retired" if p["retired"] else "")
    console.print(table)
    if report["look_alikes"]:
        console.print(
            "\n[bold]Look alike but not declared the same[/bold] — if one is a synonym, run "
            "[bold]hafiz projects alias <name> <main-name>[/bold]:"
        )
        for group in report["look_alikes"]:
            console.print("  · " + "  |  ".join(group))


def run_alias(alias: str, project: str, *, output_json: bool) -> None:
    from hafiz.core.projects import add_alias

    try:
        result = _run(add_alias, alias, project)
    except ValueError as e:
        _fail(str(e), output_json)
    if output_json:
        typer.echo(json.dumps({"ok": True, **result}))
    else:
        console.print(f"[green]✓[/green] {result['alias']!r} now means {result['project']!r}")


def run_unalias(alias: str, *, output_json: bool) -> None:
    from hafiz.core.projects import remove_alias

    result = _run(remove_alias, alias)
    if output_json:
        typer.echo(json.dumps({"ok": True, **result}))
    elif result["removed"]:
        console.print(f"[green]✓[/green] {alias!r} is no longer a synonym")
    else:
        console.print(f"[yellow]{alias!r} was not a synonym[/yellow]")


def run_retire(project: str, *, undo: bool, output_json: bool) -> None:
    from hafiz.core.projects import set_retired

    try:
        result = _run(set_retired, project, not undo)
    except ValueError as e:
        _fail(str(e), output_json)
    if output_json:
        typer.echo(json.dumps({"ok": True, **result}))
    else:
        state = "retired" if result["retired"] else "active again"
        console.print(f"[green]✓[/green] {result['project']!r} is {state}")


def run_merge(*, apply: bool, output_json: bool) -> None:
    from hafiz.core.projects import merge_aliases

    result = _run(merge_aliases, apply=apply)
    if output_json:
        typer.echo(json.dumps({"ok": True, **result}))
        return
    if not result["moves"] and not result["skipped"]:
        console.print("Nothing to merge — every synonym's rows already sit under its main name.")
        return
    verb = "Moved" if apply else "Would move"
    for m in result["moves"]:
        rows = ", ".join(f"{n} {table}" for table, n in m["rows"].items())
        console.print(f"{verb} {rows}: {m['alias']!r} → {m['project']!r}")
    for s in result["skipped"]:
        console.print(f"[yellow]Skipped[/yellow] {s['alias']!r} → {s['project']!r}: {s['reason']}")
    if not apply and result["moves"]:
        console.print("\n[dim]Preview only. Run with --apply to move them.[/dim]")
