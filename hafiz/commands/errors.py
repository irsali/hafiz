"""`hafiz errors` command handlers — list / show / clear the error log.

Presentation layer: builds on ``hafiz.core.error_log``. Keeps the
JSON shape stable (agents parse this).

Human output is terse by design — errors are debug data, not a primary
surface. Use ``hafiz errors show <id>`` when you want the full
traceback.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from hafiz.core import error_log

console = Console()


# ── list ──────────────────────────────────────────────────────────────


# Fields you can group by. Currently just exception_type — the agent
# pattern-recognition use case ("which classes have I been hitting?")
# is the only one we've validated. New options should land here only
# with a real consumer.
SUPPORTED_GROUP_BY = ("exception_type",)


def run_errors_list(
    *,
    since: str | None = None,
    limit: int = 20,
    group_by: str | None = None,
    output_json: bool = False,
) -> None:
    if group_by is not None and group_by not in SUPPORTED_GROUP_BY:
        msg = f"Unknown --group-by value {group_by!r}. Supported: {', '.join(SUPPORTED_GROUP_BY)}."
        if output_json:
            console.print_json(json.dumps({"ok": False, "error": "bad_group_by", "message": msg}))
        else:
            console.print(f"[red]{msg}[/red]")
        raise typer.Exit(2)

    if group_by is not None:
        # In grouped mode, we want to consider the full matching set
        # (counts are meaningless if records are silently truncated).
        # MAX_ENTRIES caps the log itself, so this is bounded.
        records = error_log.tail(since=since, limit=None)
    else:
        records = error_log.tail(since=since, limit=limit)

    if group_by == "exception_type":
        _render_grouped(records, since=since, output_json=output_json)
        return

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "count": len(records),
                    "since": since,
                    "errors": [r.as_jsonable() for r in records],
                }
            )
        )
        return

    if not records:
        if since:
            console.print(f"[dim]No errors recorded since [bold]{since}[/bold].[/dim]")
        else:
            console.print("[dim]No errors recorded.[/dim]")
        return

    console.print()
    tbl = Table(
        title=(f"Errors (last {len(records)}{' since ' + since if since else ''})"),
        border_style="red",
    )
    tbl.add_column("ID", style="dim", no_wrap=True)
    tbl.add_column("When", style="dim", no_wrap=True)
    tbl.add_column("Command", style="bold")
    tbl.add_column("Exception")
    tbl.add_column("Message", overflow="fold")

    for r in records:
        tbl.add_row(
            r.id[:8],
            r.timestamp.replace("T", " ").rstrip("+00:00"),
            r.command or "—",
            r.exception_type,
            (r.message[:120] + "…") if len(r.message) > 120 else r.message,
        )
    console.print(tbl)
    console.print("[dim]`hafiz errors show <id>` for full traceback + suggestion.[/dim]\n")


def _render_grouped(
    records: list,
    *,
    since: str | None,
    output_json: bool,
) -> None:
    groups = error_log.group_by_exception_type(records)
    total = len(records)
    with_suggestions = sum(1 for r in records if r.suggested_action)
    most_recent = None
    if records:
        head = records[0]
        most_recent = {
            "id": head.id,
            "exception_type": head.exception_type,
            "command": head.command,
            "timestamp": head.timestamp,
        }

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "since": since,
                    "grouped_by": "exception_type",
                    "total": total,
                    "with_suggestions": with_suggestions,
                    "most_recent": most_recent,
                    "groups": groups,
                }
            )
        )
        return

    if not records:
        if since:
            console.print(f"[dim]No errors recorded since [bold]{since}[/bold].[/dim]")
        else:
            console.print("[dim]No errors recorded.[/dim]")
        return

    console.print()
    title = f"Errors grouped by exception_type (total {total}{', since ' + since if since else ''})"
    tbl = Table(title=title, border_style="red")
    tbl.add_column("Exception", style="bold")
    tbl.add_column("Count", justify="right")
    tbl.add_column("With suggestion", justify="right", style="dim")
    tbl.add_column("Most recent", style="dim", no_wrap=True)
    tbl.add_column("Sample command", style="bold")
    tbl.add_column("Sample message", overflow="fold")

    for g in groups:
        tbl.add_row(
            g["exception_type"],
            str(g["count"]),
            str(g["with_suggestions"]),
            g["most_recent_timestamp"].replace("T", " ").rstrip("+00:00"),
            g["sample_command"] or "—",
            g["sample_message"],
        )
    console.print(tbl)
    if with_suggestions:
        console.print(
            f"[dim]{with_suggestions}/{total} record(s) have a suggested fix. "
            f"`hafiz errors show <id>` for details.[/dim]\n"
        )
    else:
        console.print("[dim]`hafiz errors show <id>` for full traceback + suggestion.[/dim]\n")


# ── show ──────────────────────────────────────────────────────────────


def run_errors_show(record_id: str, *, output_json: bool = False) -> None:
    record = error_log.get(record_id)
    if record is None:
        if output_json:
            console.print_json(
                json.dumps(
                    {
                        "ok": False,
                        "error": "not_found",
                        "message": f"No error matches id prefix {record_id!r}.",
                    }
                )
            )
        else:
            console.print(f"[red]No error matches id prefix [bold]{record_id}[/bold].[/red]")
        raise typer.Exit(1)

    if output_json:
        console.print_json(json.dumps(record.as_jsonable()))
        return

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]{record.exception_type}[/bold]: {record.message}",
            title=f"Error {record.id[:8]}",
            border_style="red",
            padding=(0, 1),
        )
    )

    # Metadata table
    meta = Table(show_header=False, border_style="cyan")
    meta.add_column("Key", style="bold")
    meta.add_column("Value")
    meta.add_row("id", record.id)
    meta.add_row("timestamp", record.timestamp)
    meta.add_row("command", record.command)
    meta.add_row("argv", " ".join(record.argv) if record.argv else "—")
    meta.add_row("cwd", record.cwd)
    if record.hafiz_version:
        meta.add_row("hafiz_version", record.hafiz_version)
    if record.git_branch:
        meta.add_row(
            "git_branch",
            record.git_branch + (" (dirty)" if record.git_dirty else ""),
        )
    if record.host_fingerprint:
        meta.add_row("host", record.host_fingerprint)
    console.print(meta)

    if record.suggested_action:
        console.print()
        console.print(
            Panel(
                record.suggested_action,
                title="Suggested fix",
                border_style="yellow",
                padding=(0, 1),
            )
        )

    if record.context:
        console.print()
        ctx = Table(title="Context", show_header=False, border_style="cyan")
        ctx.add_column("Key", style="bold")
        ctx.add_column("Value")
        for k, v in record.context.items():
            ctx.add_row(str(k), str(v))
        console.print(ctx)

    if record.traceback:
        console.print()
        console.print(
            Syntax(
                record.traceback.rstrip("\n"),
                "pytb",
                theme="ansi_dark",
                line_numbers=False,
                word_wrap=True,
            )
        )
    console.print()


# ── clear ─────────────────────────────────────────────────────────────


def run_errors_clear(*, output_json: bool = False) -> None:
    count = error_log.clear()
    if output_json:
        console.print_json(json.dumps({"ok": True, "cleared": count}))
        return
    if count:
        console.print(f"[green]Cleared {count} error record(s).[/green]")
    else:
        console.print("[dim]No error records to clear.[/dim]")
