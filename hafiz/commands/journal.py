"""hafiz journal — render a time-bounded digest of observations."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

from rich.console import Console
from rich.table import Table

from hafiz.core.database import close_engine
from hafiz.core.durations import parse_duration
from hafiz.core.journal import JournalBundle, build_journal

console = Console()

# Visual hierarchy — dim for raw capture, bold for load-bearing decisions.
OBS_TYPE_STYLE = {
    "note": "dim cyan",
    "fact": "white",
    "decision": "bold green",
    "learning": "yellow",
    "pattern": "magenta",
    "warning": "bold red",
}


def _parse_since(since: str | None) -> timedelta:
    if since is None:
        return timedelta(days=7)
    try:
        return parse_duration(since)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def _parse_day(day: str | None) -> datetime | None:
    if day is None:
        return None
    try:
        parsed = datetime.fromisoformat(day)
    except ValueError:
        console.print(
            f"[red]Error:[/red] --day must be an ISO date "
            f"(e.g. 2026-04-20), got {day!r}"
        )
        raise SystemExit(1)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def run_journal(
    *,
    since: str | None = None,
    day: str | None = None,
    project: str | None = None,
    workspace: bool = False,
    source: str | None = None,
    obs_type: str | None = None,
    session_id: str | None = None,
    task: str | None = None,
    limit: int = 500,
    output_json: bool = False,
) -> None:
    """Entry point for the ``hafiz journal`` command."""
    if since and day:
        console.print(
            "[red]Error:[/red] --since and --day are mutually exclusive."
        )
        raise SystemExit(1)

    since_td = _parse_since(since) if day is None else None
    day_dt = _parse_day(day)

    async def _run() -> JournalBundle:
        try:
            projects: str | list[str] | None = project
            if workspace:
                from hafiz.core.context import resolve_workspace_projects

                projects = await resolve_workspace_projects() or None

            return await build_journal(
                since=since_td,
                day=day_dt,
                project=projects,
                source=source,
                obs_type=obs_type,
                session_id=session_id,
                task=task,
                limit=limit,
            )
        finally:
            await close_engine()

    bundle = asyncio.run(_run())

    if output_json:
        _print_json(bundle)
    else:
        _print_rich(bundle, since_arg=since, day_arg=day)


def _print_json(bundle: JournalBundle) -> None:
    payload = {
        "window": {
            "start": bundle.window_start.isoformat(),
            "end": bundle.window_end.isoformat(),
        },
        "total": len(bundle.entries),
        "entries": [
            {
                "id": e.id,
                "content": e.content,
                "obs_type": e.obs_type,
                "source": e.source,
                "project": e.project,
                "tags": e.tags,
                "confidence": e.confidence,
                "valid_from": e.valid_from.isoformat(),
                "valid_until": (
                    e.valid_until.isoformat() if e.valid_until else None
                ),
                "session_id": e.session_id,
                "task": e.task,
                "commit_hash": e.commit_hash,
                "branch": e.metadata.get("branch"),
                "is_dirty": e.metadata.get("is_dirty"),
            }
            for e in bundle.entries
        ],
        "captures": [
            {
                "transcript_id": c.transcript_id,
                "title": c.title,
                "source_file": c.source_file,
                "turn_count": c.turn_count,
                "captured_at": c.captured_at.isoformat(),
                "source": c.source,
                "tags": c.tags,
                "project": c.project,
                "session_id": c.session_id,
                "task": c.task,
                "preview": c.preview,
            }
            for c in bundle.captures
        ],
    }
    console.print_json(json.dumps(payload))


def _print_rich(
    bundle: JournalBundle,
    *,
    since_arg: str | None,
    day_arg: str | None,
) -> None:
    if not bundle.entries and not bundle.captures:
        label = f"--day {day_arg}" if day_arg else f"--since {since_arg or '7d'}"
        console.print(f"[yellow]Nothing in window ({label}).[/yellow]")
        return

    window_label = (
        f"Day {day_arg}" if day_arg else f"Since {since_arg or '7d'}"
    )
    total_items = len(bundle.entries) + len(bundle.captures)
    totals = [f"{len(bundle.entries)} entries"]
    if bundle.captures:
        totals.append(f"{len(bundle.captures)} captures")
    console.print()
    console.print(
        f"[bold]Journal — {window_label}[/bold]  "
        f"[dim]({total_items} items · {' · '.join(totals)})[/dim]"
    )

    for day_str, entries, captures in bundle.grouped_by_day():
        if entries:
            table = Table(
                title=day_str,
                border_style="cyan",
                title_justify="left",
            )
            table.add_column("Time", style="dim", width=5)
            table.add_column("Type", width=10)
            table.add_column("Source", style="dim", width=18)
            table.add_column("Content", ratio=3)
            table.add_column("Context", style="dim", width=16)

            for e in entries:
                t = e.valid_from.astimezone(timezone.utc).strftime("%H:%M")
                type_style = OBS_TYPE_STYLE.get(e.obs_type, "white")
                content_preview = (
                    e.content if len(e.content) <= 120 else e.content[:117] + "..."
                )
                ctx_parts: list[str] = []
                if branch := e.metadata.get("branch"):
                    ctx_parts.append(branch)
                if e.commit_hash:
                    ctx_parts.append(e.commit_hash[:8])
                if e.metadata.get("is_dirty"):
                    ctx_parts.append("*")
                table.add_row(
                    t,
                    f"[{type_style}]{e.obs_type}[/{type_style}]",
                    e.source or "—",
                    content_preview,
                    " ".join(ctx_parts),
                )

            console.print()
            console.print(table)

        if captures:
            cap_table = Table(
                title=f"{day_str} — captures",
                border_style="magenta",
                title_justify="left",
            )
            cap_table.add_column("Time", style="dim", width=5)
            cap_table.add_column("Title", width=24)
            cap_table.add_column("Source", style="dim", width=18)
            cap_table.add_column("Turns", justify="right", width=6)
            cap_table.add_column("Preview", ratio=3)

            for c in captures:
                t = c.captured_at.astimezone(timezone.utc).strftime("%H:%M")
                cap_table.add_row(
                    t,
                    c.title or "—",
                    c.source or "—",
                    str(c.turn_count),
                    c.preview,
                )

            console.print()
            console.print(cap_table)
    console.print()
