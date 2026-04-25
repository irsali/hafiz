"""hafiz session start / end / show / list — session lifecycle + listing.

Phase 2 of workitems/done/communications-and-sessions.md. The on-disk
JSON is now a per-TTY *cursor* pointing at a real ``sessions`` table
row; both layers are kept in sync by :mod:`hafiz.core.session`. The
``list`` subcommand is a small UX gap-fill so callers can find slugs
without hand-rolling psql.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hafiz.core.database import close_engine
from hafiz.core.session import current_session, end_session, start_session
from hafiz.core.sessions import list_sessions

console = Console()


def _info_block(data: dict) -> str:
    inc = data.get("include_domains")
    exc = data.get("exclude_domains")
    filter_lines = ""
    if inc:
        filter_lines += f"\n  [bold]Include:[/bold]  {', '.join(inc)}"
    if exc:
        filter_lines += f"\n  [bold]Exclude:[/bold]  {', '.join(exc)}"
    return (
        f"  [bold]ID:[/bold]       {data.get('session_id')}\n"
        f"  [bold]UUID:[/bold]     {data.get('session_uuid') or '—'}\n"
        f"  [bold]Name:[/bold]     {data.get('name')}\n"
        f"  [bold]Task:[/bold]     {data.get('task') or '—'}\n"
        f"  [bold]Project:[/bold]  {data.get('project') or '—'}\n"
        f"  [bold]Started:[/bold]  {data.get('started_at')}\n"
        f"  [bold]TTY:[/bold]      {data.get('tty')}"
        f"{filter_lines}"
    )


def run_session_start(
    name: str,
    *,
    task: str | None = None,
    project: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    output_json: bool = False,
) -> None:
    try:
        data = start_session(
            name,
            task=task,
            project=project,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(2)

    if output_json:
        console.print_json(json.dumps({"action": "session_start", "session": data}))
        return

    info = (
        "[bold green]Session started[/bold green]\n\n"
        f"{_info_block(data)}\n\n"
        "Subsequent `hafiz observe` / `note` / `capture` in this terminal will\n"
        "auto-tag with this session + task unless overridden per-call."
    )
    console.print(Panel(info, border_style="cyan"))


def run_session_show(output_json: bool = False) -> None:
    data = current_session()
    if output_json:
        console.print_json(json.dumps({"session": data}))
        return
    if not data:
        console.print("[dim]No active session for this terminal.[/dim]")
        return
    info = f"[bold]Active session[/bold]\n\n{_info_block(data)}"
    console.print(Panel(info, border_style="cyan"))


def run_session_end(output_json: bool = False) -> None:
    data = end_session()
    if output_json:
        console.print_json(json.dumps({"action": "session_end", "session": data}))
        return
    if not data:
        console.print("[dim]No active session for this terminal.[/dim]")
        return
    console.print(
        f"[bold green]Session ended:[/bold green] "
        f"{data.get('session_id')} "
        f"([dim]{data.get('name')}[/dim])"
    )


def _humanize_age(when: datetime | None) -> str:
    if when is None:
        return "—"
    now = datetime.now(timezone.utc)
    delta = now - when.astimezone(timezone.utc)
    days = delta.days
    if days < 0:
        return when.astimezone(timezone.utc).strftime("%Y-%m-%d")
    if days == 0:
        h = delta.seconds // 3600
        if h == 0:
            return "just now"
        return f"{h}h ago"
    if days == 1:
        return "1d ago"
    if days < 30:
        return f"{days}d ago"
    if days < 365:
        return f"{round(days / 30)}mo ago"
    return f"{round(days / 365)}y ago"


def run_session_list(
    *,
    agent: str | None = None,
    project: str | None = None,
    include_ended: bool = True,
    limit: int = 50,
    output_json: bool = False,
) -> None:
    """Surface recent sessions so callers don't have to reach for psql.

    Filters mirror the underlying core helper:
    - ``agent`` filters by the ``sessions.agent`` column
      (``claude-code``, ``cursor``, etc.).
    - ``project`` filters by ``scope_value`` when ``scope_kind='project'``.
    - ``include_ended=False`` shows only sessions whose ``ended_at IS NULL``.
    """

    async def _do():
        try:
            return await list_sessions(
                agent=agent,
                scope_kind="project" if project else None,
                scope_value=project,
                limit=limit,
                include_ended=include_ended,
            )
        finally:
            await close_engine()

    rows = asyncio.run(_do())

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "filters": {
                        "agent": agent,
                        "project": project,
                        "include_ended": include_ended,
                        "limit": limit,
                    },
                    "total": len(rows),
                    "sessions": [
                        {
                            "id": str(r.id),
                            "slug": r.slug,
                            "name": r.name,
                            "agent": r.agent,
                            "scope_kind": r.scope_kind,
                            "scope_value": r.scope_value,
                            "task": r.task,
                            "started_at": r.started_at.isoformat(),
                            "ended_at": r.ended_at.isoformat()
                            if r.ended_at
                            else None,
                        }
                        for r in rows
                    ],
                }
            )
        )
        return

    if not rows:
        console.print("[yellow]No sessions match.[/yellow]")
        return

    title_bits = [f"{len(rows)} session(s)"]
    if agent:
        title_bits.append(f"agent={agent}")
    if project:
        title_bits.append(f"project={project}")
    if not include_ended:
        title_bits.append("active only")

    table = Table(title=" · ".join(title_bits), border_style="cyan")
    table.add_column("Slug", style="bold", no_wrap=True)
    table.add_column("Name", ratio=2)
    table.add_column("Agent", style="dim", width=14)
    table.add_column("Project", style="dim", width=18)
    table.add_column("Started", style="dim", width=10)
    table.add_column("Status", style="dim", width=10)

    for r in rows:
        status = "ended" if r.ended_at else "active"
        project_display = r.scope_value if r.scope_kind == "project" else "—"
        table.add_row(
            r.slug,
            r.name or "—",
            r.agent or "—",
            project_display or "—",
            _humanize_age(r.started_at),
            status,
        )
    console.print(table)
