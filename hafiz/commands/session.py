"""hafiz session start / end / show — per-TTY session state management."""

from __future__ import annotations

import json

from rich.console import Console
from rich.panel import Panel

from hafiz.core.session import current_session, end_session, start_session

console = Console()


def run_session_start(
    name: str,
    *,
    task: str | None = None,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    try:
        data = start_session(name, task=task, project=project)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    if output_json:
        console.print_json(json.dumps({"action": "session_start", "session": data}))
        return

    info = (
        f"[bold green]Session started[/bold green]\n\n"
        f"  [bold]ID:[/bold]       {data['session_id']}\n"
        f"  [bold]Name:[/bold]     {data['name']}\n"
        f"  [bold]Task:[/bold]     {data.get('task') or '—'}\n"
        f"  [bold]Project:[/bold]  {data.get('project') or '—'}\n"
        f"  [bold]Started:[/bold]  {data['started_at']}\n"
        f"  [bold]TTY:[/bold]      {data['tty']}\n\n"
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
    info = (
        f"[bold]Active session[/bold]\n\n"
        f"  [bold]ID:[/bold]       {data.get('session_id')}\n"
        f"  [bold]Name:[/bold]     {data.get('name')}\n"
        f"  [bold]Task:[/bold]     {data.get('task') or '—'}\n"
        f"  [bold]Project:[/bold]  {data.get('project') or '—'}\n"
        f"  [bold]Started:[/bold]  {data.get('started_at')}\n"
        f"  [bold]TTY:[/bold]      {data.get('tty')}"
    )
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
