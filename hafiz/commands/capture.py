"""hafiz capture — ingest a transcript or multi-page dump as transcript chunks."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from hafiz.core.database import close_engine
from hafiz.core.session import resolve_session_tag

console = Console()


def _resolve_text(text: str | None, file: str | None) -> str:
    """Resolve transcript content from arg / --file / stdin, or error cleanly."""
    if text and file:
        console.print("[red]Error:[/red] pass TEXT or --file, not both.")
        raise SystemExit(1)
    if text:
        return text
    if file:
        p = Path(file)
        if not p.exists():
            console.print(f"[red]Error:[/red] file not found: {file}")
            raise SystemExit(1)
        return p.read_text(encoding="utf-8")
    # Fall back to stdin if something is piped in.
    if not sys.stdin.isatty():
        return sys.stdin.read()
    console.print("[red]Error:[/red] no input — pass TEXT, use --file <path>, or pipe via stdin.")
    raise SystemExit(1)


def run_capture(
    text: str | None,
    *,
    file: str | None = None,
    title: str | None = None,
    project: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    session: str | None = None,
    task: str | None = None,
    session_key: str | None = None,
    output_json: bool = False,
) -> None:
    """Entry point for the ``hafiz capture`` command."""
    content = _resolve_text(text, file)
    if not content.strip():
        console.print("[red]Error:[/red] input is empty — nothing to capture.")
        raise SystemExit(1)

    resolved_session_id, resolved_task = resolve_session_tag(
        session_override=session, task_override=task, session_key=session_key
    )

    async def _run():
        try:
            from hafiz.core.capture import store_transcript

            return await store_transcript(
                content,
                title=title,
                project=project,
                source=source,
                tags=tags,
                session_id=resolved_session_id,
                task=resolved_task,
            )
        finally:
            await close_engine()

    try:
        summary = asyncio.run(_run())
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "action": "capture",
                    "communication_id": summary.communication_id,
                    "title": summary.title,
                    "source": source,
                    "project": project,
                    "turn_count": summary.turn_count,
                    "messages_embedded": summary.messages_embedded,
                    "session_id": resolved_session_id,
                    "task": resolved_task,
                }
            )
        )
        return

    tags_str = ", ".join(tags) if tags else "none"
    session_line = ""
    if resolved_session_id or resolved_task:
        session_line = (
            f"  [bold]Session:[/bold]  {resolved_session_id or '—'}\n"
            f"  [bold]Task:[/bold]     {resolved_task or '—'}\n"
        )
    info = (
        f"[bold green]Transcript captured[/bold green]\n\n"
        f"  [bold]ID:[/bold]       {summary.communication_id}\n"
        f"  [bold]Title:[/bold]    {summary.title or '—'}\n"
        f"  [bold]Source:[/bold]   {source or '—'}\n"
        f"  [bold]Project:[/bold]  {project or '—'}\n"
        f"  [bold]Tags:[/bold]     {tags_str}\n"
        f"{session_line}"
        f"  [bold]Turns:[/bold]    {summary.turn_count}\n"
        f"  [bold]Embedded:[/bold] {summary.messages_embedded}\n"
        f"  [dim]Source layer — surface via `hafiz recall {summary.communication_id}`[/dim]"
    )
    console.print(Panel(info, border_style="cyan"))
