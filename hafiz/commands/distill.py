"""hafiz distill — surface recent raw captures as promotable candidates."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hafiz.core.database import close_engine
from hafiz.core.distill import DistillBundle, find_distill_candidates
from hafiz.core.durations import parse_duration

console = Console()


def _parse_since(since: str | None) -> timedelta:
    if since is None:
        return timedelta(days=7)
    try:
        return parse_duration(since)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)


def run_distill(
    *,
    since: str | None = None,
    project: str | None = None,
    workspace: bool = False,
    session_id: str | None = None,
    task: str | None = None,
    include_transcripts: bool = True,
    limit: int = 200,
    output_json: bool = False,
) -> None:
    """Entry point for the ``hafiz distill`` command."""
    td = _parse_since(since)

    async def _run() -> DistillBundle:
        try:
            projects: str | list[str] | None = project
            if workspace:
                from hafiz.core.context import resolve_workspace_projects

                projects = await resolve_workspace_projects() or None
            return await find_distill_candidates(
                since=td,
                project=projects,
                session_id=session_id,
                task=task,
                include_transcripts=include_transcripts,
                limit=limit,
            )
        finally:
            await close_engine()

    bundle = asyncio.run(_run())

    if output_json:
        _print_json(bundle)
    else:
        _print_rich(bundle, since_arg=since)


def _print_json(bundle: DistillBundle) -> None:
    note_ids = [n.id for n in bundle.notes]
    transcript_ids = [t.transcript_id for t in bundle.transcripts]
    message_ids = [m.id for m in bundle.messages]
    payload = {
        "window": {
            "start": bundle.window_start.isoformat(),
            "end": bundle.window_end.isoformat(),
        },
        "total_notes": len(bundle.notes),
        "total_transcripts": len(bundle.transcripts),
        "total_messages": len(bundle.messages),
        "notes": [
            {
                "id": n.id,
                "content": n.content,
                "valid_from": n.valid_from.isoformat(),
                "source": n.source,
                "project": n.project,
                "tags": n.tags,
                "session_id": n.session_id,
                "task": n.task,
            }
            for n in bundle.notes
        ],
        "transcripts": [
            {
                "transcript_id": t.transcript_id,
                "title": t.title,
                "source_file": t.source_file,
                "turn_count": t.turn_count,
                "captured_at": t.captured_at.isoformat(),
                "project": t.project,
                "source": t.source,
                "session_id": t.session_id,
                "task": t.task,
                "preview": t.preview,
            }
            for t in bundle.transcripts
        ],
        "messages": [
            {
                "id": m.id,
                "communication_id": m.communication_id,
                "seq": m.seq,
                "role": m.role,
                "author": m.author,
                "content": m.content,
                "ts": m.ts.isoformat(),
                "marked_salient": m.marked_salient,
            }
            for m in bundle.messages
        ],
        "promotion_hint": _promotion_hint(note_ids, transcript_ids, message_ids),
    }
    console.print_json(json.dumps(payload))


def _promotion_hint(
    note_ids: list[str],
    transcript_ids: list[str],
    message_ids: list[str] | None = None,
) -> str | None:
    """Human / agent-oriented scaffold for the follow-up promote call.

    Phase 5 enrichment: ``--derived-from`` accepts message ids as well
    as annotation ids — the polymorphic ``annotation_targets`` pivot
    classifies each id at write time. We prefer message ids in the
    hint when they're available, since they're typically the load-
    bearing source of a distilled decision; notes / transcripts are
    fallbacks.
    """
    candidates: list[str] = []
    if message_ids:
        candidates = message_ids
    elif note_ids:
        candidates = note_ids
    elif transcript_ids:
        candidates = transcript_ids
    if not candidates:
        return None
    return (
        "hafiz observe '<distilled text>' --type decision "
        f"--derived-from {','.join(candidates[:5])}"
    )


def _print_rich(bundle: DistillBundle, *, since_arg: str | None) -> None:
    total = (
        len(bundle.notes) + len(bundle.transcripts) + len(bundle.messages)
    )
    if total == 0:
        console.print(
            f"[yellow]No distill candidates in the last "
            f"{since_arg or '7d'}.[/yellow]"
        )
        return

    window_label = f"Since {since_arg or '7d'}"
    console.print()
    console.print(
        f"[bold]Distill candidates — {window_label}[/bold]  "
        f"[dim]({len(bundle.notes)} notes, "
        f"{len(bundle.transcripts)} transcripts, "
        f"{len(bundle.messages)} messages)[/dim]"
    )

    if bundle.notes:
        table = Table(
            title="Notes", border_style="cyan", title_justify="left"
        )
        table.add_column("ID", style="dim", width=14, no_wrap=True)
        table.add_column("Date", style="dim", width=11)
        table.add_column("Source", style="dim", width=18)
        table.add_column("Task", width=14)
        table.add_column("Content", ratio=3)
        for n in bundle.notes:
            table.add_row(
                n.id[:12],
                n.valid_from.astimezone(timezone.utc).strftime("%Y-%m-%d"),
                n.source or "—",
                n.task or "—",
                n.content[:100] + ("..." if len(n.content) > 100 else ""),
            )
        console.print()
        console.print(table)

    if bundle.transcripts:
        table = Table(
            title="Transcripts",
            border_style="magenta",
            title_justify="left",
        )
        table.add_column("ID", style="dim", width=14, no_wrap=True)
        table.add_column("Date", style="dim", width=11)
        table.add_column("Title", width=24)
        table.add_column("Turns", justify="right", width=6)
        table.add_column("Preview", ratio=3)
        for t in bundle.transcripts:
            table.add_row(
                t.transcript_id[:12],
                t.captured_at.astimezone(timezone.utc).strftime("%Y-%m-%d"),
                t.title or "—",
                str(t.turn_count),
                t.preview,
            )
        console.print()
        console.print(table)

    if bundle.messages:
        table = Table(
            title="Messages (source layer)",
            border_style="magenta",
            title_justify="left",
        )
        table.add_column("ID", style="dim", width=14, no_wrap=True)
        table.add_column("When", style="dim", width=16)
        table.add_column("Role", width=10)
        table.add_column("Author", style="dim", width=18)
        table.add_column("Content", ratio=3)
        for m in bundle.messages:
            preview = m.content[:120]
            if len(m.content) > 120:
                preview += "…"
            table.add_row(
                m.id[:12],
                m.ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                m.role,
                m.author or "—",
                preview,
            )
        console.print()
        console.print(table)

    hint = _promotion_hint(
        [n.id for n in bundle.notes],
        [t.transcript_id for t in bundle.transcripts],
        [m.id for m in bundle.messages],
    )
    if hint:
        console.print()
        console.print(
            Panel(
                f"[bold]Promotion scaffold[/bold]\n\n"
                f"[dim]Review the candidates above, then promote "
                f"(first 5 ids cited):[/dim]\n\n"
                f"  {hint}",
                border_style="green",
            )
        )
    console.print()
