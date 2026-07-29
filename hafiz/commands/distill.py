"""hafiz distill — surface recent raw captures as promotable candidates."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, timedelta

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hafiz.core.database import close_engine
from hafiz.core.distill import DistillBundle, Theme, brief_gate_open, find_distill_candidates
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
    include_promoted: bool = False,
    limit: int = 200,
    message_limit: int | None = None,
    output_json: bool = False,
    brief: bool = False,
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
                include_promoted=include_promoted,
                limit=limit,
                message_limit=message_limit,
            )
        finally:
            await close_engine()

    bundle = asyncio.run(_run())

    if brief:
        _print_brief(bundle)
    elif output_json:
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
                "promoted": n.promoted,
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
        "themes": [
            {
                "size": t.size,
                "score": round(t.score, 4),
                "oldest": t.oldest.isoformat(),
                "newest": t.newest.isoformat(),
                "members": [
                    {
                        "id": m.id,
                        "kind": m.kind,
                        "label": m.label,
                        "ts": m.ts.isoformat(),
                        "content": m.content,
                    }
                    for m in t.members
                ],
                "scaffold": _theme_scaffold(t),
            }
            for t in bundle.themes
        ],
        "backlog": bundle.backlog.to_dict() if bundle.backlog else None,
        "promotion_hint": _promotion_hint(note_ids, transcript_ids, message_ids),
    }
    console.print_json(json.dumps(payload))


def _theme_scaffold(theme: Theme) -> str:
    """The one ``observe`` that would retire this theme from the backlog.

    Every member is cited, not just the first five: citing a capture is what
    marks it distilled, so a truncated scaffold would leave the uncited
    members in the queue forever and make the backlog look stuck.
    """
    ids = ",".join(m.id for m in theme.members)
    return f"hafiz observe '<distilled text>' --type decision --derived-from {ids}"


def _print_brief(bundle: DistillBundle) -> None:
    """Token-lean markdown for injection into an agent's context.

    Prints **nothing** unless the backlog clears the configured gate. A
    session-start hook pipes this straight through, so silence has to be the
    ordinary answer — the same contract ``query --format md`` follows on an
    empty result set.
    """
    from hafiz.core.config import load_settings

    cfg = load_settings().distill
    if not brief_gate_open(
        bundle.backlog,
        min_pending=cfg.brief_min_pending,
        min_age_days=cfg.brief_min_age_days,
    ):
        return

    backlog = bundle.backlog
    assert backlog is not None  # gate_open is False when it's None
    age = backlog.oldest_pending_age_days
    lines = [
        "## Distillation backlog",
        "",
        f"{backlog.pending} raw captures are waiting to be distilled"
        + (f", oldest {age:g}d old" if age is not None else "")
        + f" ({backlog.themes} themes).",
        "",
        "Read them, then promote the ones worth keeping with the scaffold "
        "shown — `--derived-from` is what drains the queue. Decline one with "
        "`hafiz forget <id> --annotation`.",
        "",
    ]
    for theme in bundle.themes[: cfg.brief_max_themes]:
        head = f"### Theme ({theme.size} capture{'s' if theme.size > 1 else ''}"
        head += f", similarity {theme.score:.2f})" if theme.size > 1 else ")"
        lines.append(head)
        for member in theme.members[: cfg.brief_max_members]:
            lines.append(f"- [{member.kind}] {_preview(member.content, 200)}")
        hidden = theme.size - cfg.brief_max_members
        if hidden > 0:
            lines.append(f"- _…{hidden} more in this theme_")
        lines.append("")
        lines.append(f"```\n{_theme_scaffold(theme)}\n```")
        lines.append("")

    dropped = backlog.themes - cfg.brief_max_themes
    if dropped > 0:
        lines.append(f"_{dropped} further themes not shown — `hafiz distill --json` for all._")

    # Builtin print, not the Rich console: Rich soft-wraps to terminal width,
    # which inserts hard newlines mid-command and mid-word. A wrapped scaffold
    # is a broken scaffold. Same reason `query --format md` prints directly.
    print("\n".join(lines))


def _preview(text: str, width: int) -> str:
    """One line, collapsed whitespace — raw note prose shreds the layout."""
    flat = " ".join(text.split())
    return flat[:width] + ("…" if len(flat) > width else "")


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
    total = len(bundle.notes) + len(bundle.transcripts) + len(bundle.messages)
    if total == 0:
        console.print(f"[yellow]No distill candidates in the last {since_arg or '7d'}.[/yellow]")
        return

    window_label = f"Since {since_arg or '7d'}"
    console.print()
    console.print(
        f"[bold]Distill candidates — {window_label}[/bold]  "
        f"[dim]({len(bundle.notes)} notes, "
        f"{len(bundle.transcripts)} transcripts, "
        f"{len(bundle.messages)} messages)[/dim]"
    )

    backlog = bundle.backlog
    if backlog:
        age = backlog.oldest_pending_age_days
        summary = (
            f"[dim]Backlog: {backlog.pending} pending in "
            f"{backlog.themes} themes ({backlog.clustered} clustered)"
        )
        if age is not None:
            summary += f", oldest {age:g}d"
        if backlog.promoted:
            summary += f" · {backlog.promoted} already promoted (hidden)"
        if backlog.skipped_unembedded:
            summary += f" · {backlog.skipped_unembedded} turns skipped (not embedded)"
        console.print(summary + "[/dim]")

    if bundle.themes:
        console.print()
        console.print("[bold]Themes[/bold] [dim](biggest first — one observe drains one)[/dim]")
        for n, theme in enumerate(bundle.themes, 1):
            label = f"{theme.size} capture{'s' if theme.size > 1 else ''}"
            if theme.size > 1:
                label += f", similarity {theme.score:.2f}"
            body = "\n".join(
                f"  [dim]{m.kind:>7}[/dim]  {_preview(m.content, 140)}" for m in theme.members
            )
            console.print()
            console.print(
                Panel(
                    f"{body}\n\n[dim]{_theme_scaffold(theme)}[/dim]",
                    title=f"[cyan]Theme {n}[/cyan] [dim]({label})[/dim]",
                    title_align="left",
                    border_style="cyan" if theme.size > 1 else "dim",
                )
            )

    if bundle.notes:
        table = Table(title="Notes", border_style="cyan", title_justify="left")
        table.add_column("ID", style="dim", width=14, no_wrap=True)
        table.add_column("Date", style="dim", width=11)
        table.add_column("Source", style="dim", width=18)
        table.add_column("Task", width=14)
        table.add_column("Content", ratio=3)
        for n in bundle.notes:
            # Promoted rows only appear under --include-promoted; flag them so
            # the drain is visible rather than inferred from a shorter list.
            content = _preview(n.content, 100)
            table.add_row(
                n.id[:12],
                n.valid_from.astimezone(UTC).strftime("%Y-%m-%d"),
                n.source or "—",
                n.task or "—",
                f"[dim](promoted)[/dim] {content}" if n.promoted else content,
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
                t.captured_at.astimezone(UTC).strftime("%Y-%m-%d"),
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
                m.ts.astimezone(UTC).strftime("%Y-%m-%d %H:%M"),
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
