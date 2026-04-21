"""hafiz observe / note / recall — store and search observations."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hafiz.core.database import close_engine
from hafiz.core.durations import parse_duration
from hafiz.core.session import resolve_session_tag

console = Console()


def _compute_valid_until(
    expires_in: str | None, expires: str | None
) -> datetime | None:
    """Resolve --expires-in / --expires into an absolute UTC datetime, or None.

    Mutually exclusive — providing both is a user error.
    """
    if expires_in and expires:
        console.print(
            "[red]Error:[/red] --expires-in and --expires are mutually exclusive."
        )
        raise SystemExit(1)
    if expires_in:
        try:
            return datetime.now(timezone.utc) + parse_duration(expires_in)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)
    if expires:
        try:
            parsed = datetime.fromisoformat(expires)
        except ValueError:
            console.print(
                f"[red]Error:[/red] --expires must be an ISO date/datetime "
                f"(e.g. 2026-06-01), got {expires!r}"
            )
            raise SystemExit(1)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def run_observe(
    text: str,
    *,
    obs_type: str = "fact",
    source: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 1.0,
    expires_in: str | None = None,
    expires: str | None = None,
    session: str | None = None,
    task: str | None = None,
    output_json: bool = False,
) -> None:
    """Store an observation and print confirmation."""
    valid_until = _compute_valid_until(expires_in, expires)
    resolved_session_id, resolved_task = resolve_session_tag(
        session_override=session, task_override=task
    )

    async def _store():
        try:
            from hafiz.core.observations import store_observation

            obs = await store_observation(
                text,
                obs_type=obs_type,
                source=source,
                project=project,
                tags=tags,
                confidence=confidence,
                valid_until=valid_until,
                session_id=resolved_session_id,
                task=resolved_task,
            )
            return obs
        finally:
            await close_engine()

    obs = asyncio.run(_store())

    if output_json:
        data = {
            "action": "observe",
            "observation": {
                "id": str(obs.id),
                "content": obs.content,
                "obs_type": obs.obs_type,
                "source": obs.source,
                "project": obs.project,
                "tags": obs.tags,
                "confidence": obs.confidence,
                "valid_from": obs.valid_from.isoformat(),
                "valid_until": obs.valid_until.isoformat() if obs.valid_until else None,
                "session_id": obs.session_id,
                "task": obs.task,
                "commit_hash": obs.commit_hash,
            },
        }
        console.print_json(json.dumps(data))
        return

    tags_str = ", ".join(obs.tags) if obs.tags else "none"
    session_line = ""
    if obs.session_id or obs.task:
        session_line = (
            f"  [bold]Session:[/bold]    {obs.session_id or '—'}\n"
            f"  [bold]Task:[/bold]       {obs.task or '—'}\n"
        )
    info = (
        f"[bold green]Observation stored[/bold green]\n\n"
        f"  [bold]ID:[/bold]         {obs.id}\n"
        f"  [bold]Type:[/bold]       {obs.obs_type}\n"
        f"  [bold]Source:[/bold]     {obs.source or '—'}\n"
        f"  [bold]Project:[/bold]    {obs.project or '—'}\n"
        f"  [bold]Tags:[/bold]       {tags_str}\n"
        f"  [bold]Confidence:[/bold] {obs.confidence:.0%}\n"
        f"{session_line}"
        f"  [bold]Content:[/bold]    {obs.content[:200]}"
    )
    console.print(Panel(info, border_style="cyan"))


def run_note(
    text: str,
    *,
    source: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 1.0,
    expires_in: str | None = None,
    expires: str | None = None,
    session: str | None = None,
    task: str | None = None,
    output_json: bool = False,
) -> None:
    """Store a raw thought as ``obs_type="note"`` — low-bar capture.

    Thin wrapper over :func:`run_observe`; keeps the CLI surface light
    so ``hafiz note "..."`` does not require choosing a type.
    """
    run_observe(
        text,
        obs_type="note",
        source=source,
        project=project,
        tags=tags,
        confidence=confidence,
        expires_in=expires_in,
        expires=expires,
        session=session,
        task=task,
        output_json=output_json,
    )


STALE_DAYS = 90


def _age(valid_from: datetime) -> tuple[str, int, bool]:
    """Return (human label, age in days, stale flag) for a ``valid_from``.

    Stale threshold is :data:`STALE_DAYS` — age beyond that dims the row
    in recall output so old knowledge doesn't look as authoritative.
    """
    now = datetime.now(timezone.utc)
    days = (now - valid_from.astimezone(timezone.utc)).days
    if days < 0:
        return "future", days, False
    if days == 0:
        return "today", 0, False
    if days == 1:
        return "1d ago", 1, False
    if days < 30:
        return f"{days}d ago", days, days > STALE_DAYS
    if days < 365:
        return f"{round(days / 30)}mo ago", days, days > STALE_DAYS
    return f"{round(days / 365)}y ago", days, True


def run_recall(
    query: str,
    *,
    limit: int = 10,
    project: str | None = None,
    workspace: bool = False,
    obs_type: str | None = None,
    output_json: bool = False,
) -> None:
    """Search observations by semantic similarity and display results."""

    async def _search():
        try:
            from hafiz.core.observations import search_observations

            search_project: str | list[str] | None = project
            if workspace:
                from hafiz.core.context import resolve_workspace_projects

                search_project = await resolve_workspace_projects() or None
            results = await search_observations(
                query,
                limit=limit,
                project=search_project,
                obs_type=obs_type,
            )
            return results
        finally:
            await close_engine()

    results = asyncio.run(_search())

    if output_json:
        data = {
            "query": query,
            "results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "obs_type": r.obs_type,
                    "source": r.source,
                    "project": r.project,
                    "tags": r.tags,
                    "confidence": r.confidence,
                    "valid_from": r.valid_from.isoformat(),
                    "valid_until": r.valid_until.isoformat() if r.valid_until else None,
                    "age_days": _age(r.valid_from)[1],
                    "stale": _age(r.valid_from)[2],
                    "score": r.score,
                }
                for r in results
            ],
            "total": len(results),
        }
        console.print_json(json.dumps(data))
        return

    if not results:
        console.print("[yellow]No observations found.[/yellow]")
        return

    console.print()
    table = Table(
        title=f"Recall: \"{query}\" ({len(results)} results)",
        border_style="cyan",
    )
    table.add_column("Type", style="yellow", width=10)
    table.add_column("Content", ratio=3)
    table.add_column("Source", style="dim", width=16)
    table.add_column("Age", style="dim", width=8)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Score", justify="right", width=8)

    for r in results:
        score_color = "green" if r.score > 0.7 else "yellow" if r.score > 0.5 else "red"
        content_preview = r.content[:120]
        if len(r.content) > 120:
            content_preview += "..."
        age_label, _, stale = _age(r.valid_from)
        table.add_row(
            r.obs_type,
            content_preview,
            r.source or "—",
            age_label,
            f"{r.confidence:.0%}",
            f"[{score_color}]{r.score:.2%}[/{score_color}]",
            style="dim" if stale else None,
        )

    console.print(table)
    console.print()
