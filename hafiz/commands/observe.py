"""hafiz observe / recall — store and search observations."""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hafiz.core.database import close_engine

console = Console()


def run_observe(
    text: str,
    *,
    obs_type: str = "fact",
    source: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 1.0,
    output_json: bool = False,
) -> None:
    """Store an observation and print confirmation."""

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
            },
        }
        console.print_json(json.dumps(data))
        return

    tags_str = ", ".join(obs.tags) if obs.tags else "none"
    info = (
        f"[bold green]Observation stored[/bold green]\n\n"
        f"  [bold]ID:[/bold]         {obs.id}\n"
        f"  [bold]Type:[/bold]       {obs.obs_type}\n"
        f"  [bold]Source:[/bold]     {obs.source or '—'}\n"
        f"  [bold]Project:[/bold]    {obs.project or '—'}\n"
        f"  [bold]Tags:[/bold]       {tags_str}\n"
        f"  [bold]Confidence:[/bold] {obs.confidence:.0%}\n"
        f"  [bold]Content:[/bold]    {obs.content[:200]}"
    )
    console.print(Panel(info, border_style="cyan"))


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
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Score", justify="right", width=8)

    for r in results:
        score_color = "green" if r.score > 0.7 else "yellow" if r.score > 0.5 else "red"
        content_preview = r.content[:120]
        if len(r.content) > 120:
            content_preview += "..."
        table.add_row(
            r.obs_type,
            content_preview,
            r.source or "—",
            f"{r.confidence:.0%}",
            f"[{score_color}]{r.score:.2%}[/{score_color}]",
        )

    console.print(table)
    console.print()
