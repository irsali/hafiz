"""hafiz context — synthesize relevant code, graph, and observations for a task."""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from hafiz.core.database import close_engine

console = Console()


def run_context(
    query: str,
    *,
    project: str | None = None,
    workspace: bool = False,
    limit_chunks: int = 5,
    limit_observations: int = 5,
    output_json: bool = False,
) -> None:
    """Build and display a context bundle for a task description."""

    async def _build():
        try:
            if workspace:
                from hafiz.core.context import build_workspace_context
                from hafiz.core.config import get_settings

                settings = get_settings()
                configured_projects = settings.workspace.projects or None
                bundle = await build_workspace_context(
                    query,
                    projects=configured_projects,
                    limit_chunks=limit_chunks * 2,
                    limit_observations=limit_observations * 2,
                )
            else:
                from hafiz.core.context import build_context

                bundle = await build_context(
                    query,
                    project=project,
                    limit_chunks=limit_chunks,
                    limit_observations=limit_observations,
                )
            return bundle
        finally:
            await close_engine()

    bundle = asyncio.run(_build())

    if output_json:
        console.print_json(json.dumps(bundle.to_dict(), default=str))
        return

    title = f"Context (workspace): {query[:50]}" if workspace else f"Context: {query[:60]}"
    console.print()
    md = Markdown(bundle.to_markdown())
    console.print(
        Panel(
            md,
            title=title,
            border_style="green" if workspace else "cyan",
            padding=(1, 2),
        )
    )
    console.print()

    # Summary line
    summary = (
        f"  [dim]Chunks: {len(bundle.chunks)} | "
        f"Entities: {len(bundle.entities)} | "
        f"Observations: {len(bundle.observations)}"
    )
    if bundle.project_distribution:
        projects_involved = len(bundle.project_distribution)
        summary += f" | Projects: {projects_involved}"
    summary += "[/dim]"
    console.print(summary)
    console.print()
