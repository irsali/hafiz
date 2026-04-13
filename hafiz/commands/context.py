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
    limit_chunks: int = 5,
    limit_observations: int = 5,
    output_json: bool = False,
) -> None:
    """Build and display a context bundle for a task description."""

    async def _build():
        try:
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

    console.print()
    md = Markdown(bundle.to_markdown())
    console.print(
        Panel(
            md,
            title=f"Context: {query[:60]}",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    # Summary line
    console.print(
        f"  [dim]Chunks: {len(bundle.chunks)} | "
        f"Entities: {len(bundle.entities)} | "
        f"Observations: {len(bundle.observations)}[/dim]"
    )
    console.print()
