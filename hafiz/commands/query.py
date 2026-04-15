"""hafiz query — vector similarity search over indexed chunks."""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hafiz.core.database import get_engine, close_engine
from hafiz.core.search import vector_search

console = Console()
query_app = typer.Typer(name="query", help="Search indexed content")


def _run_query(
    text: str,
    *,
    limit: int,
    project: str | None,
    workspace: bool = False,
    chunk_type: str | None,
    output_json: bool,
) -> None:
    """Run the async search and display results."""

    async def _search():
        try:
            search_project: str | list[str] | None = project
            if workspace:
                from hafiz.core.context import resolve_workspace_projects

                search_project = await resolve_workspace_projects() or None
            results = await vector_search(
                text,
                limit=limit,
                project=search_project,
                chunk_type=chunk_type,
            )
            return results
        finally:
            await close_engine()

    results = asyncio.run(_search())

    if output_json:
        data = {
            "query": text,
            "results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "source_file": r.source_file,
                    "line_start": r.line_start,
                    "line_end": r.line_end,
                    "chunk_type": r.chunk_type,
                    "language": r.language,
                    "project": r.project,
                    "score": r.score,
                }
                for r in results
            ],
            "total": len(results),
        }
        console.print_json(json.dumps(data))
        return

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print()
    panel_content = []
    for i, r in enumerate(results, 1):
        location = r.source_file
        if r.line_start and r.line_end:
            location += f" (lines {r.line_start}-{r.line_end})"

        lang = f"[dim]{r.language}[/dim] " if r.language else ""
        score_color = "green" if r.score > 0.7 else "yellow" if r.score > 0.5 else "red"

        panel_content.append(
            f"  {lang}[bold]{location}[/bold]  [{score_color}]{r.score:.2%}[/{score_color}]"
        )
        # Show a preview (first 200 chars)
        preview = r.content[:200].replace("\n", " ").strip()
        if len(r.content) > 200:
            preview += "..."
        panel_content.append(f"  [dim]{preview}[/dim]")
        panel_content.append("")

    panel_text = "\n".join(panel_content)
    console.print(
        Panel(
            panel_text,
            title=f"Results ({len(results)} chunks)",
            border_style="cyan",
        )
    )
