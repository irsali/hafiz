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
    kind: str | None,
    output_json: bool,
) -> None:
    """Run the async search and display results."""

    async def _search():
        try:
            search_project: str | list[str] | None = project
            if workspace:
                # hafiz.core.context still depends on the old schema;
                # workspace fan-out is disabled until Phase 3b rewires it.
                console.print(
                    "[yellow]--workspace fan-out is disabled until "
                    "hafiz.core.context is rewired (Phase 3b). "
                    "Falling back to --project filter.[/yellow]"
                )
            results = await vector_search(
                text,
                limit=limit,
                project=search_project,
                kind=kind,
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
                    "unit_id": r.unit_id,
                    "unit_name": r.unit_name,
                    "kind": r.kind,
                    "content": r.content,
                    "source_file": r.source_file,
                    "line_start": r.line_start,
                    "line_end": r.line_end,
                    "language": r.language,
                    "project": r.project,
                    "part_index": r.part_index,
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
    for r in results:
        location = f"{r.source_file}::{r.unit_name}"
        if r.line_start and r.line_end:
            location += f" (L{r.line_start}-{r.line_end})"

        kind_tag = f"[dim]{r.kind}[/dim] "
        score_color = (
            "green" if r.score > 0.7 else "yellow" if r.score > 0.5 else "red"
        )

        panel_content.append(
            f"  {kind_tag}[bold]{location}[/bold]  "
            f"[{score_color}]{r.score:.2%}[/{score_color}]"
        )
        preview = r.content[:200].replace("\n", " ").strip()
        if len(r.content) > 200:
            preview += "..."
        panel_content.append(f"  [dim]{preview}[/dim]")
        panel_content.append("")

    panel_text = "\n".join(panel_content)
    console.print(
        Panel(
            panel_text,
            title=f"Results ({len(results)} matches)",
            border_style="cyan",
        )
    )
