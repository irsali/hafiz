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
    include_transcripts: bool = False,
) -> None:
    """Run the async search and display results.

    When ``include_transcripts`` is set, additionally vector-search the
    source-layer ``communication_messages`` and merge the results into
    the output, clearly tagged with ``layer="source"`` so consumers
    (agents) can distinguish them from knowledge-layer hits. Default
    (off): the wisdom layer is primary.
    """

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
            transcript_hits = []
            if include_transcripts:
                from hafiz.core.communications import search_messages

                # Source-layer search has no project filter today;
                # users scope by passing --include-transcripts only
                # when they actually want them. Limit is shared so a
                # transcript-heavy query doesn't blow past the cap.
                rows = await search_messages(text, limit=limit)
                transcript_hits = [(r, score) for r, score in rows]
            return results, transcript_hits
        finally:
            await close_engine()

    results, transcript_hits = asyncio.run(_search())

    if output_json:
        data = {
            "query": text,
            "results": [
                {
                    "layer": "knowledge",
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
            "transcripts": [
                {
                    "layer": "source",
                    "kind": "chat.turn",
                    "id": msg.id,
                    "communication_id": msg.communication_id,
                    "seq": msg.seq,
                    "role": msg.role,
                    "author": msg.author,
                    "content": msg.content,
                    "ts": msg.ts.isoformat(),
                    "score": score,
                }
                for msg, score in transcript_hits
            ],
            "total": len(results) + len(transcript_hits),
            "include_transcripts": include_transcripts,
        }
        console.print_json(json.dumps(data))
        return

    if not results and not transcript_hits:
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
    if results:
        console.print(
            Panel(
                panel_text,
                title=f"Knowledge-layer results ({len(results)} matches)",
                border_style="cyan",
            )
        )

    if transcript_hits:
        ts_panel = []
        for msg, score in transcript_hits:
            score_color = (
                "green" if score > 0.7 else "yellow" if score > 0.5 else "red"
            )
            preview = msg.content[:200].replace("\n", " ").strip()
            if len(msg.content) > 200:
                preview += "..."
            ts_panel.append(
                f"  [dim]chat.turn[/dim] [bold]{msg.role}[/bold] "
                f"seq {msg.seq}  [{score_color}]{score:.2%}[/{score_color}]"
            )
            ts_panel.append(f"  [dim]{preview}[/dim]")
            ts_panel.append("")
        console.print(
            Panel(
                "\n".join(ts_panel),
                title=(
                    f"Source-layer transcripts "
                    f"({len(transcript_hits)} matches, opt-in)"
                ),
                border_style="magenta",
            )
        )
