"""hafiz query — vector similarity search over indexed chunks."""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.panel import Panel

from hafiz.core.database import close_engine
from hafiz.core.formats import OutputFormat, chunk_compact, chunk_md
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
    output_format: OutputFormat = OutputFormat.RICH,
    with_ids: bool = False,
    min_score: float | None = None,
    include_transcripts: bool = False,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
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
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                similarity_threshold=min_score or 0.0,
            )
            transcript_hits = []
            if include_transcripts:
                from hafiz.core.communications import search_messages

                # Source-layer search has no project filter today;
                # users scope by passing --include-transcripts only
                # when they actually want them. Limit is shared so a
                # transcript-heavy query doesn't blow past the cap.
                rows = await search_messages(text, limit=limit)
                transcript_hits = [
                    (r, score) for r, score in rows if min_score is None or score >= min_score
                ]
            return results, transcript_hits
        finally:
            await close_engine()

    results, transcript_hits = asyncio.run(_search())

    if output_format is OutputFormat.COMPACT:
        data = {
            "query": text,
            "results": [chunk_compact(r, with_ids=with_ids) for r in results],
            "total": len(results) + len(transcript_hits),
        }
        if include_transcripts:
            data["transcripts"] = [
                {"content": msg.content, "role": msg.role, "seq": msg.seq}
                | ({"id": msg.id} if with_ids else {})
                for msg, _ in transcript_hits
            ]
        console.print_json(json.dumps(data))
        return

    if output_format is OutputFormat.MD:
        if not results and not transcript_hits:
            return  # silence, not a placeholder — see the note in observe.py
        print(f"## Results: {text}\n")
        for r in results:
            print(chunk_md(r, with_ids=with_ids))
            print()
        for msg, _ in transcript_hits:
            print(f"### transcript · {msg.role} · seq {msg.seq}\n\n{msg.content}\n")
        return

    if output_format is OutputFormat.JSON:
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
        score_color = "green" if r.score > 0.7 else "yellow" if r.score > 0.5 else "red"

        panel_content.append(
            f"  {kind_tag}[bold]{location}[/bold]  [{score_color}]{r.score:.2%}[/{score_color}]"
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
            score_color = "green" if score > 0.7 else "yellow" if score > 0.5 else "red"
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
                title=(f"Source-layer transcripts ({len(transcript_hits)} matches, opt-in)"),
                border_style="magenta",
            )
        )
