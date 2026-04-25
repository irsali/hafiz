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
    limit_annotations: int = 5,
    include_transcripts: bool = False,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    output_json: bool = False,
) -> None:
    """Build and display a context bundle for a task description.

    When ``include_transcripts`` is set, top source-layer messages
    matching the query are appended to the bundle under a separate
    ``transcripts`` key (JSON) / "Transcripts" panel (rich) — opt-in
    by design; the wisdom layer stays primary.
    """

    async def _build():
        try:
            if workspace:
                from hafiz.core.context import (
                    build_workspace_context,
                    resolve_workspace_projects,
                )

                projects = await resolve_workspace_projects()
                if not projects:
                    console.print(
                        "[yellow]No workspace-sibling projects found in the index. "
                        "Falling back to all projects.[/yellow]"
                    )

                bundle = await build_workspace_context(
                    query,
                    projects=projects,
                    limit_chunks=limit_chunks * 2,
                    limit_annotations=limit_annotations * 2,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                )
            else:
                from hafiz.core.context import build_context

                bundle = await build_context(
                    query,
                    project=project,
                    limit_chunks=limit_chunks,
                    limit_annotations=limit_annotations,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                )
            transcripts = []
            if include_transcripts:
                from hafiz.core.communications import search_messages

                transcripts = await search_messages(
                    query, limit=limit_chunks
                )
            return bundle, transcripts
        finally:
            await close_engine()

    bundle, transcripts = asyncio.run(_build())

    if output_json:
        payload = bundle.to_dict()
        payload["transcripts"] = [
            {
                "id": msg.id,
                "communication_id": msg.communication_id,
                "seq": msg.seq,
                "role": msg.role,
                "author": msg.author,
                "content": msg.content,
                "ts": msg.ts.isoformat(),
                "score": score,
            }
            for msg, score in transcripts
        ]
        payload["include_transcripts"] = include_transcripts
        console.print_json(json.dumps(payload, default=str))
        return

    title = (
        f"Context (workspace): {query[:50]}"
        if workspace
        else f"Context: {query[:60]}"
    )
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

    summary = (
        f"  [dim]Chunks: {len(bundle.chunks)} | "
        f"Units: {len(bundle.entities)} | "
        f"Annotations: {len(bundle.annotations)}"
    )
    if bundle.project_distribution:
        projects_involved = len(bundle.project_distribution)
        summary += f" | Projects: {projects_involved}"
    if include_transcripts:
        summary += f" | Transcripts: {len(transcripts)}"
    summary += "[/dim]"
    console.print(summary)
    console.print()

    if transcripts:
        ts_lines = []
        for msg, score in transcripts:
            preview = msg.content[:160].replace("\n", " ").strip()
            if len(msg.content) > 160:
                preview += "…"
            ts_lines.append(
                f"  [bold]{msg.role}[/bold] seq {msg.seq}  "
                f"[dim]({score:.2%})[/dim]"
            )
            ts_lines.append(f"  [dim]{preview}[/dim]")
            ts_lines.append("")
        console.print(
            Panel(
                "\n".join(ts_lines),
                title=(
                    f"Source-layer transcripts ({len(transcripts)} matches, "
                    f"opt-in via --include-transcripts)"
                ),
                border_style="magenta",
            )
        )
        console.print()
