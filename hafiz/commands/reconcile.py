"""hafiz reconcile — find clusters of near-duplicate live annotations.

The after-the-fact backstop to write-time near-duplicate detection. Read-only:
it surfaces drift (duplicates that slipped through bulk writes or predate
detection) and leaves resolution to an explicit ``observe --supersedes`` or
``forget``. Hafiz reports *similarity*; the operator decides what's a real
contradiction.
"""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.panel import Panel

from hafiz.core.database import close_engine

console = Console()


def run_reconcile(
    *,
    project: str | None = None,
    kind: str | None = None,
    threshold: float | None = None,
    limit: int = 500,
    output_json: bool = False,
) -> None:
    """Surface clusters of near-duplicate live annotations for manual review."""

    async def _run():
        try:
            from hafiz.core.annotations import reconcile_duplicates

            return await reconcile_duplicates(
                project=project, kind=kind, threshold=threshold, limit=limit
            )
        finally:
            await close_engine()

    clusters = asyncio.run(_run())

    if output_json:
        data = {
            "action": "reconcile",
            "clusters": [
                {
                    "kind": c.kind,
                    "project": c.project,
                    "members": [
                        {"id": m.id, "content": m.content, "score": m.score} for m in c.members
                    ],
                }
                for c in clusters
            ],
            "total": len(clusters),
        }
        console.print_json(json.dumps(data))
        return

    if not clusters:
        console.print("[green]No near-duplicate live annotations found.[/green]")
        return

    console.print()
    console.print(
        f"[bold]Found {len(clusters)} cluster(s) of near-duplicate live annotations.[/bold]"
    )
    console.print(
        '[dim]Resolve with: hafiz observe "<text>" --supersedes <id>  ·  '
        "hafiz forget <id> --annotation[/dim]\n"
    )

    for i, c in enumerate(clusters, 1):
        lines = [
            f"[bold yellow]Cluster {i}[/bold yellow]  "
            f"[dim]kind={c.kind} · project={c.project or '—'}[/dim]",
            "",
        ]
        for m in c.members:
            preview = m.content[:90] + ("…" if len(m.content) > 90 else "")
            lines.append(f"  [cyan]{m.id}[/cyan]  [dim]({m.score:.0%})[/dim]  {preview}")
        console.print(Panel("\n".join(lines), border_style="yellow"))
