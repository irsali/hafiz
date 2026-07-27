"""hafiz reconcile — find clusters of near-duplicate live annotations.

The after-the-fact backstop to write-time near-duplicate detection. Read-only:
it surfaces drift (duplicates that slipped through bulk writes or predate
detection) and proposes a resolution, but never applies one. Hafiz reports
*similarity*; the operator decides what's a real contradiction and runs the
command themselves.
"""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.panel import Panel

from hafiz.core.database import close_engine

console = Console()

#: Enough of a record to tell two near-duplicates apart in a terminal.
_PREVIEW = 100


def _preview(content: str) -> str:
    """One line of the record. Annotations are multi-line prose; left raw they
    break the panel into ragged fragments that are harder to compare, which is
    the only thing this view is for."""
    flat = " ".join(content.split())
    return flat[:_PREVIEW] + ("…" if len(flat) > _PREVIEW else "")


def _shell_quote(text: str) -> str:
    """Single-quote for a shell, so a pasted command survives its own content."""
    return "'" + text.replace("'", "'\\''") + "'"


def _commands_for(cluster) -> list[str]:
    """The resolution to run, as literal commands, in order.

    Both actions retire everything but the primary. They differ only in what
    happens to the primary itself: under ``retire`` it survives as written;
    under ``merge`` the operator first writes text that supersedes it, because
    the newest row is shorter than this one and dropping it would lose whatever
    only it says. The ``<merged text>`` placeholder is deliberate — hafiz does
    not call an LLM, and guessing the merge is exactly the lossy step this
    branch exists to prevent.

    One ``observe`` at most: ``--supersedes`` takes a single id, so a merge is
    "supersede the primary, retire the rest" rather than one new row per member.
    """
    cmds = []
    if cluster.action == "merge":
        primary = cluster.primary
        cmds.append(
            f"hafiz observe '<merged text>' --type {cluster.kind}"
            + (f" --project {_shell_quote(cluster.project)}" if cluster.project else "")
            + (f" --source {primary.source}" if primary.source else "")
            + f" --supersedes {primary.id}"
        )
    cmds += [f"hafiz forget {m.id} --annotation" for m in cluster.others]
    return cmds


def run_reconcile(
    *,
    project: str | None = None,
    kind: str | None = None,
    threshold: float | None = None,
    limit: int | None = None,
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

    report = asyncio.run(_run())
    clusters = report.clusters

    if output_json:
        data = {
            "action": "reconcile",
            "scanned": report.scanned,
            "total_live": report.total_live,
            "truncated": report.truncated,
            "threshold": report.threshold,
            "clusters": [
                {
                    "kind": c.kind,
                    "project": c.project,
                    "suggested_action": c.action,
                    "primary_id": c.primary.id,
                    "members": [
                        {
                            "id": m.id,
                            "content": m.content,
                            "score": m.score,
                            "source": m.source,
                            "valid_from": m.valid_from.isoformat(),
                            "chars": len(m.content),
                            "primary": m.primary,
                        }
                        for m in c.members
                    ],
                    "commands": _commands_for(c),
                }
                for c in clusters
            ],
            "total": len(clusters),
        }
        console.print_json(json.dumps(data))
        return

    coverage = f"Scanned {report.scanned} of {report.total_live} live annotations"
    if report.truncated:
        coverage += " [yellow](truncated — pass --limit 0 to sweep all)[/yellow]"

    if not clusters:
        console.print(
            f"[green]No near-duplicate live annotations found.[/green] [dim]{coverage}[/dim]"
        )
        return

    console.print()
    console.print(
        f"[bold]Found {len(clusters)} cluster(s) of near-duplicate live annotations.[/bold]"
    )
    console.print(f"[dim]{coverage} · threshold {report.threshold:.2f}[/dim]")
    console.print("[dim]Nothing below has been changed. Run a command to resolve it.[/dim]\n")

    for i, c in enumerate(clusters, 1):
        lines = [
            f"[bold yellow]Cluster {i}[/bold yellow]  "
            f"[dim]kind={c.kind} · project={c.project or '—'} · suggested: {c.action}[/dim]",
            "",
        ]
        primary_label = "[green]KEEP  [/green]" if c.action == "retire" else "[cyan]MERGE [/cyan]"
        for m in c.members:
            label = primary_label if m.primary else "[red]RETIRE[/red]"
            lines.append(
                f"  {label} [cyan]{m.id}[/cyan]  [dim]({m.score:.0%} · "
                f"{len(m.content)} chars · {m.valid_from:%Y-%m-%d})[/dim]"
            )
            lines.append(f"         {_preview(m.content)}")
        lines.append("")
        if c.action == "merge":
            lines.append(
                "  [yellow]The newest row is the shorter one, so it is not safe to keep on its"
                "\n  own — write text that carries both, then retire the rest:[/yellow]"
            )
        for cmd in _commands_for(c):
            lines.append(f"  [bold]{cmd}[/bold]")
        console.print(Panel("\n".join(lines), border_style="yellow"))

    console.print(
        "[dim]Suggestions only, and nothing here is a contradiction check — hafiz measures"
        " similarity.\nRead both rows before running anything.[/dim]"
    )
