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

    Every action retires everything but the primary. They differ only in what
    happens to the primary itself: under ``retire`` it survives as written,
    because it was verified to contain every word its siblings carry. Under
    ``merge`` — and under ``review``, which is a merge whose unmatched text is
    short enough that a glance may downgrade it — no row contains the others, so
    the operator first writes text that supersedes the primary.

    ``review`` deliberately gets the *merge* commands rather than a ready-made
    ``forget``. Merging text that turned out to be mere rewording costs a
    redundant rewrite; retiring text that turned out to matter costs the fact.
    The panel names the cheaper path in prose instead of pasting it.

    The ``<merged text>`` placeholder is deliberate — hafiz does not call an LLM,
    and guessing the merge is exactly the lossy step this command exists to
    prevent.

    One ``observe`` at most: ``--supersedes`` takes a single id, so a merge is
    "supersede the primary, retire the rest" rather than one new row per member.
    """
    cmds = []
    if cluster.action in ("merge", "review"):
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
                            "overlap": m.overlap,
                            "unique_fragments": m.unique_fragments,
                            "unique_words": m.unique_words,
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
        primary_label = {
            "retire": "[green]KEEP  [/green]",
            "review": "[yellow]CHECK [/yellow]",
        }.get(c.action, "[cyan]MERGE [/cyan]")
        for m in c.members:
            label = primary_label if m.primary else "[red]RETIRE[/red]"
            # Both numbers, always: cosine is why these are grouped, overlap is
            # why the proposal is what it is, and the two disagree often enough
            # that showing only one invites the wrong action.
            metrics = f"{m.score:.0%} sim"
            if not m.primary:
                # The verdict, not just the ratio. A row fully contained in a
                # much longer keeper scores a *low* ratio (SequenceMatcher is
                # 2·matches/total, so length asymmetry drags it down) while
                # losing nothing — one real pair sits at 40% and is a strict
                # subset. Showing the ratio alone would read as "mostly
                # different" and argue against the safe action.
                # Keyed on unique_words, the count the decision itself uses, so
                # the verdict shown can never disagree with the tier assigned.
                if not m.unique_words:
                    verdict = "[green]nothing only here[/green]"
                else:
                    verdict = f"[yellow]{m.unique_words} word(s) only here[/yellow]"
                metrics += f" · {m.overlap:.0%} same words · {verdict}"
            lines.append(
                f"  {label} [cyan]{m.id}[/cyan]  [dim]({metrics} · "
                f"{len(m.content)} chars · {m.valid_from:%Y-%m-%d})[/dim]"
            )
            lines.append(f"         {_preview(m.content)}")
            # The whole point: what dies if this row is retired. Under `review`
            # the runs are single words, so one per line spends three lines
            # saying "a", "1", "a s" — joined, they read as the wording drift
            # they are. Under `merge` they are whole clauses and each earns a
            # line.
            if m.unique_fragments and c.action == "review":
                joined = " · ".join(m.unique_fragments)
                lines.append(f"         [yellow]only here:[/yellow] [dim]{_preview(joined)}[/dim]")
            else:
                for frag in m.unique_fragments[:3]:
                    lines.append(
                        f"         [yellow]only here:[/yellow] [dim]{_preview(frag)}[/dim]"
                    )
                if len(m.unique_fragments) > 3:
                    lines.append(
                        f"         [dim]…and {len(m.unique_fragments) - 3} more unique fragment(s)"
                        "[/dim]"
                    )
        lines.append("")
        if c.action == "merge":
            lines.append(
                "  [yellow]No row here contains the others — each keeps something of its own"
                "\n  (see 'only here' above). Write text that carries all of it, then retire"
                "\n  the rest:[/yellow]"
            )
        elif c.action == "review":
            lines.append(
                "  [yellow]These differ only by the stray words above. Usually that is a"
                "\n  contraction or a spelling — occasionally it is a reversal that flips"
                "\n  the meaning, which is why no plain retire is pre-pasted here. Read the"
                "\n  words. If they are rewording, retire the RETIRE ids above directly;"
                "\n  otherwise:[/yellow]"
            )
        for cmd in _commands_for(c):
            lines.append(f"  [bold]{cmd}[/bold]")
        console.print(Panel("\n".join(lines), border_style="yellow"))

    console.print(
        "[dim]Suggestions only, and nothing here is a contradiction check — hafiz measures"
        " similarity.\nRead both rows before running anything.[/dim]"
    )
