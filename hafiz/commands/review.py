"""hafiz review — analyze knowledge quality and suggest improvements."""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from hafiz.core.database import close_engine

console = Console()


def run_review(
    *,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    """Run a self-review of the hafiz knowledge base."""

    async def _review():
        try:
            from hafiz.core.review import run_review as _run

            report = await _run(project=project)
            return report
        finally:
            await close_engine()

    report = asyncio.run(_review())

    if output_json:
        console.print_json(json.dumps(report.to_dict(), default=str))
        return

    console.print()
    md = Markdown(report.to_markdown())
    title = f"Review: {project}" if project else "Review: all projects"
    console.print(
        Panel(
            md,
            title=title,
            border_style="yellow",
            padding=(1, 2),
        )
    )
    console.print()

    summary = report.to_dict()["summary"]
    console.print(
        f"  [dim]Findings: {summary['total']} "
        f"({summary['warnings']} warnings, "
        f"{summary['suggestions']} suggestions, "
        f"{summary['info']} info)[/dim]"
    )
    console.print()
