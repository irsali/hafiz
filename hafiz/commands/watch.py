"""hafiz watch — real-time file watcher with Rich status display."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

from hafiz.core.database import close_engine

logger = logging.getLogger(__name__)
console = Console()


def run_watch(
    path: str,
    *,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    """Start the file watcher with a Rich status display."""
    target = Path(path).resolve()
    if not target.is_dir():
        console.print(f"[red]Not a directory:[/red] {target}")
        raise SystemExit(1)

    from hafiz.core.watcher import start_watcher

    # Activity log for the display
    activity: list[dict] = []
    stats = {"files_reindexed": 0, "chunks_updated": 0, "files_deleted": 0}

    def on_reindex(rel_path: str, chunk_count: int) -> None:
        stats["files_reindexed"] += 1
        stats["chunks_updated"] += chunk_count
        entry = {"action": "reindex", "file": rel_path, "chunks": chunk_count}
        activity.append(entry)
        if len(activity) > 20:
            activity.pop(0)
        if output_json:
            console.print(json.dumps(entry))

    def on_delete(rel_path: str, deleted: int) -> None:
        stats["files_deleted"] += 1
        entry = {"action": "delete", "file": rel_path, "chunks_removed": deleted}
        activity.append(entry)
        if len(activity) > 20:
            activity.pop(0)
        if output_json:
            console.print(json.dumps(entry))

    observer, handler = start_watcher(
        target, project=project, on_reindex=on_reindex, on_delete=on_delete
    )

    def _build_table() -> Table:
        table = Table(title=f"Watching: {target}", show_lines=False)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Project", project or "(default)")
        table.add_row("Files re-indexed", str(stats["files_reindexed"]))
        table.add_row("Chunks updated", str(stats["chunks_updated"]))
        table.add_row("Files deleted", str(stats["files_deleted"]))
        table.add_row("", "")
        table.add_row("[dim]Recent activity[/dim]", "")
        for entry in activity[-10:]:
            if entry["action"] == "reindex":
                table.add_row(
                    f"  [green]+[/green] {entry['file']}",
                    f"{entry['chunks']} chunks",
                )
            else:
                table.add_row(
                    f"  [red]-[/red] {entry['file']}",
                    f"{entry.get('chunks_removed', 0)} removed",
                )
        return table

    if output_json:
        console.print(json.dumps({"status": "watching", "path": str(target), "project": project}))

    try:
        if output_json:
            # In JSON mode, just block until interrupted
            observer.join()
        else:
            console.print(
                f"[bold green]Watching[/bold green] {target} "
                f"(project={project or 'default'})\n"
                f"[dim]Press Ctrl+C to stop[/dim]\n"
            )
            with Live(_build_table(), console=console, refresh_per_second=2) as live:
                while observer.is_alive():
                    live.update(_build_table())
                    observer.join(timeout=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        console.print("\n[yellow]Stopping watcher...[/yellow]")
        observer.stop()
        handler.shutdown()
        observer.join(timeout=5)

        # Close database engine
        asyncio.run(close_engine())

        if output_json:
            console.print(json.dumps({"status": "stopped", **stats}))
        else:
            console.print(
                f"[green]Watcher stopped.[/green] "
                f"Re-indexed {stats['files_reindexed']} files, "
                f"{stats['chunks_updated']} chunks updated."
            )
