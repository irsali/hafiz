"""hafiz ingest — walk a path, parse each file via the Parser Registry,
and write units / revisions / embeddings to the DB.

Hash-aware and idempotent: unchanged files produce zero new revisions
and zero new embeddings; changed files re-embed only the units whose
bodies actually changed. Vanished files and vanished units get
tombstoned (``valid_until`` set) so search skips them while history
remains intact.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)

from hafiz.core.chunker import walk_files
from hafiz.core.config import get_settings
from hafiz.core.database import close_engine, get_session_factory
from hafiz.core.store import index_file, tombstone_vanished_files

logger = logging.getLogger(__name__)
console = Console()


def _emit(event: dict) -> None:
    """Write a JSON event to stdout and flush immediately."""
    print(json.dumps(event), flush=True)


def run_ingest(
    path: str,
    *,
    project: str | None = None,
    prune: bool = False,  # retained for CLI compat; the new pipeline
                          # tombstones on-the-fly, so explicit prune is a
                          # no-op here. Kept so `--prune` still parses.
    output_json: bool = False,
) -> None:
    """Run the ingestion pipeline for a path."""

    async def _ingest():
        try:
            return await _do_ingest(
                path, project=project, output_json=output_json
            )
        finally:
            await close_engine()

    asyncio.run(_ingest())


async def _do_ingest(
    path: str,
    *,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    target = Path(path).resolve()
    settings = get_settings()
    ignore_patterns = settings.workspace.ignore

    if not target.exists():
        if output_json:
            _emit({"event": "error", "message": f"Path not found: {target}"})
        else:
            console.print(f"[red]Path not found:[/red] {target}")
        raise SystemExit(1)

    # ── Step 1: enumerate files ───────────────────────────────────────────
    if output_json:
        _emit({"event": "walk", "status": "start", "path": str(target)})
    else:
        walk_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        )
        walk_ctx.__enter__()
        walk_ctx.add_task("Walking files...", total=None)

    files = list(walk_files(target, ignore_patterns=ignore_patterns))

    if not output_json:
        walk_ctx.__exit__(None, None, None)

    if not files:
        if output_json:
            _emit({"event": "complete", "files": 0, "revisions": 0})
        else:
            console.print("[yellow]No files found to index.[/yellow]")
        return

    if output_json:
        _emit({
            "event": "walk",
            "status": "done",
            "files": len(files),
        })
    else:
        console.print(
            f"Found [bold]{len(files)}[/bold] files under [bold]{target}[/bold]"
        )

    # ── Step 2: parse + embed + store, one file per transaction ──────────
    session_factory = get_session_factory()
    totals = {
        "files_processed": 0,
        "units_seen": 0,
        "revisions_created": 0,
        "embeddings_written": 0,
        "units_tombstoned": 0,
    }
    seen_paths: set[str] = set()
    failures: list[tuple[str, str]] = []

    if output_json:
        _emit({"event": "index", "status": "start", "total": len(files)})
    else:
        prog = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )
        prog.__enter__()
        task = prog.add_task("Indexing files...", total=len(files))

    for file_path in files:
        seen_paths.add(str(file_path))
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError) as e:
            failures.append((str(file_path), f"read error: {e}"))
            if not output_json:
                prog.update(task, advance=1)
            continue

        try:
            async with session_factory() as session:
                async with session.begin():
                    result = await index_file(
                        file_path,
                        content,
                        project=project,
                        session=session,
                    )
        except Exception as e:
            logger.exception("Failed to index %s", file_path)
            failures.append((str(file_path), f"{type(e).__name__}: {e}"))
            if not output_json:
                prog.update(task, advance=1)
            continue

        totals["files_processed"] += 1
        totals["units_seen"] += result.units_seen
        totals["revisions_created"] += result.revisions_created
        totals["embeddings_written"] += result.embeddings_written
        totals["units_tombstoned"] += result.units_tombstoned

        if output_json:
            _emit({
                "event": "index",
                "status": "progress",
                "path": str(file_path),
                "parser": result.parser_name,
                "units_seen": result.units_seen,
                "revisions_created": result.revisions_created,
                "embeddings_written": result.embeddings_written,
            })
        else:
            prog.update(task, advance=1)

    if not output_json:
        prog.__exit__(None, None, None)

    # ── Step 3: tombstone vanished files (project-scoped) ────────────────
    if project is not None:
        files_tombstoned = await tombstone_vanished_files(project, seen_paths)
    else:
        files_tombstoned = 0  # cross-project tombstoning is unsafe

    # ── Step 4: summary ──────────────────────────────────────────────────
    if output_json:
        _emit({
            "event": "complete",
            **totals,
            "files_tombstoned": files_tombstoned,
            "failures": [
                {"path": p, "error": e} for p, e in failures
            ],
        })
    else:
        console.print(
            f"[green]Indexed {totals['files_processed']} files[/green] — "
            f"{totals['units_seen']} units, "
            f"{totals['revisions_created']} new revisions, "
            f"{totals['embeddings_written']} new embeddings"
        )
        if totals["units_tombstoned"]:
            console.print(
                f"  [dim]Tombstoned {totals['units_tombstoned']} vanished units[/dim]"
            )
        if files_tombstoned:
            console.print(
                f"  [dim]Tombstoned {files_tombstoned} vanished files[/dim]"
            )
        if failures:
            console.print(
                f"  [yellow]{len(failures)} file(s) failed to index:[/yellow]"
            )
            for p, e in failures[:5]:
                console.print(f"    [dim]{p}: {e}[/dim]")
            if len(failures) > 5:
                console.print(f"    [dim]... and {len(failures) - 5} more[/dim]")


def run_git_hook_ingest_cmd(*, project: str | None = None) -> None:
    """Git-hook-based ingest: only files changed in the latest commit.

    Phase 5 of the structural-grounding work rewires this for diff-based
    delta ingest. Until then, fall back to a full ingest of the cwd so
    the hook keeps working.
    """
    console.print(
        "[yellow]git-hook ingest currently runs a full ingest. "
        "Phase 5 introduces proper diff-based delta ingest.[/yellow]"
    )
    run_ingest(".", project=project)
