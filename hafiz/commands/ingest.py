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
from hafiz.core.git_context import (
    changed_files_since,
    current_git_context,
    git_operation_in_progress,
    is_git_repo,
)
from hafiz.core.store import (
    index_file,
    latest_indexed_commit,
    reconcile_orphaned_commits,
    tombstone_vanished_files,
    upsert_commit,
)

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

    # ── Step 0: git-axis — race safety + resolve HEAD + diff base ────────
    repo_root = target if target.is_dir() else target.parent
    in_flight = (
        git_operation_in_progress(repo_root) if is_git_repo(repo_root) else None
    )
    if in_flight:
        msg = (
            f"Git operation in progress ({in_flight}); refusing to ingest an "
            f"intermediate tree state. Finish the rebase / merge / cherry-pick "
            f"and re-run."
        )
        if output_json:
            _emit({"event": "error", "code": "git_in_progress", "message": msg})
        else:
            console.print(f"[red]{msg}[/red]")
        raise SystemExit(2)

    git_ctx = current_git_context(repo_root) if is_git_repo(repo_root) else {}
    head_sha: str | None = git_ctx.get("commit_hash") or None

    # Diff-driven scope: if the project has been ingested before at a
    # commit reachable from HEAD, restrict this pass to changed files.
    # Otherwise walk everything (first-time ingest, branch-switched to
    # an unrelated line, or rebase orphaned the old base).
    diff_scope: set[Path] | None = None
    base_sha: str | None = None
    if head_sha and project is not None:
        base_sha = await latest_indexed_commit(project)
        if base_sha and base_sha != head_sha:
            diff_scope = changed_files_since(base_sha, repo_root)

    # ── Step 1: enumerate files ───────────────────────────────────────────
    if output_json:
        _emit(
            {
                "event": "walk",
                "status": "start",
                "path": str(target),
                "head": head_sha,
                "base": base_sha,
                "diff_driven": diff_scope is not None,
            }
        )
    else:
        walk_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        )
        walk_ctx.__enter__()
        if diff_scope is not None:
            walk_ctx.add_task(
                f"Walking files (diff-driven, base {base_sha[:8]})...",
                total=None,
            )
        else:
            walk_ctx.add_task("Walking files...", total=None)

    files = list(walk_files(target, ignore_patterns=ignore_patterns))
    if diff_scope is not None:
        files = [f for f in files if f.resolve() in diff_scope]

    if not output_json:
        walk_ctx.__exit__(None, None, None)

    if not files:
        # Still record the commit — "we're at HEAD" is useful information
        # even when the diff is empty.
        if head_sha:
            await upsert_commit(head_sha, project=project, cwd=repo_root)
        if output_json:
            _emit(
                {
                    "event": "complete",
                    "files": 0,
                    "revisions": 0,
                    "diff_driven": diff_scope is not None,
                    "head": head_sha,
                }
            )
        else:
            if diff_scope is not None:
                console.print(
                    f"[dim]No changed files since {base_sha[:8]} — "
                    f"index is up-to-date at HEAD.[/dim]"
                )
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
                        commit_hash=head_sha,
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

    # ── Step 3a: tombstone vanished files (project-scoped + full walk only) ─
    # When diff-driven, we only looked at changed files, so we can't know
    # what's vanished. Skip tombstoning in that case.
    if project is not None and diff_scope is None:
        files_tombstoned = await tombstone_vanished_files(project, seen_paths)
    else:
        files_tombstoned = 0

    # ── Step 3b: record the ingest commit in the `commits` table ──────────
    if head_sha:
        await upsert_commit(head_sha, project=project, cwd=repo_root)

    # ── Step 3c: reconcile orphaned commits (Phase 5b belt-and-braces) ────
    # If a rebase happened since last ingest without the post-rewrite hook
    # firing, stale commit rows get marked `rewritten_at=now` so they stop
    # looking like live history.
    reconciled = (
        await reconcile_orphaned_commits(project, repo_root)
        if head_sha
        else 0
    )

    # ── Step 4: summary ──────────────────────────────────────────────────
    if output_json:
        _emit({
            "event": "complete",
            **totals,
            "files_tombstoned": files_tombstoned,
            "commits_reconciled": reconciled,
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
        if reconciled:
            console.print(
                f"  [dim]Marked {reconciled} rewritten commits as orphaned[/dim]"
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
