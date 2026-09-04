"""hafiz import — agent-transcript importers.

Currently only ``hafiz import claude-code`` is implemented; the
command group is shaped so additional harnesses (Cursor, Copilot)
can slot in additively.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hafiz.core.database import close_engine
from hafiz.core.durations import parse_duration
from hafiz.core.importers.claude_code import (
    DEFAULT_PROJECTS_DIR,
    import_claude_code,
)
from hafiz.core.store import project_for_path

console = Console()


def _resolve_since(since: str | None) -> datetime | None:
    if not since:
        return None
    try:
        delta = parse_duration(since)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)
    return datetime.now(UTC) - delta


def run_import_from_hook(*, output_json: bool = False) -> None:
    """Import exactly the transcript an agent-harness hook is reporting.

    Reads the harness's hook payload as JSON on **stdin** — the same
    channel Claude Code already uses — and imports only that session's
    file. Parsing the payload here rather than in the hook body is what
    keeps the installed hook a portable one-liner: no ``jq`` dependency,
    no shell JSON handling, and the logic is unit-testable.

    Two invariants, both load-bearing for anything wired into a hook:

    * **This never fails the turn.** Every failure path exits 0 and stays
      quiet. A memory layer that can break the conversation gets removed
      by the user, and then it captures nothing at all.
    * **It imports one file, not the whole store.** ``transcript_path``
      names the session, so capture cost stays proportional to the
      session that just ended rather than to every session ever had.
    """
    try:
        raw = sys.stdin.read()
    except Exception:
        raise SystemExit(0) from None
    if not raw or not raw.strip():
        raise SystemExit(0)

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        raise SystemExit(0) from None
    if not isinstance(payload, dict):
        raise SystemExit(0)

    transcript = payload.get("transcript_path")
    if not transcript:
        # Nothing to capture — e.g. a hook event fired before the harness
        # had written a transcript. Not an error.
        raise SystemExit(0)

    target = Path(str(transcript)).expanduser()
    if not target.is_file():
        raise SystemExit(0)

    hook_cwd = payload.get("cwd")

    async def _run():
        try:
            project = await project_for_path(hook_cwd) if hook_cwd else None
            result = await import_claude_code(
                root=target,
                project=project,
                embed=True,
            )
            return result, project
        finally:
            await close_engine()

    try:
        summary, project = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 — hook safety: never fail the turn
        if output_json:
            console.print_json(json.dumps({"ok": False, "error": str(exc)}))
        raise SystemExit(0) from None

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "ok": True,
                    "action": "import_from_hook",
                    "transcript_path": str(target),
                    "project": project,
                    "summary": summary.to_dict(),
                }
            )
        )


def run_import_claude_code(
    path: str | None = None,
    *,
    project: str | None = None,
    limit: int | None = None,
    since: str | None = None,
    dry_run: bool = False,
    no_embed: bool = False,
    output_json: bool = False,
) -> None:
    root = Path(path).expanduser().resolve() if path else DEFAULT_PROJECTS_DIR
    cutoff = _resolve_since(since)

    async def _run():
        try:
            result = await import_claude_code(
                root=root,
                project=project,
                limit=limit,
                since=cutoff,
                dry_run=dry_run,
                embed=not no_embed,
            )
            # Enforce bounded retention opportunistically: this is the command
            # that grows the source layer, so it's the honest place to prune it.
            # Deliberately NOT on `ingest` — that's the code/doc subsystem, it
            # fires per-commit from a hook, and its output goes to /dev/null, so
            # a sweep there would be unattributable and unobservable.
            from hafiz.core.communications import tombstone_expired_communications
            from hafiz.core.telemetry import tombstone_expired_retrievals

            sweep = await tombstone_expired_communications(dry_run=dry_run)
            retr = await tombstone_expired_retrievals(dry_run=dry_run)
            sweep = {
                **sweep,
                "matched": sweep["matched"] + retr["matched"],
                "tombstoned": sweep["tombstoned"] + retr["tombstoned"],
                "retrievals": retr,
            }
            return result, sweep
        finally:
            await close_engine()

    summary, sweep = asyncio.run(_run())

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "action": "import_claude_code",
                    "root": str(root),
                    "project": project,
                    "since": cutoff.isoformat() if cutoff else None,
                    "dry_run": dry_run,
                    "embed": not no_embed,
                    "summary": summary.to_dict(),
                    "retention_sweep": sweep,
                }
            )
        )
        return

    table = Table(title=f"Claude Code import — {root}", border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Count", justify="right")
    s = summary
    table.add_row("Files seen", str(s.files_seen))
    table.add_row("Files skipped", str(s.files_skipped))
    table.add_row("Communications created", str(s.communications_created))
    table.add_row("Communications already present", str(s.communications_existing))
    table.add_row("Sessions created", str(s.sessions_created))
    table.add_row("Messages written", str(s.messages_written))
    table.add_row("Messages embedded", str(s.messages_embedded))
    if s.errors:
        table.add_row("Errors", str(len(s.errors)))
    if sweep["matched"]:
        table.add_row(
            "Retention-expired tombstoned",
            str(sweep["tombstoned"]) + (" (dry run)" if dry_run else ""),
        )
    console.print(table)

    if dry_run:
        console.print("[yellow]Dry run — no rows written.[/yellow]")
    if s.errors:
        console.print()
        err_panel = Panel(
            "\n".join(f"[red]{e['path']}[/red]: {e['error']}" for e in s.errors[:10]),
            title="Errors (first 10)",
            border_style="red",
        )
        console.print(err_panel)
