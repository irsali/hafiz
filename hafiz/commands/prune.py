"""hafiz prune — retained as a guarded no-op.

Under the v5 pipeline, stale-file cleanup is automatic: every ``hafiz ingest``
runs :func:`hafiz.core.store.tombstone_vanished_files`, which tombstones any
file under the project whose path wasn't seen in the pass (its units cascade).
There is no separate prune step to run.

The command is kept — rather than removed — so installed hooks and scripts that
call ``hafiz prune`` keep working. It reports that cleanup is handled on ingest
and exits 0. ``--json`` and ``--dry-run`` are preserved for shape stability.
"""

from __future__ import annotations

import json

from rich.console import Console

console = Console()

_MESSAGE = (
    "Stale-file cleanup is automatic on ingest (tombstone_vanished_files). "
    "There is nothing to prune separately — re-run `hafiz ingest` to reconcile."
)


def run_prune(
    project: str | None = None,
    dry_run: bool = False,
    output_json: bool = False,
) -> None:
    """No-op cleanup command. Cleanup happens on ingest; this just reports."""
    if output_json:
        console.print(
            json.dumps(
                {
                    "action": "prune",
                    "noop": True,
                    "reason": "handled-on-ingest",
                    "project": project,
                    "dry_run": dry_run,
                    "message": _MESSAGE,
                },
                indent=2,
            )
        )
        return

    console.print(f"[green]Nothing to prune.[/green] {_MESSAGE}")
