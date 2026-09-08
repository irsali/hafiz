"""``hafiz migrate-backend`` — move a whole store between backends.

Presentation only. Everything real is in :mod:`hafiz.core.migrate`.
"""

from __future__ import annotations

import asyncio
import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hafiz.core.config import get_settings
from hafiz.core.migrate import MigrationError, migrate_backend
from hafiz.core.redaction import redact_credentials

console = Console()


def run_migrate_backend(
    target: str, *, dry_run: bool = False, output_json: bool = False, assume_yes: bool = False
) -> None:
    source = get_settings().database.url

    if not (dry_run or output_json or assume_yes) and not _confirm(source, target):
        console.print("[dim]Nothing was copied.[/dim]")
        raise typer.Exit(code=1)

    try:
        result = asyncio.run(migrate_backend(source_url=source, target_url=target, dry_run=dry_run))
    except MigrationError as e:
        # Migration failures wrap driver errors, which carry the connection
        # URL — this is the command whose traceback was found holding a live
        # password in the error log.
        message = redact_credentials(str(e))
        if output_json:
            print(json.dumps({"ok": False, "error": message}, indent=2))
        else:
            console.print(Panel(message, title="Migration stopped", border_style="red"))
        raise typer.Exit(code=1) from e

    if output_json:
        print(json.dumps(result.to_dict(), indent=2))
        if not result.ok:
            raise typer.Exit(code=1)
        return

    table = Table(
        title=(
            f"{'Would copy' if result.dry_run else 'Copied'} — "
            f"{result.source_backend} → {result.target_backend}"
        ),
        border_style="cyan",
    )
    table.add_column("Table")
    table.add_column("Rows", justify="right")
    if not result.dry_run:
        table.add_column("Back-refs", justify="right")
    for entry in result.tables:
        row = [entry.name, f"{entry.copied:,}"]
        if not result.dry_run:
            row.append(f"{entry.back_references:,}" if entry.back_references else "")
        table.add_row(*row)
    console.print()
    console.print(table)
    console.print(f"  [bold]{result.total_rows:,}[/bold] rows total")

    if result.dry_run:
        console.print(
            "\n  [dim]Dry run — nothing was written. Re-run without --dry-run to copy.[/dim]"
        )
        return

    if result.vector_check:
        console.print(f"  [green]Vectors verified[/green] [dim]({result.vector_check})[/dim]")

    # `target_url` is left verbatim on purpose: this is a command to copy and
    # run, and `hafiz config set database.url postgresql://u:***@h/db` would
    # write a literal `***` as the password. It is the argument the user just
    # supplied, not a secret being disclosed to them.
    console.print(
        f"\n  [green]Migration complete.[/green] The source database was not modified.\n"
        f"  [dim]Point hafiz at the new store:  hafiz config set database.url {result.target_url}\n"
        f"  Then check it over:            hafiz status --diagnose[/dim]"
    )


def _confirm(source: str, target: str) -> bool:
    console.print(
        Panel(
            # `source` comes from config — it is the credential the user may
            # not remember and the one that shows up in shared diagnostics, so
            # it is redacted. `target` is what they typed on this command line
            # seconds ago; masking it would stop them verifying the thing they
            # are being asked to confirm, and reveals nothing new.
            f"Copy every row from\n  [bold]{redact_credentials(source)}[/bold]\n"
            f"into\n  [bold]{target}[/bold]\n\n"
            "The source is opened read-only and will not be modified.\n"
            "The target must be empty; this copies, it does not merge.",
            title="Migrate backend",
            border_style="yellow",
        )
    )
    return typer.confirm("Proceed?", default=False)
