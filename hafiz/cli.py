"""Hafiz CLI — the sovereign intelligence layer for your workspace.

Entry point for the `hafiz` command. Built with Typer + Rich.
"""

from __future__ import annotations

from typing import Optional

import typer

from hafiz import __version__

app = typer.Typer(
    name="hafiz",
    help="Hafiz — sovereign intelligence layer for your workspace.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"hafiz {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Hafiz — sovereign intelligence layer for your workspace."""


# ─── INIT ───────────────────────────────────────────────────────────────

@app.command()
def init() -> None:
    """Initialize the Hafiz database (create tables + pgvector extension)."""
    from hafiz.commands.maintenance import run_init

    run_init()


# ─── INGEST ─────────────────────────────────────────────────────────────

@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory to index."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Tag chunks with a project name."
    ),
    no_extract: bool = typer.Option(
        False, "--no-extract", help="Skip entity/relationship extraction."
    ),
) -> None:
    """Index files into the Hafiz knowledge base."""
    from hafiz.commands.ingest import run_ingest

    run_ingest(path, project=project, no_extract=no_extract)


# ─── QUERY ──────────────────────────────────────────────────────────────

@app.command()
def query(
    text: str = typer.Argument(..., help="Search query text."),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output results as JSON (for agents)."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter results by project."
    ),
    type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by chunk type (code, doc, note, decision)."
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of results."
    ),
) -> None:
    """Search indexed content with vector similarity."""
    from hafiz.commands.query import _run_query

    _run_query(text, limit=limit, project=project, chunk_type=type, output_json=json_output)


# ─── STATUS ─────────────────────────────────────────────────────────────

@app.command()
def status(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Show database statistics and index health."""
    from hafiz.commands.maintenance import run_status

    run_status(output_json=json_output)


# ─── DOCTOR ────────────────────────────────────────────────────────────

@app.command()
def doctor(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Run diagnostic checks on the Hafiz installation."""
    from hafiz.commands.maintenance import run_doctor

    run_doctor(output_json=json_output)


# ─── CONFIG ─────────────────────────────────────────────────────────────

config_app = typer.Typer(name="config", help="Configuration management.")
app.add_typer(config_app)


@config_app.command("show")
def config_show(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Show current Hafiz configuration."""
    from hafiz.commands.maintenance import run_config_show

    run_config_show(output_json=json_output)


# ─── GRAPH ─────────────────────────────────────────────────────────────

graph_app = typer.Typer(name="graph", help="Explore the knowledge graph.")
app.add_typer(graph_app)


@graph_app.command("show")
def graph_show(
    name: str = typer.Argument(..., help="Entity name to look up."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Show an entity and its direct connections."""
    from hafiz.commands.graph import run_graph_show

    run_graph_show(name, project=project, output_json=json_output)


@graph_app.command("deps")
def graph_deps(
    name: str = typer.Argument(..., help="Entity name to look up."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Show what an entity depends on (outgoing relations)."""
    from hafiz.commands.graph import run_graph_deps

    run_graph_deps(name, project=project, output_json=json_output)


@graph_app.command("dependents")
def graph_dependents(
    name: str = typer.Argument(..., help="Entity name to look up."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Show what depends on an entity (incoming relations)."""
    from hafiz.commands.graph import run_graph_dependents

    run_graph_dependents(name, project=project, output_json=json_output)
