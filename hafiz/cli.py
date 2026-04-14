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
    path: Optional[str] = typer.Argument(None, help="Path to file or directory to index."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Tag chunks with a project name."
    ),
    no_extract: bool = typer.Option(
        False, "--no-extract", help="Skip entity/relationship extraction."
    ),
    git_hook: bool = typer.Option(
        False, "--git-hook", help="Index only files changed in the latest commit."
    ),
) -> None:
    """Index files into the Hafiz knowledge base."""
    if git_hook:
        from hafiz.commands.ingest import run_git_hook_ingest_cmd

        run_git_hook_ingest_cmd(project=project)
    else:
        if path is None:
            typer.echo("Error: Missing argument 'PATH'. Use --git-hook or provide a path.")
            raise typer.Exit(1)
        from hafiz.commands.ingest import run_ingest

        run_ingest(path, project=project, no_extract=no_extract)


# ─── WATCH ──────────────────────────────────────────────────────────

@app.command()
def watch(
    path: str = typer.Argument(..., help="Directory to watch for changes."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Tag indexed chunks with a project name."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output events as JSON (for agents)."
    ),
) -> None:
    """Watch a directory and re-index files on change (real-time)."""
    from hafiz.commands.watch import run_watch

    run_watch(path, project=project, output_json=json_output)


# ─── PRUNE ──────────────────────────────────────────────────────────

@app.command()
def prune(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="List stale files without deleting."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Remove chunks for files that no longer exist on disk."""
    from hafiz.commands.prune import run_prune

    run_prune(project=project, dry_run=dry_run, output_json=json_output)


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


# ─── OBSERVE ──────────────────────────────────────────────────────────

@app.command()
def observe(
    text: str = typer.Argument(..., help="The observation text to store."),
    obs_type: str = typer.Option(
        "fact", "--type", "-t", help="Type: fact, decision, learning, pattern, warning."
    ),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Origin (e.g. agent:bilal, user:manual)."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Tag with a project name."
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", help="Comma-separated tags."
    ),
    confidence: float = typer.Option(
        1.0, "--confidence", "-c", help="Confidence score 0.0–1.0."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Store a fact, decision, or learning as an observation."""
    from hafiz.commands.observe import run_observe

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    run_observe(
        text,
        obs_type=obs_type,
        source=source,
        project=project,
        tags=tag_list,
        confidence=confidence,
        output_json=json_output,
    )


# ─── RECALL ───────────────────────────────────────────────────────────

@app.command()
def recall(
    query: str = typer.Argument(..., help="Search query for observations."),
    obs_type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by observation type."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of results."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Recall observations by semantic similarity."""
    from hafiz.commands.observe import run_recall

    run_recall(
        query,
        limit=limit,
        project=project,
        obs_type=obs_type,
        output_json=json_output,
    )


# ─── CONTEXT ──────────────────────────────────────────────────────────

@app.command()
def context(
    query: str = typer.Argument(..., help="Task description or question."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Synthesize relevant code, graph, and observations for a task."""
    from hafiz.commands.context import run_context

    run_context(query, project=project, output_json=json_output)


# ─── CHUNKS ────────────────────────────────────────────────────────

chunks_app = typer.Typer(name="chunks", help="Manage indexed chunks.")
app.add_typer(chunks_app)


@chunks_app.command("export")
def chunks_export(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    path: Optional[str] = typer.Option(
        None, "--path", help="Filter by source-file path prefix."
    ),
    limit: int = typer.Option(
        200, "--limit", "-l", help="Maximum chunks to export."
    ),
    offset: int = typer.Option(
        0, "--offset", help="Skip the first N chunks."
    ),
) -> None:
    """Export indexed chunks as JSON (for agent-driven extraction)."""
    from hafiz.commands.chunks import run_chunks_export

    run_chunks_export(project=project, path_prefix=path, limit=limit, offset=offset)


# ─── EXTRACT ───────────────────────────────────────────────────────

extract_app = typer.Typer(name="extract", help="Entity & relationship extraction.")
app.add_typer(extract_app)


@extract_app.command("import")
def extract_import_cmd(
    file: Optional[str] = typer.Option(
        None, "--file", "-f", help="JSON file (reads stdin if omitted)."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Project tag for stored entities."
    ),
) -> None:
    """Import extraction results from JSON (file or stdin)."""
    from hafiz.commands.extract import run_extract_import

    run_extract_import(file, project=project)


# ─── HOOKS ─────────────────────────────────────────────────────────

hooks_app = typer.Typer(name="hooks", help="Git hook management.")
app.add_typer(hooks_app)


@hooks_app.command("install")
def hooks_install(
    repo_path: str = typer.Argument(".", help="Path to the git repository."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Project name to pass to the hook."
    ),
) -> None:
    """Install the Hafiz post-commit hook into a git repository."""
    from hafiz.commands.hooks import run_hooks_install

    run_hooks_install(repo_path, project=project)


# ─── AGENT ─────────────────────────────────────────────────────────

agent_app = typer.Typer(name="agent", help="Agent integration management.")
app.add_typer(agent_app)


@agent_app.command("install")
def agent_install(
    name: Optional[str] = typer.Argument(
        None, help="Agent name (claude-code, cursor, github-copilot)."
    ),
    local: bool = typer.Option(
        False, "--local", "-l", help="Install in current project instead of globally."
    ),
    path: Optional[str] = typer.Option(
        None, "--path", help="Override destination directory."
    ),
    file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Override destination filename."
    ),
) -> None:
    """Install hafiz skills for an AI coding agent."""
    from hafiz.commands.agent import run_agent_install

    run_agent_install(name, local=local, path_override=path, file_override=file)


@agent_app.command("uninstall")
def agent_uninstall(
    name: Optional[str] = typer.Argument(
        None, help="Agent name to uninstall."
    ),
    local: bool = typer.Option(
        False, "--local", "-l", help="Uninstall from current project."
    ),
    path: Optional[str] = typer.Option(
        None, "--path", help="Override destination directory."
    ),
    file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Override destination filename."
    ),
    force: bool = typer.Option(
        False, "--force", help="Remove even if not installed by hafiz."
    ),
) -> None:
    """Remove hafiz skills for an AI coding agent."""
    from hafiz.commands.agent import run_agent_uninstall

    run_agent_uninstall(name, local=local, path_override=path, file_override=file, force=force)


@agent_app.command("list")
def agent_list() -> None:
    """List available agents and their installation status."""
    from hafiz.commands.agent import run_agent_list

    run_agent_list()
