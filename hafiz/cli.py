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
    git_hook: bool = typer.Option(
        False, "--git-hook", help="Index only files changed in the latest commit."
    ),
    prune: bool = typer.Option(
        False, "--prune", help="Remove stale chunks before indexing."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Emit newline-delimited JSON progress events."
    ),
) -> None:
    """Index files into the Hafiz knowledge base (chunk + embed + store)."""
    if git_hook:
        from hafiz.commands.ingest import run_git_hook_ingest_cmd

        run_git_hook_ingest_cmd(project=project)
    else:
        if path is None:
            typer.echo("Error: Missing argument 'PATH'. Use --git-hook or provide a path.")
            raise typer.Exit(1)
        from hafiz.commands.ingest import run_ingest

        run_ingest(path, project=project, prune=prune, output_json=json_output)


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
    workspace: bool = typer.Option(
        False, "--workspace", "-w", help="Scope to sibling projects in parent directory."
    ),
    type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by type (chunk: code/doc/note/decision; observation: fact/decision/learning/pattern/warning)."
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of results."
    ),
    recall: bool = typer.Option(
        False, "--recall", help="Search observations instead of code chunks."
    ),
    include_superseded: bool = typer.Option(
        False,
        "--include-superseded",
        help="(with --recall) Also return superseded/expired observations, dimmed.",
    ),
) -> None:
    """Search indexed content with vector similarity.

    By default, searches code chunks. Use --recall to search observations
    (decisions, facts, learnings, patterns, warnings).
    """
    if project and workspace:
        typer.echo("Error: --project and --workspace are mutually exclusive.")
        raise typer.Exit(1)

    if recall:
        from hafiz.commands.observe import run_recall

        run_recall(
            text,
            limit=limit,
            project=project,
            workspace=workspace,
            kind=type,
            include_superseded=include_superseded,
            output_json=json_output,
        )
    else:
        from hafiz.commands.query import _run_query

        _run_query(text, limit=limit, project=project, workspace=workspace, kind=type, output_json=json_output)


# ─── STATUS ─────────────────────────────────────────────────────────────

@app.command()
def status(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
    diagnose: bool = typer.Option(
        False, "--diagnose", help="Run diagnostic checks (config, DB, pgvector, embeddings)."
    ),
) -> None:
    """Show database statistics and index health.

    Use --diagnose to also run full diagnostic checks (replaces 'hafiz doctor').
    """
    from hafiz.commands.maintenance import run_status

    if diagnose:
        from hafiz.commands.maintenance import run_doctor

        run_doctor(output_json=json_output)
    else:
        run_status(output_json=json_output)


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
    depth: int = typer.Option(
        1, "--depth", "-d", min=0, help="Include neighbors up to N hops (undirected)."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Show an entity and its N-hop neighborhood (undirected walk)."""
    from hafiz.commands.graph import run_graph_show

    run_graph_show(name, depth=depth, project=project, output_json=json_output)


@graph_app.command("deps")
def graph_deps(
    name: str = typer.Argument(..., help="Entity name to look up."),
    depth: int = typer.Option(
        1, "--depth", "-d", min=0, help="Walk outgoing edges up to N hops."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Show what an entity depends on, transitively (outgoing walk)."""
    from hafiz.commands.graph import run_graph_deps

    run_graph_deps(name, depth=depth, project=project, output_json=json_output)


@graph_app.command("impact")
def graph_impact(
    name: str = typer.Argument(..., help="Entity name to look up."),
    depth: int = typer.Option(
        1, "--depth", "-d", min=0, help="Walk incoming edges up to N hops."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Show the blast radius — what breaks if this entity changes (incoming walk)."""
    from hafiz.commands.graph import run_graph_impact

    run_graph_impact(name, depth=depth, project=project, output_json=json_output)


@graph_app.command("path")
def graph_path(
    source: str = typer.Argument(..., help="Source entity name."),
    target: str = typer.Argument(..., help="Target entity name."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Find the shortest directed path from SOURCE to TARGET."""
    from hafiz.commands.graph import run_graph_path

    run_graph_path(source, target, project=project, output_json=json_output)


@graph_app.command("rank")
def graph_rank(
    metric: str = typer.Option(
        "pagerank",
        "--metric",
        "-m",
        help="Centrality metric: pagerank, betweenness, degree, in_degree, out_degree.",
    ),
    top: int = typer.Option(
        20, "--top", "-n", min=1, help="Number of results to show."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Rank entities by a centrality metric (importance ranking)."""
    from hafiz.commands.graph import run_graph_rank

    run_graph_rank(metric=metric, top_n=top, project=project, output_json=json_output)


@graph_app.command("stats")
def graph_stats(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    top_central: int = typer.Option(
        5,
        "--top-central",
        min=0,
        help="Number of top-central nodes to include (by PageRank).",
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Show graph-level health: counts, density, components, top-central nodes."""
    from hafiz.commands.graph import run_graph_stats

    run_graph_stats(project=project, top_central=top_central, output_json=json_output)


# ─── SESSION ──────────────────────────────────────────────────────────

session_app = typer.Typer(
    name="session",
    help="Per-TTY session state — tags subsequent observations and captures.",
)
app.add_typer(session_app)


@session_app.command("start")
def session_start(
    name: str = typer.Argument(..., help="Human-readable session name (a slug is auto-generated)."),
    task: Optional[str] = typer.Option(
        None, "--task", help="Default task for this session."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Default project for this session."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Start a named session for this terminal."""
    from hafiz.commands.session import run_session_start

    run_session_start(name, task=task, project=project, output_json=json_output)


@session_app.command("end")
def session_end(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """End the active session for this terminal."""
    from hafiz.commands.session import run_session_end

    run_session_end(output_json=json_output)


@session_app.command("show")
def session_show(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Show the active session for this terminal."""
    from hafiz.commands.session import run_session_show

    run_session_show(output_json=json_output)


# ─── OBSERVE ──────────────────────────────────────────────────────────

@app.command()
def observe(
    text: str = typer.Argument(..., help="The annotation text to store."),
    kind: str = typer.Option(
        "fact",
        "--type",
        "-t",
        help="Kind: fact, decision, learning, pattern, warning, note.",
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
    expires_in: Optional[str] = typer.Option(
        None,
        "--expires-in",
        help="Expire after a duration (e.g. 30d, 2w, 6m, 1y).",
    ),
    expires: Optional[str] = typer.Option(
        None,
        "--expires",
        help="Expire at an ISO date/datetime (e.g. 2026-06-01).",
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session",
        help="Session id to tag (overrides any active `hafiz session`).",
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Task label to tag (overrides any active `hafiz session`).",
    ),
    supersedes: Optional[str] = typer.Option(
        None,
        "--supersedes",
        help="UUID of an observation this one replaces — marks the old row inactive atomically.",
    ),
    derived_from: Optional[str] = typer.Option(
        None,
        "--derived-from",
        help="Comma-separated UUIDs this observation was distilled from (lineage, stored in metadata).",
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
        kind=kind,
        source=source,
        project=project,
        tags=tag_list,
        confidence=confidence,
        expires_in=expires_in,
        expires=expires,
        session=session,
        task=task,
        supersedes=supersedes,
        derived_from=derived_from,
        output_json=json_output,
    )


# ─── CAPTURE ──────────────────────────────────────────────────────


@app.command()
def capture(
    text: Optional[str] = typer.Argument(
        None, help="Transcript text. Omit to read from --file or stdin."
    ),
    file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Read transcript from a file."
    ),
    title: Optional[str] = typer.Option(
        None, "--title", help="Human-readable title (used in the synthetic path)."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Tag chunks with a project name."
    ),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Origin (e.g. agent:claude-code, user:you)."
    ),
    tags: Optional[str] = typer.Option(
        None, "--tags", help="Comma-separated tags."
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session",
        help="Session id to tag (overrides any active `hafiz session`).",
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Task label to tag (overrides any active `hafiz session`).",
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Ingest a transcript / multi-page dump as searchable transcript chunks."""
    from hafiz.commands.capture import run_capture

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    run_capture(
        text,
        file=file,
        title=title,
        project=project,
        source=source,
        tags=tag_list,
        session=session,
        task=task,
        output_json=json_output,
    )


# ─── NOTE ──────────────────────────────────────────────────────────


@app.command()
def note(
    text: str = typer.Argument(..., help="The note text to store."),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Origin (e.g. agent:claude-code, user:you)."
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
    expires_in: Optional[str] = typer.Option(
        None,
        "--expires-in",
        help="Expire after a duration (e.g. 30d, 2w, 6m, 1y).",
    ),
    expires: Optional[str] = typer.Option(
        None,
        "--expires",
        help="Expire at an ISO date/datetime (e.g. 2026-06-01).",
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session",
        help="Session id to tag (overrides any active `hafiz session`).",
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Task label to tag (overrides any active `hafiz session`).",
    ),
    supersedes: Optional[str] = typer.Option(
        None,
        "--supersedes",
        help="UUID of a note this one replaces.",
    ),
    derived_from: Optional[str] = typer.Option(
        None,
        "--derived-from",
        help="Comma-separated UUIDs this note was distilled from.",
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Capture a raw thought as a note — low-bar lane, distill later."""
    from hafiz.commands.observe import run_note

    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    run_note(
        text,
        source=source,
        project=project,
        tags=tag_list,
        confidence=confidence,
        expires_in=expires_in,
        expires=expires,
        session=session,
        task=task,
        supersedes=supersedes,
        derived_from=derived_from,
        output_json=json_output,
    )


# ─── JOURNAL ──────────────────────────────────────────────────────────


@app.command()
def journal(
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Duration window ending now (e.g. 7d, 2w, 6h). Default: 7d.",
    ),
    day: Optional[str] = typer.Option(
        None,
        "--day",
        help="Specific UTC day (ISO date, e.g. 2026-04-20). Exclusive with --since.",
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    workspace: bool = typer.Option(
        False,
        "--workspace",
        "-w",
        help="Scope to sibling projects in parent directory.",
    ),
    source: Optional[str] = typer.Option(
        None, "--source", help="Filter by source (e.g. agent:claude-code)."
    ),
    type: Optional[str] = typer.Option(
        None, "--type", "-t", help="Filter by observation type."
    ),
    session: Optional[str] = typer.Option(
        None, "--session", help="Filter by session id."
    ),
    task: Optional[str] = typer.Option(
        None, "--task", help="Filter by task label."
    ),
    limit: int = typer.Option(
        500, "--limit", "-l", help="Maximum entries (default 500)."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Time-bounded digest of observations — what you captured recently."""
    if project and workspace:
        typer.echo("Error: --project and --workspace are mutually exclusive.")
        raise typer.Exit(1)

    from hafiz.commands.journal import run_journal

    run_journal(
        since=since,
        day=day,
        project=project,
        workspace=workspace,
        source=source,
        kind=type,
        session_id=session,
        task=task,
        limit=limit,
        output_json=json_output,
    )


# ─── DISTILL ──────────────────────────────────────────────────────────


@app.command()
def distill(
    since: Optional[str] = typer.Option(
        None,
        "--since",
        help="Duration window (e.g. 7d, 2w, 6h). Default: 7d.",
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    workspace: bool = typer.Option(
        False,
        "--workspace",
        "-w",
        help="Scope to sibling projects in parent directory.",
    ),
    session: Optional[str] = typer.Option(
        None, "--session", help="Filter by session id."
    ),
    task: Optional[str] = typer.Option(
        None, "--task", help="Filter by task label."
    ),
    no_transcripts: bool = typer.Option(
        False,
        "--no-transcripts",
        help="Only surface notes; skip transcript candidates.",
    ),
    limit: int = typer.Option(
        200, "--limit", "-l", help="Maximum notes (default 200)."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Surface recent notes + transcripts as promotable candidates.

    Distill is a SCANNER, not a promoter. Read the candidates, then
    promote via `hafiz observe '<distilled>' --type decision --derived-from <ids>`.
    """
    if project and workspace:
        typer.echo("Error: --project and --workspace are mutually exclusive.")
        raise typer.Exit(1)

    from hafiz.commands.distill import run_distill

    run_distill(
        since=since,
        project=project,
        workspace=workspace,
        session_id=session,
        task=task,
        include_transcripts=not no_transcripts,
        limit=limit,
        output_json=json_output,
    )


# ─── CONTEXT ──────────────────────────────────────────────────────────

@app.command()
def context(
    query: str = typer.Argument(..., help="Task description or question."),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    workspace: bool = typer.Option(
        False, "--workspace", "-w", help="Scope to sibling projects in parent directory."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Synthesize relevant code, graph, and observations for a task."""
    if project and workspace:
        typer.echo("Error: --project and --workspace are mutually exclusive.")
        raise typer.Exit(1)

    from hafiz.commands.context import run_context

    run_context(query, project=project, workspace=workspace, output_json=json_output)


# ─── REVIEW ──────────────────────────────────────────────────────────

@app.command()
def review(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Review a specific project."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Review knowledge quality and get improvement suggestions."""
    from hafiz.commands.review import run_review

    run_review(project=project, output_json=json_output)


# ─── EXTRACT ───────────────────────────────────────────────────────

extract_app = typer.Typer(name="extract", help="Entity & relationship extraction.")
app.add_typer(extract_app)


@extract_app.command("export")
def extract_export_cmd(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter by project."
    ),
    limit: int = typer.Option(
        500, "--limit", "-l", help="Maximum units to surface."
    ),
    pretty: bool = typer.Option(
        False, "--pretty", help="Human-readable summary instead of JSON."
    ),
) -> None:
    """Emit the AST-known units + structural edges agents can annotate
    against. JSON by default (agents parse it); --pretty for humans."""
    from hafiz.commands.extract import run_extract_export

    run_extract_export(project=project, limit=limit, output_json=not pretty)


@extract_app.command("import")
def extract_import_cmd(
    file: Optional[str] = typer.Option(
        None, "--file", "-f", help="JSON file (reads stdin if omitted)."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Project tag for stored rows."
    ),
) -> None:
    """Import agent extraction v2 (annotations + semantic edges)
    from JSON (file or stdin)."""
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
    """Install Hafiz git hooks (post-commit + post-merge) into a repository."""
    from hafiz.commands.hooks import run_hooks_install

    run_hooks_install(repo_path, project=project)


# ─── EMBEDDING ──────────────────────────────────────────────────────

embedding_app = typer.Typer(
    name="embedding",
    help="Inspect and retry the embedding device selection (GPU vs CPU).",
)
app.add_typer(embedding_app)


@embedding_app.command("status")
def embedding_status(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Show the current embedding device and its provenance (config / sticky / probe)."""
    from hafiz.commands.embedding import run_embedding_status

    run_embedding_status(output_json=json_output)


@embedding_app.command("retry")
def embedding_retry(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Clear sticky state and re-probe the embedding device (use after freeing VRAM, upgrading drivers, etc.)."""
    from hafiz.commands.embedding import run_embedding_retry

    run_embedding_retry(output_json=json_output)


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


# ─── PARSERS ──────────────────────────────────────────────────────────

parsers_app = typer.Typer(name="parsers", help="Parser registry observability.")
app.add_typer(parsers_app)


@parsers_app.command("list")
def parsers_list_cmd(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """List registered parsers (in-tree + entry-point-loaded) and their
    language coverage. Useful for answering "is AST active for my .go
    files?" without inspecting config."""
    from hafiz.commands.parsers import run_parsers_list

    run_parsers_list(output_json=json_output)
