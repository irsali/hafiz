"""Hafiz CLI — the sovereign intelligence layer for your workspace.

Entry point for the `hafiz` command. Built with Typer + Rich.
"""

from __future__ import annotations

from typing import Optional

import typer

from hafiz import __version__

app = typer.Typer(
    name="hafiz",
    help=(
        "Hafiz — sovereign intelligence layer for your workspace.\n\n"
        "[bold]Getting started:[/bold] hafiz init  →  hafiz status --diagnose"
        "  →  hafiz doctor --probe  →  hafiz ingest <path> --project <name>\n"
        "[bold]Day-to-day:[/bold]    hafiz context \"<task>\"  ·  hafiz query \"<text>\""
        "  ·  hafiz observe \"<decision>\" --type decision  ·  hafiz note \"<thought>\"\n"
        "[bold]When stuck:[/bold]    hafiz errors list  ·  hafiz status --diagnose"
        "  ·  hafiz doctor"
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"hafiz {__version__}")
        raise typer.Exit()


@app.callback()
def _top_level_callback(
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
    # Empty body by design — Typer invokes this for global flags
    # (currently just --version) before dispatching to subcommands.
    # The real entry point lives in `main()` at the bottom of this
    # file (see pyproject `[project.scripts] hafiz = "hafiz.cli:main"`).


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
    source: Optional[str] = typer.Option(
        None,
        "--source",
        help="(with --recall) Filter by source (e.g. user:anjum, agent:claude-code).",
    ),
    include_superseded: bool = typer.Option(
        False,
        "--include-superseded",
        help="(with --recall) Also return superseded/expired observations, dimmed.",
    ),
    include_transcripts: bool = typer.Option(
        False,
        "--include-transcripts",
        help=(
            "Opt in to source-layer search: include matching messages "
            "from imported agent transcripts alongside knowledge-layer "
            "results. Off by default — the wisdom layer must remain primary."
        ),
    ),
) -> None:
    """Search indexed content with vector similarity.

    By default, searches code chunks. Use --recall to search observations
    (decisions, facts, learnings, patterns, warnings). Use
    --include-transcripts to additionally search the source layer
    (imported agent transcripts).
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
            source=source,
            include_superseded=include_superseded,
            output_json=json_output,
        )
    else:
        from hafiz.commands.query import _run_query

        _run_query(
            text,
            limit=limit,
            project=project,
            workspace=workspace,
            kind=type,
            include_transcripts=include_transcripts,
            output_json=json_output,
        )


# ─── RECALL ─────────────────────────────────────────────────────────

@app.command()
def recall(
    target: str = typer.Argument(
        ...,
        help=(
            "Session slug, session uuid, or communication uuid to recall."
        ),
    ),
    role: Optional[str] = typer.Option(
        None, "--role", help="Filter to a single role (user/assistant/tool/system)."
    ),
    seq_from: Optional[int] = typer.Option(
        None, "--from", help="Start at this seq (inclusive)."
    ),
    seq_to: Optional[int] = typer.Option(
        None, "--to", help="Stop at this seq (inclusive)."
    ),
    has_tool_call: Optional[bool] = typer.Option(
        None,
        "--has-tool-call/--no-tool-call",
        help="Filter to messages that do (or don't) carry tool_calls.",
    ),
    query_text: Optional[str] = typer.Option(
        None,
        "--query",
        "-q",
        help="Vector search across the session's messages instead of a linear walk.",
    ),
    limit: int = typer.Option(
        1000, "--limit", "-l", help="Maximum messages to return."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Surface source-layer messages from a session or communication.

    Source-layer rows are hidden from default `hafiz query` /
    `hafiz context`; this command is the explicit, opt-in path. Pass
    `--query` to vector-search across the session's turns; otherwise
    the result is a linear, ordered walk.
    """
    from hafiz.commands.recall import run_recall as run_messages_recall

    run_messages_recall(
        target,
        role=role,
        has_tool_call=has_tool_call,
        seq_from=seq_from,
        seq_to=seq_to,
        limit=limit,
        query_text=query_text,
        output_json=json_output,
    )


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

    Use --diagnose to also run full diagnostic checks (shortcut to
    `hafiz doctor` without per-tunable probing).
    """
    from hafiz.commands.maintenance import run_status

    if diagnose:
        from hafiz.commands.maintenance import run_doctor

        run_doctor(output_json=json_output)
    else:
        run_status(output_json=json_output)


@app.command()
def doctor(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (stable agent-consumable shape)."
    ),
    probe: bool = typer.Option(
        False,
        "--probe",
        help=(
            "Run per-tunable probers to recommend values for this host. "
            "Slow (loads the embedding model, runs several forward passes) "
            "but gives concrete recommendations for `hafiz config set`."
        ),
    ),
    apply_: bool = typer.Option(
        False,
        "--apply",
        help=(
            "Persist recommendations to the sticky tuning cache "
            "(~/.cache/hafiz/tuning_state.json). Implies --probe."
        ),
    ),
) -> None:
    """Diagnose install health, show host capabilities, and (with
    --probe) recommend per-tunable values for this machine.

    Every user-visible field is documented in docs/commands.md; the `--json`
    shape is stable for agents that want to act on the recommendations.
    """
    from hafiz.commands.maintenance import run_doctor

    run_doctor(output_json=json_output, probe=probe or apply_, apply=apply_)


# ─── CONFIG ─────────────────────────────────────────────────────────────

config_app = typer.Typer(name="config", help="Configuration management.")
app.add_typer(config_app)


@config_app.command("show")
def config_show(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Show current Hafiz configuration and per-tunable resolution sources."""
    from hafiz.commands.maintenance import run_config_show

    run_config_show(output_json=json_output)


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Tunable key, e.g. embedding.max_part_chars"),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Print a single tunable's effective value and its source layer."""
    from hafiz.commands.maintenance import run_config_get

    run_config_get(key, output_json=json_output)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Tunable key, e.g. embedding.max_part_chars"),
    value: str = typer.Argument(..., help="Value to set (string; coerced by type)"),
    local: bool = typer.Option(
        False,
        "--local",
        help=(
            "Write to ./hafiz.toml (project scope) instead of the user config "
            "at ~/.config/hafiz/hafiz.toml."
        ),
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Persist a tunable value to hafiz.toml.

    By default writes to the user-scope config (~/.config/hafiz/hafiz.toml).
    Use --local to target the project's ./hafiz.toml instead.
    """
    from hafiz.commands.maintenance import run_config_set

    run_config_set(key, value, local=local, output_json=json_output)


@config_app.command("unset")
def config_unset(
    key: str = typer.Argument(..., help="Tunable key to remove from hafiz.toml"),
    local: bool = typer.Option(
        False, "--local", help="Target ./hafiz.toml instead of user scope."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Remove a tunable from hafiz.toml so it falls through to sticky / default."""
    from hafiz.commands.maintenance import run_config_unset

    run_config_unset(key, local=local, output_json=json_output)


@config_app.command("apply")
def config_apply(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Run all probers and persist their recommendations to sticky state.

    Equivalent to `hafiz doctor --probe --apply`. Does not modify
    hafiz.toml — sticky state is user-scope cache, separate from
    checked-in configuration.
    """
    from hafiz.commands.maintenance import run_config_apply

    run_config_apply(output_json=json_output)


@config_app.command("clear-sticky")
def config_clear_sticky(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Delete the sticky tuning-state cache (re-probe is required to repopulate)."""
    from hafiz.commands.maintenance import run_config_clear_sticky

    run_config_clear_sticky(output_json=json_output)


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


@session_app.command("list")
def session_list(
    agent: Optional[str] = typer.Option(
        None, "--agent", help="Filter by agent (claude-code, cursor, ...)."
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Filter to sessions scoped to a project."
    ),
    active: bool = typer.Option(
        False,
        "--active",
        help="Only show sessions whose ended_at is NULL.",
    ),
    limit: int = typer.Option(
        50, "--limit", "-l", help="Maximum sessions to return (default 50)."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """List recent sessions, newest first.

    Useful for finding a slug without dropping into psql, e.g. before
    `hafiz recall <slug>` or `hafiz forget <slug>`.
    """
    from hafiz.commands.session import run_session_list

    run_session_list(
        agent=agent,
        project=project,
        include_ended=not active,
        limit=limit,
        output_json=json_output,
    )


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
    include_transcripts: bool = typer.Option(
        False,
        "--include-transcripts",
        help=(
            "Append top source-layer transcript matches to the bundle. "
            "Off by default — wisdom layer stays primary."
        ),
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

    run_context(
        query,
        project=project,
        workspace=workspace,
        include_transcripts=include_transcripts,
        output_json=json_output,
    )


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



# ─── FORGET ───────────────────────────────────────────────────────────

@app.command()
def forget(
    target: Optional[str] = typer.Argument(
        None,
        help=(
            "Communication uuid, session uuid, or session slug to redact. "
            "Omit when using --all-expired."
        ),
    ),
    hard: bool = typer.Option(
        False,
        "--hard",
        help=(
            "Permanently delete the communication and its messages. "
            "Default is soft tombstone (sets valid_until = now)."
        ),
    ),
    all_expired: bool = typer.Option(
        False,
        "--all-expired",
        help=(
            "Sweep mode: tombstone every communication past its "
            "retention_until. Use without a target."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="(with --all-expired) Report counts without modifying rows.",
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Redact source-layer rows.

    Two modes:

    * Targeted: ``hafiz forget <id>`` — soft tombstone by default;
      ``--hard`` deletes content. Works on a communication uuid,
      a session uuid, or a session slug.
    * Sweep: ``hafiz forget --all-expired`` — tombstones every
      communication past its retention window (default 90 days).
    """
    if all_expired and target:
        typer.echo("Error: --all-expired and a target are mutually exclusive.")
        raise typer.Exit(1)
    if not all_expired and not target:
        typer.echo("Error: provide a target or use --all-expired.")
        raise typer.Exit(1)

    if all_expired:
        from hafiz.commands.forget import run_forget_sweep

        run_forget_sweep(dry_run=dry_run, output_json=json_output)
    else:
        from hafiz.commands.forget import run_forget_target

        run_forget_target(target, hard=hard, output_json=json_output)


# ─── IMPORT ───────────────────────────────────────────────────────────

import_app = typer.Typer(
    name="import",
    help="Import agent transcripts and other source-layer data.",
)
app.add_typer(import_app)


@import_app.command("claude-code")
def import_claude_code_cmd(
    path: Optional[str] = typer.Argument(
        None,
        help=(
            "Path to a JSONL file or a directory of session JSONL files. "
            "Defaults to ~/.claude/projects (every session you've ever had)."
        ),
    ),
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Tag stored communications with this project."
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Stop after N JSONL files."
    ),
    since: Optional[str] = typer.Option(
        None, "--since", help="Only import sessions ending after this duration ago (e.g. 7d)."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Parse and report counts without writing."
    ),
    no_embed: bool = typer.Option(
        False, "--no-embed", help="Skip embedding (text only). Useful for fast imports."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (for agents)."
    ),
) -> None:
    """Import Claude Code session JSONL files into the source layer.

    Idempotent — re-running is a no-op for already-seen sessions.
    Source-layer rows are hidden from default `hafiz query` /
    `hafiz context`; surface them with `hafiz recall <session>` or the
    `--include-transcripts` flag.
    """
    from hafiz.commands.import_cmd import run_import_claude_code

    run_import_claude_code(
        path,
        project=project,
        limit=limit,
        since=since,
        dry_run=dry_run,
        no_embed=no_embed,
        output_json=json_output,
    )


# ─── ERRORS ───────────────────────────────────────────────────────────

errors_app = typer.Typer(
    name="errors",
    help="Inspect the hafiz error log (~/.cache/hafiz/errors.log).",
)
app.add_typer(errors_app)


@errors_app.command("list")
def errors_list(
    since: Optional[str] = typer.Option(
        None, "--since", help="Relative duration: 1h, 30m, 2d, 1w. Default: all."
    ),
    limit: int = typer.Option(
        20, "--limit", "-n", min=1, help="Max records to return."
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON (agent-consumable)."
    ),
) -> None:
    """Show recent errors, newest first.

    Use `--since 1d` to limit scope, and `hafiz errors show <id>` for
    the full traceback of one entry.
    """
    from hafiz.commands.errors import run_errors_list

    run_errors_list(since=since, limit=limit, output_json=json_output)


@errors_app.command("show")
def errors_show(
    record_id: str = typer.Argument(..., help="Full or unique-prefix error id."),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Show the full structured record (including traceback) for one error."""
    from hafiz.commands.errors import run_errors_show

    run_errors_show(record_id, output_json=json_output)


@errors_app.command("clear")
def errors_clear(
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output as JSON."
    ),
) -> None:
    """Delete the error log. Returns the count of records discarded."""
    from hafiz.commands.errors import run_errors_clear

    run_errors_clear(output_json=json_output)


# ─── Top-level entry point ────────────────────────────────────────────
#
# `pyproject.toml` points `hafiz` at this `main()` so every unhandled
# exception lands in the error log. SystemExit (including Typer's
# controlled-exit path via `typer.Exit`) passes through unaltered —
# we only want to capture bugs, not propagate tracebacks from
# legitimate non-zero exits.


def main() -> None:
    import sys as _sys

    try:
        app()
    except SystemExit:
        raise
    except KeyboardInterrupt:
        # User-initiated — don't log, don't decorate; just exit quietly.
        _sys.exit(130)
    except Exception as exc:  # noqa: BLE001 — this is the backstop
        from rich.console import Console as _Console

        from hafiz.core.error_log import log_exception

        record = log_exception(exc, argv=_sys.argv[1:])
        err = _Console(stderr=True)
        err.print(
            f"[red]hafiz hit an unexpected error:[/red] "
            f"[bold]{record.exception_type}[/bold]: {record.message}"
        )
        if record.suggested_action:
            err.print(f"[yellow]Suggested fix:[/yellow] {record.suggested_action}")
        err.print(
            f"[dim]Saved. Run [bold]hafiz errors show {record.id[:8]}[/bold] "
            f"for the traceback.[/dim]"
        )
        _sys.exit(1)
