"""hafiz init, status, config, doctor — maintenance commands."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from hafiz.core.config import CONFIG_FILENAME, find_config_file, get_settings
from hafiz.core.database import close_engine, create_tables, get_session_factory

console = Console()


def run_init() -> None:
    """Initialize the Hafiz database — create pgvector extension and all tables."""

    async def _init():
        try:
            settings = get_settings()
            console.print(f"Connecting to [bold]{settings.database.url}[/bold]")
            await create_tables()
            console.print("[green]Database initialized successfully.[/green]")
            console.print("  - pgvector extension enabled")
            console.print(
                "  - Tables created: files, units, unit_revisions, "
                "embeddings, edges, annotations, commits"
            )
            console.print(
                "  [yellow]Note: migration 0005 replaces the old schema. "
                "Re-ingest is required after upgrade from an older Hafiz.[/yellow]"
            )

            # Check for config file
            config_path = find_config_file()
            if config_path:
                console.print(f"  - Config loaded from: {config_path}")
            else:
                console.print(
                    f"  [yellow]No {CONFIG_FILENAME} found. Using defaults + env vars.[/yellow]"
                )
                console.print(
                    f"  Run [bold]cp hafiz.toml.example {CONFIG_FILENAME}[/bold] to create one."
                )
        finally:
            await close_engine()

    asyncio.run(_init())


def _embedding_device_summary() -> dict:
    """Sync, DB-independent summary of the embedding-device selection."""
    from hafiz.core import device_state as dstate

    settings = get_settings()
    sticky = dstate.load_state()
    configured = settings.embedding.device

    if configured in ("cpu", "gpu"):
        source = "config"
        effective = configured
    elif sticky is not None:
        source = "sticky"
        effective = sticky.device
    else:
        source = "not-probed"
        effective = "(not probed)"

    return {
        "configured": configured,
        "source": source,
        "effective": effective,
        "sticky_probed_at": sticky.probed_at if sticky else None,
        "sticky_reason_category": sticky.reason_category if sticky else None,
    }


async def _index_staleness(last_commit_per_project: dict[str | None, str]) -> dict:
    """For each project, how far its indexed commit trails the repo's HEAD.

    Fills the gap that let four repos drift 31-64 commits behind while hooks
    were installed and firing: nothing ever compared the index to the repo.

    Thin wrapper over :func:`hafiz.core.freshness.index_staleness`, which search
    results share — the probe belongs in core, and two copies would drift. The
    already-computed map is handed over rather than re-derived.
    """
    from hafiz.core.freshness import index_staleness

    return await index_staleness(list(last_commit_per_project), last_commit=last_commit_per_project)


def _staleness_note(entry: dict) -> tuple[str, str]:
    """(label, rich style) summarising one project's index freshness."""
    behind = entry.get("commits_behind")
    if entry.get("is_ancestor") is False:
        # Indexed commit isn't in HEAD's history — rebased away or force-pushed
        # over. A commit count here would be meaningless, not merely imprecise.
        return "diverged", "red"
    if behind is None:
        return "unknown", "dim"
    if behind == 0:
        return "current", "green"
    return f"{behind} behind", "red" if behind >= 10 else "yellow"


def run_status(*, output_json: bool = False) -> None:
    """Show database statistics and index health."""

    async def _status():
        try:
            from sqlalchemy import func, select

            from hafiz.core.database import (
                Annotation,
                Commit,
                Edge,
                Embedding,
                File,
                Unit,
                UnitRevision,
            )

            session_factory = get_session_factory()
            async with session_factory() as session:
                # ── Current-state counts (tombstoned / superseded excluded) ─
                files_count = (
                    await session.execute(
                        select(func.count()).select_from(File).where(File.valid_until.is_(None))
                    )
                ).scalar() or 0
                units_count = (
                    await session.execute(
                        select(func.count()).select_from(Unit).where(Unit.valid_until.is_(None))
                    )
                ).scalar() or 0
                current_revisions_count = (
                    await session.execute(
                        select(func.count())
                        .select_from(UnitRevision)
                        .where(UnitRevision.superseded_at.is_(None))
                    )
                ).scalar() or 0
                embeddings_count = (
                    await session.execute(select(func.count()).select_from(Embedding))
                ).scalar() or 0
                edges_count = (
                    await session.execute(
                        select(func.count()).select_from(Edge).where(Edge.superseded_at.is_(None))
                    )
                ).scalar() or 0
                annotations_count = (
                    await session.execute(select(func.count()).select_from(Annotation))
                ).scalar() or 0
                commits_count = (
                    await session.execute(select(func.count()).select_from(Commit))
                ).scalar() or 0

                # ── Historical totals (include tombstoned for context) ──
                total_units = (
                    await session.execute(select(func.count()).select_from(Unit))
                ).scalar() or 0
                total_revisions = (
                    await session.execute(select(func.count()).select_from(UnitRevision))
                ).scalar() or 0

                # ── Breakdowns by project and kind (current only) ──
                project_rows = (
                    await session.execute(
                        select(File.project, func.count())
                        .where(File.valid_until.is_(None))
                        .group_by(File.project)
                        .order_by(func.count().desc())
                    )
                ).all()

                kind_rows = (
                    await session.execute(
                        select(Unit.kind, func.count())
                        .where(Unit.valid_until.is_(None))
                        .group_by(Unit.kind)
                        .order_by(func.count().desc())
                    )
                ).all()

            # ── Most-recent commit per project, and how stale it is ──
            # Ordered by commits.committed_at, NOT max(hash): hashes are hex,
            # so a lexicographic max picks a commit at random and silently
            # masked genuinely stale indexes across six repos for months.
            from hafiz.core.store import last_indexed_commit_per_project

            last_commit_per_project = await last_indexed_commit_per_project()
            staleness = await _index_staleness(last_commit_per_project)

            # Bounded retention is an outward-facing commitment; the sweep only
            # runs on `import`, which stops firing exactly when it's needed. The
            # count is what makes an unenforced policy visible.
            from hafiz.core.communications import count_overdue_communications
            from hafiz.core.telemetry import count_overdue_retrievals

            overdue_comms = await count_overdue_communications()
            overdue_retr = await count_overdue_retrievals()
            overdue = overdue_comms + overdue_retr

            stats = {
                "files": files_count,
                "units": units_count,
                "revisions_current": current_revisions_count,
                "revisions_total": total_revisions,
                "units_total": total_units,
                "units_tombstoned": total_units - units_count,
                "embeddings": embeddings_count,
                "edges": edges_count,
                "annotations": annotations_count,
                "commits": commits_count,
                "by_project": {p or "(none)": c for p, c in project_rows},
                "by_kind": {k or "(none)": c for k, c in kind_rows},
                "last_commit_per_project": {
                    p or "(none)": c for p, c in last_commit_per_project.items()
                },
                "staleness": staleness,
                "retention": {
                    "overdue": overdue,
                    "communications": overdue_comms,
                    "retrievals": overdue_retr,
                },
                # A project-less ingest can't update a project's rows — `files`
                # is unique on (project, path) — so it writes a parallel
                # untagged copy that search then returns alongside the real one.
                # 1,956 such rows accumulated unnoticed on a real deployment
                # because nothing counted them.
                "untagged": {"files": dict(project_rows).get(None, 0)},
            }
            return stats
        finally:
            await close_engine()

    device_info = _embedding_device_summary()
    stats = asyncio.run(_status())
    stats["embedding_device"] = device_info

    if output_json:
        console.print_json(json.dumps(stats))
        return

    table = Table(title="Hafiz Status", show_header=False, border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Files (current)", str(stats["files"]))
    table.add_row("Units (current)", str(stats["units"]))
    if stats["units_tombstoned"]:
        table.add_row(
            "  [dim]tombstoned[/dim]",
            f"[dim]{stats['units_tombstoned']}[/dim]",
        )
    table.add_row(
        "Revisions",
        f"{stats['revisions_current']} current / {stats['revisions_total']} total",
    )
    table.add_row("Embeddings", str(stats["embeddings"]))
    table.add_row("Edges (current)", str(stats["edges"]))
    table.add_row("Annotations", str(stats["annotations"]))
    table.add_row("Commits (indexed)", str(stats["commits"]))
    overdue = stats["retention"]["overdue"]
    table.add_row(
        "Retention overdue",
        f"[red]{overdue}[/red]" if overdue else "0",
    )
    dev = stats["embedding_device"]
    table.add_row(
        "Embedding device",
        f"{dev['effective']} [dim]({dev['source']})[/dim]",
    )

    console.print()
    console.print(table)

    if overdue:
        console.print(
            f"  [yellow]{overdue} communication(s) past their retention window.[/yellow]\n"
            f"  [dim]Sweep them with: hafiz forget --all-expired[/dim]"
        )

    if stats["by_project"]:
        console.print()
        proj_table = Table(title="Files by Project", border_style="cyan")
        proj_table.add_column("Project")
        proj_table.add_column("Files", justify="right")
        for proj, count in stats["by_project"].items():
            proj_table.add_row(proj, str(count))
        console.print(proj_table)

    untagged = stats["untagged"]["files"]
    if untagged:
        console.print(
            f"  [red]{untagged} file(s) indexed with no project[/red] — a duplicate\n"
            f"  shadow index that search returns alongside the real rows.\n"
            f"  [dim]Cause: an ingest with no --project. Fix the source with:\n"
            f"  hafiz hooks install <repo> --project <name>[/dim]"
        )

    if stats["by_kind"]:
        console.print()
        kind_table = Table(title="Units by Kind", border_style="cyan")
        kind_table.add_column("Kind")
        kind_table.add_column("Units", justify="right")
        for kind, count in stats["by_kind"].items():
            kind_table.add_row(kind, str(count))
        console.print(kind_table)

    if stats["last_commit_per_project"]:
        console.print()
        commit_table = Table(title="Index freshness per project", border_style="cyan")
        commit_table.add_column("Project")
        commit_table.add_column("Indexed at", style="dim")
        commit_table.add_column("Status", justify="right")
        stale_count = 0
        for proj, sha in stats["last_commit_per_project"].items():
            if proj == "(none)":
                continue  # not a repo; covered by the untagged warning above
            entry = stats["staleness"].get(proj, {})
            label, style = _staleness_note(entry)
            if entry.get("commits_behind") or entry.get("is_ancestor") is False:
                stale_count += 1
            commit_table.add_row(
                proj,
                sha[:12] if sha else "—",
                f"[{style}]{label}[/{style}]",
            )
        console.print(commit_table)
        if stale_count:
            console.print(
                f"  [yellow]{stale_count} project(s) behind HEAD — "
                f"search may return code that no longer exists.[/yellow]\n"
                f"  [dim]Re-index with: hafiz ingest <path> --project <name>[/dim]"
            )


def run_config_show(*, output_json: bool = False) -> None:
    """Show the current Hafiz configuration + per-tunable resolution sources."""
    settings = get_settings()

    # Per-tunable resolution chain — independent of pydantic settings.
    from hafiz.core import tunables as _tunables

    tunable_rows = [
        {
            "key": t.key,
            "value": _tunables.resolve_with_source(t.key)[0],
            "source": _tunables.resolve_with_source(t.key)[1],
            "default": t.default,
            "is_policy": t.is_policy,
        }
        for t in _tunables.all_tunables()
    ]

    if output_json:
        payload = json.loads(settings.model_dump_json())
        payload["tunables"] = tunable_rows
        console.print_json(json.dumps(payload))
        return

    config_path = find_config_file()

    console.print()
    if config_path:
        console.print(f"Config file: [bold]{config_path}[/bold]")
    else:
        console.print(f"[yellow]No {CONFIG_FILENAME} found — using defaults + env vars[/yellow]")

    console.print()

    # Database
    db_table = Table(title="Database", show_header=False, border_style="cyan")
    db_table.add_column("Key", style="bold")
    db_table.add_column("Value")
    db_table.add_row("url", settings.database.url)
    console.print(db_table)

    # Embedding
    console.print()
    emb_table = Table(title="Embedding", show_header=False, border_style="cyan")
    emb_table.add_column("Key", style="bold")
    emb_table.add_column("Value")
    emb_table.add_row("model", settings.embedding.model)
    emb_table.add_row("provider", settings.embedding.provider)
    emb_table.add_row("dimensions", str(settings.embedding.dimensions))
    emb_table.add_row("device", settings.embedding.device)
    console.print(emb_table)

    # LLM
    console.print()
    llm_table = Table(title="LLM", show_header=False, border_style="cyan")
    llm_table.add_column("Key", style="bold")
    llm_table.add_column("Value")
    llm_table.add_row("provider", settings.llm.provider)
    llm_table.add_row("model", settings.llm.model)
    console.print(llm_table)

    # Workspace
    console.print()
    ws_table = Table(title="Workspace", show_header=False, border_style="cyan")
    ws_table.add_column("Key", style="bold")
    ws_table.add_column("Value")
    ws_table.add_row("root", settings.workspace.root)
    ws_table.add_row("projects", ", ".join(settings.workspace.projects) or "(none)")
    ws_table.add_row("ignore", ", ".join(settings.workspace.ignore))
    console.print(ws_table)

    # Tunables — the source-aware view, so users can see *why* a value is what it is.
    console.print()
    tun_table = Table(title="Tunables", border_style="cyan")
    tun_table.add_column("Key", style="bold")
    tun_table.add_column("Value")
    tun_table.add_column("Source")
    tun_table.add_column("Default", style="dim")
    for row in tunable_rows:
        source_style = {
            "env": "[magenta]env[/magenta]",
            "toml": "[cyan]toml[/cyan]",
            "sticky": "[green]sticky[/green]",
            "default": "[dim]default[/dim]",
        }[row["source"]]
        tun_table.add_row(row["key"], str(row["value"]), source_style, str(row["default"]))
    console.print(tun_table)
    console.print()


# ── `hafiz config get` ────────────────────────────────────────────────


def run_config_get(key: str, *, output_json: bool = False) -> None:
    from hafiz.core import tunables as _tunables

    try:
        t = _tunables.get(key)
    except KeyError:
        return _config_error(
            "unknown_tunable",
            f"No tunable registered for key {key!r}. "
            f"Run `hafiz doctor --json` to list registered tunables.",
            output_json,
            exit_code=1,
        )

    value, source = _tunables.resolve_with_source(key)
    if output_json:
        console.print_json(
            json.dumps(
                {
                    "key": key,
                    "value": value,
                    "source": source,
                    "default": t.default,
                    "is_policy": t.is_policy,
                }
            )
        )
        return

    console.print()
    console.print(f"[bold]{key}[/bold] = {value}")
    console.print(f"  source:  [dim]{source}[/dim]")
    if value != t.default:
        console.print(f"  default: [dim]{t.default}[/dim]")
    console.print()


# ── `hafiz config set` / `unset` ──────────────────────────────────────


def _resolve_config_target(*, local: bool) -> Path:
    """Decide where `config set` / `unset` writes. User scope by default."""
    if local:
        return Path.cwd() / CONFIG_FILENAME
    return Path.home() / ".config" / "hafiz" / CONFIG_FILENAME


def _read_toml(path: Path) -> dict:
    """Parse a TOML file if it exists; return empty dict otherwise."""
    if not path.is_file():
        return {}
    import tomllib

    with open(path, "rb") as f:
        return tomllib.load(f)


def _write_toml(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` (tomli-w). Creates parents as needed."""
    import tomli_w  # project dep, available at runtime

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)


def _dig(data: dict, parts: list[str]) -> tuple[dict, str] | None:
    """Walk to the parent table of the leaf key, creating intermediate
    tables as needed. Returns (parent_table, leaf_key) or None on
    malformed paths (an existing non-dict in the way)."""
    obj = data
    for p in parts[:-1]:
        if p not in obj or obj[p] is None:
            obj[p] = {}
        if not isinstance(obj[p], dict):
            return None
        obj = obj[p]
    return obj, parts[-1]


def run_config_set(
    key: str, raw_value: str, *, local: bool = False, output_json: bool = False
) -> None:
    from hafiz.core import tunables as _tunables

    try:
        t = _tunables.get(key)
    except KeyError:
        return _config_error(
            "unknown_tunable",
            f"No tunable registered for key {key!r}.",
            output_json,
            exit_code=1,
        )

    try:
        value = _tunables._coerce(t, raw_value)
    except (ValueError, TypeError) as e:
        return _config_error(
            "coerce_failed",
            f"Cannot interpret {raw_value!r} as {t.type_.__name__}: {e}",
            output_json,
            exit_code=1,
        )

    if t.validator is not None:
        try:
            t.validator(value)
        except ValueError as e:
            return _config_error(
                "validation_failed",
                f"Invalid value for {key}: {e}",
                output_json,
                exit_code=1,
            )

    target = _resolve_config_target(local=local)
    data = _read_toml(target)
    dug = _dig(data, key.split("."))
    if dug is None:
        return _config_error(
            "malformed_toml",
            f"Existing {target} has a non-table entry blocking {key!r}.",
            output_json,
            exit_code=1,
        )
    parent, leaf = dug
    parent[leaf] = value
    _write_toml(target, data)

    # Drop the pydantic-settings cache so subsequent reads pick up the
    # new TOML. The sticky layer is unchanged.
    from hafiz.core.config import reset_settings

    reset_settings()

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "ok": True,
                    "key": key,
                    "value": value,
                    "target": str(target),
                    "scope": "local" if local else "user",
                }
            )
        )
        return

    scope = "project" if local else "user"
    console.print(
        f"[green]Set[/green] [bold]{key}[/bold] = {value} in the {scope}-scope "
        f"config ([bold]{target}[/bold])."
    )


def run_config_unset(key: str, *, local: bool = False, output_json: bool = False) -> None:
    from hafiz.core import tunables as _tunables

    try:
        _tunables.get(key)
    except KeyError:
        return _config_error(
            "unknown_tunable",
            f"No tunable registered for key {key!r}.",
            output_json,
            exit_code=1,
        )

    target = _resolve_config_target(local=local)
    if not target.is_file():
        if output_json:
            console.print_json(
                json.dumps({"ok": True, "key": key, "target": str(target), "no_op": True})
            )
        else:
            console.print(f"[dim]No-op: {target} does not exist, nothing to remove.[/dim]")
        return

    data = _read_toml(target)
    parts = key.split(".")
    # Walk down, collecting parent chain so we can prune empty tables.
    chain: list[tuple[dict, str]] = []
    obj: Any = data
    for p in parts:
        if not isinstance(obj, dict) or p not in obj:
            break
        chain.append((obj, p))
        obj = obj[p]
    else:
        parent, leaf = chain[-1]
        del parent[leaf]
        # Walk up, deleting emptied parent tables we created along the way.
        for table, tkey in reversed(chain[:-1]):
            child = table[tkey]
            if isinstance(child, dict) and not child:
                del table[tkey]
            else:
                break

    _write_toml(target, data)
    from hafiz.core.config import reset_settings

    reset_settings()

    if output_json:
        console.print_json(json.dumps({"ok": True, "key": key, "target": str(target)}))
    else:
        console.print(f"[green]Unset[/green] [bold]{key}[/bold] from [bold]{target}[/bold].")


# ── `hafiz config apply` / `clear-sticky` ─────────────────────────────


def run_config_apply(*, output_json: bool = False, assume_yes: bool = False) -> None:
    """Run all probers and persist recommendations to sticky state.

    Interactive by default: prompts per recommendation so the user can
    accept, skip, or supply a custom value. ``--yes`` (``assume_yes=True``)
    persists every recommendation without prompting; ``--json`` is also
    silent (machine consumers don't get prompted). When stdin/stdout
    aren't both TTYs we degrade to ``--yes`` semantics so piped runs
    don't hang on an unanswerable prompt.

    Equivalent to ``hafiz doctor --apply`` but with a narrower,
    apply-focused JSON summary agents can act on directly.
    """
    from hafiz.core.host_probe import probe_host

    host = probe_host()
    rows = _collect_tuning(host, probe=True)

    interactive = (
        not output_json
        and not assume_yes
        and _is_interactive()
        and _has_pending_recommendations(rows)
    )
    if interactive:
        console.print()
        console.print(
            "[bold]Per-tunable review.[/bold] [dim]Each prompt: [Y]es to apply, "
            "[n]o to skip, [c]ustom to enter your own value.[/dim]"
        )
        rows = _interactive_filter(rows)

    applied = _apply_tuning(host, rows)

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "ok": True,
                    "applied": applied,
                    "host_fingerprint": host.fingerprint,
                    "interactive": interactive,
                }
            )
        )
        return

    if not applied:
        if interactive:
            console.print("\n[dim]Nothing applied — every recommendation was skipped.[/dim]")
        else:
            console.print(
                "[yellow]No probed recommendations to apply.[/yellow] "
                "Run `hafiz doctor --probe` to inspect per-tunable probe_error details."
            )
        return
    console.print()
    for a in applied:
        console.print(
            f"[green]Applied[/green] [bold]{a['key']}[/bold] = {a['value']} "
            f"(confidence {a['confidence']})"
        )
        if a.get("rationale"):
            console.print(f"  [dim]{a['rationale']}[/dim]")
    console.print(
        "\n[dim]Persisted to sticky cache. "
        "Run [bold]hafiz config clear-sticky[/bold] to revert.[/dim]"
    )


def run_config_clear_sticky(*, output_json: bool = False) -> None:
    from hafiz.core.tuning_state import clear_state

    removed = clear_state()
    if output_json:
        console.print_json(json.dumps({"ok": True, "removed": removed}))
        return
    if removed:
        console.print("[green]Cleared sticky tuning cache.[/green]")
    else:
        console.print("[dim]No sticky tuning cache to clear.[/dim]")


# ── helpers ───────────────────────────────────────────────────────────


def _config_error(code: str, message: str, output_json: bool, *, exit_code: int = 1) -> None:
    import typer as _typer

    if output_json:
        console.print_json(json.dumps({"ok": False, "error": code, "message": message}))
    else:
        console.print(f"[red]Error:[/red] {message}")
    raise _typer.Exit(exit_code)


def datetime_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def run_doctor(
    *,
    output_json: bool = False,
    probe: bool = False,
    apply: bool = False,
    assume_yes: bool = False,
) -> None:
    """Run diagnostic checks on the Hafiz installation.

    When ``probe`` is True, additionally runs each registered tunable's
    prober and reports recommended values. When ``apply`` is True,
    the recommendations that differ from current are persisted to the
    sticky tuning cache. ``apply`` implies ``probe``.
    """

    checks: list[dict] = []

    def _check(name: str, passed: bool, detail: str = "", fix: str = "") -> None:
        checks.append({"name": name, "passed": passed, "detail": detail, "fix": fix})

    # 1. Config file
    config_path = find_config_file()
    _check(
        "Config file",
        config_path is not None,
        detail=str(config_path) if config_path else "not found",
        fix="Create hafiz.toml in your project root (see hafiz.toml.example).",
    )

    # 2. Database URL valid
    settings = get_settings()
    db_url = settings.database.url
    url_valid = db_url.startswith("postgresql") and "@" in db_url
    _check(
        "Database URL valid",
        url_valid,
        detail=db_url,
        fix="Set HAFIZ_DATABASE__URL or update hafiz.toml [database] section.",
    )

    # 3. (removed — ANTHROPIC_API_KEY no longer needed, extraction is agent-driven)

    # Async checks
    async def _async_checks():
        try:
            from sqlalchemy import func, select, text

            from hafiz.core.database import (
                Annotation,
                Commit,
                Edge,
                Embedding,
                File,
                Unit,
                UnitRevision,
            )

            # 4. Database connectivity
            try:
                session_factory = get_session_factory()
                async with session_factory() as session:
                    await session.execute(text("SELECT 1"))
                _check("Database connectivity", True, detail="connected")
            except Exception as e:
                _check(
                    "Database connectivity",
                    False,
                    detail=str(e)[:120],
                    fix="Ensure PostgreSQL is running and the database URL is correct.",
                )
                return  # Can't proceed without DB

            # 5. pgvector extension
            try:
                async with session_factory() as session:
                    result = await session.execute(
                        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                    )
                    has_pgvector = result.scalar() is not None
                _check(
                    "pgvector extension",
                    has_pgvector,
                    detail="installed" if has_pgvector else "not installed",
                    fix="Run: hafiz init (or CREATE EXTENSION vector in psql).",
                )
            except Exception as e:
                _check(
                    "pgvector extension",
                    False,
                    detail=str(e)[:120],
                    fix="Run: hafiz init",
                )

            # 6. Tables exist (structural-grounding schema)
            expected_tables = {
                "files",
                "units",
                "unit_revisions",
                "embeddings",
                "edges",
                "annotations",
                "commits",
            }
            try:
                async with session_factory() as session:
                    result = await session.execute(
                        text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
                    )
                    existing_tables = {row[0] for row in result.fetchall()}

                missing = expected_tables - existing_tables
                _check(
                    "All tables exist",
                    not missing,
                    detail=f"found: {', '.join(sorted(existing_tables & expected_tables))}"
                    + (f" | missing: {', '.join(sorted(missing))}" if missing else ""),
                    fix="Run: hafiz init" if missing else "",
                )
            except Exception as e:
                _check(
                    "All tables exist",
                    False,
                    detail=str(e)[:120],
                    fix="Run: hafiz init",
                )

            # 7. Table row counts (current state)
            try:
                async with session_factory() as session:
                    files_count = (
                        await session.execute(
                            select(func.count()).select_from(File).where(File.valid_until.is_(None))
                        )
                    ).scalar() or 0
                    units_count = (
                        await session.execute(
                            select(func.count()).select_from(Unit).where(Unit.valid_until.is_(None))
                        )
                    ).scalar() or 0
                    rev_count = (
                        await session.execute(
                            select(func.count())
                            .select_from(UnitRevision)
                            .where(UnitRevision.superseded_at.is_(None))
                        )
                    ).scalar() or 0
                    emb_count = (
                        await session.execute(select(func.count()).select_from(Embedding))
                    ).scalar() or 0
                    edge_count = (
                        await session.execute(
                            select(func.count())
                            .select_from(Edge)
                            .where(Edge.superseded_at.is_(None))
                        )
                    ).scalar() or 0
                    ann_count = (
                        await session.execute(select(func.count()).select_from(Annotation))
                    ).scalar() or 0
                    commit_count = (
                        await session.execute(select(func.count()).select_from(Commit))
                    ).scalar() or 0

                _check(
                    "Table row counts",
                    True,
                    detail=(
                        f"files={files_count}, units={units_count}, "
                        f"revisions={rev_count}, embeddings={emb_count}, "
                        f"edges={edge_count}, annotations={ann_count}, "
                        f"commits={commit_count}"
                    ),
                )
            except Exception as e:
                _check(
                    "Table row counts",
                    False,
                    detail=str(e)[:120],
                    fix="Run: hafiz init",
                )

            # 7b. Retention enforcement. Bounded retention is an outward-facing
            # commitment, and the sweep only runs on `import` — which stops
            # firing exactly when it's most needed (358 rows sat overdue for
            # four weeks in a real deployment, four weeks after imports
            # stopped). Reported here as well as in `status` because
            # visibility, not the trigger, is what actually enforces it.
            try:
                from hafiz.core.communications import count_overdue_communications
                from hafiz.core.telemetry import count_overdue_retrievals

                overdue = await count_overdue_communications() + await count_overdue_retrievals()
                _check(
                    "Retention enforced",
                    overdue == 0,
                    detail=(
                        "no source-layer rows past retention"
                        if overdue == 0
                        else f"{overdue} source-layer row(s) past retention_until"
                    ),
                    fix="Run: hafiz forget --all-expired" if overdue else "",
                )
            except Exception as e:
                _check("Retention enforced", False, detail=str(e)[:120])

            # 7c. Untagged file rows. `files` is unique on (project, path), so a
            # project-less ingest can't update a project's rows — it writes a
            # parallel copy that search then returns alongside the real one, and
            # that copy is never diff-scoped or tombstoned. Nothing counted them,
            # so 1,956 accumulated on a real deployment.
            try:
                async with session_factory() as session:
                    untagged = (
                        await session.execute(
                            select(func.count())
                            .select_from(File)
                            .where(File.project.is_(None))
                            .where(File.valid_until.is_(None))
                        )
                    ).scalar() or 0
                _check(
                    "Every file has a project",
                    untagged == 0,
                    detail=(
                        "no untagged files"
                        if untagged == 0
                        else f"{untagged} file(s) with project=NULL — a duplicate shadow index"
                    ),
                    fix=(
                        "Re-run: hafiz hooks install <repo> --project <name>, then re-ingest"
                        if untagged
                        else ""
                    ),
                )
            except Exception as e:
                _check("Every file has a project", False, detail=str(e)[:120])

        finally:
            await close_engine()

    asyncio.run(_async_checks())

    # 8. Embedding model loadable (sync check — separate from DB)
    try:
        _check("Embedding model loadable", True, detail=settings.embedding.model)
    except Exception as e:
        _check(
            "Embedding model loadable",
            False,
            detail=str(e)[:120],
            fix="Run: pip install fastembed",
        )

    # 9. Parser coverage — "which languages will hafiz ingest see as AST?"
    try:
        from hafiz.core.parsers import get_registry

        registry = get_registry()
        parsers = registry.all_parsers()
        coverage = ", ".join(f"{p.name}({'+'.join(registry.extensions_for(p))})" for p in parsers)
        _check(
            "Parser registry",
            bool(parsers),
            detail=coverage or "(no parsers registered)",
            fix="Re-install hafiz — in-tree parsers should self-register." if not parsers else "",
        )
    except Exception as e:
        _check(
            "Parser registry",
            False,
            detail=str(e)[:120],
            fix="Re-install hafiz.",
        )

    # 10. Runtime deps importable — catch "pipx install predates a new
    #     dependency" cases before the user hits them from a real
    #     command. Only the deps we've actually seen break in the wild
    #     are checked; keep the list small on purpose.
    _critical_runtime_modules = [
        "scipy",
        "networkx",
        "fastembed",
        "sqlalchemy",
        "pgvector",
        "pydantic",
        "pydantic_settings",
        "tomli_w",
    ]
    missing_modules = []
    for mod in _critical_runtime_modules:
        try:
            __import__(mod)
        except ImportError:
            missing_modules.append(mod)
    _check(
        "Runtime deps importable",
        not missing_modules,
        detail=("all present" if not missing_modules else f"missing: {', '.join(missing_modules)}"),
        fix=(
            f"`pipx inject hafiz {' '.join(missing_modules)}` (or `pipx reinstall hafiz`)"
            if missing_modules
            else ""
        ),
    )

    # 10b. Accelerator availability. An installed GPU extra is not proof the
    #      GPU is in use: onnxruntime, onnxruntime-gpu and onnxruntime-openvino
    #      all install the same `onnxruntime` import package, so whichever lands
    #      last wins and the others' providers vanish while their metadata
    #      remains. Naming that case matters — the obvious "install the extra"
    #      advice appears to succeed and changes nothing.
    try:
        from hafiz.core.accelerators import diagnose_accelerators
        from hafiz.core.host_probe import probe_host

        host = probe_host()
        for finding in diagnose_accelerators(providers=host.onnx_providers, gpu_name=host.gpu_name):
            if finding.state == "no-hardware":
                continue  # nothing to report on a host that can't use it
            _check(
                f"Accelerator: {finding.name}",
                finding.ok,
                detail=finding.detail,
                fix=finding.fix,
            )
    except Exception as e:
        _check("Accelerator availability", False, detail=str(e)[:120])

    # 11. Recent errors — informational, not a fail. Agents can drill
    #     in with `hafiz errors list --since 1d --json`.
    try:
        from hafiz.core import error_log

        recent = error_log.tail(since="1d", limit=1)
        count_24h = error_log.count_recent(since="1d")
        if count_24h == 0:
            detail = "none in last 24h"
        else:
            most_recent = recent[0]
            detail = (
                f"{count_24h} in last 24h; most recent: "
                f"{most_recent.exception_type} in `{most_recent.command}` "
                f"({most_recent.id[:8]})"
            )
        _check("Recent errors", True, detail=detail)
    except Exception as e:
        _check(
            "Recent errors",
            False,
            detail=str(e)[:120],
            fix="Error log unreadable — run `hafiz errors clear` to reset.",
        )

    # Host probe + tuning (phase 2 of the tunable-registry work item).
    # Host probe is cheap (/proc/meminfo + nvidia-smi); tuning
    # recommendations only populate when probe=True, so the default
    # `hafiz doctor` stays interactive.
    from hafiz.core.host_probe import probe_host

    host = probe_host()
    tuning = _collect_tuning(host, probe=probe)

    # Persist recommendations to sticky state when asked. We only write
    # rows that actually produced a recommendation (skip policy caps and
    # failed probes); sticky never silently lowers a value below default.
    applied: list[dict] = []
    interactive = False
    if apply:
        interactive = (
            not output_json
            and not assume_yes
            and _is_interactive()
            and _has_pending_recommendations(tuning)
        )
        if interactive:
            console.print()
            console.print(
                "[bold]Per-tunable review.[/bold] [dim]Each prompt: [Y]es to "
                "apply, [n]o to skip, [c]ustom to enter your own value.[/dim]"
            )
            tuning = _interactive_filter(tuning)
        applied = _apply_tuning(host, tuning)

    # ── Output ─────────────────────────────────────────────────────────

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "checks": checks,
                    "host": host.as_dict(),
                    "tuning": tuning,
                    "applied": applied,
                }
            )
        )
        return

    console.print()
    all_passed = True
    for chk in checks:
        icon = "\u2705" if chk["passed"] else "\u274c"
        line = f"{icon} [bold]{chk['name']}[/bold]"
        if chk["detail"]:
            line += f"  [dim]{chk['detail']}[/dim]"
        console.print(line)
        if not chk["passed"] and chk["fix"]:
            console.print(f"   [yellow]\u2192 {chk['fix']}[/yellow]")
            all_passed = False

    _render_host_table(host)
    _render_tuning_table(tuning, probe=probe)

    console.print()
    if all_passed:
        console.print("[green]All checks passed.[/green]")
    else:
        console.print("[yellow]Some checks failed — see suggestions above.[/yellow]")
    if apply and applied:
        console.print()
        for a in applied:
            console.print(
                f"[green]Applied[/green] [bold]{a['key']}[/bold] = {a['value']} "
                f"(confidence {a['confidence']})"
            )
        console.print(
            "[dim]Persisted to sticky cache. "
            "Run [bold]hafiz config clear-sticky[/bold] to revert.[/dim]"
        )
    elif apply and not applied:
        console.print()
        console.print("[yellow]No probed recommendations to apply.[/yellow]")
    if not probe:
        console.print(
            "[dim]Run [bold]hafiz doctor --probe[/bold] to measure this host "
            "and get per-tunable recommendations.[/dim]"
        )
    console.print()


# ── Tuning helpers ─────────────────────────────────────────────────────


def _is_interactive() -> bool:
    """True when both stdin and stdout are a real TTY.

    Used to decide whether to prompt for per-tunable review. Piped or
    redirected runs (CI, pre-commit, ``hafiz config apply | tee``)
    silently fall back to ``--yes`` semantics — prompting in that
    context would hang on an unanswerable read.
    """
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _has_pending_recommendations(rows: list[dict]) -> bool:
    """True when at least one row would actually be applied — i.e. has a
    recommendation that differs from the current effective value and
    didn't error out. Used to skip the prompt header when there's
    nothing to ask about."""
    for r in rows:
        if r.get("probe_error"):
            continue
        rec = r.get("recommended")
        if rec is None:
            continue
        if rec == r.get("current"):
            continue
        return True
    return False


def _interactive_filter(rows: list[dict]) -> list[dict]:
    """Walk recommendation rows; ask the user accept / skip / custom for
    each one. Returns the (possibly modified) row list — skipped rows
    have ``recommended`` cleared to None so ``_apply_tuning`` ignores
    them; custom rows have ``recommended`` replaced with the user's
    value, ``confidence`` flagged as ``user``, and ``rationale``
    rewritten to record the override.

    Rows that have no actionable recommendation (policy caps, probe
    errors, recommendation == current) pass through untouched.
    """
    from hafiz.core import tunables as _tunables

    out: list[dict] = []
    for r in rows:
        rec = r.get("recommended")
        if r.get("probe_error") or rec is None or rec == r.get("current"):
            out.append(r)
            continue

        console.print()
        console.print(
            f"[bold]{r['key']}[/bold]: "
            f"current=[yellow]{r['current']}[/yellow] → "
            f"recommended=[green]{rec}[/green] "
            f"[dim](confidence: {r.get('confidence') or '—'})[/dim]"
        )
        if r.get("rationale"):
            console.print(f"  [dim]{r['rationale']}[/dim]")

        choice = Prompt.ask(
            "  Apply?",
            choices=["y", "n", "c"],
            default="y",
            show_choices=True,
        )

        r2 = dict(r)
        if choice == "n":
            r2["recommended"] = None
            r2["user_choice"] = "skip"
            console.print("  [dim]skipped[/dim]")
        elif choice == "c":
            try:
                t = _tunables.get(r["key"])
            except KeyError:
                console.print(f"  [red]Unknown tunable {r['key']!r}; skipping.[/red]")
                r2["recommended"] = None
                r2["user_choice"] = "skip"
                out.append(r2)
                continue
            while True:
                raw = Prompt.ask(f"  Enter custom value for {r['key']}")
                try:
                    val = _tunables._coerce(t, raw)
                except (ValueError, TypeError) as exc:
                    console.print(f"  [red]Invalid:[/red] {exc}")
                    continue
                if t.validator is not None:
                    try:
                        t.validator(val)
                    except ValueError as exc:
                        console.print(f"  [red]Invalid:[/red] {exc}")
                        continue
                break
            r2["recommended"] = val
            r2["confidence"] = "user"
            r2["rationale"] = f"User-supplied value (probe originally recommended {rec})."
            r2["measured"] = {"path": "user_override", "probe_recommended": rec}
            r2["user_choice"] = "custom"
        else:
            r2["user_choice"] = "accept"

        out.append(r2)

    return out


def _apply_tuning(host, rows: list[dict]) -> list[dict]:
    """Persist probed recommendations to sticky state and return a
    summary of what was applied.

    Skips policy caps, failed probes, and rows whose recommendation
    matches the current effective value (no point writing a no-op).
    The returned summary mirrors the JSON shape agents expect to read.
    """
    from hafiz.core.tuning_state import (
        TuningEntry,
        load_state,
        merge_into_state,
        save_state,
    )

    new_entries: dict[str, TuningEntry] = {}
    summary: list[dict] = []
    now_iso = datetime_now_iso()

    for r in rows:
        if r.get("recommended") is None:
            continue
        if r.get("probe_error"):
            continue
        if r["recommended"] == r["current"]:
            # Already effective — don't write a sticky entry that just
            # duplicates current state.
            continue
        entry = TuningEntry(
            value=r["recommended"],
            rationale=r.get("rationale"),
            confidence=r.get("confidence"),
            probed_at=now_iso,
            measured=r.get("measured") or {},
        )
        new_entries[r["key"]] = entry
        summary.append(
            {
                "key": r["key"],
                "value": entry.value,
                "rationale": entry.rationale,
                "confidence": entry.confidence,
            }
        )

    if not new_entries:
        return summary

    existing = load_state()
    merged = merge_into_state(
        existing,
        fingerprint=host.fingerprint,
        ort_version=host.onnxruntime_version,
        new_entries=new_entries,
    )
    save_state(merged)
    return summary


def _collect_tuning(host, *, probe: bool) -> list[dict]:
    """Build per-tunable report rows for both JSON and Rich output.

    Shape (stable for agents; additive changes only):

        {
          "key": "embedding.max_part_chars",
          "current": 2000,
          "default": 2000,
          "description": "...",
          "is_policy": False,
          "recommended": int | None,   # only populated when probe=True
          "rationale": str | None,
          "confidence": "high|medium|low" | None,
          "measured": dict | None,
          "probe_error": str | None,
        }
    """
    from hafiz.core import tunables as _tunables

    rows: list[dict] = []
    for t in _tunables.all_tunables():
        current = _tunables.resolve(t.key)
        row: dict = {
            "key": t.key,
            "current": current,
            "default": t.default,
            "description": t.description,
            "is_policy": t.is_policy,
            "recommended": None,
            "rationale": None,
            "confidence": None,
            "measured": None,
            "probe_error": None,
        }
        if probe and t.prober is not None:
            try:
                result = t.prober(host)
                row["recommended"] = result.recommended_value
                row["rationale"] = result.rationale
                row["confidence"] = result.confidence
                row["measured"] = dict(result.measured)
            except Exception as e:
                row["probe_error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
    return rows


def _render_host_table(host) -> None:
    console.print()
    tbl = Table(title="Host", show_header=False, border_style="cyan")
    tbl.add_column("Key", style="bold")
    tbl.add_column("Value")

    def _mb(n):
        return f"{n:,} MB" if isinstance(n, int) else "—"

    tbl.add_row("platform", host.platform)
    tbl.add_row("cpu_count", str(host.cpu_count) if host.cpu_count else "—")
    tbl.add_row("ram_total", _mb(host.ram_total_mb))
    tbl.add_row("ram_available", _mb(host.ram_available_mb))
    if host.swap_total_mb:
        tbl.add_row(
            "swap",
            f"{host.swap_used_mb or 0:,}/{host.swap_total_mb:,} MB used",
        )
    tbl.add_row(
        "onnx_providers",
        ", ".join(host.onnx_providers) if host.onnx_providers else "—",
    )
    if host.onnxruntime_version:
        tbl.add_row("onnxruntime", host.onnxruntime_version)
    if host.gpu_name:
        tbl.add_row("gpu", host.gpu_name)
        tbl.add_row(
            "gpu_vram",
            f"{host.gpu_vram_free_mb or 0:,} MB free / {host.gpu_vram_total_mb or 0:,} MB total",
        )
    tbl.add_row("fingerprint", host.fingerprint)
    console.print(tbl)


def _render_tuning_table(rows: list[dict], *, probe: bool) -> None:
    console.print()
    title = "Tuning recommendations" if probe else "Tuning — current values"
    tbl = Table(title=title, border_style="cyan")
    tbl.add_column("Key", style="bold")
    tbl.add_column("Current")
    if probe:
        tbl.add_column("Recommended")
        tbl.add_column("Confidence")
    tbl.add_column("Notes", style="dim")

    for r in rows:
        key = r["key"]
        current = str(r["current"])
        notes: list[str] = []
        if r["is_policy"]:
            notes.append("policy cap (not probed)")
        if probe and r.get("probe_error"):
            notes.append(f"probe error: {r['probe_error']}")
        elif probe and r.get("rationale"):
            notes.append(r["rationale"])
        notes_s = " — ".join(notes) or "—"

        if probe:
            rec = r["recommended"]
            if rec is None:
                rec_cell = "—"
            elif rec != r["current"]:
                rec_cell = f"[green]{rec}[/green]"
            else:
                rec_cell = str(rec)
            tbl.add_row(key, current, rec_cell, r["confidence"] or "—", notes_s)
        else:
            tbl.add_row(key, current, notes_s)

    console.print(tbl)
