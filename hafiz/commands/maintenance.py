"""hafiz init, status, config, doctor — maintenance commands."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hafiz.core.config import get_settings, find_config_file, CONFIG_FILENAME
from hafiz.core.database import create_tables, close_engine, get_session_factory

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
                        select(func.count())
                        .select_from(File)
                        .where(File.valid_until.is_(None))
                    )
                ).scalar() or 0
                units_count = (
                    await session.execute(
                        select(func.count())
                        .select_from(Unit)
                        .where(Unit.valid_until.is_(None))
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
                    await session.execute(
                        select(func.count()).select_from(Embedding)
                    )
                ).scalar() or 0
                edges_count = (
                    await session.execute(
                        select(func.count())
                        .select_from(Edge)
                        .where(Edge.superseded_at.is_(None))
                    )
                ).scalar() or 0
                annotations_count = (
                    await session.execute(
                        select(func.count()).select_from(Annotation)
                    )
                ).scalar() or 0
                commits_count = (
                    await session.execute(
                        select(func.count()).select_from(Commit)
                    )
                ).scalar() or 0

                # ── Historical totals (include tombstoned for context) ──
                total_units = (
                    await session.execute(
                        select(func.count()).select_from(Unit)
                    )
                ).scalar() or 0
                total_revisions = (
                    await session.execute(
                        select(func.count()).select_from(UnitRevision)
                    )
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

                # ── Most-recent commit per project ──
                last_commit_rows = (
                    await session.execute(
                        select(
                            File.project,
                            func.max(File.last_seen_commit),
                        )
                        .where(File.valid_until.is_(None))
                        .where(File.last_seen_commit.is_not(None))
                        .group_by(File.project)
                    )
                ).all()

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
                "by_project": {
                    p or "(none)": c for p, c in project_rows
                },
                "by_kind": {k or "(none)": c for k, c in kind_rows},
                "last_commit_per_project": {
                    p or "(none)": c for p, c in last_commit_rows
                },
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
    dev = stats["embedding_device"]
    table.add_row(
        "Embedding device",
        f"{dev['effective']} [dim]({dev['source']})[/dim]",
    )

    console.print()
    console.print(table)

    if stats["by_project"]:
        console.print()
        proj_table = Table(title="Files by Project", border_style="cyan")
        proj_table.add_column("Project")
        proj_table.add_column("Files", justify="right")
        for proj, count in stats["by_project"].items():
            proj_table.add_row(proj, str(count))
        console.print(proj_table)

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
        commit_table = Table(
            title="Last indexed commit per project", border_style="cyan"
        )
        commit_table.add_column("Project")
        commit_table.add_column("Commit", style="dim")
        for proj, sha in stats["last_commit_per_project"].items():
            commit_table.add_row(proj, sha[:12] if sha else "—")
        console.print(commit_table)


def run_config_show(*, output_json: bool = False) -> None:
    """Show the current Hafiz configuration."""
    settings = get_settings()

    if output_json:
        console.print_json(settings.model_dump_json())
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
    console.print()


def run_doctor(*, output_json: bool = False, probe: bool = False) -> None:
    """Run diagnostic checks on the Hafiz installation.

    When ``probe`` is True, additionally runs each registered tunable's
    prober and reports recommended values. Probes can be slow (the
    embedding prober loads the model and runs several forward passes);
    the default is False so ``hafiz doctor`` stays fast.
    """

    checks: list[dict] = []

    def _check(name: str, passed: bool, detail: str = "", fix: str = "") -> None:
        checks.append(
            {"name": name, "passed": passed, "detail": detail, "fix": fix}
        )

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
                        text(
                            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                        )
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
                        text(
                            "SELECT tablename FROM pg_tables "
                            "WHERE schemaname = 'public'"
                        )
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
                            select(func.count())
                            .select_from(File)
                            .where(File.valid_until.is_(None))
                        )
                    ).scalar() or 0
                    units_count = (
                        await session.execute(
                            select(func.count())
                            .select_from(Unit)
                            .where(Unit.valid_until.is_(None))
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
                        await session.execute(
                            select(func.count()).select_from(Embedding)
                        )
                    ).scalar() or 0
                    edge_count = (
                        await session.execute(
                            select(func.count())
                            .select_from(Edge)
                            .where(Edge.superseded_at.is_(None))
                        )
                    ).scalar() or 0
                    ann_count = (
                        await session.execute(
                            select(func.count()).select_from(Annotation)
                        )
                    ).scalar() or 0
                    commit_count = (
                        await session.execute(
                            select(func.count()).select_from(Commit)
                        )
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

        finally:
            await close_engine()

    asyncio.run(_async_checks())

    # 8. Embedding model loadable (sync check — separate from DB)
    try:
        from fastembed import TextEmbedding

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
        coverage = ", ".join(
            f"{p.name}({'+'.join(registry.extensions_for(p))})" for p in parsers
        )
        _check(
            "Parser registry",
            bool(parsers),
            detail=coverage or "(no parsers registered)",
            fix="Re-install hafiz — in-tree parsers should self-register."
            if not parsers
            else "",
        )
    except Exception as e:
        _check(
            "Parser registry",
            False,
            detail=str(e)[:120],
            fix="Re-install hafiz.",
        )

    # Host probe + tuning (phase 2 of the tunable-registry work item).
    # Host probe is cheap (/proc/meminfo + nvidia-smi); tuning
    # recommendations only populate when probe=True, so the default
    # `hafiz doctor` stays interactive.
    from hafiz.core.host_probe import probe_host

    host = probe_host()
    tuning = _collect_tuning(host, probe=probe)

    # ── Output ─────────────────────────────────────────────────────────

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "checks": checks,
                    "host": host.as_dict(),
                    "tuning": tuning,
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
    if not probe:
        console.print(
            "[dim]Run [bold]hafiz doctor --probe[/bold] to measure this host "
            "and get per-tunable recommendations.[/dim]"
        )
    console.print()


# ── Tuning helpers ─────────────────────────────────────────────────────


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
            f"{host.gpu_vram_free_mb or 0:,} MB free / "
            f"{host.gpu_vram_total_mb or 0:,} MB total",
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
