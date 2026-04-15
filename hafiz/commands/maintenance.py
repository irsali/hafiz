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
            console.print("  - Tables created: chunks, entities, relations, observations")

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


def run_status(*, output_json: bool = False) -> None:
    """Show database statistics and index health."""

    async def _status():
        try:
            from sqlalchemy import func, select
            from hafiz.core.database import Chunk, Entity, Relation, Observation

            session_factory = get_session_factory()
            async with session_factory() as session:
                # Count all tables
                chunk_count = (
                    await session.execute(select(func.count()).select_from(Chunk))
                ).scalar() or 0
                entity_count = (
                    await session.execute(select(func.count()).select_from(Entity))
                ).scalar() or 0
                relation_count = (
                    await session.execute(select(func.count()).select_from(Relation))
                ).scalar() or 0
                observation_count = (
                    await session.execute(select(func.count()).select_from(Observation))
                ).scalar() or 0

                # Chunks by project
                project_counts = (
                    await session.execute(
                        select(Chunk.project, func.count())
                        .group_by(Chunk.project)
                        .order_by(func.count().desc())
                    )
                ).all()

                # Chunks by type
                type_counts = (
                    await session.execute(
                        select(Chunk.chunk_type, func.count())
                        .group_by(Chunk.chunk_type)
                        .order_by(func.count().desc())
                    )
                ).all()

                # Unique source files
                file_count = (
                    await session.execute(
                        select(func.count(func.distinct(Chunk.source_file)))
                    )
                ).scalar() or 0

            stats = {
                "chunks": chunk_count,
                "entities": entity_count,
                "relations": relation_count,
                "observations": observation_count,
                "files": file_count,
                "by_project": {p or "(none)": c for p, c in project_counts},
                "by_type": {t or "(none)": c for t, c in type_counts},
            }
            return stats
        finally:
            await close_engine()

    stats = asyncio.run(_status())

    if output_json:
        console.print_json(json.dumps(stats))
        return

    # Rich display
    table = Table(title="Hafiz Status", show_header=False, border_style="cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Chunks", str(stats["chunks"]))
    table.add_row("Source files", str(stats["files"]))
    table.add_row("Entities", str(stats["entities"]))
    table.add_row("Relations", str(stats["relations"]))
    table.add_row("Observations", str(stats["observations"]))

    console.print()
    console.print(table)

    if stats["by_project"]:
        console.print()
        proj_table = Table(title="Chunks by Project", border_style="cyan")
        proj_table.add_column("Project")
        proj_table.add_column("Chunks", justify="right")
        for proj, count in stats["by_project"].items():
            proj_table.add_row(proj, str(count))
        console.print(proj_table)

    if stats["by_type"]:
        console.print()
        type_table = Table(title="Chunks by Type", border_style="cyan")
        type_table.add_column("Type")
        type_table.add_column("Chunks", justify="right")
        for ctype, count in stats["by_type"].items():
            type_table.add_row(ctype, str(count))
        console.print(type_table)


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


def run_doctor(*, output_json: bool = False) -> None:
    """Run diagnostic checks on the Hafiz installation."""

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
            from sqlalchemy import func, inspect, select, text
            from hafiz.core.database import Chunk, Entity, Relation, Observation

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

            # 6. Tables exist
            expected_tables = {"chunks", "entities", "relations", "observations"}
            try:
                from sqlalchemy import inspect as sa_inspect

                engine = session_factory.kw.get("bind") or get_session_factory().kw.get("bind")
                # Use a raw connection to inspect tables
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

            # 7. Table row counts
            try:
                async with session_factory() as session:
                    chunk_count = (
                        await session.execute(select(func.count()).select_from(Chunk))
                    ).scalar() or 0
                    entity_count = (
                        await session.execute(select(func.count()).select_from(Entity))
                    ).scalar() or 0
                    relation_count = (
                        await session.execute(
                            select(func.count()).select_from(Relation)
                        )
                    ).scalar() or 0
                    observation_count = (
                        await session.execute(
                            select(func.count()).select_from(Observation)
                        )
                    ).scalar() or 0

                _check(
                    "Table row counts",
                    True,
                    detail=(
                        f"chunks={chunk_count}, entities={entity_count}, "
                        f"relations={relation_count}, observations={observation_count}"
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

    # ── Output ─────────────────────────────────────────────────────────

    if output_json:
        console.print_json(json.dumps({"checks": checks}))
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

    console.print()
    if all_passed:
        console.print("[green]All checks passed.[/green]")
    else:
        console.print("[yellow]Some checks failed — see suggestions above.[/yellow]")
    console.print()
