"""hafiz init, status, config — maintenance commands."""

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
