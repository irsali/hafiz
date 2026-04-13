"""hafiz graph — explore entities and relationships in the knowledge graph."""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from hafiz.core.database import close_engine, get_session_factory, Entity, Relation

console = Console()


# ── Helpers ────────────────────────────────────────────────────────────────


async def _find_entity(name: str, project: str | None = None) -> Entity | None:
    """Look up an entity by name (case-insensitive)."""
    from sqlalchemy import select, func

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Entity).where(func.lower(Entity.name) == name.lower())
        if project:
            stmt = stmt.where(Entity.project == project)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()


async def _get_outgoing(entity_id, session) -> list[tuple[Relation, Entity]]:
    """Get outgoing relations (this entity -> others)."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    result = await session.execute(
        select(Relation)
        .where(Relation.source_id == entity_id)
        .options(selectinload(Relation.target))
    )
    relations = result.scalars().all()
    return [(r, r.target) for r in relations]


async def _get_incoming(entity_id, session) -> list[tuple[Relation, Entity]]:
    """Get incoming relations (others -> this entity)."""
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    result = await session.execute(
        select(Relation)
        .where(Relation.target_id == entity_id)
        .options(selectinload(Relation.source))
    )
    relations = result.scalars().all()
    return [(r, r.source) for r in relations]


def _entity_label(entity: Entity) -> str:
    """Format an entity for display."""
    return f"[bold]{entity.name}[/bold] [dim]({entity.entity_type})[/dim]"


def _entity_dict(entity: Entity) -> dict:
    """Serialize an entity for JSON output."""
    return {
        "id": str(entity.id),
        "name": entity.name,
        "entity_type": entity.entity_type,
        "description": entity.description,
        "project": entity.project,
        "source_file": entity.source_file,
    }


def _relation_dict(relation: Relation, other: Entity) -> dict:
    """Serialize a relation + connected entity for JSON output."""
    return {
        "relation_type": relation.relation_type,
        "weight": relation.weight,
        "evidence": relation.evidence,
        "entity": _entity_dict(other),
    }


# ── Commands ───────────────────────────────────────────────────────────────


def run_graph_show(
    name: str,
    *,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    """Show an entity and all its direct connections."""

    async def _show():
        try:
            entity = await _find_entity(name, project=project)
            if entity is None:
                console.print(f"[red]Entity not found:[/red] {name}")
                raise SystemExit(1)

            session_factory = get_session_factory()
            async with session_factory() as session:
                # Re-fetch within session for relationship loading
                from sqlalchemy import select

                entity = (
                    await session.execute(
                        select(Entity).where(Entity.id == entity.id)
                    )
                ).scalar_one()

                outgoing = await _get_outgoing(entity.id, session)
                incoming = await _get_incoming(entity.id, session)

            if output_json:
                data = {
                    "entity": _entity_dict(entity),
                    "outgoing": [
                        _relation_dict(r, target) for r, target in outgoing
                    ],
                    "incoming": [
                        _relation_dict(r, source) for r, source in incoming
                    ],
                }
                console.print_json(json.dumps(data, default=str))
                return

            # Rich display
            console.print()
            desc = entity.description or "(no description)"
            info = (
                f"[bold cyan]{entity.name}[/bold cyan] "
                f"[dim]({entity.entity_type})[/dim]\n"
                f"{desc}"
            )
            if entity.source_file:
                info += f"\n[dim]Source: {entity.source_file}[/dim]"
            console.print(Panel(info, title="Entity", border_style="cyan"))

            if outgoing:
                tree = Tree(f"[bold]Outgoing[/bold] ({len(outgoing)})")
                for rel, target in outgoing:
                    label = f"--[yellow]{rel.relation_type}[/yellow]--> {_entity_label(target)}"
                    branch = tree.add(label)
                    if rel.evidence:
                        branch.add(f"[dim]{rel.evidence[:120]}[/dim]")
                console.print(tree)

            if incoming:
                tree = Tree(f"[bold]Incoming[/bold] ({len(incoming)})")
                for rel, source in incoming:
                    label = f"{_entity_label(source)} --[yellow]{rel.relation_type}[/yellow]-->"
                    branch = tree.add(label)
                    if rel.evidence:
                        branch.add(f"[dim]{rel.evidence[:120]}[/dim]")
                console.print(tree)

            if not outgoing and not incoming:
                console.print("[dim]No connections found.[/dim]")

            console.print()

        finally:
            await close_engine()

    asyncio.run(_show())


def run_graph_deps(
    name: str,
    *,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    """Show what an entity depends on (outgoing relations)."""

    async def _deps():
        try:
            entity = await _find_entity(name, project=project)
            if entity is None:
                console.print(f"[red]Entity not found:[/red] {name}")
                raise SystemExit(1)

            session_factory = get_session_factory()
            async with session_factory() as session:
                outgoing = await _get_outgoing(entity.id, session)

            if output_json:
                data = {
                    "entity": name,
                    "dependencies": [
                        _relation_dict(r, target) for r, target in outgoing
                    ],
                }
                console.print_json(json.dumps(data, default=str))
                return

            console.print()
            if not outgoing:
                console.print(f"[dim]{name} has no outgoing dependencies.[/dim]")
                return

            table = Table(
                title=f"Dependencies of {name}",
                border_style="cyan",
            )
            table.add_column("Relation", style="yellow")
            table.add_column("Target", style="bold")
            table.add_column("Type", style="dim")
            table.add_column("Evidence")

            for rel, target in outgoing:
                evidence = (rel.evidence or "")[:80]
                table.add_row(
                    rel.relation_type,
                    target.name,
                    target.entity_type,
                    evidence,
                )

            console.print(table)
            console.print()

        finally:
            await close_engine()

    asyncio.run(_deps())


def run_graph_dependents(
    name: str,
    *,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    """Show what depends on an entity (incoming relations)."""

    async def _dependents():
        try:
            entity = await _find_entity(name, project=project)
            if entity is None:
                console.print(f"[red]Entity not found:[/red] {name}")
                raise SystemExit(1)

            session_factory = get_session_factory()
            async with session_factory() as session:
                incoming = await _get_incoming(entity.id, session)

            if output_json:
                data = {
                    "entity": name,
                    "dependents": [
                        _relation_dict(r, source) for r, source in incoming
                    ],
                }
                console.print_json(json.dumps(data, default=str))
                return

            console.print()
            if not incoming:
                console.print(f"[dim]Nothing depends on {name}.[/dim]")
                return

            table = Table(
                title=f"Dependents of {name}",
                border_style="cyan",
            )
            table.add_column("Source", style="bold")
            table.add_column("Type", style="dim")
            table.add_column("Relation", style="yellow")
            table.add_column("Evidence")

            for rel, source in incoming:
                evidence = (rel.evidence or "")[:80]
                table.add_row(
                    source.name,
                    source.entity_type,
                    rel.relation_type,
                    evidence,
                )

            console.print(table)
            console.print()

        finally:
            await close_engine()

    asyncio.run(_dependents())
