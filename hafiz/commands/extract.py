"""hafiz extract import / export — agent extraction v2.

The agent workflow narrows post-structural-grounding: parsers own
structural facts (entities, calls, imports, inherits); agents own
semantic meaning (annotations, concepts, patterns, workarounds).

``hafiz extract export`` surfaces the AST-known units so agents can see
the structure that's already captured and attach their annotations to
it instead of re-deriving. ``hafiz extract import`` accepts the v2
contract (see :mod:`hafiz.core.extractor`) and loudly rejects v1
payloads with a migration message.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from rich.console import Console
from sqlalchemy import select

from hafiz.core.database import (
    Edge,
    File,
    Unit,
    UnitRevision,
    close_engine,
    get_session_factory,
)
from hafiz.core.extractor import (
    EXTRACT_CONTRACT_VERSION,
    ExtractContractError,
    parse_extraction_payload,
    store_extraction,
)

console = Console()


# ── import ─────────────────────────────────────────────────────────────────


def run_extract_import(
    file: str | None = None,
    *,
    project: str | None = None,
) -> None:
    """Import an agent extraction payload from a file or stdin."""

    async def _run():
        try:
            raw = _read_json(file)
            try:
                result = parse_extraction_payload(raw)
            except ExtractContractError as exc:
                console.print(f"[red]Contract error:[/red] {exc}")
                raise SystemExit(2)

            ann_count, edge_count, unresolved = await store_extraction(
                result, project=project
            )

            console.print(
                f"[green]Imported {ann_count} annotations, {edge_count} edges[/green]"
            )
            if unresolved:
                console.print(
                    f"  [yellow]{unresolved} reference(s) could not be "
                    f"resolved to a unit — stored unresolved.[/yellow]"
                )
            for w in result.warnings:
                console.print(f"  [yellow]warning:[/yellow] {w}")
        finally:
            await close_engine()

    asyncio.run(_run())


def _read_json(file: str | None) -> dict[str, Any]:
    if file:
        with open(file) as f:
            return json.load(f)
    return json.load(sys.stdin)


# ── export ─────────────────────────────────────────────────────────────────


def run_extract_export(
    *,
    project: str | None = None,
    limit: int = 500,
    output_json: bool = True,
) -> None:
    """Emit the AST-known units/edges the agent can attach annotations to.

    Replaces the old "export unextracted chunks" flow: there are no
    unextracted chunks anymore — parsing happens at ingest time, not
    at extract time. This export is pure read-side: here's what's
    already in the graph, annotate it.
    """

    async def _run() -> dict[str, Any]:
        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                unit_stmt = (
                    select(Unit, File, UnitRevision)
                    .join(File, File.id == Unit.file_id)
                    .join(
                        UnitRevision,
                        (UnitRevision.unit_id == Unit.id)
                        & (UnitRevision.superseded_at.is_(None)),
                    )
                    .where(Unit.valid_until.is_(None))
                    .where(File.valid_until.is_(None))
                    .order_by(File.path, Unit.line_start)
                    .limit(limit)
                )
                if project is not None:
                    unit_stmt = unit_stmt.where(File.project == project)
                unit_rows = (await session.execute(unit_stmt)).all()

                unit_ids = [u.id for u, _, _ in unit_rows]

                edge_stmt = (
                    select(Edge)
                    .where(Edge.superseded_at.is_(None))
                    .where(Edge.source == "ast")
                )
                if unit_ids:
                    edge_stmt = edge_stmt.where(
                        Edge.source_unit_id.in_(unit_ids)
                    )
                else:
                    edge_stmt = edge_stmt.where(
                        Edge.source_unit_id.in_([])
                    )
                edges = (await session.execute(edge_stmt)).scalars().all()

            return {
                "version": EXTRACT_CONTRACT_VERSION,
                "project": project,
                "units": [
                    {
                        "identity_key": u.identity_key,
                        "name": u.name,
                        "parent_name": u.parent_name,
                        "kind": u.kind,
                        "source_file": f.path,
                        "line_start": rev.line_start,
                        "line_end": rev.line_end,
                    }
                    for u, f, rev in unit_rows
                ],
                "edges": [
                    {
                        "source_unit_id": str(e.source_unit_id),
                        "target_unit_id": (
                            str(e.target_unit_id)
                            if e.target_unit_id
                            else None
                        ),
                        "target_name": e.target_name,
                        "relation": e.relation,
                    }
                    for e in edges
                ],
            }
        finally:
            await close_engine()

    payload = asyncio.run(_run())

    if output_json:
        console.print_json(json.dumps(payload))
    else:
        console.print(
            f"[bold]{len(payload['units'])}[/bold] units, "
            f"[bold]{len(payload['edges'])}[/bold] AST edges "
            f"(project: {project or 'all'})"
        )
