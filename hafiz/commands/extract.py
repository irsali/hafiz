"""hafiz extract import — import entity/relation extraction results from JSON.

Designed for agent-driven extraction: Claude Code (or any LLM) analyses
chunks exported by ``hafiz chunks export`` and produces a JSON payload that
this command stores into the knowledge graph.
"""

from __future__ import annotations

import asyncio
import json
import sys

from rich.console import Console

from hafiz.core.database import close_engine
from hafiz.core.extractor import (
    ExtractionResult,
    ExtractedEntity,
    ExtractedRelation,
    store_extraction,
)

console = Console()


def _parse_extraction_json(data: dict) -> ExtractionResult:
    """Parse a JSON payload into an ExtractionResult."""
    entities = [
        ExtractedEntity(
            name=e["name"],
            entity_type=e["entity_type"],
            description=e.get("description", ""),
            source_file=e.get("source_file"),
            chunk_id=e.get("chunk_id"),
        )
        for e in data.get("entities", [])
    ]
    relations = [
        ExtractedRelation(
            source_name=r["source_name"],
            source_type=r["source_type"],
            target_name=r["target_name"],
            target_type=r["target_type"],
            relation_type=r["relation_type"],
            evidence=r.get("evidence", ""),
        )
        for r in data.get("relations", [])
    ]
    return ExtractionResult(entities=entities, relations=relations)


def run_extract_import(
    file: str | None = None,
    *,
    project: str | None = None,
) -> None:
    """Import extraction results from a JSON file or stdin."""

    async def _run():
        try:
            if file:
                with open(file) as f:
                    data = json.load(f)
            else:
                data = json.load(sys.stdin)

            result = _parse_extraction_json(data)
            ent_count, rel_count = await store_extraction(result, project=project)
            console.print(
                f"[green]Imported {ent_count} entities, {rel_count} relations[/green]"
            )
        finally:
            await close_engine()

    asyncio.run(_run())
