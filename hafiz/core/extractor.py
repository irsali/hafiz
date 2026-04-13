"""LLM-powered entity and relationship extraction from code chunks.

Uses the Anthropic Python SDK directly with structured output (tool_use)
to extract entities and relationships from ingested chunks.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from hafiz.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

ENTITY_TYPES = [
    "class",
    "function",
    "module",
    "api_endpoint",
    "database_table",
    "concept",
    "config",
    "service",
]

RELATION_TYPES = [
    "calls",
    "imports",
    "inherits",
    "depends_on",
    "defines",
    "reads",
    "writes",
    "configures",
    "implements",
]

EXTRACTION_BATCH_SIZE = 3

# ── Data classes ───────────────────────────────────────────────────────────


@dataclass
class ExtractedEntity:
    name: str
    entity_type: str
    description: str
    source_file: str | None = None
    chunk_id: str | None = None


@dataclass
class ExtractedRelation:
    source_name: str
    source_type: str
    target_name: str
    target_type: str
    relation_type: str
    evidence: str = ""


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)


# ── Tool schema for Claude structured output ───────────────────────────────

EXTRACTION_TOOL = {
    "name": "extract_graph",
    "description": (
        "Extract entities and relationships from code chunks. "
        "Identify classes, functions, modules, APIs, database tables, concepts, "
        "configs, and services — and the relationships between them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "description": "Entities found in the code chunks.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The entity name (e.g. class name, function name).",
                        },
                        "entity_type": {
                            "type": "string",
                            "enum": ENTITY_TYPES,
                            "description": "The type of entity.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Brief description of what this entity does.",
                        },
                        "chunk_index": {
                            "type": "integer",
                            "description": "Index of the chunk (0-based) this entity was found in.",
                        },
                    },
                    "required": ["name", "entity_type", "description", "chunk_index"],
                },
            },
            "relations": {
                "type": "array",
                "description": "Relationships between entities found in the code.",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_name": {
                            "type": "string",
                            "description": "Name of the source entity.",
                        },
                        "source_type": {
                            "type": "string",
                            "enum": ENTITY_TYPES,
                            "description": "Type of the source entity.",
                        },
                        "target_name": {
                            "type": "string",
                            "description": "Name of the target entity.",
                        },
                        "target_type": {
                            "type": "string",
                            "enum": ENTITY_TYPES,
                            "description": "Type of the target entity.",
                        },
                        "relation_type": {
                            "type": "string",
                            "enum": RELATION_TYPES,
                            "description": "The type of relationship.",
                        },
                        "evidence": {
                            "type": "string",
                            "description": "The code snippet or text that proves this relationship.",
                        },
                    },
                    "required": [
                        "source_name",
                        "source_type",
                        "target_name",
                        "target_type",
                        "relation_type",
                        "evidence",
                    ],
                },
            },
        },
        "required": ["entities", "relations"],
    },
}


# ── Extraction prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a code analysis assistant. Your job is to extract entities and \
relationships from code chunks.

Entity types: class, function, module, api_endpoint, database_table, \
concept, config, service.

Relation types: calls, imports, inherits, depends_on, defines, reads, \
writes, configures, implements.

Rules:
- Only extract entities that are DEFINED or CLEARLY REFERENCED in the code.
- Use the exact name as it appears in the code (e.g. "MyClass", "get_user").
- For relations, provide the actual code snippet as evidence.
- Be precise — do not invent entities or relationships not in the code.
- If a chunk has no meaningful entities, return empty arrays.\
"""


def _build_user_message(chunks: list[dict]) -> str:
    """Build the user message for a batch of chunks."""
    parts = []
    for i, chunk in enumerate(chunks):
        source = chunk.get("source_file", "unknown")
        language = chunk.get("language", "")
        content = chunk["content"]
        parts.append(
            f"--- Chunk {i} (file: {source}, language: {language}) ---\n{content}"
        )
    return "\n\n".join(parts)


# ── Client ─────────────────────────────────────────────────────────────────


def _get_client():
    """Get the Anthropic client. Returns None if API key is not set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    import anthropic

    return anthropic.Anthropic(api_key=api_key)


# ── Core extraction ────────────────────────────────────────────────────────


async def extract_from_chunks(
    chunks: list[dict],
) -> ExtractionResult:
    """Extract entities and relationships from a batch of chunks using Claude.

    Each chunk dict should have: content, source_file, language, chunk_id.
    Batch size should be <= EXTRACTION_BATCH_SIZE (3).

    Returns ExtractionResult with entities and relations.
    """
    client = _get_client()
    if client is None:
        logger.warning("ANTHROPIC_API_KEY not set — skipping extraction")
        return ExtractionResult()

    settings = get_settings()
    model = settings.llm.model

    user_message = _build_user_message(chunks)

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "extract_graph"},
            messages=[{"role": "user", "content": user_message}],
        )
    except Exception as e:
        logger.warning("LLM extraction failed: %s", e)
        return ExtractionResult()

    # Parse the tool_use response
    result = ExtractionResult()

    for block in response.content:
        if block.type != "tool_use" or block.name != "extract_graph":
            continue

        data = block.input

        for ent in data.get("entities", []):
            chunk_idx = ent.get("chunk_index", 0)
            source_chunk = chunks[chunk_idx] if chunk_idx < len(chunks) else chunks[0]
            result.entities.append(
                ExtractedEntity(
                    name=ent["name"],
                    entity_type=ent["entity_type"],
                    description=ent.get("description", ""),
                    source_file=source_chunk.get("source_file"),
                    chunk_id=source_chunk.get("chunk_id"),
                )
            )

        for rel in data.get("relations", []):
            result.relations.append(
                ExtractedRelation(
                    source_name=rel["source_name"],
                    source_type=rel["source_type"],
                    target_name=rel["target_name"],
                    target_type=rel["target_type"],
                    relation_type=rel["relation_type"],
                    evidence=rel.get("evidence", ""),
                )
            )

    return result


# ── Database storage ───────────────────────────────────────────────────────


async def store_extraction(
    result: ExtractionResult,
    *,
    project: str | None = None,
) -> tuple[int, int]:
    """Store extracted entities and relations into the database.

    Upserts entities by (name, entity_type, project).
    Returns (entities_stored, relations_stored).
    """
    if not result.entities and not result.relations:
        return 0, 0

    from sqlalchemy import select
    from hafiz.core.database import Entity, Relation, get_session_factory

    session_factory = get_session_factory()
    entity_count = 0
    relation_count = 0

    # Map (name, type) -> Entity id for relation linking
    entity_map: dict[tuple[str, str], uuid.UUID] = {}

    async with session_factory() as session:
        async with session.begin():
            # Upsert entities
            for ent in result.entities:
                # Check if entity already exists
                existing = (
                    await session.execute(
                        select(Entity).where(
                            Entity.name == ent.name,
                            Entity.entity_type == ent.entity_type,
                            Entity.project == project,
                        )
                    )
                ).scalar_one_or_none()

                if existing:
                    # Update description and properties
                    existing.description = ent.description
                    existing.source_file = ent.source_file
                    props = existing.properties or {}
                    if ent.chunk_id:
                        chunk_ids = props.get("chunk_ids", [])
                        if ent.chunk_id not in chunk_ids:
                            chunk_ids.append(ent.chunk_id)
                        props["chunk_ids"] = chunk_ids
                    existing.properties = props
                    existing.updated_at = datetime.now(timezone.utc)
                    entity_map[(ent.name, ent.entity_type)] = existing.id
                else:
                    props = {}
                    if ent.chunk_id:
                        props["chunk_ids"] = [ent.chunk_id]
                    new_entity = Entity(
                        id=uuid.uuid4(),
                        name=ent.name,
                        entity_type=ent.entity_type,
                        description=ent.description,
                        project=project,
                        source_file=ent.source_file,
                        properties=props,
                    )
                    session.add(new_entity)
                    entity_map[(ent.name, ent.entity_type)] = new_entity.id
                    entity_count += 1

            # Flush to ensure entity IDs are available
            await session.flush()

            # Store relations
            for rel in result.relations:
                source_key = (rel.source_name, rel.source_type)
                target_key = (rel.target_name, rel.target_type)

                source_id = entity_map.get(source_key)
                target_id = entity_map.get(target_key)

                # If either entity wasn't in this batch, look it up
                if source_id is None:
                    row = (
                        await session.execute(
                            select(Entity.id).where(
                                Entity.name == rel.source_name,
                                Entity.entity_type == rel.source_type,
                                Entity.project == project,
                            )
                        )
                    ).scalar_one_or_none()
                    if row:
                        source_id = row
                        entity_map[source_key] = source_id

                if target_id is None:
                    row = (
                        await session.execute(
                            select(Entity.id).where(
                                Entity.name == rel.target_name,
                                Entity.entity_type == rel.target_type,
                                Entity.project == project,
                            )
                        )
                    ).scalar_one_or_none()
                    if row:
                        target_id = row
                        entity_map[target_key] = target_id

                if source_id is None or target_id is None:
                    logger.debug(
                        "Skipping relation %s->%s: entity not found",
                        rel.source_name,
                        rel.target_name,
                    )
                    continue

                new_relation = Relation(
                    id=uuid.uuid4(),
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=rel.relation_type,
                    evidence=rel.evidence,
                )
                session.add(new_relation)
                relation_count += 1

    return entity_count, relation_count


# ── Batch orchestrator ─────────────────────────────────────────────────────


async def run_extraction(
    chunks: list[dict],
    *,
    project: str | None = None,
    on_progress: callable | None = None,
) -> tuple[int, int]:
    """Run extraction on a list of chunk dicts, batching by EXTRACTION_BATCH_SIZE.

    Each chunk dict should have: content, source_file, language, chunk_id.

    Args:
        chunks: List of chunk dicts.
        project: Project tag for entity storage.
        on_progress: Optional callback(batch_size) called after each batch.

    Returns (total_entities, total_relations).
    """
    if not chunks:
        return 0, 0

    # Check API key upfront
    if not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set — skipping graph extraction")
        return 0, 0

    total_entities = 0
    total_relations = 0

    for i in range(0, len(chunks), EXTRACTION_BATCH_SIZE):
        batch = chunks[i : i + EXTRACTION_BATCH_SIZE]

        result = await extract_from_chunks(batch)
        ent_count, rel_count = await store_extraction(result, project=project)

        total_entities += ent_count
        total_relations += rel_count

        if on_progress:
            on_progress(len(batch))

    return total_entities, total_relations
