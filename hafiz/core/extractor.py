"""Entity and relationship extraction — data classes and storage.

Extraction is agent-driven: the agent reads chunks (via ``hafiz chunks export``),
identifies entities and relationships, and pipes the result into
``hafiz extract import``. No external API key is needed — the agent IS the brain.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

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
