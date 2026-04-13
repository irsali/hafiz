"""Context synthesizer — combines chunks, graph, and observations into a unified bundle.

The killer feature: `hafiz context "task description"` pulls together everything
Hafiz knows that's relevant to a task.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from hafiz.core.database import Entity, Relation, get_session_factory
from hafiz.core.observations import ObservationResult, search_observations
from hafiz.core.search import SearchResult, vector_search


@dataclass
class ContextBundle:
    """Everything Hafiz knows about a query, in one place."""

    query: str
    chunks: list[SearchResult] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    observations: list[ObservationResult] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the context bundle as Markdown."""
        sections = [f"# Context: {self.query}"]

        # Relevant Code
        sections.append("\n## Relevant Code")
        if self.chunks:
            for c in self.chunks:
                location = c.source_file
                if c.line_start and c.line_end:
                    location += f":{c.line_start}-{c.line_end}"
                lang = f" ({c.language})" if c.language else ""
                sections.append(f"\n### {location}{lang}  — similarity {c.score:.2%}")
                sections.append(f"```{c.language or ''}\n{c.content}\n```")
        else:
            sections.append("\n_No relevant code chunks found._")

        # Knowledge Graph
        sections.append("\n## Knowledge Graph")
        if self.entities:
            for ent in self.entities:
                sections.append(
                    f"\n**{ent['name']}** ({ent['entity_type']})"
                )
                if ent.get("description"):
                    sections.append(f"  {ent['description']}")
                for conn in ent.get("connections", []):
                    sections.append(
                        f"  - {conn['direction']} **{conn['entity']}** "
                        f"via _{conn['relation']}_"
                    )
        else:
            sections.append("\n_No related entities found._")

        # Decisions & Facts
        sections.append("\n## Decisions & Facts")
        if self.observations:
            for o in self.observations:
                source = f" (source: {o.source})" if o.source else ""
                sections.append(
                    f"\n- **[{o.obs_type}]** {o.content}  "
                    f"— confidence {o.confidence:.0%}, "
                    f"similarity {o.score:.2%}{source}"
                )
        else:
            sections.append("\n_No matching observations._")

        return "\n".join(sections)

    def to_dict(self) -> dict:
        """Serialize the context bundle for JSON output."""
        return {
            "query": self.query,
            "chunks": [
                {
                    "id": c.id,
                    "content": c.content,
                    "source_file": c.source_file,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "chunk_type": c.chunk_type,
                    "language": c.language,
                    "project": c.project,
                    "score": c.score,
                }
                for c in self.chunks
            ],
            "entities": self.entities,
            "observations": [
                {
                    "id": o.id,
                    "content": o.content,
                    "obs_type": o.obs_type,
                    "source": o.source,
                    "project": o.project,
                    "tags": o.tags,
                    "confidence": o.confidence,
                    "score": o.score,
                }
                for o in self.observations
            ],
        }


async def build_context(
    query: str,
    *,
    project: str | None = None,
    limit_chunks: int = 5,
    limit_observations: int = 5,
) -> ContextBundle:
    """Build a context bundle by combining chunks, graph, and observations.

    1. Vector search over chunks
    2. Find entities mentioned in top chunk source files, load their connections
    3. Semantic search over observations

    Args:
        query: The task description or question.
        project: Filter all sources by project.
        limit_chunks: Max code chunks to include.
        limit_observations: Max observations to include.

    Returns:
        A ContextBundle with all relevant context.
    """
    # 1. Relevant chunks
    chunks = await vector_search(
        query, limit=limit_chunks, project=project
    )

    # 2. Graph neighbours — find entities in files that produced top chunks
    entities = await _graph_from_chunks(chunks, project=project)

    # 3. Matching observations
    observations = await search_observations(
        query, limit=limit_observations, project=project
    )

    return ContextBundle(
        query=query,
        chunks=chunks,
        entities=entities,
        observations=observations,
    )


async def _graph_from_chunks(
    chunks: list[SearchResult],
    *,
    project: str | None = None,
) -> list[dict]:
    """Find entities whose source_file matches the retrieved chunks, plus connections."""
    source_files = {c.source_file for c in chunks if c.source_file}
    if not source_files:
        return []

    session_factory = get_session_factory()
    async with session_factory() as session:
        # Find entities in those files
        stmt = select(Entity).where(Entity.source_file.in_(source_files))
        if project:
            stmt = stmt.where(Entity.project == project)

        result = await session.execute(stmt)
        found_entities = result.scalars().all()

        if not found_entities:
            return []

        entity_ids = [e.id for e in found_entities]

        # Load outgoing relations for these entities
        out_result = await session.execute(
            select(Relation)
            .where(Relation.source_id.in_(entity_ids))
            .options(selectinload(Relation.target))
        )
        outgoing = out_result.scalars().all()

        # Load incoming relations for these entities
        in_result = await session.execute(
            select(Relation)
            .where(Relation.target_id.in_(entity_ids))
            .options(selectinload(Relation.source))
        )
        incoming = in_result.scalars().all()

        # Build lookup: entity_id -> connections
        connections: dict[str, list[dict]] = {str(e.id): [] for e in found_entities}

        for rel in outgoing:
            eid = str(rel.source_id)
            if eid in connections:
                connections[eid].append(
                    {
                        "direction": "-->",
                        "relation": rel.relation_type,
                        "entity": rel.target.name,
                        "entity_type": rel.target.entity_type,
                    }
                )

        for rel in incoming:
            eid = str(rel.target_id)
            if eid in connections:
                connections[eid].append(
                    {
                        "direction": "<--",
                        "relation": rel.relation_type,
                        "entity": rel.source.name,
                        "entity_type": rel.source.entity_type,
                    }
                )

        return [
            {
                "name": e.name,
                "entity_type": e.entity_type,
                "description": e.description,
                "source_file": e.source_file,
                "connections": connections.get(str(e.id), []),
            }
            for e in found_entities
        ]
