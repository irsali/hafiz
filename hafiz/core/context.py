"""Context synthesizer — combines chunks, graph, and observations into a unified bundle.

The killer feature: `hafiz context "task description"` pulls together everything
Hafiz knows that's relevant to a task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from hafiz.core.database import Chunk, Entity, Relation, get_session_factory
from hafiz.core.observations import ObservationResult, search_observations
from hafiz.core.search import SearchResult, vector_search


@dataclass
class ContextBundle:
    """Everything Hafiz knows about a query, in one place."""

    query: str
    chunks: list[SearchResult] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    observations: list[ObservationResult] = field(default_factory=list)
    project_distribution: dict[str, int] | None = None

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

        # Project Distribution (workspace mode)
        if self.project_distribution:
            sections.append("\n## Project Distribution")
            for proj, count in sorted(
                self.project_distribution.items(), key=lambda x: x[1], reverse=True
            ):
                sections.append(f"- **{proj}**: {count} chunks")

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
        result = {
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
        if self.project_distribution is not None:
            result["project_distribution"] = self.project_distribution
        return result


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


async def _all_indexed_projects() -> set[str]:
    """Return the set of all project names that have indexed chunks."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(Chunk.project)
            .where(Chunk.project.isnot(None))
            .group_by(Chunk.project)
        )
        return {row[0] for row in result.all()}


def _normalize(name: str) -> str:
    """Normalize a name for fuzzy matching: lowercase, strip spaces/hyphens/underscores."""
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def _match_dirs_to_projects(dir_names: set[str], indexed: set[str]) -> list[str]:
    """Match directory names to indexed project names.

    Tries exact match first, then normalized (case-insensitive, ignore
    spaces/hyphens/underscores) to handle common mismatches like
    'Admin Portal' dir -> 'AdminPortal' project tag.
    """
    matched: set[str] = set()

    # Exact matches
    matched |= dir_names & indexed

    # Normalized matching for the rest
    remaining = indexed - matched
    if remaining:
        norm_to_project = {_normalize(p): p for p in remaining}
        for d in dir_names:
            norm = _normalize(d)
            if norm in norm_to_project:
                matched.add(norm_to_project[norm])

    return sorted(matched)


async def resolve_workspace_projects(cwd: Path | None = None) -> list[str]:
    """Resolve workspace-sibling projects from the filesystem.

    Logic:
    1. Get the parent directory of cwd (the "workspace root").
    2. List its subdirectories (sibling projects).
    3. Match directory names against indexed project tags in the DB,
       using normalized matching (case-insensitive, ignore spaces/hyphens).

    Edge case: if cwd itself has no matching project in the DB but its
    children do, treat cwd as the workspace root (use children instead
    of siblings).

    Returns:
        List of matched project names (DB names, not dir names).
    """
    if cwd is None:
        cwd = Path.cwd()

    indexed = await _all_indexed_projects()
    if not indexed:
        return []

    # Try siblings first: parent's children
    parent = cwd.parent
    sibling_names = {
        d.name for d in parent.iterdir() if d.is_dir() and not d.name.startswith(".")
    }
    matched = _match_dirs_to_projects(sibling_names, indexed)

    if matched:
        return matched

    # Fallback: maybe cwd IS the workspace root — check its children
    child_names = {
        d.name for d in cwd.iterdir() if d.is_dir() and not d.name.startswith(".")
    }
    matched = _match_dirs_to_projects(child_names, indexed)

    return matched


async def build_workspace_context(
    query: str,
    *,
    projects: list[str],
    limit_chunks: int = 10,
    limit_observations: int = 10,
) -> ContextBundle:
    """Build context scoped to workspace-sibling projects.

    Searches chunks and observations filtered to the given project list.
    Includes project distribution to show which projects contributed.

    Args:
        query: The task description or question.
        projects: Project names to search across (resolved from filesystem).
        limit_chunks: Max code chunks (higher default for cross-project).
        limit_observations: Max observations (higher default for cross-project).

    Returns:
        A ContextBundle with cross-project context and distribution info.
    """
    # Search across workspace projects
    chunks = await vector_search(query, limit=limit_chunks, project=projects)

    # Graph neighbours from matched chunks
    entities = await _graph_from_chunks(chunks, project=projects)

    # Observations across workspace projects
    observations = await search_observations(
        query, limit=limit_observations, project=projects
    )

    # Compute project distribution from the returned chunks
    distribution: dict[str, int] = {}
    for c in chunks:
        proj = c.project or "(untagged)"
        distribution[proj] = distribution.get(proj, 0) + 1

    return ContextBundle(
        query=query,
        chunks=chunks,
        entities=entities,
        observations=observations,
        project_distribution=distribution,
    )


async def _graph_from_chunks(
    chunks: list[SearchResult],
    *,
    project: str | list[str] | None = None,
) -> list[dict]:
    """Find entities whose source_file matches the retrieved chunks, plus connections."""
    source_files = {c.source_file for c in chunks if c.source_file}
    if not source_files:
        return []

    session_factory = get_session_factory()
    async with session_factory() as session:
        # Find entities in those files
        stmt = select(Entity).where(Entity.source_file.in_(source_files))
        if isinstance(project, list):
            stmt = stmt.where(Entity.project.in_(project))
        elif project:
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
