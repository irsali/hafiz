"""Context synthesizer — combines chunks, graph, and observations into a unified bundle.

The killer feature: `hafiz context "task description"` pulls together everything
Hafiz knows that's relevant to a task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
from sqlalchemy import select

from hafiz.core import graph_analysis as ga
from hafiz.core.config import get_settings
from hafiz.core.database import Chunk, get_session_factory
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
                badges = []
                if ent.get("is_seed"):
                    badges.append("seed")
                if (dist := ent.get("distance")) is not None and not ent.get("is_seed"):
                    badges.append(f"{dist} hop{'s' if dist != 1 else ''} away")
                if (pr := ent.get("pagerank_score")) is not None:
                    badges.append(f"PR {pr:.4f}")
                badge_str = f" _[{' · '.join(badges)}]_" if badges else ""
                sections.append(
                    f"\n**{ent['name']}** ({ent['entity_type']}){badge_str}"
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


def _in_project(attrs: dict, project: str | list[str] | None) -> bool:
    """True if entity `attrs` passes the project scope filter."""
    if project is None:
        return True
    ent_project = attrs.get("project")
    if isinstance(project, str):
        return ent_project == project
    return ent_project in project


def _connections_for(G: nx.MultiDiGraph, node_id: str) -> list[dict]:
    """Flatten a node's incoming + outgoing parallel edges into display dicts."""
    out: list[dict] = []
    for _, target, data in G.out_edges(node_id, data=True):
        out.append(
            {
                "direction": "-->",
                "relation": data.get("relation_type"),
                "entity": G.nodes[target].get("name"),
                "entity_type": G.nodes[target].get("entity_type"),
            }
        )
    for source, _, data in G.in_edges(node_id, data=True):
        out.append(
            {
                "direction": "<--",
                "relation": data.get("relation_type"),
                "entity": G.nodes[source].get("name"),
                "entity_type": G.nodes[source].get("entity_type"),
            }
        )
    return out


async def _graph_from_chunks(
    chunks: list[SearchResult],
    *,
    project: str | list[str] | None = None,
    depth: int | None = None,
    max_entities: int | None = None,
) -> list[dict]:
    """Expand from retrieved chunks into the knowledge graph.

    Algorithm:
      1. Seed: every entity whose `source_file` matches one of the chunks'.
      2. Multi-source BFS up to `depth` hops (undirected) — distance tracked as
         min distance to any seed.
      3. Rank by (distance asc, PageRank desc) — nearest + most central first.
      4. Cap to `max_entities` so a hub entity can't flood the bundle.

    Each returned entity carries `distance`, `is_seed`, `pagerank_score`, and
    its 1-hop `connections` list (preserving the existing display contract).
    """
    settings = get_settings().graph
    if depth is None:
        depth = settings.context_depth
    if max_entities is None:
        max_entities = settings.context_max_entities

    source_files = {c.source_file for c in chunks if c.source_file}
    if not source_files:
        return []

    # Load the cached graph. A project-scoped cache is only possible for a single
    # string project; list or None falls back to the global graph and filters
    # via `_in_project`. (See Hafiz observation "graph cache scoping" for context
    # on future multi-project cache scopes.)
    cache_scope = project if isinstance(project, str) else None
    G, _ = await ga.get_cached_graph(project=cache_scope)
    if G.number_of_nodes() == 0:
        return []

    # Seed: entities in the retrieved source files, respecting project scope
    seed_ids = [
        nid
        for nid, attrs in G.nodes(data=True)
        if attrs.get("source_file") in source_files and _in_project(attrs, project)
    ]
    if not seed_ids:
        return []

    # Multi-source BFS — distance = min over seeds
    distances: dict[str, int] = {}
    for seed in seed_ids:
        for nid, dist in ga.walk(G, seed, depth=depth, direction="both").items():
            prev = distances.get(nid)
            if prev is None or dist < prev:
                distances[nid] = dist

    # For workspace (list) mode we're using the global graph, so drop walked
    # neighbors that landed outside the project scope.
    if isinstance(project, list):
        distances = {
            nid: d for nid, d in distances.items() if _in_project(G.nodes[nid], project)
        }

    # PageRank on the (scope-appropriate) graph
    if G.number_of_edges() > 0:
        pr = nx.pagerank(G, weight="weight")
    else:
        pr = {}

    # Rank: nearest first, then most structurally central
    ranked = sorted(
        distances.items(),
        key=lambda kv: (kv[1], -pr.get(kv[0], 0.0)),
    )[:max_entities]

    seed_set = set(seed_ids)
    return [
        {
            "name": G.nodes[nid].get("name"),
            "entity_type": G.nodes[nid].get("entity_type"),
            "description": G.nodes[nid].get("description"),
            "source_file": G.nodes[nid].get("source_file"),
            "project": G.nodes[nid].get("project"),
            "distance": dist,
            "is_seed": nid in seed_set,
            "pagerank_score": round(pr.get(nid, 0.0), 6),
            "connections": _connections_for(G, nid),
        }
        for nid, dist in ranked
    ]
