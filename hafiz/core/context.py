"""Context synthesizer — combines retrieved chunks, graph neighborhood,
and annotations into a unified bundle.

The killer feature: ``hafiz context "task description"`` pulls together
everything Hafiz knows that's relevant to a task. Post-structural-grounding:

  - "Chunks" are ``embeddings`` rows joined back to ``unit_revisions →
    units → files``. Each result is one embedding part (typically the
    whole unit, sometimes a sub-part for oversized content).
  - "Entities" are current ``Unit`` rows (``kind``, ``name``,
    ``parent_name``, ``source_file``, ``project``). Graph expansion
    walks current ``Edge`` rows.
  - "Observations" are ``Annotation`` rows — same wisdom layer, renamed.

Transcripts now live in the source layer (``hafiz.core.capture`` writes
``communications`` + messages). They are excluded from default context
by design and surfaced only via ``--include-transcripts`` / ``hafiz
recall``; the old chunk-based turn-neighbor expansion is retired.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
from sqlalchemy import select

from hafiz.core import graph_analysis as ga
from hafiz.core import telemetry
from hafiz.core.annotations import AnnotationResult, search_annotations
from hafiz.core.config import get_settings
from hafiz.core.database import File, get_session_factory
from hafiz.core.durations import age_label
from hafiz.core.search import SearchResult, require_query, vector_search


def _score_label(a: AnnotationResult) -> str:
    """Name the score being shown, so reranked output isn't mistaken for vector."""
    if a.rerank_score is None:
        return f"similarity {a.score:.2%}"
    return f"relevance {a.rerank_score:.0%}"


@dataclass
class ContextBundle:
    """Everything Hafiz knows about a query, in one place."""

    query: str
    chunks: list[SearchResult] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    annotations: list[AnnotationResult] = field(default_factory=list)
    project_distribution: dict[str, int] | None = None

    def to_markdown(self) -> str:
        """Render the context bundle as Markdown."""
        sections = [f"# Context: {self.query}"]

        # ── Relevant Code / Content ──
        sections.append("\n## Relevant Content")
        if self.chunks:
            for c in self.chunks:
                location = f"{c.source_file}::{c.unit_name}"
                if c.line_start and c.line_end:
                    location += f":{c.line_start}-{c.line_end}"
                lang = f" ({c.language})" if c.language else ""
                part_marker = f" — part {c.part_index}" if c.part_index > 0 else ""
                sections.append(f"\n### {location}{lang}{part_marker}  — similarity {c.score:.2%}")
                sections.append(f"```{c.language or ''}\n{c.content}\n```")
        else:
            sections.append("\n_No relevant content found._")

        # ── Knowledge Graph ──
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
                sections.append(f"\n**{ent['name']}** ({ent['kind']}){badge_str}")
                if ent.get("parent_name"):
                    sections.append(f"  parent: `{ent['parent_name']}`")
                for conn in ent.get("connections", []):
                    sections.append(
                        f"  - {conn['direction']} **{conn['unit']}** via _{conn['relation']}_"
                    )
        else:
            sections.append("\n_No related units found._")

        # ── Project Distribution (workspace mode) ──
        if self.project_distribution:
            sections.append("\n## Project Distribution")
            for proj, count in sorted(
                self.project_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                sections.append(f"- **{proj}**: {count} matches")

        # ── Annotations ──
        sections.append("\n## Decisions & Facts")
        if self.annotations:
            for a in self.annotations:
                source = f" (source: {a.source})" if a.source else ""
                sections.append(
                    f"\n- **[{a.kind}]** {a.content}  "
                    f"— confidence {a.confidence:.0%}, "
                    f"{_score_label(a)}{source}"
                )
        else:
            sections.append("\n_No matching annotations._")

        return "\n".join(sections)

    def to_dict(self) -> dict:
        """Serialize the context bundle for JSON output."""
        result = {
            "query": self.query,
            "chunks": [
                {
                    "id": c.id,
                    "unit_id": c.unit_id,
                    "unit_name": c.unit_name,
                    "kind": c.kind,
                    "content": c.content,
                    "source_file": c.source_file,
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "language": c.language,
                    "project": c.project,
                    "part_index": c.part_index,
                    "score": c.score,
                }
                for c in self.chunks
            ],
            "entities": self.entities,
            "annotations": [
                {
                    "id": a.id,
                    "content": a.content,
                    "kind": a.kind,
                    "source": a.source,
                    "project": a.project,
                    "tags": a.tags,
                    "confidence": a.confidence,
                    "unit_id": a.unit_id,
                    "score": a.score,
                    "rerank_score": a.rerank_score,
                }
                for a in self.annotations
            ],
        }
        if self.project_distribution is not None:
            result["project_distribution"] = self.project_distribution
        return result

    def to_compact(self, *, with_ids: bool = False) -> dict:
        """Serialize for token-efficient injection into a context window.

        Same rows as :meth:`to_dict`, stripped to the fields a consuming model
        actually reads: content plus just enough provenance to judge and cite
        it. Drops uuids, timestamps, null fields, float scores, and the graph
        neighbourhood's edge lists (entities collapse to ``name (kind)``).

        ``with_ids`` re-adds the annotation id. Without it an agent can read a
        decision but not ``--supersedes`` it, so pass it whenever the consumer
        might write back.
        """
        return {
            "query": self.query,
            "chunks": [
                {
                    "content": c.content,
                    "kind": c.kind,
                    "unit_name": c.unit_name,
                    "source_file": c.source_file,
                    **({"id": c.id} if with_ids else {}),
                }
                for c in self.chunks
            ],
            "units": [f"{e.get('name')} ({e.get('kind')})" for e in self.entities if e.get("name")],
            "annotations": [
                {
                    "content": a.content,
                    "kind": a.kind,
                    "source": a.source,
                    "age": age_label(a.valid_from)[0],
                    **({"id": a.id} if with_ids else {}),
                }
                for a in self.annotations
            ],
        }


async def build_context(
    query: str,
    *,
    project: str | None = None,
    limit_chunks: int = 5,
    limit_annotations: int = 5,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    min_score: float | None = None,
) -> ContextBundle:
    """Build a context bundle by combining embeddings, graph, and annotations.

    1. Vector search over embeddings (current revisions of current units).
    2. Graph neighborhood seeded by the files that produced those hits.
    3. Semantic search over annotations.

    ``include_domains`` / ``exclude_domains`` are forwarded to
    :func:`vector_search` and used to filter the seed chunks; downstream
    graph expansion and annotation search are not domain-filtered (graph
    walks already follow only edges from the seeded files, and
    annotations are kind-agnostic in this layer).

    ``min_score`` is a 0–1 relevance floor applied to both stages — cosine
    similarity for chunks, the reranked score for annotations. Because the
    graph neighbourhood is seeded from the surviving chunks, raising the floor
    narrows the whole bundle, not just its first section.

    Raises:
        EmptyQueryError: if ``query`` is blank.
    """
    query = require_query(query)
    chunks = await vector_search(
        query,
        limit=limit_chunks,
        project=project,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        similarity_threshold=min_score or 0.0,
        telemetry_command=telemetry.CONTEXT,
    )
    entities = await _graph_from_chunks(chunks, project=project)
    annotations = await search_annotations(
        query,
        limit=limit_annotations,
        project=project,
        min_score=min_score,
        telemetry_command=telemetry.CONTEXT_OBSERVATIONS,
    )

    return ContextBundle(
        query=query,
        chunks=chunks,
        entities=entities,
        annotations=annotations,
    )


async def _all_indexed_projects() -> set[str]:
    """Return the set of all project names with at least one current file."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            select(File.project)
            .where(File.project.isnot(None))
            .where(File.valid_until.is_(None))
            .group_by(File.project)
        )
        return {row[0] for row in result.all()}


def _normalize(name: str) -> str:
    """Fuzzy-match helper: lowercase, strip spaces/hyphens/underscores."""
    return name.lower().replace(" ", "").replace("-", "").replace("_", "")


def _match_dirs_to_projects(dir_names: set[str], indexed: set[str]) -> list[str]:
    """Match directory names to indexed project names. Exact then normalized."""
    matched: set[str] = set()
    matched |= dir_names & indexed

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

    1. Get cwd.parent (the "workspace root").
    2. List its subdirectories (sibling projects).
    3. Match directory names against indexed project tags in the DB.

    Fallback: if no sibling matches but cwd's own children do, treat cwd
    as the workspace root (use children instead of siblings).
    """
    if cwd is None:
        cwd = Path.cwd()

    indexed = await _all_indexed_projects()
    if not indexed:
        return []

    parent = cwd.parent
    sibling_names = {d.name for d in parent.iterdir() if d.is_dir() and not d.name.startswith(".")}
    matched = _match_dirs_to_projects(sibling_names, indexed)

    if matched:
        return matched

    child_names = {d.name for d in cwd.iterdir() if d.is_dir() and not d.name.startswith(".")}
    return _match_dirs_to_projects(child_names, indexed)


async def build_workspace_context(
    query: str,
    *,
    projects: list[str],
    limit_chunks: int = 10,
    limit_annotations: int = 10,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    min_score: float | None = None,
) -> ContextBundle:
    """Build context scoped to workspace-sibling projects.

    Raises:
        EmptyQueryError: if ``query`` is blank.
    """
    query = require_query(query)
    chunks = await vector_search(
        query,
        limit=limit_chunks,
        project=projects,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        similarity_threshold=min_score or 0.0,
        telemetry_command=telemetry.CONTEXT,
    )
    entities = await _graph_from_chunks(chunks, project=projects)
    annotations = await search_annotations(
        query,
        limit=limit_annotations,
        project=projects,
        min_score=min_score,
        telemetry_command=telemetry.CONTEXT_OBSERVATIONS,
    )

    distribution: dict[str, int] = {}
    for c in chunks:
        proj = c.project or "(untagged)"
        distribution[proj] = distribution.get(proj, 0) + 1

    return ContextBundle(
        query=query,
        chunks=chunks,
        entities=entities,
        annotations=annotations,
        project_distribution=distribution,
    )


def _in_project(attrs: dict, project: str | list[str] | None) -> bool:
    """True if a graph node's attrs pass the project scope filter."""
    if project is None:
        return True
    unit_project = attrs.get("project")
    if isinstance(project, str):
        return unit_project == project
    return unit_project in project


def _connections_for(G: nx.MultiDiGraph, node_id: str) -> list[dict]:
    """Flatten a node's in + out parallel edges into display dicts."""
    out: list[dict] = []
    for _, target, data in G.out_edges(node_id, data=True):
        out.append(
            {
                "direction": "-->",
                "relation": data.get("relation"),
                "unit": G.nodes[target].get("name"),
                "kind": G.nodes[target].get("kind"),
            }
        )
    for source, _, data in G.in_edges(node_id, data=True):
        out.append(
            {
                "direction": "<--",
                "relation": data.get("relation"),
                "unit": G.nodes[source].get("name"),
                "kind": G.nodes[source].get("kind"),
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

    1. Seed: every unit whose ``source_file`` matches one of the chunks'.
    2. Multi-source BFS up to ``depth`` hops (undirected).
    3. Rank by (distance asc, PageRank desc).
    4. Cap to ``max_entities`` so a hub unit can't flood the bundle.
    """
    settings = get_settings().graph
    if depth is None:
        depth = settings.context_depth
    if max_entities is None:
        max_entities = settings.context_max_entities

    source_files = {c.source_file for c in chunks if c.source_file}
    if not source_files:
        return []

    cache_scope = project if isinstance(project, str) else None
    G, _ = await ga.get_cached_graph(project=cache_scope)
    if G.number_of_nodes() == 0:
        return []

    seed_ids = [
        nid
        for nid, attrs in G.nodes(data=True)
        if attrs.get("source_file") in source_files and _in_project(attrs, project)
    ]
    if not seed_ids:
        return []

    distances: dict[str, int] = {}
    for seed in seed_ids:
        for nid, dist in ga.walk(G, seed, depth=depth, direction="both").items():
            prev = distances.get(nid)
            if prev is None or dist < prev:
                distances[nid] = dist

    if isinstance(project, list):
        distances = {nid: d for nid, d in distances.items() if _in_project(G.nodes[nid], project)}

    if G.number_of_edges() > 0:
        pr = nx.pagerank(G, weight="weight")
    else:
        pr = {}

    ranked = sorted(
        distances.items(),
        key=lambda kv: (kv[1], -pr.get(kv[0], 0.0)),
    )[:max_entities]

    seed_set = set(seed_ids)
    return [
        {
            "name": G.nodes[nid].get("name"),
            "kind": G.nodes[nid].get("kind"),
            "parent_name": G.nodes[nid].get("parent_name"),
            "source_file": G.nodes[nid].get("source_file"),
            "project": G.nodes[nid].get("project"),
            "distance": dist,
            "is_seed": nid in seed_set,
            "pagerank_score": round(pr.get(nid, 0.0), 6),
            "connections": _connections_for(G, nid),
        }
        for nid, dist in ranked
    ]
