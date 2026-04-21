"""Graph analysis — load entities + relations from Postgres into NetworkX.

Builds a directed graph suitable for multi-hop traversal, centrality, and
community detection. Caches the graph as a pickle with automatic staleness
detection based on a three-part signature: (max_updated_at, entity_count,
relation_count). The signature catches both mutations and deletions without
requiring explicit invalidation hooks.
"""

from __future__ import annotations

import asyncio
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
from sqlalchemy import func, select

from hafiz.core.database import Entity, Relation, get_session_factory


CACHE_DIR = Path.home() / ".cache" / "hafiz"
CACHE_VERSION = 2  # bump when pickle format or meta schema changes (v2: MultiDiGraph)


# ── Cache signature ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraphSignature:
    """Three-part fingerprint of the graph state in the database.

    A cache is valid iff its signature equals the current DB signature. Including
    counts (not just max_updated_at) catches deletions, which don't bump any
    remaining row's timestamp.
    """

    max_updated: datetime | None
    entity_count: int
    relation_count: int


@dataclass
class GraphMeta:
    """Metadata stored alongside a pickled graph."""

    project: str | None
    built_at: datetime
    signature: GraphSignature
    version: int
    node_count: int
    edge_count: int


# ── Public API ──────────────────────────────────────────────────────────────


async def load_graph(project: str | None = None) -> nx.MultiDiGraph:
    """Build a fresh NetworkX MultiDiGraph from the database.

    A multigraph is used so that multiple relations between the same
    (source, target) pair are preserved faithfully — e.g. entity A both
    `calls` and `imports` entity B becomes two parallel edges rather than a
    single edge with lossy last-wins semantics.

    Node attributes: name, entity_type, description, project, source_file
    Edge attributes: relation_type, weight, evidence
    Edge key: the relation's UUID (stringified), so each DB relation maps to a
        distinct edge key and can be retrieved or removed deterministically.

    Project scoping: when `project` is given, only entities tagged to that project
    are included, and only relations whose BOTH endpoints fall inside that scope.
    """
    session_factory = get_session_factory()

    async with session_factory() as session:
        ent_stmt = select(Entity)
        if project is not None:
            ent_stmt = ent_stmt.where(Entity.project == project)
        entities = (await session.execute(ent_stmt)).scalars().all()
        entity_ids = {e.id for e in entities}

        rel_stmt = select(Relation)
        relations = (await session.execute(rel_stmt)).scalars().all()

    G: nx.MultiDiGraph = nx.MultiDiGraph()
    for e in entities:
        G.add_node(
            str(e.id),
            name=e.name,
            entity_type=e.entity_type,
            description=e.description,
            project=e.project,
            source_file=e.source_file,
        )
    for r in relations:
        if r.source_id in entity_ids and r.target_id in entity_ids:
            G.add_edge(
                str(r.source_id),
                str(r.target_id),
                key=str(r.id),
                relation_type=r.relation_type,
                weight=r.weight,
                evidence=r.evidence,
            )
    return G


async def current_signature(project: str | None = None) -> GraphSignature:
    """Compute the current (max_updated, entity_count, relation_count) signature.

    Relations aren't tagged with a project directly — we treat their timestamps
    as global. This over-invalidates a project-scoped cache when an unrelated
    project's relations change, which is a cheap price for correctness.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        ent_upd_stmt = select(func.max(Entity.updated_at))
        ent_cnt_stmt = select(func.count()).select_from(Entity)
        if project is not None:
            ent_upd_stmt = ent_upd_stmt.where(Entity.project == project)
            ent_cnt_stmt = ent_cnt_stmt.where(Entity.project == project)

        max_e = (await session.execute(ent_upd_stmt)).scalar()
        max_r = (await session.execute(select(func.max(Relation.updated_at)))).scalar()
        ent_count = (await session.execute(ent_cnt_stmt)).scalar() or 0
        rel_count = (
            await session.execute(select(func.count()).select_from(Relation))
        ).scalar() or 0

    candidates = [t for t in (max_e, max_r) if t is not None]
    max_updated = max(candidates) if candidates else None
    return GraphSignature(
        max_updated=max_updated,
        entity_count=ent_count,
        relation_count=rel_count,
    )


async def get_cached_graph(
    project: str | None = None,
    *,
    force_rebuild: bool = False,
) -> tuple[nx.MultiDiGraph, GraphMeta]:
    """Return the NetworkX graph, using a disk-cached pickle when fresh.

    Staleness is detected by comparing the cached signature against the current
    DB signature. On mismatch (or `force_rebuild=True`, or corrupt cache), the
    graph is rebuilt and the cache is rewritten atomically via a temp file.
    """
    cache_path = _cache_path(project)
    signature = await current_signature(project)

    if not force_rebuild and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                G, meta = pickle.load(f)
            if (
                isinstance(meta, GraphMeta)
                and meta.version == CACHE_VERSION
                and meta.project == project
                and meta.signature == signature
            ):
                return G, meta
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError):
            pass  # corrupt or schema-drift — rebuild

    G = await load_graph(project)
    meta = GraphMeta(
        project=project,
        built_at=datetime.now(timezone.utc),
        signature=signature,
        version=CACHE_VERSION,
        node_count=G.number_of_nodes(),
        edge_count=G.number_of_edges(),
    )
    _write_cache_atomic(cache_path, G, meta)
    return G, meta


def invalidate_cache(project: str | None = None) -> None:
    """Delete the cache file for a project scope (or global cache if None).

    Not normally needed — auto-staleness detection handles mutation and deletion.
    Exposed for manual use (tests, debugging, schema migrations).
    """
    cache_path = _cache_path(project)
    if cache_path.exists():
        cache_path.unlink()


def invalidate_all_caches() -> None:
    """Delete every cached graph file. Safe to call if CACHE_DIR doesn't exist."""
    if not CACHE_DIR.exists():
        return
    for p in CACHE_DIR.glob("graph-*.pkl"):
        p.unlink()


# ── Internals ───────────────────────────────────────────────────────────────


def _cache_path(project: str | None) -> Path:
    """Filesystem-safe cache path for a project scope."""
    key = project or "__global__"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return CACHE_DIR / f"graph-{safe}.pkl"


def _write_cache_atomic(path: Path, G: nx.MultiDiGraph, meta: GraphMeta) -> None:
    """Write (G, meta) to `path` via temp file + rename. Failures are non-fatal."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump((G, meta), f)
        tmp.replace(path)
    except OSError:
        # Cache write is best-effort; we still return the graph to the caller.
        if "tmp" in locals() and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


# ── Sync wrapper for convenience in CLI commands ────────────────────────────


def get_cached_graph_sync(
    project: str | None = None,
    *,
    force_rebuild: bool = False,
) -> tuple[nx.MultiDiGraph, GraphMeta]:
    """Blocking wrapper over `get_cached_graph` for synchronous CLI code paths."""
    return asyncio.run(get_cached_graph(project, force_rebuild=force_rebuild))


# ── Query helpers (pure, operate on an already-loaded graph) ─────────────────


def find_nodes_by_name(
    G: nx.MultiDiGraph,
    name: str,
    *,
    project: str | None = None,
) -> list[str]:
    """Return node IDs whose `name` attribute matches (case-insensitive).

    If `project` is given, only nodes tagged to that project are returned. Names
    are not unique in Hafiz — two entities of different types can share a name
    across files — so this always returns a list.
    """
    needle = name.lower()
    return [
        nid
        for nid, attrs in G.nodes(data=True)
        if (attrs.get("name") or "").lower() == needle
        and (project is None or attrs.get("project") == project)
    ]


def walk(
    G: nx.MultiDiGraph,
    source: str,
    *,
    depth: int = 1,
    direction: str = "out",
) -> dict[str, int]:
    """BFS from `source` up to `depth` hops, returning `{node_id: distance}`.

    direction:
      - "out"  → follow outgoing edges (A depends on B → walk from A finds B)
      - "in"   → follow incoming edges (blast-radius / impact analysis)
      - "both" → undirected walk (show mode)

    The source is always included at distance 0. `depth=0` returns only the source.
    """
    if source not in G:
        raise ValueError(f"source node {source!r} not in graph")
    if direction not in ("out", "in", "both"):
        raise ValueError(f"direction must be 'out', 'in', or 'both' — got {direction!r}")
    if depth < 0:
        raise ValueError(f"depth must be >= 0 — got {depth}")

    distances: dict[str, int] = {source: 0}
    frontier = [source]
    for d in range(depth):
        next_frontier: list[str] = []
        for node in frontier:
            if direction == "out":
                neighbors = (v for _, v in G.out_edges(node))
            elif direction == "in":
                neighbors = (u for u, _ in G.in_edges(node))
            else:  # both
                neighbors = (
                    n
                    for n in (
                        [v for _, v in G.out_edges(node)]
                        + [u for u, _ in G.in_edges(node)]
                    )
                )
            for n in neighbors:
                if n not in distances:
                    distances[n] = d + 1
                    next_frontier.append(n)
        if not next_frontier:
            break
        frontier = next_frontier
    return distances


def edges_between(
    G: nx.MultiDiGraph,
    u: str,
    v: str,
) -> list[dict]:
    """Return all parallel edge attribute dicts from `u` → `v`. Empty if none."""
    if not G.has_edge(u, v):
        return []
    # MultiDiGraph: G[u][v] is a dict keyed by edge-key
    return [dict(attrs) for attrs in G[u][v].values()]


def shortest_path_between(
    G: nx.MultiDiGraph,
    source: str,
    target: str,
) -> list[str] | None:
    """Directed shortest path as a list of node IDs. Returns None if no path exists."""
    try:
        return nx.shortest_path(G, source=source, target=target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ── Centrality (importance ranking) ─────────────────────────────────────────


VALID_METRICS: tuple[str, ...] = ("pagerank", "betweenness", "degree", "in_degree", "out_degree")


def rank_nodes(
    G: nx.MultiDiGraph,
    *,
    metric: str = "pagerank",
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Rank nodes by a centrality metric. Returns `[(node_id, score), ...]` sorted desc.

    Supported metrics:
      - "pagerank"   — weighted PageRank (uses edge `weight` attribute)
      - "betweenness"— fraction of shortest paths that pass through each node
      - "degree"     — total degree (in + out), normalized to [0, 1]
      - "in_degree"  — incoming edges only
      - "out_degree" — outgoing edges only

    Betweenness is O(V·E) and may be slow on graphs larger than a few thousand
    nodes; the caller should gate it behind a flag or sampling for large inputs.
    """
    if metric not in VALID_METRICS:
        raise ValueError(
            f"metric must be one of {VALID_METRICS} — got {metric!r}"
        )
    if G.number_of_nodes() == 0:
        return []

    if metric == "pagerank":
        scores = nx.pagerank(G, weight="weight")
    elif metric == "betweenness":
        scores = nx.betweenness_centrality(G)
    elif metric == "degree":
        scores = nx.degree_centrality(G)
    elif metric == "in_degree":
        scores = nx.in_degree_centrality(G)
    else:  # out_degree
        scores = nx.out_degree_centrality(G)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked


# ── Graph-level stats ───────────────────────────────────────────────────────


@dataclass
class GraphStats:
    """Snapshot of graph-level health metrics."""

    node_count: int
    edge_count: int
    density: float
    weakly_connected_components: int
    largest_component_size: int
    isolated_nodes: int
    entity_type_counts: dict[str, int]
    relation_type_counts: dict[str, int]
    top_by_pagerank: list[tuple[str, float]]  # [(node_id, score), ...]


def graph_stats(G: nx.MultiDiGraph, *, top_central: int = 5) -> GraphStats:
    """Compute overall graph statistics in a single pass.

    `top_central` controls how many top-PageRank nodes are returned. PageRank
    is skipped when the graph is empty.
    """
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()

    # Type distributions
    entity_types: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        t = attrs.get("entity_type") or "(none)"
        entity_types[t] = entity_types.get(t, 0) + 1

    relation_types: dict[str, int] = {}
    for _, _, attrs in G.edges(data=True):
        t = attrs.get("relation_type") or "(none)"
        relation_types[t] = relation_types.get(t, 0) + 1

    # Connectivity
    if node_count > 0:
        components = list(nx.weakly_connected_components(G))
        wcc_count = len(components)
        largest = max((len(c) for c in components), default=0)
        isolated = sum(1 for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) == 0)
        density = nx.density(G)
    else:
        wcc_count = 0
        largest = 0
        isolated = 0
        density = 0.0

    # Centrality preview
    top_ranked = rank_nodes(G, metric="pagerank", top_n=top_central) if node_count else []

    return GraphStats(
        node_count=node_count,
        edge_count=edge_count,
        density=density,
        weakly_connected_components=wcc_count,
        largest_component_size=largest,
        isolated_nodes=isolated,
        entity_type_counts=dict(
            sorted(entity_types.items(), key=lambda kv: kv[1], reverse=True)
        ),
        relation_type_counts=dict(
            sorted(relation_types.items(), key=lambda kv: kv[1], reverse=True)
        ),
        top_by_pagerank=top_ranked,
    )
