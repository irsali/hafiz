"""Graph analysis — load units + edges from Postgres into NetworkX.

Builds a directed graph suitable for multi-hop traversal, centrality, and
community detection. Caches the graph as a pickle with automatic staleness
detection based on a three-part signature: (max_observed_at, unit_count,
edge_count). The signature catches both mutations and deletions without
requiring explicit invalidation hooks.

Graph model (post-structural-grounding):

  Nodes — Units with ``valid_until IS NULL`` (currently present).
    Attributes: ``name``, ``kind``, ``parent_name``, ``project``,
    ``source_file``.

  Edges — Edges with ``superseded_at IS NULL`` where both
    ``source_unit_id`` and ``target_unit_id`` resolved to in-scope units.
    External references (``target_unit_id IS NULL``, ``target_name`` set)
    are excluded from the graph — they're visible via raw DB queries
    but don't participate in traversal / centrality.
    Attributes: ``relation`` (e.g. ``calls``, ``imports``, ``inherits``),
    ``weight``, ``evidence``, ``source`` (``ast`` / ``agent`` / ``user``).
"""

from __future__ import annotations

import asyncio
import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
from sqlalchemy import func, select

from hafiz.core.database import Edge, File, Unit, UnitRevision, get_session_factory

CACHE_DIR = Path.home() / ".cache" / "hafiz"
# Bump when pickle payload shape changes. v3: post-structural-grounding
# (Unit/Edge instead of Entity/Relation; attrs renamed).
CACHE_VERSION = 3


# ── Cache signature ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GraphSignature:
    """Three-part fingerprint of the graph state in the database.

    A cache is valid iff its signature equals the current DB signature.
    Including counts (not just max_observed) catches deletions, which
    don't bump any remaining row's timestamp.
    """

    max_observed: datetime | None
    unit_count: int
    edge_count: int


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
    (source, target) pair are preserved — e.g. unit A both ``calls`` and
    ``imports`` unit B becomes two parallel edges rather than a single
    edge with lossy last-wins semantics.

    Project scoping filters units through ``File.project`` and drops
    edges whose endpoints fall outside the resulting scope.
    """
    session_factory = get_session_factory()

    async with session_factory() as session:
        unit_stmt = (
            select(Unit, File)
            .join(File, File.id == Unit.file_id)
            .where(Unit.valid_until.is_(None))
            .where(File.valid_until.is_(None))
        )
        if project is not None:
            unit_stmt = unit_stmt.where(File.project == project)
        unit_rows = (await session.execute(unit_stmt)).all()
        unit_ids = {u.id for u, _ in unit_rows}

        edge_stmt = select(Edge).where(
            Edge.superseded_at.is_(None),
            Edge.target_unit_id.is_not(None),
        )
        edges = (await session.execute(edge_stmt)).scalars().all()

    G: nx.MultiDiGraph = nx.MultiDiGraph()
    for unit, file in unit_rows:
        G.add_node(
            str(unit.id),
            name=unit.name,
            kind=unit.kind,
            parent_name=unit.parent_name,
            project=file.project,
            source_file=file.path,
        )
    for edge in edges:
        if edge.source_unit_id in unit_ids and edge.target_unit_id in unit_ids:
            G.add_edge(
                str(edge.source_unit_id),
                str(edge.target_unit_id),
                key=str(edge.id),
                relation=edge.relation,
                weight=edge.weight,
                evidence=edge.evidence,
                source=edge.source,
            )
    return G


async def current_signature(project: str | None = None) -> GraphSignature:
    """Compute the current (max_observed, unit_count, edge_count) signature.

    ``max_observed`` samples the newest revision ``observed_at`` and the
    newest edge ``observed_at``. Edges aren't project-tagged directly —
    their timestamps are treated as global; a project-scoped cache
    therefore over-invalidates when an unrelated project's edges change.
    Cheap correctness.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        unit_count_stmt = (
            select(func.count())
            .select_from(Unit)
            .join(File, File.id == Unit.file_id)
            .where(Unit.valid_until.is_(None))
            .where(File.valid_until.is_(None))
        )
        rev_max_stmt = (
            select(func.max(UnitRevision.observed_at))
            .select_from(UnitRevision)
            .join(Unit, Unit.id == UnitRevision.unit_id)
            .join(File, File.id == Unit.file_id)
            .where(UnitRevision.superseded_at.is_(None))
            .where(Unit.valid_until.is_(None))
            .where(File.valid_until.is_(None))
        )
        if project is not None:
            unit_count_stmt = unit_count_stmt.where(File.project == project)
            rev_max_stmt = rev_max_stmt.where(File.project == project)

        unit_count = (await session.execute(unit_count_stmt)).scalar() or 0
        max_rev = (await session.execute(rev_max_stmt)).scalar()
        max_edge = (
            await session.execute(
                select(func.max(Edge.observed_at)).where(Edge.superseded_at.is_(None))
            )
        ).scalar()
        edge_count = (
            await session.execute(
                select(func.count())
                .select_from(Edge)
                .where(Edge.superseded_at.is_(None))
                .where(Edge.target_unit_id.is_not(None))
            )
        ).scalar() or 0

    candidates = [t for t in (max_rev, max_edge) if t is not None]
    max_observed = max(candidates) if candidates else None
    return GraphSignature(
        max_observed=max_observed,
        unit_count=unit_count,
        edge_count=edge_count,
    )


async def get_cached_graph(
    project: str | None = None,
    *,
    force_rebuild: bool = False,
) -> tuple[nx.MultiDiGraph, GraphMeta]:
    """Return the NetworkX graph, using a disk-cached pickle when fresh."""
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
        built_at=datetime.now(UTC),
        signature=signature,
        version=CACHE_VERSION,
        node_count=G.number_of_nodes(),
        edge_count=G.number_of_edges(),
    )
    _write_cache_atomic(cache_path, G, meta)
    return G, meta


def invalidate_cache(project: str | None = None) -> None:
    """Delete the cache file for a project scope (or global cache if None)."""
    cache_path = _cache_path(project)
    if cache_path.exists():
        cache_path.unlink()


def invalidate_all_caches() -> None:
    """Delete every cached graph file. Safe if CACHE_DIR doesn't exist."""
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
    """Write (G, meta) to `path` via temp file + rename. Failures non-fatal."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump((G, meta), f)
        tmp.replace(path)
    except OSError:
        if "tmp" in locals() and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def get_cached_graph_sync(
    project: str | None = None,
    *,
    force_rebuild: bool = False,
) -> tuple[nx.MultiDiGraph, GraphMeta]:
    """Blocking wrapper over ``get_cached_graph`` for synchronous CLI code."""
    return asyncio.run(get_cached_graph(project, force_rebuild=force_rebuild))


# ── Query helpers (pure, operate on an already-loaded graph) ─────────────────


def find_nodes_by_name(
    G: nx.MultiDiGraph,
    name: str,
    *,
    project: str | None = None,
) -> list[str]:
    """Return node IDs whose ``name`` attribute matches (case-insensitive).

    Names are not unique — two units of different kinds can share a name
    across files — so this always returns a list. Scoped by project when
    given.
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
    """BFS from ``source`` up to ``depth`` hops, returning ``{node_id: distance}``.

    direction:
      - ``"out"``  → follow outgoing edges (A depends on B → walk from A finds B)
      - ``"in"``   → follow incoming edges (blast-radius / impact analysis)
      - ``"both"`` → undirected walk (show mode)
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
            else:
                neighbors = (
                    n
                    for n in ([v for _, v in G.out_edges(node)] + [u for u, _ in G.in_edges(node)])
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
    """Return all parallel edge attribute dicts from ``u`` → ``v``."""
    if not G.has_edge(u, v):
        return []
    return [dict(attrs) for attrs in G[u][v].values()]


def shortest_path_between(
    G: nx.MultiDiGraph,
    source: str,
    target: str,
) -> list[str] | None:
    """Directed shortest path. Returns None if no path exists."""
    try:
        return nx.shortest_path(G, source=source, target=target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


# ── Centrality (importance ranking) ─────────────────────────────────────────


VALID_METRICS: tuple[str, ...] = (
    "pagerank",
    "betweenness",
    "degree",
    "in_degree",
    "out_degree",
)


def rank_nodes(
    G: nx.MultiDiGraph,
    *,
    metric: str = "pagerank",
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Rank nodes by a centrality metric. ``[(node_id, score), ...]`` desc."""
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {VALID_METRICS} — got {metric!r}")
    if G.number_of_nodes() == 0:
        return []

    if metric == "pagerank":
        # networkx.pagerank delegates to scipy internally. If scipy isn't
        # importable (common after a pipx install that predates scipy being
        # added to deps), the resulting ModuleNotFoundError bubbles up as a
        # scary traceback. Surface it cleanly with the fix — the top-level
        # handler would otherwise log it, which is fine, but a clear
        # remediation is better than a suggestion-after-the-fact.
        try:
            import scipy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "pagerank needs scipy, which isn't installed in this "
                "environment. Fix: `pipx inject hafiz scipy` (or "
                "`pipx reinstall hafiz` if hafiz was installed via pipx "
                "before scipy was added to dependencies)."
            ) from exc
        scores = nx.pagerank(G, weight="weight")
    elif metric == "betweenness":
        scores = nx.betweenness_centrality(G)
    elif metric == "degree":
        scores = nx.degree_centrality(G)
    elif metric == "in_degree":
        scores = nx.in_degree_centrality(G)
    else:
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
    kind_counts: dict[str, int]
    relation_counts: dict[str, int]
    top_by_pagerank: list[tuple[str, float]]


def graph_stats(G: nx.MultiDiGraph, *, top_central: int = 5) -> GraphStats:
    """Compute overall graph statistics in a single pass."""
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()

    kinds: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        k = attrs.get("kind") or "(none)"
        kinds[k] = kinds.get(k, 0) + 1

    relations: dict[str, int] = {}
    for _, _, attrs in G.edges(data=True):
        r = attrs.get("relation") or "(none)"
        relations[r] = relations.get(r, 0) + 1

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

    top_ranked = rank_nodes(G, metric="pagerank", top_n=top_central) if node_count else []

    return GraphStats(
        node_count=node_count,
        edge_count=edge_count,
        density=density,
        weakly_connected_components=wcc_count,
        largest_component_size=largest,
        isolated_nodes=isolated,
        kind_counts=dict(sorted(kinds.items(), key=lambda kv: kv[1], reverse=True)),
        relation_counts=dict(sorted(relations.items(), key=lambda kv: kv[1], reverse=True)),
        top_by_pagerank=top_ranked,
    )
