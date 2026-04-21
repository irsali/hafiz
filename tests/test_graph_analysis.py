"""Tests for hafiz.core.graph_analysis.

DB-free tests cover the cache machinery, signature comparison, and path
sanitization. DB-dependent tests (load_graph, current_signature) are skipped
when Postgres is unavailable — matching the pattern used in test_search.py.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import pytest

from hafiz.core import graph_analysis as ga


# ── Cache path sanitization ─────────────────────────────────────────────────


def test_cache_path_global_when_no_project():
    p = ga._cache_path(None)
    assert p.name == "graph-__global__.pkl"


def test_cache_path_sanitizes_unsafe_chars():
    p = ga._cache_path("weird/name with:spaces")
    # Slash, space, and colon should all be replaced with underscore
    assert "/" not in p.name
    assert " " not in p.name
    assert ":" not in p.name
    assert p.name.startswith("graph-")
    assert p.name.endswith(".pkl")


def test_cache_path_preserves_safe_chars():
    p = ga._cache_path("my-project_01")
    assert p.name == "graph-my-project_01.pkl"


# ── Signature equality ─────────────────────────────────────────────────────


def test_signature_equality_same_values():
    t = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
    s1 = ga.GraphSignature(max_updated=t, entity_count=5, relation_count=7)
    s2 = ga.GraphSignature(max_updated=t, entity_count=5, relation_count=7)
    assert s1 == s2


def test_signature_inequality_on_count_change():
    t = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
    s1 = ga.GraphSignature(max_updated=t, entity_count=5, relation_count=7)
    s2 = ga.GraphSignature(max_updated=t, entity_count=4, relation_count=7)
    # Deletion scenario: count drops but max_updated is unchanged
    assert s1 != s2


def test_signature_inequality_on_timestamp_change():
    t1 = datetime(2026, 4, 21, 12, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 4, 21, 13, 0, tzinfo=timezone.utc)
    s1 = ga.GraphSignature(max_updated=t1, entity_count=5, relation_count=7)
    s2 = ga.GraphSignature(max_updated=t2, entity_count=5, relation_count=7)
    assert s1 != s2


# ── Cache write / read roundtrip ────────────────────────────────────────────


def test_cache_write_and_read_roundtrip(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ga, "CACHE_DIR", tmp_path)

    G = nx.MultiDiGraph()
    G.add_node("a", name="Alpha", entity_type="function")
    G.add_node("b", name="Beta", entity_type="function")
    # Two parallel edges between a → b — verifies multi-edge semantics survive pickling
    G.add_edge("a", "b", key="r1", relation_type="calls", weight=1.0, evidence=None)
    G.add_edge("a", "b", key="r2", relation_type="imports", weight=1.0, evidence=None)

    sig = ga.GraphSignature(
        max_updated=datetime(2026, 4, 21, tzinfo=timezone.utc),
        entity_count=2,
        relation_count=2,
    )
    meta = ga.GraphMeta(
        project="demo",
        built_at=datetime.now(timezone.utc),
        signature=sig,
        version=ga.CACHE_VERSION,
        node_count=2,
        edge_count=2,
    )

    path = ga._cache_path("demo")
    ga._write_cache_atomic(path, G, meta)
    assert path.exists()

    with path.open("rb") as f:
        loaded_G, loaded_meta = pickle.load(f)

    assert isinstance(loaded_G, nx.MultiDiGraph)
    assert loaded_G.number_of_nodes() == 2
    assert loaded_G.number_of_edges() == 2
    # MultiDiGraph edge access: G[u][v] returns {key: attrs}
    edges_between = loaded_G["a"]["b"]
    assert set(edges_between.keys()) == {"r1", "r2"}
    assert edges_between["r1"]["relation_type"] == "calls"
    assert edges_between["r2"]["relation_type"] == "imports"
    assert loaded_meta.signature == sig
    assert loaded_meta.project == "demo"


def test_invalidate_cache_removes_file(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ga, "CACHE_DIR", tmp_path)

    G = nx.MultiDiGraph()
    meta = ga.GraphMeta(
        project="x",
        built_at=datetime.now(timezone.utc),
        signature=ga.GraphSignature(None, 0, 0),
        version=ga.CACHE_VERSION,
        node_count=0,
        edge_count=0,
    )
    path = ga._cache_path("x")
    ga._write_cache_atomic(path, G, meta)
    assert path.exists()

    ga.invalidate_cache("x")
    assert not path.exists()


def test_invalidate_all_caches(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ga, "CACHE_DIR", tmp_path)

    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "graph-a.pkl").write_bytes(b"x")
    (tmp_path / "graph-b.pkl").write_bytes(b"y")
    # Non-graph file should be left alone
    (tmp_path / "other.pkl").write_bytes(b"z")

    ga.invalidate_all_caches()
    assert not (tmp_path / "graph-a.pkl").exists()
    assert not (tmp_path / "graph-b.pkl").exists()
    assert (tmp_path / "other.pkl").exists()


def test_invalidate_all_caches_missing_dir_is_safe(tmp_path: Path, monkeypatch):
    missing = tmp_path / "does-not-exist"
    monkeypatch.setattr(ga, "CACHE_DIR", missing)
    # Must not raise
    ga.invalidate_all_caches()


# ── get_cached_graph: cache-hit / rebuild paths (mocked DB) ────────────────


@pytest.mark.asyncio
async def test_get_cached_graph_uses_cache_when_signature_matches(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(ga, "CACHE_DIR", tmp_path)

    # Seed a cache
    G_cached = nx.MultiDiGraph()
    G_cached.add_node("a", name="Cached", entity_type="function")
    sig = ga.GraphSignature(
        max_updated=datetime(2026, 4, 21, tzinfo=timezone.utc),
        entity_count=1,
        relation_count=0,
    )
    meta = ga.GraphMeta(
        project=None,
        built_at=datetime.now(timezone.utc),
        signature=sig,
        version=ga.CACHE_VERSION,
        node_count=1,
        edge_count=0,
    )
    ga._write_cache_atomic(ga._cache_path(None), G_cached, meta)

    # current_signature returns the matching signature — load_graph must NOT be called
    async def fake_signature(project=None):
        return sig

    async def fake_load(project=None):
        raise AssertionError("load_graph should not be called on cache hit")

    with patch.object(ga, "current_signature", side_effect=fake_signature), patch.object(
        ga, "load_graph", side_effect=fake_load
    ):
        G, returned_meta = await ga.get_cached_graph()

    assert G.number_of_nodes() == 1
    assert G.nodes["a"]["name"] == "Cached"
    assert returned_meta.signature == sig


@pytest.mark.asyncio
async def test_get_cached_graph_rebuilds_on_signature_mismatch(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(ga, "CACHE_DIR", tmp_path)

    # Seed a cache with an old signature
    G_cached = nx.MultiDiGraph()
    G_cached.add_node("old")
    old_sig = ga.GraphSignature(
        max_updated=datetime(2026, 4, 20, tzinfo=timezone.utc),
        entity_count=1,
        relation_count=0,
    )
    meta = ga.GraphMeta(
        project=None,
        built_at=datetime.now(timezone.utc),
        signature=old_sig,
        version=ga.CACHE_VERSION,
        node_count=1,
        edge_count=0,
    )
    ga._write_cache_atomic(ga._cache_path(None), G_cached, meta)

    # Current signature differs — should rebuild
    new_sig = ga.GraphSignature(
        max_updated=datetime(2026, 4, 21, tzinfo=timezone.utc),
        entity_count=2,
        relation_count=1,
    )

    async def fake_signature(project=None):
        return new_sig

    G_fresh = nx.MultiDiGraph()
    G_fresh.add_node("new-a")
    G_fresh.add_node("new-b")
    G_fresh.add_edge(
        "new-a", "new-b", key="r1", relation_type="calls", weight=1.0, evidence=None
    )

    async def fake_load(project=None):
        return G_fresh

    with patch.object(ga, "current_signature", side_effect=fake_signature), patch.object(
        ga, "load_graph", side_effect=fake_load
    ):
        G, returned_meta = await ga.get_cached_graph()

    assert G.number_of_nodes() == 2
    assert "old" not in G.nodes
    assert "new-a" in G.nodes
    assert returned_meta.signature == new_sig

    # And the cache on disk should now reflect the fresh data
    with ga._cache_path(None).open("rb") as f:
        persisted_G, persisted_meta = pickle.load(f)
    assert persisted_meta.signature == new_sig
    assert persisted_G.number_of_nodes() == 2


@pytest.mark.asyncio
async def test_get_cached_graph_force_rebuild(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ga, "CACHE_DIR", tmp_path)

    # Seed a valid cache whose signature matches
    G_cached = nx.MultiDiGraph()
    G_cached.add_node("cached-only")
    sig = ga.GraphSignature(None, 1, 0)
    meta = ga.GraphMeta(
        project=None,
        built_at=datetime.now(timezone.utc),
        signature=sig,
        version=ga.CACHE_VERSION,
        node_count=1,
        edge_count=0,
    )
    ga._write_cache_atomic(ga._cache_path(None), G_cached, meta)

    async def fake_signature(project=None):
        return sig

    G_fresh = nx.MultiDiGraph()
    G_fresh.add_node("fresh")

    load_called = {"count": 0}

    async def fake_load(project=None):
        load_called["count"] += 1
        return G_fresh

    with patch.object(ga, "current_signature", side_effect=fake_signature), patch.object(
        ga, "load_graph", side_effect=fake_load
    ):
        G, _ = await ga.get_cached_graph(force_rebuild=True)

    assert load_called["count"] == 1
    assert "fresh" in G.nodes
    assert "cached-only" not in G.nodes


@pytest.mark.asyncio
async def test_get_cached_graph_rebuilds_on_corrupt_cache(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(ga, "CACHE_DIR", tmp_path)

    # Write a garbage file at the cache path
    tmp_path.mkdir(exist_ok=True)
    ga._cache_path(None).write_bytes(b"not a pickle")

    sig = ga.GraphSignature(None, 1, 0)

    async def fake_signature(project=None):
        return sig

    G_fresh = nx.MultiDiGraph()
    G_fresh.add_node("rebuilt")

    async def fake_load(project=None):
        return G_fresh

    with patch.object(ga, "current_signature", side_effect=fake_signature), patch.object(
        ga, "load_graph", side_effect=fake_load
    ):
        G, meta = await ga.get_cached_graph()

    assert "rebuilt" in G.nodes
    assert meta.signature == sig


# ── Query helpers: find_nodes_by_name ───────────────────────────────────────


def _toy_graph() -> nx.MultiDiGraph:
    """Fixture: a small hand-built MultiDiGraph for traversal tests.

    Layout:
        alpha (proj=A) --calls--> beta (proj=A) --calls--> gamma (proj=B)
        alpha --imports--> beta   (parallel edge)
        delta (proj=A) --calls--> beta
        gamma --calls--> alpha    (cycle back)
        isolated (proj=A)         (no edges)
        DUPLICATE (proj=A), DUPLICATE (proj=B)  (name collision across projects)
    """
    G = nx.MultiDiGraph()
    G.add_node("n-alpha", name="alpha", entity_type="function", project="A")
    G.add_node("n-beta", name="beta", entity_type="function", project="A")
    G.add_node("n-gamma", name="gamma", entity_type="function", project="B")
    G.add_node("n-delta", name="delta", entity_type="function", project="A")
    G.add_node("n-iso", name="isolated", entity_type="function", project="A")
    G.add_node("n-dup1", name="DUPLICATE", entity_type="function", project="A")
    G.add_node("n-dup2", name="DUPLICATE", entity_type="class", project="B")

    G.add_edge("n-alpha", "n-beta", key="r1", relation_type="calls")
    G.add_edge("n-alpha", "n-beta", key="r2", relation_type="imports")
    G.add_edge("n-beta", "n-gamma", key="r3", relation_type="calls")
    G.add_edge("n-delta", "n-beta", key="r4", relation_type="calls")
    G.add_edge("n-gamma", "n-alpha", key="r5", relation_type="calls")
    return G


def test_find_nodes_by_name_case_insensitive():
    G = _toy_graph()
    assert ga.find_nodes_by_name(G, "ALPHA") == ["n-alpha"]
    assert ga.find_nodes_by_name(G, "alpha") == ["n-alpha"]


def test_find_nodes_by_name_no_match():
    G = _toy_graph()
    assert ga.find_nodes_by_name(G, "does-not-exist") == []


def test_find_nodes_by_name_multiple_matches():
    G = _toy_graph()
    matches = ga.find_nodes_by_name(G, "duplicate")
    assert set(matches) == {"n-dup1", "n-dup2"}


def test_find_nodes_by_name_project_filter():
    G = _toy_graph()
    assert ga.find_nodes_by_name(G, "duplicate", project="A") == ["n-dup1"]
    assert ga.find_nodes_by_name(G, "duplicate", project="B") == ["n-dup2"]
    assert ga.find_nodes_by_name(G, "alpha", project="B") == []


# ── Query helpers: walk ─────────────────────────────────────────────────────


def test_walk_depth_zero_returns_only_source():
    G = _toy_graph()
    result = ga.walk(G, "n-alpha", depth=0, direction="out")
    assert result == {"n-alpha": 0}


def test_walk_out_single_hop():
    G = _toy_graph()
    result = ga.walk(G, "n-alpha", depth=1, direction="out")
    assert result == {"n-alpha": 0, "n-beta": 1}


def test_walk_out_two_hops():
    G = _toy_graph()
    result = ga.walk(G, "n-alpha", depth=2, direction="out")
    assert result["n-alpha"] == 0
    assert result["n-beta"] == 1
    assert result["n-gamma"] == 2


def test_walk_in_shows_dependents():
    G = _toy_graph()
    result = ga.walk(G, "n-beta", depth=1, direction="in")
    # alpha and delta both call beta directly
    assert result["n-beta"] == 0
    assert result["n-alpha"] == 1
    assert result["n-delta"] == 1
    assert "n-gamma" not in result


def test_walk_both_is_undirected():
    G = _toy_graph()
    result = ga.walk(G, "n-beta", depth=1, direction="both")
    # Direct neighbors in either direction: alpha (in), delta (in), gamma (out)
    assert set(result.keys()) == {"n-beta", "n-alpha", "n-delta", "n-gamma"}


def test_walk_handles_cycles():
    """The cycle alpha → beta → gamma → alpha must not cause infinite loop."""
    G = _toy_graph()
    result = ga.walk(G, "n-alpha", depth=5, direction="out")
    # Each node visited at shortest distance only
    assert result["n-alpha"] == 0
    assert result["n-beta"] == 1
    assert result["n-gamma"] == 2


def test_walk_invalid_direction_raises():
    G = _toy_graph()
    with pytest.raises(ValueError, match="direction"):
        ga.walk(G, "n-alpha", depth=1, direction="sideways")


def test_walk_invalid_depth_raises():
    G = _toy_graph()
    with pytest.raises(ValueError, match="depth"):
        ga.walk(G, "n-alpha", depth=-1, direction="out")


def test_walk_missing_source_raises():
    G = _toy_graph()
    with pytest.raises(ValueError, match="not in graph"):
        ga.walk(G, "nonexistent", depth=1, direction="out")


def test_walk_isolated_node():
    G = _toy_graph()
    result = ga.walk(G, "n-iso", depth=10, direction="both")
    assert result == {"n-iso": 0}


# ── Query helpers: edges_between ────────────────────────────────────────────


def test_edges_between_returns_all_parallel_edges():
    G = _toy_graph()
    edges = ga.edges_between(G, "n-alpha", "n-beta")
    rel_types = {e["relation_type"] for e in edges}
    assert rel_types == {"calls", "imports"}


def test_edges_between_single_edge():
    G = _toy_graph()
    edges = ga.edges_between(G, "n-beta", "n-gamma")
    assert len(edges) == 1
    assert edges[0]["relation_type"] == "calls"


def test_edges_between_no_edge():
    G = _toy_graph()
    assert ga.edges_between(G, "n-alpha", "n-iso") == []


def test_edges_between_wrong_direction():
    G = _toy_graph()
    # beta → alpha does not exist (only alpha → beta)
    assert ga.edges_between(G, "n-beta", "n-alpha") == []


# ── Query helpers: shortest_path_between ────────────────────────────────────


def test_shortest_path_direct():
    G = _toy_graph()
    assert ga.shortest_path_between(G, "n-alpha", "n-beta") == ["n-alpha", "n-beta"]


def test_shortest_path_multi_hop():
    G = _toy_graph()
    path = ga.shortest_path_between(G, "n-alpha", "n-gamma")
    assert path == ["n-alpha", "n-beta", "n-gamma"]


def test_shortest_path_no_path():
    G = _toy_graph()
    # isolated has no edges at all
    assert ga.shortest_path_between(G, "n-alpha", "n-iso") is None


def test_shortest_path_missing_node():
    G = _toy_graph()
    assert ga.shortest_path_between(G, "n-alpha", "not-a-node") is None
    assert ga.shortest_path_between(G, "not-a-node", "n-alpha") is None


def test_shortest_path_self_loop():
    """Path to self is trivial (single-node path)."""
    G = _toy_graph()
    assert ga.shortest_path_between(G, "n-alpha", "n-alpha") == ["n-alpha"]


# ── Centrality: rank_nodes ──────────────────────────────────────────────────


def test_rank_nodes_invalid_metric_raises():
    G = _toy_graph()
    with pytest.raises(ValueError, match="metric must be one of"):
        ga.rank_nodes(G, metric="bogus")


def test_rank_nodes_empty_graph_returns_empty():
    G = nx.MultiDiGraph()
    assert ga.rank_nodes(G, metric="pagerank") == []


def test_rank_nodes_pagerank_sums_to_one():
    G = _toy_graph()
    ranked = ga.rank_nodes(G, metric="pagerank")
    total = sum(score for _, score in ranked)
    # PageRank scores sum to 1.0 by construction
    assert abs(total - 1.0) < 1e-6


def test_rank_nodes_pagerank_sorted_desc():
    G = _toy_graph()
    ranked = ga.rank_nodes(G, metric="pagerank")
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_nodes_pagerank_beta_is_most_central():
    """beta has 3 incoming edges (alpha x2 parallel, delta) — highest PR."""
    G = _toy_graph()
    ranked = ga.rank_nodes(G, metric="pagerank")
    top_node = ranked[0][0]
    assert top_node == "n-beta"


def test_rank_nodes_top_n_limits_results():
    G = _toy_graph()
    ranked = ga.rank_nodes(G, metric="pagerank", top_n=3)
    assert len(ranked) == 3


def test_rank_nodes_in_degree_identifies_beta():
    """beta has 3 incoming edges — the most of any node."""
    G = _toy_graph()
    ranked = ga.rank_nodes(G, metric="in_degree")
    assert ranked[0][0] == "n-beta"


def test_rank_nodes_out_degree_identifies_alpha():
    """alpha has 2 outgoing (parallel edges to beta) — most out-edges."""
    G = _toy_graph()
    ranked = ga.rank_nodes(G, metric="out_degree")
    assert ranked[0][0] == "n-alpha"


def test_rank_nodes_betweenness_works_on_multidigraph():
    """Smoke test that betweenness computes and returns something sensible."""
    G = _toy_graph()
    ranked = ga.rank_nodes(G, metric="betweenness")
    assert len(ranked) == G.number_of_nodes()
    # Isolated node must have zero betweenness
    iso_score = next(s for nid, s in ranked if nid == "n-iso")
    assert iso_score == 0.0


# ── Graph-level stats ───────────────────────────────────────────────────────


def test_graph_stats_empty_graph_safe():
    G = nx.MultiDiGraph()
    stats = ga.graph_stats(G)
    assert stats.node_count == 0
    assert stats.edge_count == 0
    assert stats.density == 0.0
    assert stats.weakly_connected_components == 0
    assert stats.top_by_pagerank == []


def test_graph_stats_counts_toy_graph():
    G = _toy_graph()
    stats = ga.graph_stats(G)
    assert stats.node_count == 7
    assert stats.edge_count == 5


def test_graph_stats_isolated_node_counted():
    G = _toy_graph()
    stats = ga.graph_stats(G)
    # Three nodes have zero in + zero out: n-iso and the two DUPLICATE fixtures
    assert stats.isolated_nodes == 3


def test_graph_stats_components_include_isolated_nodes():
    """Each isolated node and each connected subgraph counts as one component."""
    G = _toy_graph()
    stats = ga.graph_stats(G)
    # Components: {alpha, beta, gamma, delta} (all connected via beta), {iso},
    # {dup1}, {dup2}. Total: 4 components.
    assert stats.weakly_connected_components == 4
    assert stats.largest_component_size == 4


def test_graph_stats_entity_type_counts_sorted_desc():
    G = _toy_graph()
    stats = ga.graph_stats(G)
    # 6 functions + 1 class
    assert stats.entity_type_counts["function"] == 6
    assert stats.entity_type_counts["class"] == 1
    # Sorted by count desc
    counts = list(stats.entity_type_counts.values())
    assert counts == sorted(counts, reverse=True)


def test_graph_stats_relation_type_counts():
    G = _toy_graph()
    stats = ga.graph_stats(G)
    # 4 calls, 1 imports
    assert stats.relation_type_counts["calls"] == 4
    assert stats.relation_type_counts["imports"] == 1


def test_graph_stats_top_central_respects_limit():
    G = _toy_graph()
    stats = ga.graph_stats(G, top_central=3)
    assert len(stats.top_by_pagerank) == 3


# ── DB-dependent integration tests (skipped without Postgres) ───────────────


async def _db_available() -> bool:
    """Return True iff a live Postgres with our schema is reachable."""
    try:
        from sqlalchemy import text

        from hafiz.core.database import close_engine, get_session_factory

        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(text("SELECT 1 FROM entities LIMIT 1"))
        return True
    except Exception:
        return False
    finally:
        try:
            from hafiz.core.database import close_engine

            await close_engine()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_load_graph_live_db_returns_multidigraph():
    """Against a live DB, load_graph returns a MultiDiGraph whose counts match the signature."""
    if not await _db_available():
        pytest.skip("No live Postgres with hafiz schema available")

    from hafiz.core.database import close_engine

    try:
        sig = await ga.current_signature()
        G = await ga.load_graph()
        assert isinstance(G, nx.MultiDiGraph)
        assert G.number_of_nodes() == sig.entity_count
        # Edge count may be <= relation_count because relations with
        # missing/out-of-scope endpoints are filtered out. It must never exceed.
        assert G.number_of_edges() <= sig.relation_count
    finally:
        await close_engine()


@pytest.mark.asyncio
async def test_current_signature_live_db_matches_counts():
    """Signature's counts must match direct SELECT COUNT(*) on the same tables."""
    if not await _db_available():
        pytest.skip("No live Postgres with hafiz schema available")

    from sqlalchemy import func, select

    from hafiz.core.database import Entity, Relation, close_engine, get_session_factory

    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            direct_entities = (
                await session.execute(select(func.count()).select_from(Entity))
            ).scalar() or 0
            direct_relations = (
                await session.execute(select(func.count()).select_from(Relation))
            ).scalar() or 0

        sig = await ga.current_signature()
        assert sig.entity_count == direct_entities
        assert sig.relation_count == direct_relations
    finally:
        await close_engine()
