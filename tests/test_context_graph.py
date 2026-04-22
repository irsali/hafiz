"""Tests for _graph_from_chunks — the Phase-4 graph expansion used by hafiz context.

Exercises the N-hop walk + (distance, PageRank) ranking without touching the
database. A hand-built MultiDiGraph is injected via a mocked `get_cached_graph`.
"""

from __future__ import annotations

from unittest.mock import patch

import networkx as nx
import pytest

from hafiz.core.context import _graph_from_chunks
from hafiz.core.search import SearchResult


# ── Fixtures ────────────────────────────────────────────────────────────────


def _chunk(source_file: str, score: float = 0.9) -> SearchResult:
    """Minimal SearchResult for testing — only ``source_file`` is used
    downstream by ``_graph_from_chunks``."""
    return SearchResult(
        id=f"emb-{source_file}",
        unit_id=f"unit-{source_file}",
        unit_name=source_file.rsplit(".", 1)[0],
        kind="code.function",
        content="irrelevant",
        source_file=source_file,
        line_start=1,
        line_end=10,
        language="python",
        project="demo",
        part_index=0,
        score=score,
    )


def _context_graph() -> nx.MultiDiGraph:
    """A richer toy graph for testing context expansion.

    Layout (undirected hops from seed `handler`):
       handler (seed, file=api.py)
         ├── router (file=router.py)            1 hop
         │     └── auth (file=auth.py)          2 hops
         │           └── secrets (file=sec.py)  3 hops
         │                 └── vault (file=v.py) 4 hops  [outside default depth=3]
         └── db_client (file=db.py)             1 hop
               └── pool (file=db.py)            2 hops
       orphan (file=other.py, no edges)         unreachable

    All entities are in project "demo" except `out_of_scope`, which is in
    project "other" and shares a file with `handler` (used for project-filter tests).
    """
    G = nx.MultiDiGraph()
    G.add_node("n-handler", name="handler", kind="function",
               source_file="api.py", project="demo")
    G.add_node("n-router", name="router", kind="function",
               source_file="router.py", project="demo")
    G.add_node("n-auth", name="auth", kind="function",
               source_file="auth.py", project="demo")
    G.add_node("n-secrets", name="secrets", kind="module",
               source_file="sec.py", project="demo")
    G.add_node("n-vault", name="vault", kind="service",
               source_file="v.py", project="demo")
    G.add_node("n-db", name="db_client", kind="class",
               source_file="db.py", project="demo")
    G.add_node("n-pool", name="pool", kind="class",
               source_file="db.py", project="demo")
    G.add_node("n-orphan", name="orphan", kind="function",
               source_file="other.py", project="demo")
    G.add_node("n-oos", name="out_of_scope", kind="function",
               source_file="api.py", project="other")

    G.add_edge("n-handler", "n-router", key="r1", relation="calls", weight=1.0)
    G.add_edge("n-router", "n-auth", key="r2", relation="calls", weight=1.0)
    G.add_edge("n-auth", "n-secrets", key="r3", relation="reads", weight=1.0)
    G.add_edge("n-secrets", "n-vault", key="r4", relation="depends_on", weight=1.0)
    G.add_edge("n-handler", "n-db", key="r5", relation="calls", weight=1.0)
    G.add_edge("n-db", "n-pool", key="r6", relation="depends_on", weight=1.0)
    return G


def _mock_cached_graph(G: nx.MultiDiGraph):
    """Return a patch context manager that hands back (G, fake_meta)."""
    from hafiz.core import graph_analysis as ga

    async def fake_get(project=None, *, force_rebuild=False):
        return G, None  # meta is unused by _graph_from_chunks

    return patch.object(ga, "get_cached_graph", side_effect=fake_get)


# ── No seeds / empty cases ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_returns_empty_when_no_chunks():
    G = _context_graph()
    with _mock_cached_graph(G):
        result = await _graph_from_chunks([])
    assert result == []


@pytest.mark.asyncio
async def test_returns_empty_when_no_source_files_match():
    G = _context_graph()
    chunks = [_chunk("does-not-exist.py")]
    with _mock_cached_graph(G):
        result = await _graph_from_chunks(chunks)
    assert result == []


@pytest.mark.asyncio
async def test_returns_empty_when_graph_empty():
    empty = nx.MultiDiGraph()
    chunks = [_chunk("api.py")]
    with _mock_cached_graph(empty):
        result = await _graph_from_chunks(chunks)
    assert result == []


# ── Seed identification ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_seed_entities_have_distance_zero_and_is_seed_true():
    G = _context_graph()
    chunks = [_chunk("api.py")]
    with _mock_cached_graph(G):
        result = await _graph_from_chunks(chunks, project="demo", depth=0)
    # depth=0 → only the seed itself (in project demo, that's `handler`)
    assert len(result) == 1
    assert result[0]["name"] == "handler"
    assert result[0]["distance"] == 0
    assert result[0]["is_seed"] is True


@pytest.mark.asyncio
async def test_project_filter_excludes_out_of_scope_seeds():
    """`out_of_scope` lives in api.py but belongs to a different project."""
    G = _context_graph()
    chunks = [_chunk("api.py")]
    with _mock_cached_graph(G):
        result = await _graph_from_chunks(chunks, project="demo", depth=0)
    names = {e["name"] for e in result}
    assert "handler" in names
    assert "out_of_scope" not in names


# ── Depth behavior ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_depth_one_reaches_direct_neighbors():
    G = _context_graph()
    chunks = [_chunk("api.py")]
    with _mock_cached_graph(G):
        result = await _graph_from_chunks(chunks, project="demo", depth=1, max_entities=50)
    distances = {e["name"]: e["distance"] for e in result}
    assert distances["handler"] == 0
    assert distances["router"] == 1
    assert distances["db_client"] == 1
    # 2-hop entities must NOT appear
    assert "auth" not in distances
    assert "pool" not in distances


@pytest.mark.asyncio
async def test_depth_three_matches_default_semantic():
    """At depth=3 the walk reaches secrets but not vault (4 hops away)."""
    G = _context_graph()
    chunks = [_chunk("api.py")]
    with _mock_cached_graph(G):
        result = await _graph_from_chunks(chunks, project="demo", depth=3, max_entities=50)
    names = {e["name"] for e in result}
    assert {"handler", "router", "auth", "secrets", "db_client", "pool"}.issubset(names)
    assert "vault" not in names


@pytest.mark.asyncio
async def test_orphan_node_never_reached():
    G = _context_graph()
    chunks = [_chunk("api.py")]
    with _mock_cached_graph(G):
        result = await _graph_from_chunks(chunks, project="demo", depth=10, max_entities=50)
    names = {e["name"] for e in result}
    assert "orphan" not in names


# ── Multi-source BFS ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_source_uses_minimum_distance():
    """Two seeds → every node records its minimum distance to any seed."""
    G = _context_graph()
    # Seed from auth.py AND db.py — pool is 1 hop from db_client (which is in db.py),
    # not reached from auth at depth 2 (auth -> secrets -> vault, no path to pool).
    chunks = [_chunk("auth.py"), _chunk("db.py")]
    with _mock_cached_graph(G):
        result = await _graph_from_chunks(chunks, project="demo", depth=1, max_entities=50)
    distances = {e["name"]: e["distance"] for e in result}
    # auth itself: seed (dist 0); db_client and pool: seeds from db.py (dist 0)
    assert distances["auth"] == 0
    assert distances["db_client"] == 0
    assert distances["pool"] == 0
    # router is 1 hop from auth (inbound); handler is 1 hop from both auth and db
    assert distances["router"] == 1
    assert distances["handler"] == 1


# ── Ranking: (distance asc, pagerank desc) ──────────────────────────────────


@pytest.mark.asyncio
async def test_ranking_sorted_by_distance_then_pagerank():
    G = _context_graph()
    chunks = [_chunk("api.py")]
    with _mock_cached_graph(G):
        result = await _graph_from_chunks(chunks, project="demo", depth=3, max_entities=50)
    # First entry must be a seed (distance 0)
    assert result[0]["distance"] == 0
    # Non-decreasing distances overall
    dists = [e["distance"] for e in result]
    assert dists == sorted(dists)
    # Within the same distance, pagerank_score must be non-increasing
    for i in range(1, len(result)):
        if result[i]["distance"] == result[i - 1]["distance"]:
            assert result[i]["pagerank_score"] <= result[i - 1]["pagerank_score"]


# ── max_entities cap ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_entities_caps_output():
    G = _context_graph()
    chunks = [_chunk("api.py")]
    with _mock_cached_graph(G):
        result = await _graph_from_chunks(chunks, project="demo", depth=5, max_entities=3)
    assert len(result) == 3
    # Cap preserves ranking — first result must be the seed at distance 0
    assert result[0]["distance"] == 0


# ── Connections preserved ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connections_include_both_directions():
    G = _context_graph()
    chunks = [_chunk("router.py")]
    with _mock_cached_graph(G):
        result = await _graph_from_chunks(chunks, project="demo", depth=0, max_entities=10)
    router = next(e for e in result if e["name"] == "router")
    directions = {c["direction"] for c in router["connections"]}
    # router has both incoming (from handler) and outgoing (to auth)
    assert directions == {"-->", "<--"}


# ── Defaults come from config ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_depth_and_max_entities_default_to_settings(monkeypatch):
    """When depth and max_entities aren't passed, use GraphSettings defaults."""
    from hafiz.core import config as cfg

    # Override defaults
    cfg.reset_settings()
    monkeypatch.setenv("HAFIZ_GRAPH__CONTEXT_DEPTH", "1")
    monkeypatch.setenv("HAFIZ_GRAPH__CONTEXT_MAX_ENTITIES", "2")
    try:
        G = _context_graph()
        chunks = [_chunk("api.py")]
        with _mock_cached_graph(G):
            result = await _graph_from_chunks(chunks, project="demo")
        # With depth=1 and cap=2, we get at most 2 entries, none beyond 1 hop
        assert len(result) <= 2
        assert all(e["distance"] <= 1 for e in result)
    finally:
        cfg.reset_settings()
