"""Tests for the cross-encoder reranker's contract.

The load-bearing guarantee: reranking is strictly a *reordering* of vector
results — on any failure it returns the input order unchanged, never raising,
so it can improve recall precision but never break recall. Model inference
itself is covered by dogfooding (the A/B against real queries).
"""

from __future__ import annotations

import hafiz.core.reranker as reranker


async def test_rerank_empty_and_blank_query_passthrough():
    items = [{"c": "a"}, {"c": "b"}]
    assert await reranker.rerank("", items, text_of=lambda r: r["c"]) == items
    assert await reranker.rerank("q", [], text_of=lambda r: r["c"]) == []


async def test_rerank_falls_back_to_input_order_on_model_failure(monkeypatch):
    """If the model can't load / scores blow up, return the vector order."""

    def _boom():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(reranker, "get_reranker", _boom)
    items = [{"c": "first"}, {"c": "second"}, {"c": "third"}]
    out = await reranker.rerank("query", items, text_of=lambda r: r["c"])
    assert out == items  # unchanged order, no exception


async def test_rerank_respects_top_n_on_fallback(monkeypatch):
    monkeypatch.setattr(
        reranker, "get_reranker", lambda: (_ for _ in ()).throw(RuntimeError("x"))
    )
    items = [{"c": str(i)} for i in range(10)]
    out = await reranker.rerank("q", items, text_of=lambda r: r["c"], top_n=3)
    assert out == items[:3]


async def test_rerank_reorders_by_model_score(monkeypatch):
    """With a stub model, items are sorted by descending score and truncated."""

    class StubModel:
        def rerank(self, query, docs):
            # Score = reverse of position, so the LAST doc ranks first.
            return [float(i) for i in range(len(docs))]

    monkeypatch.setattr(reranker, "get_reranker", lambda: StubModel())
    items = [{"c": "a"}, {"c": "b"}, {"c": "c"}]
    out = await reranker.rerank("q", items, text_of=lambda r: r["c"], top_n=2)
    # Highest score (last item) first.
    assert [r["c"] for r in out] == ["c", "b"]


async def test_rerank_truncates_long_docs_before_scoring(monkeypatch):
    """Docs are capped to the relevance-bearing head before the model sees them."""
    seen = {}

    class StubModel:
        def rerank(self, query, docs):
            seen["docs"] = docs
            return [1.0 for _ in docs]

    monkeypatch.setattr(reranker, "get_reranker", lambda: StubModel())
    long_doc = "x" * 5000
    await reranker.rerank("q", [{"c": long_doc}], text_of=lambda r: r["c"])
    assert len(seen["docs"][0]) == reranker._RERANK_DOC_CHARS
