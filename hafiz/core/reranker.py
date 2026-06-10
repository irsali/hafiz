"""Cross-encoder reranking — second-stage precision for recall.

Vector similarity is a *bi-encoder* score: query and document are embedded
independently, so it compresses genuinely-relevant rows and near-random noise
into a narrow band (measured on nomic-embed: on-topic ~0.47-0.67, off-topic
~0.45-0.58 — barely separable). A *cross-encoder* scores each (query, document)
pair jointly and separates them sharply.

We use it as a **reordering** stage, never a replacement: vector search
over-fetches K candidates, the cross-encoder re-scores them, and we return the
top N. If the reranker is unavailable or errors, callers fall back to the
vector order — reranking can only improve precision, never break recall.

The model ships with fastembed (``TextCrossEncoder``) — no extra dependency —
and loads lazily on first use, cached to the same persistent dir as the
embedding model with the same self-heal-on-corruption behavior.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TypeVar

from hafiz.core.config import load_settings
from hafiz.core.embeddings import _model_cache_dir, _purge_if_incomplete

logger = logging.getLogger(__name__)

_reranker = None
_reranker_lock = threading.Lock()
# Serialize ONNX inference calls — one warm model shared across async callers;
# cross-encoder Run() concurrency safety isn't guaranteed across fastembed
# versions, and a single user's recall load is low, so a lock is the safe call.
_infer_lock = asyncio.Lock()

T = TypeVar("T")

# Cross-encoder cost scales with (query + doc) token length: reranking 24
# full-length annotation bodies (~2 KB each) costs ~400-850ms, but 24 short
# docs ~18ms. The relevance signal lives in the opening sentences, and that's
# all we surface anyway — so cap the text sent to the model. Cuts rerank
# latency ~5-10x with negligible quality loss.
_RERANK_DOC_CHARS = 400


def rerank_enabled() -> bool:
    return load_settings().rerank.enabled


def _build_reranker():
    """Construct the fastembed cross-encoder against the persistent cache.

    Self-heals a corrupt/partial download before loading (reusing the embed
    model's cache-repair logic), and raises an actionable error on failure
    rather than leaking a bare ONNX traceback.
    """
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    model_name = load_settings().rerank.model
    cache_dir = _model_cache_dir()
    _purge_if_incomplete(cache_dir, model_name)
    try:
        return TextCrossEncoder(model_name=model_name, cache_dir=str(cache_dir))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load reranker model '{model_name}' from {cache_dir}: "
            f"{exc}\nThe download is likely corrupt or incomplete. Delete "
            f"{cache_dir} and retry, or disable reranking with "
            f"`hafiz config set rerank.enabled false`."
        ) from exc


def get_reranker():
    """Lazy singleton accessor for the cross-encoder model."""
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is None:
            _reranker = _build_reranker()
    return _reranker


async def warm_reranker() -> None:
    """Load the reranker now (used by the daemon at startup when enabled)."""
    await asyncio.to_thread(get_reranker)


async def rerank(
    query: str,
    items: list[T],
    *,
    text_of,
    top_n: int | None = None,
) -> list[T]:
    """Reorder ``items`` by cross-encoder relevance to ``query``.

    ``text_of(item)`` extracts the string to score for each item. Returns the
    items sorted most-relevant first, truncated to ``top_n`` if given. On any
    reranker failure (model load, inference), returns ``items`` unchanged (the
    vector order) — reranking is strictly additive.
    """
    if not items or not query.strip():
        return items[:top_n] if top_n else items
    # Truncate to the relevance-bearing head — cross-encoder cost scales with
    # doc length, and the opening carries the signal (see _RERANK_DOC_CHARS).
    docs = [(text_of(it) or "")[:_RERANK_DOC_CHARS] for it in items]
    try:
        model = get_reranker()
        async with _infer_lock:
            scores = await asyncio.to_thread(lambda: list(model.rerank(query, docs)))
    except Exception as exc:  # noqa: BLE001 — degrade to vector order, never raise
        logger.warning("rerank failed (%s); falling back to vector order", exc)
        return items[:top_n] if top_n else items
    ranked = [it for _, it in sorted(zip(scores, items), key=lambda p: p[0], reverse=True)]
    return ranked[:top_n] if top_n else ranked
