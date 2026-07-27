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

The cross-encoder score is **surfaced, not discarded** (:func:`rerank_scored`).
It is the only score that separates signal from noise, so it is what a
relevance floor must filter on; hiding it made reranked output indistinguishable
from vector output and invited callers to re-sort on the wrong number.

The model ships with fastembed (``TextCrossEncoder``) — no extra dependency —
and loads lazily on first use, cached to the same persistent dir as the
embedding model with the same self-heal-on-corruption behavior.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading

from hafiz.core.config import load_settings
from hafiz.core.embeddings import _model_cache_dir, _purge_if_incomplete

logger = logging.getLogger(__name__)

_reranker = None
_reranker_lock = threading.Lock()
# Serialize ONNX inference calls — one warm model shared across async callers;
# cross-encoder Run() concurrency safety isn't guaranteed across fastembed
# versions, and a single user's recall load is low, so a lock is the safe call.
_infer_lock = asyncio.Lock()

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


def normalize_score(logit: float) -> float:
    """Map a raw cross-encoder logit onto a 0–1 relevance probability.

    The model emits **unbounded logits** (measured on
    ``ms-marco-MiniLM-L-6-v2``: a relevant doc scored ``+3.32``, two
    irrelevant ones ``-11.35`` / ``-11.31``). Vector similarity is already
    0–1, so surfacing raw logits would give ``--min-score`` two different
    scales depending on whether reranking ran — a footgun. A logistic
    squash puts both on one scale (``+3.32 → 0.965``, ``-11.35 → 0.00001``)
    and *preserves order*, since sigmoid is monotonic.
    """
    # math.exp overflows for very negative logits; clamp to the range where
    # the result is indistinguishable from the asymptote anyway.
    if logit < -60:
        return 0.0
    if logit > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-logit))


async def score_passages(query: str, passages: list[str]) -> list[float] | None:
    """Normalized 0–1 relevance for each passage, **in input order**.

    The primitive underneath :func:`rerank_scored`, for callers that need the
    scores positionally rather than sorted — picking which span of a long
    record to show, say, where the answer is "which index won", not "what is
    the new order".

    Returns ``None`` if scoring could not run, so the caller can fall back
    rather than mistake a failure for a uniform score. Passages are *not*
    truncated to ``_RERANK_DOC_CHARS``: a caller scoring spans has already
    made them short, and silently trimming them here would score a prefix and
    report it as the span's score.
    """
    if not passages or not query.strip():
        return None
    try:
        model = get_reranker()
        async with _infer_lock:
            raw = await asyncio.to_thread(lambda: list(model.rerank(query, passages)))
    except Exception as exc:  # noqa: BLE001 — degrade, never raise into a search
        logger.warning("passage scoring failed (%s)", exc)
        return None
    if len(raw) != len(passages):
        logger.warning(
            "passage scoring returned %d scores for %d passages", len(raw), len(passages)
        )
        return None
    return [normalize_score(float(s)) for s in raw]


async def rerank_scored[T](
    query: str,
    items: list[T],
    *,
    text_of,
    top_n: int | None = None,
) -> list[tuple[T, float | None]]:
    """Reorder ``items`` by cross-encoder relevance, returning the scores.

    Each element is ``(item, score)`` where ``score`` is the 0–1 normalized
    cross-encoder relevance (see :func:`normalize_score`). ``score`` is
    ``None`` when reranking did not run — because the query was blank, the
    item list was empty, or the model failed — which is how callers tell
    "reranked" from "vector order" apart rather than having to guess.
    """
    if not items or not query.strip():
        return [(it, None) for it in (items[:top_n] if top_n else items)]
    # Truncate to the relevance-bearing head — cross-encoder cost scales with
    # doc length, and the opening carries the signal (see _RERANK_DOC_CHARS).
    docs = [(text_of(it) or "")[:_RERANK_DOC_CHARS] for it in items]
    try:
        model = get_reranker()
        async with _infer_lock:
            scores = await asyncio.to_thread(lambda: list(model.rerank(query, docs)))
    except Exception as exc:  # noqa: BLE001 — degrade to vector order, never raise
        logger.warning("rerank failed (%s); falling back to vector order", exc)
        return [(it, None) for it in (items[:top_n] if top_n else items)]
    ranked = sorted(
        ((it, normalize_score(float(s))) for s, it in zip(scores, items, strict=False)),
        key=lambda p: p[1],
        reverse=True,
    )
    return ranked[:top_n] if top_n else ranked


async def rerank[T](
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

    Score-free convenience wrapper over :func:`rerank_scored`; callers that
    need to filter or display relevance should use that instead.
    """
    return [it for it, _ in await rerank_scored(query, items, text_of=text_of, top_n=top_n)]
