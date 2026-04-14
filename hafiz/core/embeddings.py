"""Embedding service wrapping llama-index-embeddings-fastembed.

Uses nomic-embed-text-v1.5 (768 dims) by default, running locally via ONNX.
Attempts GPU (CUDA) acceleration automatically, falls back to CPU if it fails.
"""

from __future__ import annotations

import logging

from llama_index.embeddings.fastembed import FastEmbedEmbedding

from hafiz.core.config import get_settings

logger = logging.getLogger(__name__)

_embed_model: FastEmbedEmbedding | None = None


def _cuda_available() -> bool:
    """Check if CUDA provider is available in onnxruntime."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def _try_cuda_embed(model_name: str) -> FastEmbedEmbedding | None:
    """Try creating embedding model with CUDA. Returns None on failure."""
    try:
        model = FastEmbedEmbedding(
            model_name=model_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # Force a test embedding to catch runtime CUDA errors (e.g. unsupported GPU arch)
        model.get_text_embedding("test")
        logger.info("Embedding model using GPU (CUDA)")
        return model
    except Exception:
        logger.warning("GPU not compatible with current onnxruntime — falling back to CPU")
        return None


def get_embed_model() -> FastEmbedEmbedding:
    """Get or create the embedding model (lazy singleton).

    Tries CUDA first if available, falls back to CPU.
    """
    global _embed_model
    if _embed_model is None:
        settings = get_settings()

        if _cuda_available():
            _embed_model = _try_cuda_embed(settings.embedding.model)

        if _embed_model is None:
            _embed_model = FastEmbedEmbedding(
                model_name=settings.embedding.model,
                providers=["CPUExecutionProvider"],
            )
            logger.info("Embedding model using CPU")

    return _embed_model


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts, returning a list of vectors."""
    model = get_embed_model()
    # FastEmbedEmbedding supports async batch embedding
    embeddings = await model.aget_text_embedding_batch(texts)
    return embeddings


async def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    model = get_embed_model()
    return await model.aget_query_embedding(query)
