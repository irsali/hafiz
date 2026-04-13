"""Embedding service wrapping llama-index-embeddings-fastembed.

Uses nomic-embed-text-v1.5 (768 dims) by default, running locally via ONNX.
"""

from __future__ import annotations

from llama_index.embeddings.fastembed import FastEmbedEmbedding

from hafiz.core.config import get_settings

_embed_model: FastEmbedEmbedding | None = None


def get_embed_model() -> FastEmbedEmbedding:
    """Get or create the embedding model (lazy singleton)."""
    global _embed_model
    if _embed_model is None:
        settings = get_settings()
        _embed_model = FastEmbedEmbedding(model_name=settings.embedding.model)
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
