"""Tests for ``embed_texts`` internal chunking — runtime safety brake.

The probe-derived ``embedding.max_part_chars`` recommendation is only
half the story. Real ingest can submit hundreds of parts in one
``embed_texts(...)`` call (a long markdown doc or a large Python
module produces many parts at the configured part size). Without
internal chunking, peak RSS scales with the part count and OOMs the
host even when the per-part size is "safe".

Concrete failure mode (incident 2026-04-27, second occurrence):
``hafiz ingest`` was killed by the OOM killer at 51 GB RSS while
ingesting a single file with sticky ``max_part_chars=4000`` —
~100 parts × 4 K chars × ~500 MB per-doc peak = ~50 GB.

These tests pin down the chunking contract:

  - Long lists are split into sub-batches and re-assembled in order.
  - Sub-batch size shrinks as the part size grows (``_safe_batch_size``
    table mirrors probe data on RSS-per-doc).
  - The model's batch endpoint is called multiple times, never with
    the full list when it exceeds the sub-batch size.
"""

from __future__ import annotations

import pytest

from hafiz.core import embeddings

# ── _safe_batch_size table ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "max_chars,expected",
    [
        (1_000, 32),
        (2_000, 24),
        (4_000, 8),
        (8_000, 2),
        (16_000, 1),
        (20_000, 1),  # above the largest threshold still maps to smallest band
    ],
)
def test_safe_batch_size_table(max_chars, expected):
    assert embeddings._safe_batch_size(max_chars) == expected


def test_safe_batch_size_is_monotone_non_increasing():
    """Larger ``max_part_chars`` must NEVER license a larger batch.
    Per-doc memory grows with chars (quadratic for attention), so
    the safe batch must shrink — anything else is the OOM trap."""
    sizes = [embeddings._safe_batch_size(c) for c in (1_000, 2_000, 4_000, 8_000, 16_000)]
    assert sizes == sorted(sizes, reverse=True)


# ── embed_texts chunking ───────────────────────────────────────────────


class _FakeModel:
    """Records each batch handed to it. Returns deterministic vectors
    so we can assert order is preserved across sub-batches.

    Mimics ``fastembed.TextEmbedding.embed`` — sync, returns an iterable
    of vectors. Production wraps this in ``asyncio.to_thread``.
    """

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        return [[float(len(t)), float(i)] for i, t in enumerate(texts)]


@pytest.fixture(autouse=True)
def _reset_embed_singleton():
    embeddings.reset_cache()
    yield
    embeddings.reset_cache()


def _stub_model(monkeypatch) -> _FakeModel:
    fake = _FakeModel()
    monkeypatch.setattr(embeddings, "get_embed_model", lambda: fake)
    return fake


def _stub_max_chars(monkeypatch, value: int) -> None:
    """Stub the tunable resolver so we control the sub-batch decision
    independent of any sticky cache on the dev machine."""
    from hafiz.core import tunables

    monkeypatch.setattr(tunables, "resolve", lambda key: value)


@pytest.mark.asyncio
async def test_embed_texts_short_list_runs_in_one_call(monkeypatch):
    """A list shorter than the sub-batch size must NOT be artificially
    split — that would just add overhead without any safety benefit."""
    fake = _stub_model(monkeypatch)
    _stub_max_chars(monkeypatch, 4_000)  # sub-batch = 8

    texts = [f"text-{i}" for i in range(5)]
    result = await embeddings.embed_texts(texts)

    assert len(fake.calls) == 1
    assert len(result) == 5


@pytest.mark.asyncio
async def test_embed_texts_chunks_long_list(monkeypatch):
    """The driving regression: 100 parts at 4 K chars used to be one
    50 GB embedding call. Must split into 13 sub-batches of ≤8."""
    fake = _stub_model(monkeypatch)
    _stub_max_chars(monkeypatch, 4_000)  # sub-batch = 8

    texts = [f"text-{i}" for i in range(100)]
    result = await embeddings.embed_texts(texts)

    assert len(fake.calls) == 13  # ceil(100 / 8)
    for call in fake.calls[:-1]:
        assert len(call) == 8
    assert len(fake.calls[-1]) == 4
    assert len(result) == 100


@pytest.mark.asyncio
async def test_embed_texts_preserves_order_across_sub_batches(monkeypatch):
    """Sub-batches must be concatenated in order — a vector misalignment
    between texts and embeddings would silently corrupt the index."""
    _stub_model(monkeypatch)
    _stub_max_chars(monkeypatch, 4_000)

    texts = [f"text-of-length-{i:03d}" for i in range(20)]
    result = await embeddings.embed_texts(texts)

    assert len(result) == 20
    # First component of each fake vector encodes len(text). If chunking
    # reordered, this would diverge from the input lengths.
    for vec, txt in zip(result, texts):
        assert vec[0] == float(len(txt))


@pytest.mark.asyncio
async def test_embed_texts_uses_smaller_batch_for_larger_chars(monkeypatch):
    """At max_part_chars=8000, sub-batch must be 2 (not 8). Otherwise a
    100-part file would peak ~125 GB (100 × 1.25 GB/doc). The probe
    recommendation alone doesn't catch this — runtime chunking does."""
    fake = _stub_model(monkeypatch)
    _stub_max_chars(monkeypatch, 8_000)  # sub-batch = 2

    texts = [f"text-{i}" for i in range(10)]
    await embeddings.embed_texts(texts)

    assert len(fake.calls) == 5  # ceil(10 / 2)
    for call in fake.calls:
        assert len(call) <= 2


@pytest.mark.asyncio
async def test_embed_texts_empty_list_short_circuits(monkeypatch):
    """An empty list must not load the model or call the embedder."""

    def explode():
        pytest.fail("embed_texts called the model for an empty list")

    monkeypatch.setattr(embeddings, "get_embed_model", lambda: explode())

    result = await embeddings.embed_texts([])
    assert result == []
