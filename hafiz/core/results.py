"""Search result types, importable without the database layer.

The warm daemon returns these over its socket, so a CLI call it serves only
needs the dataclasses — not SQLAlchemy, pgvector and fastembed, which
:mod:`hafiz.core.annotations` and :mod:`hafiz.core.search` pull in and which
cost ~0.3s per hook-driven call before any work was done.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class AnnotationResult:
    """A single annotation search result with its relevance scores.

    Two scores, deliberately both surfaced:

    * ``score`` — cosine similarity from the vector stage, 0–1. Kept as-is
      for back-compat; live agent hooks read this field.
    * ``rerank_score`` — the cross-encoder's 0–1 relevance, or ``None`` when
      reranking did not run. This is the score the results are *ordered* by
      when it is present, and the one a relevance floor must filter on.

    ``score`` alone cannot tell you which stage ranked the row: under
    reranking it is non-monotonic down the result list, so filtering on it
    fights the ordering.
    """

    id: str
    content: str
    kind: str
    source: str | None
    project: str | None
    tags: list[str] | None
    confidence: float
    valid_from: datetime
    valid_until: datetime | None
    unit_id: str | None
    metadata: dict
    score: float
    rerank_score: float | None = None
    #: Best-matching excerpt, set only for ``compact``/``md`` output and only
    #: when ``content`` exceeded the snippet budget. ``None`` means the full
    #: record is what gets rendered — see :mod:`hafiz.core.snippets`.
    snippet: str | None = None

    @property
    def ranking_score(self) -> float:
        """The score this row was actually ordered by — rerank if it ran."""
        return self.score if self.rerank_score is None else self.rerank_score


@dataclass
class SearchResult:
    """A single search result. Fields are populated from the joined
    ``embeddings`` → ``unit_revisions`` → ``units`` → ``files`` row."""

    id: str  # embedding row id
    unit_id: str
    unit_name: str
    kind: str  # namespaced: code.function, doc.heading, file.raw, …
    content: str  # the embedded part (may be a slice of the unit body)
    source_file: str
    line_start: int | None
    line_end: int | None
    language: str | None
    project: str | None
    part_index: int
    score: float
    is_neighbor: bool = False
