"""Read-time snippet extraction.

DB-free. Two properties matter more than the trimming itself:

* **It never fails a search.** Snippets are computed on the read path, and a
  memory layer that can break recall gets removed. Every failure mode here has
  to degrade to *some* excerpt, never an exception.
* **A truncation is never silent.** A decision read without its scope
  qualifier is worse than one that was never retrieved, so an excerpt has to
  be identifiable as an excerpt from the rendered output alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from hafiz.core.formats import annotation_compact, annotation_md
from hafiz.core.snippets import (
    _FINALISTS,
    _lexical_scores,
    _shortlist,
    attach_snippets,
    build_window,
    split_spans,
)


@dataclass
class FakeResult:
    content: str
    snippet: str | None = None
    kind: str = "decision"
    source: str = "agent:claude-code"
    id: str = "11111111-1111-1111-1111-111111111111"
    valid_from: datetime = datetime(2026, 7, 1, tzinfo=UTC)


# ── splitting ────────────────────────────────────────────────────────


def test_paragraphs_are_the_default_unit():
    spans = split_spans("First block.\n\nSecond block.\n\nThird block.", budget=100)
    assert spans == ["First block.", "Second block.", "Third block."]


def test_an_oversized_paragraph_is_split_into_sentences():
    """A single-paragraph wall is the shape this exists to handle."""
    para = "One sentence here. Two sentence here. Three sentence here."
    assert len(split_spans(para, budget=25)) == 3


def test_a_paragraph_under_budget_is_left_whole_even_with_sentences():
    para = "One sentence here. Two sentence here."
    assert split_spans(para, budget=500) == [para]


def test_text_with_no_sentence_punctuation_survives_as_one_span():
    """No punctuation means no split point; the window builder clips instead
    of this returning nothing."""
    assert split_spans("x" * 300, budget=50) == ["x" * 300]


def test_blank_paragraphs_are_dropped():
    assert split_spans("A.\n\n\n\n   \n\nB.", budget=100) == ["A.", "B."]


# ── lexical shortlisting ─────────────────────────────────────────────


def test_lexical_score_ignores_stopwords():
    """Otherwise every span scores alike and the shortlist is arbitrary."""
    spans = ["the and of with", "cookie consent banner"]
    scores = _lexical_scores("what is the consent banner", spans)
    assert scores[1] > scores[0] == 0.0


def test_a_query_of_only_stopwords_scores_everything_equally():
    assert _lexical_scores("what is the", ["abc def", "ghi jkl"]) == [1.0, 1.0]


def test_shortlist_always_includes_the_head_span():
    """Agents put the claim first, so span 0 is a strong prior worth one slot
    even when a trailing aside happens to carry the query's words."""
    spans = ["head", "a", "b", "c", "match match match"]
    lexical = [0.0, 0.0, 0.0, 0.0, 1.0]
    assert 0 in _shortlist(spans, lexical)
    assert 4 in _shortlist(spans, lexical)


def test_shortlist_is_bounded():
    spans = [f"s{i}" for i in range(20)]
    lexical = [i / 20 for i in range(20)]
    assert len(_shortlist(spans, lexical)) <= _FINALISTS + 1


# ── window building ──────────────────────────────────────────────────


def test_window_anchors_on_the_explicit_best_span():
    spans = ["aaa", "bbb", "ccc"]
    assert "bbb" in build_window(spans, [0.0] * 3, budget=3, best=1)


def test_window_marks_a_clipped_start_and_end():
    spans = ["aaa", "bbb", "ccc"]
    window = build_window(spans, [0.0, 1.0, 0.0], budget=3, best=1)
    assert window.startswith("…")
    assert window.endswith("…")


def test_a_window_covering_everything_is_not_marked():
    """No ellipsis when nothing was dropped — the marker has to mean something."""
    spans = ["aaa", "bbb"]
    window = build_window(spans, [1.0, 1.0], budget=100, best=0)
    assert "…" not in window
    assert window == "aaa bbb"


def test_window_grows_toward_the_higher_scoring_neighbour():
    # Equal-length neighbours, so the score is the only thing deciding —
    # otherwise this passes on whichever happens to be the shorter string.
    spans = ["lft.", "anchor", "rgt."]
    budget = len("anchor") + len("lft.") + 1
    assert "lft." in build_window(spans, [1.0, 0.0, 0.0], budget=budget, best=1)
    assert "rgt." in build_window(spans, [0.0, 0.0, 1.0], budget=budget, best=1)


def test_a_single_span_wider_than_the_budget_is_clipped_not_dropped():
    window = build_window(["x" * 500], [1.0], budget=50)
    assert len(window) <= 50
    assert window.endswith("…")


def test_empty_spans_produce_an_empty_window():
    assert build_window([], [], budget=100) == ""


# ── attach_snippets ──────────────────────────────────────────────────


async def test_records_under_budget_are_left_alone():
    results = [FakeResult(content="short record")]
    await attach_snippets("anything", results, budget=480)
    assert results[0].snippet is None


async def test_a_zero_budget_disables_extraction():
    results = [FakeResult(content="x" * 5000)]
    await attach_snippets("anything", results, budget=0)
    assert results[0].snippet is None


async def test_an_oversized_record_gets_a_bounded_snippet(monkeypatch):
    from hafiz.core import reranker

    async def _scores(query, passages):
        return [float(i) for i in range(len(passages))]

    monkeypatch.setattr(reranker, "score_passages", _scores)
    content = "\n\n".join(f"Paragraph {i} about consent." for i in range(20))
    results = [FakeResult(content=content)]
    await attach_snippets("consent", results, budget=200)
    assert results[0].snippet is not None
    assert len(results[0].snippet) <= 200 + len("… ") + len(" …")


async def test_scoring_failure_still_produces_a_snippet(monkeypatch):
    """`score_passages` returning None is the documented degrade path."""
    from hafiz.core import reranker

    async def _none(query, passages):
        return None

    monkeypatch.setattr(reranker, "score_passages", _none)
    results = [FakeResult(content="Head claim.\n\n" + "tail. " * 200)]
    await attach_snippets("head claim", results, budget=100)
    assert results[0].snippet
    assert "Head claim." in results[0].snippet


async def test_a_raising_scorer_does_not_break_recall(monkeypatch):
    from hafiz.core import reranker

    async def _boom(query, passages):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(reranker, "score_passages", _boom)
    results = [FakeResult(content="Head claim.\n\n" + "tail. " * 200)]
    await attach_snippets("head claim", results, budget=100)
    assert results[0].snippet  # degraded, not raised


async def test_scores_are_distributed_to_the_right_records(monkeypatch):
    """One batched call covers every record; an off-by-one in the cursor
    would excerpt record B using record A's scores."""
    from hafiz.core import reranker

    seen: list[list[str]] = []

    async def _scores(query, passages):
        seen.append(passages)
        return [1.0 if "WANTED" in p else 0.0 for p in passages]

    monkeypatch.setattr(reranker, "score_passages", _scores)
    a = "filler a.\n\n" * 30 + "\n\nWANTED alpha."
    b = "WANTED beta.\n\n" + "filler b.\n\n" * 30
    results = [FakeResult(content=a), FakeResult(content=b)]
    await attach_snippets("wanted", results, budget=120)

    assert len(seen) == 1  # a single batched call
    assert "WANTED alpha" in results[0].snippet
    assert "WANTED beta" in results[1].snippet


# ── rendering marks the excerpt ──────────────────────────────────────


def test_compact_flags_an_excerpt_and_reports_the_full_size():
    row = annotation_compact(FakeResult(content="x" * 900, snippet="… middle …"))
    assert row["content"] == "… middle …"
    assert row["excerpt"] is True
    assert row["full_chars"] == 900


def test_compact_omits_the_flag_when_nothing_was_trimmed():
    row = annotation_compact(FakeResult(content="short"))
    assert row["content"] == "short"
    assert "excerpt" not in row
    assert "full_chars" not in row


def test_md_labels_an_excerpt_in_its_metadata_line():
    out = annotation_md(FakeResult(content="x" * 900, snippet="… middle …"))
    assert "excerpt of 900 chars" in out
    assert "… middle …" in out


def test_md_of_a_whole_record_says_nothing_about_excerpts():
    assert "excerpt" not in annotation_md(FakeResult(content="short"))


@pytest.mark.parametrize("renderer", [annotation_compact, annotation_md])
def test_renderers_tolerate_a_result_without_a_snippet_attribute(renderer):
    """`context` builds its own result objects; a missing attribute must not
    be an AttributeError on the read path."""

    @dataclass
    class Bare:
        content: str = "body"
        kind: str = "fact"
        source: str = "user:anjum"
        id: str = "x"
        valid_from: datetime = datetime(2026, 7, 1, tzinfo=UTC)

    assert renderer(Bare()) is not None
