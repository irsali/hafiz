"""Tests for how retrieval results reach a caller.

Three coupled guarantees, all measured against a real deployment before being
written down:

1. **A blank query is an error, not a result set.** An empty query embeds to a
   near-zero vector against which empty documents score a perfect 1.0, so a
   caller whose variable interpolation broke gets confidently-ranked garbage.
2. **The relevance floor filters the score results are ranked by.** Under
   reranking the vector ``score`` is non-monotonic down the result list, so a
   floor applied to it would drop rank 4 and keep rank 7.
3. **``--json`` keeps its shape.** Live agent hooks parse it; new formats are
   opt-in and new fields are additive.

DB-free: the core search/annotation functions are stubbed. The point under test
is the delivery contract, not the vector math.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hafiz.core.annotations import AnnotationResult
from hafiz.core.formats import (
    OutputFormat,
    annotation_compact,
    chunk_compact,
    error_payload,
    resolve_format,
)
from hafiz.core.reranker import normalize_score
from hafiz.core.search import EmptyQueryError, SearchResult, require_query


def _ann(content: str, score: float, rerank: float | None = None, **kw) -> AnnotationResult:
    return AnnotationResult(
        id=kw.get("id", "11111111-1111-1111-1111-111111111111"),
        content=content,
        kind=kw.get("kind", "decision"),
        source=kw.get("source", "agent:claude-code"),
        project=kw.get("project"),
        tags=None,
        confidence=1.0,
        valid_from=kw.get("valid_from", datetime.now(UTC) - timedelta(days=3)),
        valid_until=None,
        unit_id=None,
        metadata={},
        score=score,
        rerank_score=rerank,
    )


# ── 1. Empty-query guard (P0-3) ─────────────────────────────────────────


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t  \n"])
def test_require_query_rejects_blank(blank):
    with pytest.raises(EmptyQueryError):
        require_query(blank)


def test_require_query_rejects_none():
    with pytest.raises(EmptyQueryError):
        require_query(None)


def test_require_query_strips_and_returns():
    assert require_query("  consent storage  ") == "consent storage"


def test_empty_query_error_is_a_valueerror():
    """CLI handlers catch ValueError to map onto exit code 2."""
    assert issubclass(EmptyQueryError, ValueError)


def test_empty_query_error_names_the_likely_cause():
    """The message has to point at broken interpolation — that's the real case."""
    assert "interpolation" in str(EmptyQueryError())


def test_require_query_hint_adapts_to_writes():
    """Same guard covers the write side, so the wording has to fit both."""
    with pytest.raises(EmptyQueryError) as exc:
        require_query("", what="annotation content", hint="nothing to store")
    assert "empty annotation content — nothing to store" in str(exc.value)


async def test_store_annotation_refuses_blank_content():
    """A blank annotation embeds to a near-zero vector and then scores
    near-perfectly against *every* later query — a permanent noise magnet.

    Guarded before the embed call, so this needs no DB.
    """
    from hafiz.core.annotations import store_annotation

    for blank in ("", "   ", "\n\t"):
        with pytest.raises(EmptyQueryError):
            await store_annotation(blank, kind="note")


# ── 2. Score semantics + relevance floor (P0-1) ─────────────────────────


def test_normalize_score_maps_logits_onto_0_1():
    """Measured cross-encoder logits: +3.32 relevant, -11.35 irrelevant."""
    assert normalize_score(3.32) == pytest.approx(0.965, abs=0.005)
    assert normalize_score(-11.35) == pytest.approx(0.0, abs=1e-4)
    assert normalize_score(0.0) == pytest.approx(0.5)


def test_normalize_score_is_monotonic():
    """Order must survive normalization or reranking would be corrupted."""
    logits = [-11.3, -6.0, -1.0, 0.0, 1.5, 3.32, 9.0]
    scores = [normalize_score(x) for x in logits]
    assert scores == sorted(scores)


def test_normalize_score_does_not_overflow_on_extremes():
    """math.exp overflows around -710; clamp rather than raise."""
    assert normalize_score(-1e9) == 0.0
    assert normalize_score(1e9) == 1.0


def test_ranking_score_prefers_rerank_when_present():
    assert _ann("x", 0.61, 0.88).ranking_score == 0.88


def test_ranking_score_falls_back_to_vector_when_rerank_absent():
    assert _ann("x", 0.61, None).ranking_score == 0.61


def test_vector_score_is_non_monotonic_under_rerank():
    """The regression this whole design exists to prevent.

    Real series from the deployed index for the query
    "user preferences workflow conventions rules": reranking reorders rows so
    the vector scores no longer descend. A floor on ``score`` would therefore
    keep a lower-ranked row and drop a higher-ranked one.
    """
    ranked = [
        _ann("r1", 0.645, 0.881),
        _ann("r2", 0.573, 0.473),
        _ann("r3", 0.580, 0.067),
        _ann("r4", 0.510, 0.013),
        _ann("r5", 0.551, 0.010),
        _ann("r6", 0.559, 0.008),
        _ann("r7", 0.624, 0.005),
    ]
    vector = [r.score for r in ranked]
    assert vector != sorted(vector, reverse=True), "expected non-monotonic vector scores"

    # A floor on the vector score is incoherent: it keeps rank 7 (0.624) while
    # dropping ranks 4-6 that outrank it.
    kept_wrong = [r.content for r in ranked if r.score >= 0.60]
    assert kept_wrong == ["r1", "r7"]

    # A floor on the ranking score is a clean prefix of the ranking.
    kept_right = [r.content for r in ranked if r.ranking_score >= 0.05]
    assert kept_right == ["r1", "r2", "r3"]


def test_ranking_score_floor_yields_a_prefix_of_the_ranking():
    """Any floor on ranking_score must cut a prefix, never a hole."""
    ranked = [_ann(f"r{i}", 0.5, s) for i, s in enumerate([0.9, 0.7, 0.4, 0.1, 0.01])]
    for floor in (0.0, 0.05, 0.2, 0.5, 0.8, 0.95):
        kept = [i for i, r in enumerate(ranked) if r.ranking_score >= floor]
        assert kept == list(range(len(kept))), f"floor {floor} cut a hole: {kept}"


# ── 3. Output formats (P0-2) ────────────────────────────────────────────


def test_json_flag_still_selects_json():
    assert resolve_format(None, json_flag=True) is OutputFormat.JSON


def test_no_flags_defaults_to_rich():
    assert resolve_format(None, json_flag=False) is OutputFormat.RICH


def test_explicit_format_wins_over_json_flag():
    """Lets a caller migrate with --json --format compact in one step."""
    assert resolve_format(OutputFormat.COMPACT, json_flag=True) is OutputFormat.COMPACT


def test_only_rich_is_non_machine():
    assert not OutputFormat.RICH.is_machine
    for fmt in (OutputFormat.JSON, OutputFormat.COMPACT, OutputFormat.MD):
        assert fmt.is_machine


def test_error_payload_uses_the_project_standard_shape():
    assert error_payload("boom") == {"ok": False, "error": "boom"}


def test_md_emits_nothing_when_a_floor_filters_everything_out(capsys, monkeypatch):
    """`md` is for prompt injection, so empty must mean *empty*.

    With a relevance floor, "no rows" is the normal answer to an off-topic
    prompt — it fires on most prompts a per-task hook sees. A placeholder line
    there is text the agent pays for on every turn and a string every integrator
    has to filter, so the format stays silent instead.
    """
    from hafiz.commands import observe

    async def _none(*a, **kw):
        return []

    monkeypatch.setattr("hafiz.core.annotations.search_annotations", _none)
    monkeypatch.setattr(observe, "close_engine", _none)
    observe.run_recall("nothing on topic here", output_format=OutputFormat.MD)
    assert capsys.readouterr().out == ""


def test_annotation_compact_drops_metadata_keeps_meaning():
    row = annotation_compact(_ann("we chose Postgres", 0.61, 0.88))
    assert row == {
        "content": "we chose Postgres",
        "kind": "decision",
        "source": "agent:claude-code",
        "age": "3d ago",
    }


def test_annotation_compact_omits_id_by_default():
    assert "id" not in annotation_compact(_ann("x", 0.5))


def test_annotation_compact_includes_id_on_request():
    """Without the id an agent can read a decision but never supersede it."""
    row = annotation_compact(_ann("x", 0.5, id="abc"), with_ids=True)
    assert row["id"] == "abc"


def test_compact_is_a_large_fraction_smaller_than_json():
    """Sanity-check the token claim on a payload shaped like the real one."""
    import json

    anns = [_ann("c" * 900, 0.6, 0.5, id=f"id-{i}") for i in range(50)]
    full = json.dumps(
        [
            {
                "id": a.id,
                "content": a.content,
                "kind": a.kind,
                "source": a.source,
                "project": a.project,
                "tags": a.tags,
                "confidence": a.confidence,
                "valid_from": a.valid_from.isoformat(),
                "valid_until": None,
                "unit_id": None,
                "age_days": 3,
                "stale": False,
                "inactive": False,
                "score": a.score,
                "rerank_score": a.rerank_score,
            }
            for a in anns
        ]
    )
    compact = json.dumps([annotation_compact(a) for a in anns])
    assert len(compact) < len(full)
    # Content dominates once ids/timestamps/scores are gone.
    assert len(compact) < len(full) * 0.95


def test_chunk_compact_keeps_location_drops_scores():
    chunk = SearchResult(
        id="e1",
        unit_id="u1",
        unit_name="build_context",
        kind="code.function",
        content="def build_context(): ...",
        source_file="/repo/hafiz/core/context.py",
        line_start=10,
        line_end=20,
        language="python",
        project="hafiz",
        part_index=0,
        score=0.71,
    )
    row = chunk_compact(chunk)
    assert row == {
        "content": "def build_context(): ...",
        "kind": "code.function",
        "unit_name": "build_context",
        "source_file": "/repo/hafiz/core/context.py",
    }
    assert "score" not in row
    assert chunk_compact(chunk, with_ids=True)["unit_id"] == "u1"
