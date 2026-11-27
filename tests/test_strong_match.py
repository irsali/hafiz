"""The strong-match safety net under the ``--min-score`` floor, and the log fields.

The cross-encoder can score a plainly worded memory near zero against a prompt
that is about exactly it, and the floor then drops the one row that mattered.
``[recall] strong_match`` keeps the closest vector match anyway when its cosine
is high enough. What these pin:

1. The rescue fires only for the single closest row, only above the threshold,
   only when the reranker actually scored it, and never displaces a row that
   cleared the floor on its own.
2. It is marked (``strong_match``) so a caller can tell why a row under the
   floor is there, and the retrieval log counts it.
3. Rows only the keyword channel found are marked ``found_by="keyword"`` and
   counted in the log too, so each channel's worth can be read back.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from hafiz.core.annotations import _strong_match, search_annotations, store_annotation
from hafiz.core.database import close_engine, get_session_factory
from hafiz.core.results import AnnotationResult

MARK = "strongmatchtest"


def _row(id_: str, score: float, rerank: float | None) -> AnnotationResult:
    return AnnotationResult(
        id=id_,
        content=id_,
        kind="decision",
        source=None,
        project=None,
        tags=None,
        confidence=1.0,
        valid_from=datetime.now(UTC),
        valid_until=None,
        unit_id=None,
        metadata={},
        score=score,
        rerank_score=rerank,
    )


# ---------------------------------------------------------------------------
# The rule — pure
# ---------------------------------------------------------------------------


def _whole(*rows: AnnotationResult) -> dict[str, float]:
    """Each row's cosine to the whole prompt — what search_annotations passes."""
    return {r.id: r.score for r in rows}


def test_the_closest_row_under_the_floor_is_rescued_when_similar_enough():
    close = _row("close", 0.82, 0.001)
    other = _row("other", 0.60, 0.002)
    assert _strong_match([other, close], [], _whole(other, close), 0.70) is close


def test_nothing_is_rescued_below_the_threshold():
    row = _row("a", 0.69, 0.001)
    assert _strong_match([row], [], _whole(row), 0.70) is None


def test_no_rescue_when_the_closest_row_already_cleared_the_floor():
    """The second-closest is never rescued in its place, even above the bar."""
    best = _row("best", 0.80, 0.9)
    second = _row("second", 0.75, 0.001)
    assert _strong_match([best, second], [best], _whole(best, second), 0.70) is None


def test_without_a_rerank_score_the_floor_already_is_the_cosine():
    row = _row("a", 0.9, None)
    assert _strong_match([row], [], _whole(row), 0.70) is None


def test_only_similarity_to_the_whole_prompt_counts():
    """On a condensed prompt a row can score high against one sentence alone; the
    threshold was calibrated on whole prompts, so that score is not used."""
    whole = _row("whole", 0.66, 0.001)
    by_sentence = _row("sentence", 0.81, 0.001)  # matched only one sentence
    assert _strong_match([whole, by_sentence], [], {"whole": 0.66}, 0.70) is None
    assert _strong_match([whole, by_sentence], [], {"whole": 0.74}, 0.70) is whole


def test_one_point_zero_turns_it_off_even_for_an_identical_text():
    row = _row("a", 1.0, 0.001)
    assert _strong_match([row], [], _whole(row), 1.0) is None


def test_the_default_threshold():
    from hafiz.core.config import RecallSettings

    assert RecallSettings().strong_match == 0.70


# ---------------------------------------------------------------------------
# Through the public search — real embeddings, reranker stubbed to say "no"
# ---------------------------------------------------------------------------


async def _wipe() -> None:
    async with get_session_factory()() as s:
        await s.execute(text("DELETE FROM annotations WHERE content LIKE :p"), {"p": f"{MARK}%"})
        await s.commit()


@pytest.fixture
async def clean():
    try:
        async with get_session_factory()() as s:
            await s.execute(text("SELECT 1 FROM annotations LIMIT 1"))
    except Exception:
        pytest.skip("database not reachable")
    await _wipe()
    yield
    await _wipe()
    await close_engine()


@pytest.fixture
def reranker_says_no(monkeypatch):
    """Every candidate scores 0.001 — the failure the safety net exists for."""
    import hafiz.core.reranker as reranker

    async def low(query, items, *, text_of, top_n=None):
        return [(it, 0.001) for it in items]

    monkeypatch.setattr(reranker, "rerank_scored", low)


@pytest.fixture
def threshold(monkeypatch):
    import hafiz.core.config as config

    real = config.load_settings

    def set_to(value: float) -> None:
        def patched():
            settings = real()
            settings.recall.strong_match = value
            return settings

        monkeypatch.setattr(config, "load_settings", patched)

    return set_to


QUESTION = f"{MARK}: which CI system runs our builds and tests"
MEMORY = f"{MARK}: our builds and tests run on the CI system GitHub Actions"


async def _store() -> None:
    await store_annotation(MEMORY, kind="decision", source="user:test")
    await store_annotation(
        f"{MARK}: the cafeteria serves lentil soup on Thursdays", kind="fact", source="user:test"
    )


async def test_a_close_match_the_reranker_rejects_is_kept_and_marked(
    clean, reranker_says_no, threshold
):
    await _store()
    threshold(0.70)
    got = await search_annotations(QUESTION, limit=6, min_score=0.05, telemetry_command=None)
    assert [r.content for r in got] == [MEMORY], (
        "exactly the closest row is kept — if the fixture's cosine fell under 0.70 the "
        "embedding model changed; pick a closer paraphrase"
    )
    assert got[0].strong_match is True
    assert got[0].score >= 0.70


async def test_off_means_the_floor_alone_decides(clean, reranker_says_no, threshold):
    await _store()
    threshold(1.0)
    assert await search_annotations(QUESTION, limit=6, min_score=0.05, telemetry_command=None) == []


async def test_without_a_floor_nothing_is_marked(clean, reranker_says_no, threshold):
    await _store()
    threshold(0.70)
    got = await search_annotations(QUESTION, limit=6, telemetry_command=None)
    assert got and not any(r.strong_match for r in got)


async def test_the_log_counts_the_rescue(clean, reranker_says_no, threshold, monkeypatch):
    import hafiz.core.telemetry as telemetry

    seen: list[dict] = []

    async def capture(**kwargs):
        seen.append(kwargs["filters"])

    monkeypatch.setattr(telemetry, "record_retrieval", capture)
    await _store()
    threshold(0.70)
    await search_annotations(QUESTION, limit=6, min_score=0.05)
    assert seen and seen[-1]["strong_match"] == 1
    assert seen[-1]["keyword_hits"] is None


async def test_a_keyword_only_row_is_marked_and_counted(clean, monkeypatch):
    """Reuses the keyword channel's own proof fixture: a token among decoys."""
    import hafiz.core.telemetry as telemetry

    token = "zqx7-9913"
    for i in range(24):
        await store_annotation(
            f"{MARK} decoy {i}: how the retry configuration backs off after a failed request",
            kind="learning",
            source="user:test",
        )
    await store_annotation(
        f"{MARK} target: the {token} gateway is pinned to a single region",
        kind="fact",
        source="user:test",
    )
    seen: list[dict] = []

    async def capture(**kwargs):
        seen.append(kwargs["filters"])

    monkeypatch.setattr(telemetry, "record_retrieval", capture)
    got = await search_annotations(f"what is the retry configuration for {token}", limit=3)
    target = [r for r in got if token in r.content]
    assert target and target[0].found_by == "keyword"
    assert all(r.found_by == "vector" for r in got if r is not target[0])
    assert seen[-1]["keyword_hits"] == 1
