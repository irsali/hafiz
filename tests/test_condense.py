"""Condensing long and dictated prompts before observation recall.

A 2026-11 audit found prompts over 600 chars came back empty 94% of the time and
dictated ones missed answers the store held. A replay of 120 real long prompts
took the hit rate from 47% to 82% with short prompts unchanged. These tests pin
the shape that made that work, and the merge that keeps it from losing anything.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hafiz.core import annotations as ann_mod
from hafiz.core import telemetry
from hafiz.core.condense import MAX_SEGMENTS, WHOLE_QUERY_CHARS, clean, segments
from hafiz.core.config import reset_settings
from hafiz.core.results import AnnotationResult

# ── Cleaning ────────────────────────────────────────────────────────────


def test_a_short_clean_prompt_is_searched_exactly_as_typed():
    assert segments("commit your changes") == ["commit your changes"]


def test_spoken_fillers_go_and_sentence_breaks_stay():
    got = clean("an issue with uh, Google Tag Manager or something. Where it was denied")
    assert got == "an issue with Google Tag Manager. Where it was denied"


def test_phrases_that_can_carry_meaning_are_kept():
    assert "you know that it is not yet implemented" in clean(
        "from your prompt, you know that it is not yet implemented"
    )


def test_an_element_picker_dump_keeps_the_users_words_and_the_elements_text():
    prompt = (
        "DOM Path: div#wpwrap > div.cn-footer > a[2]\n"
        "Position: top=777px, left=1225px, width=46px, height=16px\n"
        'HTML Element: <a href="https://x.co/support/" target="_blank">Support</a>'
        " this support link is wrong"
    )
    assert clean(prompt) == "Support this support link is wrong"


def test_urls_keep_their_host_and_path_words():
    assert clean("update https://example.com/pricing/ page") == ("update example.com pricing page")


def test_code_fences_and_stack_traces_are_dropped():
    prompt = "it fails here\n```\nconst x = 1;\n```\nTraceback (most recent call last):"
    assert clean(prompt) == "it fails here"


def test_nothing_searchable_left_means_no_segments():
    assert segments("DOM Path: div#a > div.b\nPosition: top=1px, left=2px") == []


# ── Splitting ───────────────────────────────────────────────────────────


LONG = (
    "Before merging, look into the Google Tag Manager consent-mode issue where denied was not "
    "respected. Something. Go. "
    + "We must check whether v2.4 fixed it in hu-banner.min.js before the release. " * 3
    + "Also confirm the Braintree settlement job for ft2479 still runs nightly. "
    + "And keep the billing-portal payment modal as it is for now please."
)


def test_a_long_prompt_is_searched_whole_and_by_its_key_sentences():
    got = segments(LONG)
    assert len(LONG) > WHOLE_QUERY_CHARS
    assert got[0] == clean(LONG)  # the whole is never dropped
    assert 1 < len(got) <= 1 + MAX_SEGMENTS
    sentences = got[1:]
    assert not any(s in ("Something.", "Go.") for s in sentences)  # too short to name a topic
    assert any("Braintree" in s for s in sentences)
    assert len(sentences) == len(set(sentences))  # a pasted repeat costs one slot, not three


def test_key_sentences_keep_the_prompts_order():
    got = segments(LONG)[1:]
    positions = [LONG.find(s[:40]) for s in got]
    assert positions == sorted(positions)


# ── Merging inside search_annotations ───────────────────────────────────


def _row(rid: str, score: float) -> AnnotationResult:
    return AnnotationResult(
        id=rid,
        content=rid,
        kind="fact",
        source=None,
        project=None,
        tags=None,
        confidence=1.0,
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        valid_until=None,
        unit_id=None,
        metadata={},
        score=score,
        rerank_score=score,
    )


@pytest.fixture
def searched(monkeypatch):
    """Stub the per-text search: each text returns a fixed scored set."""
    calls: list[str] = []
    recorded: dict = {}

    async def fake(text, **_):
        calls.append(text)
        if text == clean(LONG):  # the whole prompt matches weakly
            return [_row("a", 0.02), _row("b", 0.30)]
        if "Braintree" in text:
            return [_row("a", 0.90)]
        return [_row("c", 0.10)]

    async def record(**kw):
        recorded.update(kw)

    monkeypatch.setattr(ann_mod, "_ranked_candidates", fake)
    monkeypatch.setattr(telemetry, "record_retrieval", record)
    reset_settings()
    yield calls, recorded
    reset_settings()


async def test_each_row_keeps_its_best_score_across_segments(searched):
    got = await ann_mod.search_annotations(LONG, limit=6, min_score=0.05)
    assert [r.id for r in got] == ["a", "b", "c"]
    assert got[0].ranking_score == 0.90  # beat its 0.02 from the whole prompt


async def test_the_floor_applies_after_merging(searched):
    got = await ann_mod.search_annotations(LONG, limit=6, min_score=0.2)
    assert [r.id for r in got] == ["a", "b"]


async def test_the_log_says_how_many_texts_were_searched(searched):
    calls, recorded = searched
    await ann_mod.search_annotations(LONG, limit=6, min_score=0.05)
    assert recorded["filters"]["segments"] == len(calls) > 1
    assert recorded["query"] == LONG.strip()  # the log keeps what was asked


async def test_a_short_query_is_one_search_and_logs_no_segments(searched):
    calls, recorded = searched
    await ann_mod.search_annotations("braintree settlement", limit=6)
    assert calls == ["braintree settlement"]
    assert recorded["filters"]["segments"] is None


async def test_condensing_can_be_switched_off(searched, monkeypatch):
    calls, _ = searched
    monkeypatch.setenv("HAFIZ_RERANK__CONDENSE_QUERIES", "false")
    reset_settings()
    await ann_mod.search_annotations(LONG, limit=6)
    assert calls == [LONG.strip()]
