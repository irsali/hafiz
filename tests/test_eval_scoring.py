"""Scoring logic of the recall eval (eval/scoring.py) — no database needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval import scoring

DATA = Path(__file__).resolve().parent.parent / "eval" / "data"

ID_MAP = {"a": "uuid-a", "b": "uuid-b", "c": "uuid-c", "old": "uuid-old"}


def _q(**kw):
    base = {"id": "q1", "measure": "exact", "query": "what port is it", "expect": ["a"]}
    return {**base, **kw}


def test_shipped_dataset_is_valid():
    memories = scoring.load_jsonl(DATA / "memories.jsonl")
    questions = scoring.load_jsonl(DATA / "questions.jsonl")
    assert scoring.validate_dataset(memories, questions) == []
    assert 150 <= len(memories) <= 250
    # The long set is what lets the eval show channels the short set cannot:
    # long multi-topic memories, with needles past the reranker's 400 chars.
    by_id = {m["id"]: m for m in memories}
    assert sum(len(m["content"]) > 600 for m in memories) >= 50
    deep = [
        q
        for q in questions
        if q["measure"] == "exact"
        and q.get("needle")
        and any(by_id[e]["content"].find(q["needle"]) > 400 for e in q["expect"])
    ]
    assert len(deep) >= 25
    assert sum(1 for m in memories if m.get("supersedes")) >= 15
    by_measure = {m: sum(1 for q in questions if q["measure"] == m) for m in scoring.MEASURES}
    assert by_measure["offtopic"] >= 15
    assert by_measure["exact"] + by_measure["decision"] >= 90


def test_validate_catches_forward_supersede_and_bad_refs():
    mems = [
        {"id": "b", "kind": "decision", "content": "new", "supersedes": "a"},
        {"id": "a", "kind": "decision", "content": "old"},
    ]
    qs = [
        {"id": "q", "measure": "decision", "query": "what is the decision", "expect": ["zzz"]},
        {"id": "q2", "measure": "weird", "query": "short", "expect": []},
        {"id": "q2", "measure": "offtopic", "query": "a long enough query", "expect": []},
    ]
    problems = scoring.validate_dataset(mems, qs)
    text = "\n".join(problems)
    assert "supersedes 'a'" in text
    assert "unknown memory id 'zzz'" in text
    assert "duplicated" in text
    assert "unknown measure" in text
    assert "shorter than" in text


def test_validate_needle_must_appear_in_expected_memory():
    mems = [{"id": "a", "kind": "fact", "content": "port 8123"}]
    ok = {"id": "q", "measure": "exact", "query": "which port is used", "expect": ["a"]}
    assert scoring.validate_dataset(mems, [{**ok, "needle": "8123"}]) == []
    assert scoring.validate_dataset(mems, [{**ok, "needle": "9999"}])


def test_estimate_tokens_is_chars_over_four_rounded_up():
    assert scoring.estimate_tokens("") == 0
    assert scoring.estimate_tokens("abcd") == 1
    assert scoring.estimate_tokens("abcde") == 2


def test_rank_and_hits():
    r = scoring.score_question(_q(), ["uuid-b", "uuid-a"], "text", ID_MAP)
    assert (r["rank"], r["hit1"], r["hitk"]) == (2, False, True)
    r = scoring.score_question(_q(), ["uuid-a"], "text", ID_MAP)
    assert (r["rank"], r["hit1"]) == (1, True)
    r = scoring.score_question(_q(), ["uuid-b"], "text", ID_MAP)
    assert (r["rank"], r["hitk"]) == (None, False)


def test_hits_beyond_k_do_not_count():
    returned = [f"x{i}" for i in range(scoring.K)] + ["uuid-a"]
    r = scoring.score_question(_q(), returned, "text", ID_MAP)
    assert r["rank"] is None and not r["hitk"]


def test_any_of_several_expected_ids_counts():
    q = _q(expect=["a", "b"])
    assert scoring.score_question(q, ["uuid-b"], "", ID_MAP)["hit1"]


def test_stale_leak_is_detected_inside_top_k_only():
    q = _q(measure="decision", expect=["a"], forbid=["old"])
    r = scoring.score_question(q, ["uuid-a", "uuid-old"], "t", ID_MAP)
    assert r["hit1"] and r["leaked"] == ["uuid-old"] and r["had_forbid"]
    far = [f"x{i}" for i in range(scoring.K)] + ["uuid-old"]
    assert scoring.score_question(q, far, "t", ID_MAP)["leaked"] == []


def test_needle_checks_the_injected_text_not_the_ids():
    q = _q(needle="PORT_X")
    assert scoring.score_question(q, ["uuid-a"], "see PORT_X here", ID_MAP)["needle_surfaced"]
    assert scoring.score_question(q, ["uuid-a"], "nothing", ID_MAP)["needle_surfaced"] is False
    assert scoring.score_question(_q(), ["uuid-a"], "x", ID_MAP)["needle_surfaced"] is None


def test_silent_means_no_results_and_no_text():
    q = _q(measure="offtopic", expect=[])
    assert scoring.score_question(q, [], "", ID_MAP)["silent"]
    assert not scoring.score_question(q, ["uuid-a"], "## Recall\n- x", ID_MAP)["silent"]


def test_percentile_nearest_rank():
    assert scoring.percentile([], 95) == 0
    assert scoring.percentile([5], 95) == 5
    assert scoring.percentile(list(range(1, 101)), 95) == 95
    assert scoring.percentile([10, 20, 30], 50) == 20


def _rows():
    mk = scoring.score_question
    return [
        mk(_q(id="e1", needle="n"), ["uuid-a"], "n" * 8, ID_MAP),  # hit@1, 2 tokens
        mk(_q(id="e2"), ["uuid-b", "uuid-a"], "y" * 40, ID_MAP),  # hit@6 only, 10 tokens
        mk(
            _q(id="d1", measure="decision", forbid=["old"]),
            ["uuid-a", "uuid-old"],
            "z" * 20,
            ID_MAP,
        ),  # leak, 5 tokens
        mk(_q(id="d2", measure="decision"), ["uuid-c"], "w" * 4, ID_MAP),  # miss, 1 token
        mk(_q(id="o1", measure="offtopic", expect=[]), [], "", ID_MAP),  # silent
        mk(_q(id="o2", measure="offtopic", expect=[]), ["uuid-a"], "q" * 12, ID_MAP),  # noisy
    ]


def test_aggregate():
    agg = scoring.aggregate(_rows())
    assert agg["exact"]["n"] == 2
    assert agg["exact"]["hit_at_1"] == 0.5
    assert agg["exact"]["hit_at_6"] == 1.0
    assert agg["exact"]["mrr"] == pytest.approx(0.75)
    assert agg["exact"]["needle_surfaced"] == 1.0  # only e1 carries a needle, and it shows
    assert agg["decision"]["n"] == 2
    assert agg["decision"]["hit_at_1"] == 0.5
    assert agg["decision"]["n_superseded_chain"] == 1
    assert agg["decision"]["stale_leak_rate"] == 1.0
    t = agg["tokens"]
    assert t["on_topic_n"] == 4
    assert t["on_topic_mean"] == pytest.approx((2 + 10 + 5 + 1) / 4)
    assert t["on_topic_max"] == 10
    assert t["off_topic_n"] == 2
    assert t["off_topic_silent_rate"] == 0.5
    assert t["off_topic_mean"] == 1.5


def test_aggregate_handles_empty_measures():
    agg = scoring.aggregate([])
    assert agg["exact"]["hit_at_1"] is None
    assert agg["decision"]["stale_leak_rate"] is None
    assert agg["tokens"]["on_topic_p95"] == 0


def test_rendering_is_stable_and_labelled():
    agg = scoring.aggregate(_rows())
    result = {"meta": {"label": "x", "commit": "abc1234", "dirty": False}, "aggregate": agg}
    md = scoring.render_summary(result)
    assert "commit abc1234" in md and "Stale-belief leak rate" in md
    table = scoring.render_comparison(
        [result, {**result, "meta": {**result["meta"], "label": "y"}}]
    )
    header = table.splitlines()[0]
    assert "x (`abc1234`)" in header and "y (`abc1234`)" in header
    assert all(line.count("|") == header.count("|") for line in table.strip().splitlines())
    assert "No runs" in scoring.render_comparison([])


def test_failures_lists_misses_leaks_and_noisy_offtopic():
    rows = _rows()
    questions = {
        "e1": _q(id="e1", needle="n"),
        "e2": _q(id="e2"),
        "d1": _q(id="d1", measure="decision", forbid=["old"]),
        "d2": _q(id="d2", measure="decision"),
        "o1": _q(id="o1", measure="offtopic", expect=[]),
        "o2": _q(id="o2", measure="offtopic", expect=[], query="thanks that works fine"),
    }
    reverse = {v: k for k, v in ID_MAP.items()}
    text = "\n".join(scoring.failures(rows, questions, reverse))
    assert "[e2]" in text and "rank 2" in text
    assert "[d1]" in text and "STALE LEAK old" in text
    assert "[d2]" in text and f"not in top {scoring.K}" in text
    assert "[o2]" in text and "[o1]" not in text and "[e1]" not in text
