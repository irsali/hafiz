"""Reconcile's clustering and its resolution proposals.

DB-free: everything here is pure logic over embeddings that are already in
hand. The DB-backed half — scan coverage and the ``doctor`` count — lives in
``test_cli.py``.

The thing under test is a *safety* property as much as a correctness one.
Reconcile proposes destructive commands, and the near-duplicate pairs in a real
store are not all restatements: a 91%-similar pair turned out to be one row
holding a test account's email and another holding its password. Retiring the
wrong one destroys a fact nothing else carries.

The rule that guards it has been rewritten three times, because each earlier
version read plausibly and each had a case where it proposed exactly that:

1. *Never keep a row materially shorter than one it would retire.* Length is a
   proxy for containment, and it inverts — on a same-day pair it proposed
   retiring the only row carrying "supersedes the earlier same-task decision",
   because the row lacking that sentence was a few characters longer.
2. *Retire when no run of four-plus words is unique.* A four-word floor cannot
   see a one-word difference, and one word is a commit sha, a port, or a
   password.
3. *Retire when the keeper contains every word* — right in principle, but the
   tokenizer decided what "a word" was, and it dropped everything outside
   ``[a-z0-9]``. So any non-Latin row tokenized to nothing and then compared as
   *fully contained*, and ``latency >= 500`` compared equal to ``latency <=
   500``. A safety check is only as strong as its comparison unit.

What holds now: ``retire`` requires that every word of the retired row appear in
the keeper, in order, with a tokenizer that keeps scripts and comparison
operators — and when nothing can be aligned at all, containment is reported as
*unproven* rather than total. Anything else is a human's call, split into
``review`` (stray words — a glance) and ``merge`` (whole claims).

Two limits are pinned deliberately rather than papered over:
``test_a_high_cosine_pair_is_never_retired_on_similarity_alone`` fixes that the
tiers grade evidence and not meaning, and the ``_text_delta`` tests fix that
containment is a word-level *subsequence* — which cannot rule out a reversal, so
even ``retire`` is the cheapest tier to check rather than one that skips it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from hafiz.commands.reconcile import _commands_for, _preview, _shell_quote
from hafiz.core.annotations import (
    _build_cluster,
    _similarity_matrix,
    _single_linkage,
    _text_delta,
    _words,
)

BASE = datetime(2026, 7, 1, tzinfo=UTC)


def _ann(content: str, *, days: int = 0, source: str | None = "agent:claude-code"):
    """A stand-in carrying only the fields the clustering reads."""
    return SimpleNamespace(
        id=uuid.uuid4(),
        content=content,
        kind="decision",
        source=source,
        valid_from=BASE + timedelta(days=days),
    )


def _sim(n: int, value: float = 0.95):
    """A similarity matrix where every distinct pair is ``value``."""
    mat = np.full((n, n), value, dtype=np.float32)
    np.fill_diagonal(mat, 1.0)
    return mat


# ── similarity matrix ────────────────────────────────────────────────


def test_identical_vectors_score_one_and_orthogonal_score_zero():
    sim = _similarity_matrix([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert sim[0, 1] == pytest.approx(1.0)
    assert sim[0, 2] == pytest.approx(0.0)


def test_magnitude_does_not_affect_similarity():
    """Cosine, not dot product — a longer vector is not a closer one."""
    sim = _similarity_matrix([[3.0, 0.0], [0.5, 0.0]])
    assert sim[0, 1] == pytest.approx(1.0)


def test_zero_vector_is_similar_to_nothing_rather_than_nan():
    """A degenerate embedding must not propagate NaN, which compares false
    against the threshold and would silently drop the whole group."""
    sim = _similarity_matrix([[0.0, 0.0], [1.0, 0.0]])
    assert not np.isnan(sim).any()
    assert sim[0, 1] == pytest.approx(0.0)


# ── single-linkage clustering ────────────────────────────────────────


def test_linkage_is_transitive():
    """A~B and B~C put A and C in one cluster even though they don't match."""
    sim = np.array(
        [
            [1.00, 0.95, 0.10],
            [0.95, 1.00, 0.95],
            [0.10, 0.95, 1.00],
        ],
        dtype=np.float32,
    )
    groups = [sorted(g) for g in _single_linkage(sim, 0.88)]
    assert sorted(groups) == [[0, 1, 2]]


def test_rows_below_threshold_stay_singletons():
    sim = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    assert sorted(sorted(g) for g in _single_linkage(sim, 0.88)) == [[0], [1]]


# ── the tokenizer and the delta (where the safety property actually lives) ──
#
# Every safety assertion used to go through _build_cluster on well-formed
# English pairs, and two ways to fake total containment survived that: any
# non-Latin script tokenized to nothing, and comparison operators were stripped.
# Both produced a green "nothing only here" and a paste-ready forget. These test
# the primitive directly.


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("chinese", "日志保留期为三十天"),
        ("russian", "срок хранения логов тридцать дней"),
        ("greek", "διατήρηση αρχείων τριάντα ημέρες"),
        ("arabic", "الاحتفاظ بالسجلات ثلاثين يوما"),
    ],
)
def test_non_latin_scripts_survive_tokenizing(label, text):
    """``[^a-z0-9]`` erased these wholesale, so the row compared as empty and
    then as fully contained — the safety check was simply off for those
    locales."""
    assert _words(text), f"{label} tokenized to nothing"


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("日志保留期为三十天", "缓存层已切换到新的集群"),
        ("срок хранения логов тридцать дней", "кеш переключён на новый кластер"),
        ("✅", "the local run is the gate"),
        ("!!! --- ???", "the local run is the gate"),
    ],
)
def test_unrelated_content_is_never_reported_as_contained(a, b):
    """The blocker: two unrelated facts must not report zero unique words. The
    last two cases tokenize to nothing at all, where the honest answer is "no
    alignment, so containment is unproven" — not "contained"."""
    _, _, unique_words = _text_delta(a, b)
    assert unique_words > 0


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("abort the job if latency >= 500", "abort the job if latency <= 500"),
        ("skip when status != 202", "skip when status == 202"),
        ("blocked when score > 0.88", "blocked when score < 0.88"),
    ],
)
def test_reversed_operators_are_not_the_same_words(a, b):
    """Stripping symbols made a reversed bound identical to its opposite. That
    is the stage-vs-prod hazard in a store full of thresholds and ports."""
    _, _, unique_words = _text_delta(a, b)
    assert unique_words > 0


def test_identical_text_is_contained_even_when_unspaced():
    """The conservative fixes must not make containment unreachable: identical
    content still has to grade as contained, including in a script the
    tokenizer treats as one run."""
    for text in ("日志保留期为三十天", "the local run is the gate"):
        overlap, fragments, unique_words = _text_delta(text, text)
        assert (overlap, fragments, unique_words) == (1.0, [], 0)


def test_a_subset_is_contained_and_reports_no_unique_words():
    overlap, fragments, unique_words = _text_delta(
        "the local run is the gate",
        "the local run is the gate and the pipeline is not",
    )
    assert unique_words == 0
    assert fragments == []
    assert 0.0 < overlap < 1.0  # 2M/T: shorter side drags the ratio down


def test_fragments_are_every_unmatched_run_longest_first():
    """No four-word floor here — a caller shown nothing cannot check anything,
    and the one-word case is the dangerous one. The floor only grades the tier."""
    _, fragments, unique_words = _text_delta(
        "alpha beta gamma delta epsilon zulu and one more here", "zulu"
    )
    assert unique_words == 9
    assert fragments  # includes short runs
    assert [len(f.split()) for f in fragments] == sorted(
        (len(f.split()) for f in fragments), reverse=True
    )


def test_operator_sentinels_do_not_leak_into_reported_fragments():
    """The sentinels are an internal encoding; an operator reading "only here"
    must not see control characters."""
    _, fragments, _ = _text_delta("fail when latency >= 500 always", "fail when")
    assert fragments
    assert not any("\x00" in f for f in fragments)


# ── the proposal ─────────────────────────────────────────────────────


def test_newest_row_is_the_primary_when_it_loses_nothing():
    """A restatement that adds a clause: the newer row carries the older whole,
    so the older is redundant and the newer survives untouched."""
    anns = [
        _ann("the local run is the gate", days=0),
        _ann("the local run is the gate and the pipeline is not", days=5),
    ]
    cluster = _build_cluster("decision", "p", anns, [0, 1], _sim(2))
    assert cluster.action == "retire"
    assert cluster.primary.id == str(anns[1].id)
    assert [m.id for m in cluster.others] == [str(anns[0].id)]


def test_the_credential_row_wins_even_though_it_is_the_older_one():
    """The pair this file was written around — one row holding a test account,
    another holding its password. The old length-and-recency rule could pick
    either; containment always picks the row that carries both, so the fact
    survives and the redundant row is the one retired."""
    anns = [
        _ann("stage login is claude at noble-wave and the password is Dev1234", days=0),
        _ann("stage login is claude at noble-wave", days=5),
    ]
    cluster = _build_cluster("decision", "p", anns, [0, 1], _sim(2))
    assert cluster.action == "retire"
    assert cluster.primary.id == str(anns[0].id), "the row with the password must survive"
    assert "Dev1234" in cluster.primary.content
    assert [m.id for m in cluster.others] == [str(anns[1].id)]


def test_the_superset_wins_even_when_it_is_neither_newest_nor_longest():
    """Primary selection under containment, on the pair this file was written
    around: one row carries a test account's password, the other does not. The
    row that carries it must survive whichever way the dates fall — length and
    recency each point the wrong way in one of these two arrangements.

    The "one unmatched word refuses a retire" property is pinned separately, in
    the `_text_delta` tests above."""
    anns = [
        _ann("the stage admin account for the portal is claude at noble-wave", days=0),
        _ann("the stage admin account for the portal is claude at noble-wave Dev1234", days=5),
    ]
    # Newest is the superset here, so retiring the older is safe.
    assert _build_cluster("fact", None, anns, [0, 1], _sim(2)).action == "retire"
    # Reverse the dates and the superset is now the OLDER row. Length and
    # recency both point the wrong way; containment does not.
    flipped = [
        _ann("the stage admin account for the portal is claude at noble-wave Dev1234", days=0),
        _ann("the stage admin account for the portal is claude at noble-wave", days=5),
    ]
    cluster = _build_cluster("fact", None, flipped, [0, 1], _sim(2))
    assert cluster.action == "retire"
    assert cluster.primary.id == str(flipped[0].id), "the superset must win, not the newest"


def test_stray_word_differences_are_review_not_merge():
    """Contractions and spellings are the common case and are cheap to check, so
    they get their own tier. Collapsing them into `merge` made 70 clusters look
    like equal work."""
    anns = [
        _ann("the local run is the gate and cannot be deferred to the pipeline", days=0),
        _ann("the local run is the gate and can not be deferred to the pipeline", days=5),
    ]
    cluster = _build_cluster("learning", None, anns, [0, 1], _sim(2))
    assert cluster.action == "review"
    # Still reports what differs — a tier with nothing to look at is useless.
    assert any(m.unique_fragments for m in cluster.others)


def test_a_high_cosine_pair_is_never_retired_on_similarity_alone():
    """The stage-vs-prod pipeline pair that nearly got bulk-retired: 0.99 cosine,
    two different facts. The tier must not be `retire`.

    Deliberately not asserting *which* non-retire tier. The words that differ
    here are short ("stage" against "prod"), so this grades `review` even though
    the meaning difference is total — evidence and importance are not the same
    axis, and the tiers only claim the first. `review` still means "a human
    looks", which is the property that matters."""
    anns = [
        _ann("v2 stage pipeline targetFolder moves from banner to banner slash v2", days=0),
        _ann("v2 prod pipeline targetFolder moves from empty to v2 not banner slash v2", days=5),
    ]
    cluster = _build_cluster("decision", None, anns, [0, 1], _sim(2, 0.99))
    assert cluster.action != "retire", "0.99 cosine must not imply the same statement"
    assert any(m.unique_words for m in cluster.others)


def test_primary_sorts_first_and_carries_length_and_date():
    """The operator overrules the suggestion from the rendered row, so the
    fields they'd judge on have to be on it."""
    anns = [_ann("a" * 50, days=0), _ann("b" * 60, days=5)]
    cluster = _build_cluster("decision", None, anns, [0, 1], _sim(2))
    assert cluster.members[0].primary is True
    assert [m.primary for m in cluster.members[1:]] == [False]
    assert cluster.primary.valid_from == BASE + timedelta(days=5)
    assert len(cluster.primary.content) == 60


def test_member_score_is_best_similarity_to_any_sibling():
    sim = np.array(
        [
            [1.00, 0.90, 0.99],
            [0.90, 1.00, 0.91],
            [0.99, 0.91, 1.00],
        ],
        dtype=np.float32,
    )
    anns = [_ann("aaa", days=i) for i in range(3)]
    cluster = _build_cluster("decision", None, anns, [0, 1, 2], sim)
    by_id = {m.id: m.score for m in cluster.members}
    assert by_id[str(anns[0].id)] == pytest.approx(0.99, abs=1e-4)
    assert by_id[str(anns[1].id)] == pytest.approx(0.91, abs=1e-4)


# ── the emitted commands ─────────────────────────────────────────────


def test_retire_emits_one_forget_per_redundant_row_and_spares_the_primary():
    anns = [
        _ann("the purge guard reconciles JobLog", days=0),
        _ann("the purge guard reconciles JobLog rows", days=5),
        _ann("the purge guard reconciles JobLog rows and alerts on drift", days=9),
    ]
    cluster = _build_cluster("decision", "p", anns, [0, 1, 2], _sim(3))
    assert cluster.action == "retire"
    cmds = _commands_for(cluster)
    assert len(cmds) == 2
    assert all(c.startswith("hafiz forget ") and c.endswith(" --annotation") for c in cmds)
    assert cluster.primary.id not in " ".join(cmds)


def test_merge_emits_exactly_one_observe_then_retires_the_rest():
    """``--supersedes`` takes a single id, so a merge is one new row superseding
    the primary — not one new row per member, which is what a naive
    command-per-member rendering produced and would have tripled the cluster."""
    anns = [
        _ann("the purge guard reconciles JobLog rows and alerts on drift", days=0),
        _ann("the purge guard also writes a staging table for each date", days=1),
        _ann("the purge guard reconciles JobLog rows", days=5),
    ]
    cluster = _build_cluster("learning", "proj", anns, [0, 1, 2], _sim(3))
    cmds = _commands_for(cluster)
    assert cluster.action == "merge"
    assert sum(c.startswith("hafiz observe ") for c in cmds) == 1
    assert cmds[0].startswith("hafiz observe '<merged text>' --type learning")
    assert f"--supersedes {cluster.primary.id}" in cmds[0]
    assert sorted(cmds[1:]) == sorted(f"hafiz forget {m.id} --annotation" for m in cluster.others)


def test_merge_command_carries_project_and_source_so_the_new_row_keeps_them():
    anns = [
        _ann("banner blocks scripts before consent on every region", days=0, source="user:anjum"),
        _ann("banner releases scripts after consent is recorded", days=5),
    ]
    cmd = _commands_for(_build_cluster("fact", "cookie-notice", anns, [0, 1], _sim(2)))[0]
    assert "--source user:anjum" in cmd
    # Always quoted: real project names have spaces in them ("Admin Portal").
    assert "--project 'cookie-notice'" in cmd


def test_project_name_with_a_space_survives_as_one_argument():
    anns = [
        _ann("portal collapses the geolocation section when it is off", days=0),
        _ann("portal gates the geolocation method radio behind the toggle", days=5),
    ]
    cmd = _commands_for(_build_cluster("fact", "Admin Portal", anns, [0, 1], _sim(2)))[0]
    assert "--project 'Admin Portal'" in cmd


def test_merge_command_omits_project_when_the_rows_have_none():
    anns = [
        _ann("consent mode default fires before the category map arrives", days=0),
        _ann("consent mode update never reaches the dataLayer for GPC", days=5),
    ]
    assert "--project" not in _commands_for(_build_cluster("fact", None, anns, [0, 1], _sim(2)))[0]


def test_project_with_an_apostrophe_stays_one_shell_argument():
    assert _shell_quote("it's") == "'it'\\''s'"


# ── rendering ────────────────────────────────────────────────────────


def test_preview_collapses_newlines_into_one_line():
    """Annotations are multi-line prose; a raw preview shreds the panel."""
    assert _preview("first line\n\n  second   line") == "first line second line"


def test_preview_is_truncated_with_an_ellipsis():
    out = _preview("x" * 500)
    assert out.endswith("…")
    assert len(out) <= 101


#: One pair per tier, so the rich path is rendered for each. Kept as data
#: because the tier is derived from the text — there is no way to ask for a
#: tier directly, which is the point.
_TIER_FIXTURES = {
    "retire": (
        "the local run is the gate",
        "the local run is the gate and the pipeline is not",
    ),
    "review": (
        "the local run is the gate and cannot be deferred to the pipeline",
        "the local run is the gate and can not be deferred to the pipeline",
    ),
    "merge": (
        "the purge guard reconciles JobLog rows and alerts on drift",
        "the purge guard writes a staging table for each date it processes",
    ),
}


@pytest.mark.parametrize("action", ["retire", "review", "merge"])
def test_human_output_renders_every_action(action, monkeypatch, capsys):
    """The rich path is not exercised by the ``--json`` tests, and a rename
    once left an undefined name in it that every JSON test still passed over."""
    from hafiz.commands import reconcile as mod

    older, newer = _TIER_FIXTURES[action]
    anns = [_ann(older, days=0), _ann(newer, days=5)]
    cluster = _build_cluster("decision", "proj", anns, [0, 1], _sim(2))
    assert cluster.action == action

    report = SimpleNamespace(
        clusters=[cluster], scanned=2, total_live=9, truncated=True, threshold=0.88
    )

    def _fake_run(coro):
        coro.close()  # bypassing the DB call — don't leave it unawaited
        return report

    monkeypatch.setattr(mod.asyncio, "run", _fake_run)
    mod.run_reconcile()

    out = capsys.readouterr().out
    assert "Scanned 2 of 9 live annotations" in out
    assert "truncated" in out
    assert {"retire": "KEEP", "review": "CHECK", "merge": "MERGE"}[action] in out
    assert "RETIRE" in out
    assert f"hafiz forget {cluster.others[0].id} --annotation" in out
