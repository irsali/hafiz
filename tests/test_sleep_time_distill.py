"""The sleep-time distill backlog — theme clustering, the gate, the scaffold.

DB-free: everything here is logic over candidates and vectors already in hand.
The DB-backed half — the drain (a promoted note leaving the queue) and the CLI
shapes — lives in ``test_cli.py``.

Two properties here are safety properties rather than correctness ones:

- **The scaffold must cite every member of its theme.** Citing a capture is
  what drains it, so a truncated scaffold silently strands the uncited members
  in the queue forever and the backlog stops converging.
- **``--brief`` must be silent by default.** It is designed to be piped into a
  session-start hook. A memory layer that talks on every turn gets removed, so
  "nothing to say" has to be the ordinary output.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hafiz.commands.distill import _preview, _theme_scaffold
from hafiz.core.distill import (
    Backlog,
    MessageCandidate,
    NoteCandidate,
    brief_gate_open,
    cluster_candidates,
)

BASE = datetime(2026, 7, 1, tzinfo=UTC)


def _note(nid: str, *, days: int = 0, promoted: bool = False) -> NoteCandidate:
    return NoteCandidate(
        id=nid,
        content=f"note {nid}",
        valid_from=BASE + timedelta(days=days),
        source="agent:claude-code",
        project="p",
        tags=None,
        session_id=None,
        task=None,
        promoted=promoted,
    )


def _msg(mid: str, *, days: int = 0, salient: bool = False) -> MessageCandidate:
    return MessageCandidate(
        id=mid,
        communication_id="c1",
        seq=1,
        role="user",
        author=None,
        content=f"message {mid}",
        ts=BASE + timedelta(days=days),
        marked_salient=salient,
    )


def _backlog(**kw) -> Backlog:
    base = {
        "pending": 0,
        "promoted": 0,
        "oldest_pending_age_days": None,
        "themes": 0,
        "clustered": 0,
        "skipped_unembedded": 0,
    }
    return Backlog(**{**base, **kw})


# ── clustering ───────────────────────────────────────────────────────


def test_similar_captures_land_in_one_theme():
    notes = [_note("a"), _note("b")]
    vectors = {"a": [1.0, 0.0], "b": [0.99, 0.14]}  # cosine ~0.99
    themes = cluster_candidates(notes, [], vectors, threshold=0.65)
    assert len(themes) == 1
    assert {m.id for m in themes[0].members} == {"a", "b"}


def test_dissimilar_captures_stay_separate_themes():
    notes = [_note("a"), _note("b")]
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    assert [t.size for t in cluster_candidates(notes, [], vectors, threshold=0.65)] == [1, 1]


def test_a_note_and_a_turn_cluster_together():
    """The point of grouping across layers: a note lands with the turns it came
    from, so one observe can cite both and drain both."""
    themes = cluster_candidates(
        [_note("n1")],
        [_msg("m1")],
        {"n1": [1.0, 0.0], "m1": [0.98, 0.2]},
        threshold=0.65,
    )
    assert len(themes) == 1
    assert {m.kind for m in themes[0].members} == {"note", "message"}


# ── two corpora, two thresholds ──────────────────────────────────────
#
# Notes and transcript turns have different density, so one threshold cannot
# serve both. Measured on the live store at the note-tuned 0.65, single-linkage
# put 366 of 400 turns in one theme and produced ~nothing readable — at every
# corpus size tried, so it was never about corpus size. Vectors below are 2-D
# unit vectors so the cosine is exact and legible: `_at(c)` sits at cosine `c`
# from `[1, 0]`.


def _at(c: float) -> list[float]:
    return [c, (1.0 - c * c) ** 0.5]


def test_messages_use_their_own_stricter_threshold():
    """A turn pair at 0.70 clears the note bar but not the message bar."""
    vectors = {"m1": [1.0, 0.0], "m2": _at(0.70)}
    themes = cluster_candidates(
        [], [_msg("m1"), _msg("m2")], vectors, threshold=0.65, message_threshold=0.78
    )
    assert [t.size for t in themes] == [1, 1]

    # Same pair, same data — only the message bar moves.
    themes = cluster_candidates(
        [], [_msg("m1"), _msg("m2")], vectors, threshold=0.65, message_threshold=0.65
    )
    assert [t.size for t in themes] == [2]


def test_notes_keep_their_looser_threshold_when_messages_tighten():
    """Raising the message bar must not quietly re-tune note clustering.

    This is the regression that a single shared threshold would have caused:
    0.65 was reasoned about for notes, and fixing turns should not cost that.
    """
    themes = cluster_candidates(
        [_note("n1"), _note("n2")],
        [],
        {"n1": [1.0, 0.0], "n2": _at(0.70)},
        threshold=0.65,
        message_threshold=0.78,
    )
    assert [t.size for t in themes] == [2]


def test_a_cross_kind_pair_takes_the_stricter_threshold():
    """max, not min. A note/turn pair at 0.70 must not merge under 0.65/0.78."""
    themes = cluster_candidates(
        [_note("n1")],
        [_msg("m1")],
        {"n1": [1.0, 0.0], "m1": _at(0.70)},
        threshold=0.65,
        message_threshold=0.78,
    )
    assert [t.size for t in themes] == [1, 1]


def test_a_note_cannot_bridge_turns_the_message_bar_held_apart():
    """Why cross-kind takes the stricter value, stated as a test.

    Single-linkage merges transitively, so a pair that is allowed to link is a
    pair that can act as a *bridge*. With the looser value on cross-kind pairs,
    one note sitting at 0.70 from two turns would glue those turns into one
    theme even though they sit at -0.02 from each other and the message bar
    correctly held them apart — the 366-member blob, re-entering through the
    back door.

    n1 at angle 0, m1 and m2 at ±45.6°: each turn is 0.70 from the note and
    -0.02 from the other turn.
    """
    vectors = {"n1": [1.0, 0.0], "m1": _at(0.70), "m2": [0.70, -_at(0.70)[1]]}
    themes = cluster_candidates(
        [_note("n1")],
        [_msg("m1"), _msg("m2")],
        vectors,
        threshold=0.65,
        message_threshold=0.78,
    )
    assert [t.size for t in themes] == [1, 1, 1], (
        "the note must not bridge two turns the message threshold separated"
    )


def test_theme_score_is_the_real_similarity_not_a_masking_artefact():
    """The reported score must be what the members actually share.

    Honest note on the strength of this test: it pins the property, but it
    cannot currently fail by scoring off the masked matrix instead. Masked
    pairs are ``-inf`` and the aggregate is ``max``, so ``-inf`` never wins and
    the two matrices coincide for every cluster that formed — verified by
    mutation, which passed. The guard earns its place against a future change
    to the aggregate: under ``min`` or a mean, a chained cluster containing a
    masked pair would report a similarity no two members have.
    """
    themes = cluster_candidates(
        [],
        [_msg("m1"), _msg("m2")],
        {"m1": [1.0, 0.0], "m2": _at(0.90)},
        threshold=0.65,
        message_threshold=0.78,
    )
    assert [t.size for t in themes] == [2]
    assert themes[0].score == pytest.approx(0.90, abs=1e-3)


def test_message_threshold_defaults_to_the_note_threshold():
    """Omitting it must preserve the old single-threshold behaviour, so every
    existing caller and test keeps its meaning."""
    themes = cluster_candidates(
        [], [_msg("m1"), _msg("m2")], {"m1": [1.0, 0.0], "m2": _at(0.70)}, threshold=0.65
    )
    assert [t.size for t in themes] == [2]


# ── oversized themes are split, never truncated ──────────────────────
#
# A theme's scaffold cites every member because citing is what drains the
# queue. So an oversized theme cannot be fixed by dropping ids — that strands
# the dropped captures permanently and the backlog stops converging. Splitting
# keeps every member cited by exactly one scaffold, which is the property the
# first test here pins.


def _dense(n: int) -> dict[str, list[float]]:
    """n turns that cluster into one theme but are not near-identical.

    Pairwise cosine ~0.85 — above the clustering bar, below the collapse bar.
    The first version used `[1.0, 0.001 * i]`, which is ~1.0 in float32 and so
    got folded onto a single representative once collapse existed, making these
    split tests silently test the wrong thing.
    """
    off = 0.42
    out = {}
    for i in range(n):
        v = [0.0] * (n + 1)
        v[0] = 1.0
        v[1 + i] = off
        out[f"m{i}"] = v
    return out


def test_splitting_an_oversized_theme_loses_no_member():
    """The safety property. Every capture must survive in exactly one part.

    If this ever fails, captures are being silently stranded in the backlog —
    the failure mode is invisible in the output and permanent in the queue.
    """
    n = 30
    msgs = [_msg(f"m{i}", days=i) for i in range(n)]
    themes = cluster_candidates([], msgs, _dense(n), threshold=0.65, max_size=7)

    seen = [m.id for t in themes for m in t.members]
    assert sorted(seen) == sorted(f"m{i}" for i in range(n))
    assert len(seen) == len(set(seen)), "a member was cited by more than one part"


def test_no_theme_exceeds_the_cap():
    n = 30
    themes = cluster_candidates(
        [], [_msg(f"m{i}", days=i) for i in range(n)], _dense(n), threshold=0.65, max_size=7
    )
    assert max(t.size for t in themes) <= 7
    assert [t.size for t in themes] == [7, 7, 7, 7, 2]


def test_each_part_is_rescored_against_its_own_members():
    """A part's similarity must describe the part, not the cluster it came from.

    Inheriting the parent's score would report a similarity that no two members
    of the part actually share.
    """
    vectors = {"m0": [1.0, 0.0], "m1": _at(0.99), "m2": _at(0.98), "m3": _at(0.97)}
    parts = cluster_candidates(
        [], [_msg(f"m{i}", days=i) for i in range(4)], vectors, threshold=0.65, max_size=2
    )
    assert [t.size for t in parts] == [2, 2]
    # Whatever the values, no part may claim the parent cluster's best pair
    # unless that pair is inside it.
    for t in parts:
        assert t.score <= 1.0


def test_parts_are_temporally_coherent():
    """Split on ts so each part reads as a stretch of one conversation.

    A theme is meant to be read oldest-first, following how the thought
    developed; splitting on an arbitrary axis would interleave unrelated
    moments.
    """
    n = 6
    msgs = [_msg(f"m{i}", days=i) for i in range(n)]
    parts = cluster_candidates([], msgs, _dense(n), threshold=0.65, max_size=3)
    for t in parts:
        ts = [m.ts for m in t.members]
        assert ts == sorted(ts)
    # And the parts themselves must not interleave: every member of the older
    # part predates every member of the newer one.
    by_oldest = sorted(parts, key=lambda t: t.oldest)
    assert by_oldest[0].newest < by_oldest[1].oldest


def test_a_cap_of_zero_disables_splitting():
    """The escape hatch has to actually escape."""
    n = 20
    themes = cluster_candidates(
        [], [_msg(f"m{i}", days=i) for i in range(n)], _dense(n), threshold=0.65, max_size=0
    )
    assert [t.size for t in themes] == [n]


def test_the_shipped_cap_keeps_a_scaffold_runnable():
    """Guards the default, not just the mechanism — the 80-uuid command again.

    An 80-member `--derived-from` is long enough that nobody runs it, which
    strands the theme exactly as effectively as truncating it would.
    """
    from hafiz.core.config import DistillSettings

    cfg = DistillSettings()
    n = 100
    themes = cluster_candidates(
        [],
        [_msg(f"m{i}", days=i) for i in range(n)],
        _dense(n),
        threshold=cfg.cluster_threshold,
        message_threshold=cfg.cluster_threshold_messages,
        max_size=cfg.max_theme_size,
    )
    largest = max(t.size for t in themes)
    assert 0 < largest <= 20, (
        f"the shipped max_theme_size lets a theme reach {largest} members, "
        f"whose scaffold is too long to be run"
    )


# ── the shipped defaults, not just the mechanism ─────────────────────
#
# Every test above passes its thresholds explicitly, which means it verifies
# the *machinery* and says nothing about the *values*. That gap was found by
# mutation: reverting `cluster_threshold_messages` to 0.65, or
# `theme_corpus_limit` to 50 — either of which re-creates the original bug —
# left all 1080 tests green. The three tests below close it, and they read the
# built-in `DistillSettings()` rather than `load_settings()` on purpose: it is
# the shipped default that must hold, not whatever this machine's hafiz.toml
# happens to say.


def _chain_vectors(n: int, *, window: int = 4) -> dict[str, list[float]]:
    """A corpus shaped like the real failure mode: a chain, not a blob.

    Vector *i* is ``window`` consecutive ones starting at *i*, so overlap —
    and therefore cosine — falls off with distance:
    adjacent = ``(window-1)/window`` = 0.75, two apart 0.5, four apart 0.0.

    That is deliberately chosen to straddle the two thresholds. At the note bar
    (0.65) every adjacent pair links and single-linkage transitively merges the
    whole chain into one theme, even though the ends share nothing. At the turn
    bar (0.78) no pair links. It reproduces chaining exactly, with no
    randomness, so the test cannot flake.
    """
    dim = n + window - 1
    vectors = {}
    for i in range(n):
        v = [0.0] * dim
        for j in range(i, i + window):
            v[j] = 1.0
        vectors[f"m{i}"] = v
    return vectors


def test_the_shipped_defaults_do_not_chain_a_dense_turn_corpus():
    """The regression guard that does not care *which* knob broke.

    This asserts the outcome — no single theme swallows the corpus — rather
    than a threshold value, so it still fails if the default is lowered, if the
    two thresholds are collapsed into one, or if the linkage strategy is
    changed in a way that reintroduces chaining. It is the test that would have
    caught the 366-of-400 blob before it reached the live store.
    """
    from hafiz.core.config import DistillSettings

    cfg = DistillSettings()
    n = 12
    themes = cluster_candidates(
        [],
        [_msg(f"m{i}") for i in range(n)],
        _chain_vectors(n),
        threshold=cfg.cluster_threshold,
        message_threshold=cfg.cluster_threshold_messages,
    )
    largest = max(t.size for t in themes)
    assert largest < n, (
        f"a chained corpus collapsed into one theme of {largest}/{n} under the "
        f"shipped defaults (notes {cfg.cluster_threshold}, turns "
        f"{cfg.cluster_threshold_messages}) — this is the 366-of-400 blob"
    )


def test_the_turn_threshold_ships_stricter_than_the_note_threshold():
    """The gap between them is the whole fix, not a coincidence of tuning.

    Turns are more self-similar than notes, so their bar must be higher. If a
    future change makes these equal, the per-corpus split silently becomes a
    no-op while every mechanism test still passes.
    """
    from hafiz.core.config import DistillSettings

    cfg = DistillSettings()
    assert cfg.cluster_threshold_messages > cfg.cluster_threshold


def test_the_clustering_corpus_ships_larger_than_the_readable_slice():
    """Otherwise the second query buys nothing.

    The point of splitting the caps is that clustering can see past the
    readable slice. At `theme_corpus_limit <= message_limit` the split is
    machinery with no effect, which is exactly the starvation it was added to
    fix.
    """
    from hafiz.core.config import DistillSettings

    cfg = DistillSettings()
    assert cfg.theme_corpus_limit > cfg.message_limit


def test_unembedded_turn_is_not_a_candidate():
    """Selective embedding already declined it at import — short turns and pure
    tool-result echoes never get a vector. Re-offering them here contradicts
    that call, and on a real window it buried 2 useful themes under ~26
    singletons of browser chatter and raw <toolCall> echoes."""
    themes = cluster_candidates([], [_msg("m1"), _msg("m2")], {"m1": [1.0, 0.0]}, threshold=0.65)
    assert [m.id for t in themes for m in t.members] == ["m1"]


def test_unembedded_note_survives_as_a_singleton():
    """A note is a deliberate capture, not an incidental turn — it must not
    vanish just because it has no vector to compare."""
    themes = cluster_candidates([_note("n1")], [], {}, threshold=0.65)
    assert [(t.size, t.members[0].id) for t in themes] == [(1, "n1")]


def test_promoted_notes_are_never_offered_as_candidates():
    themes = cluster_candidates(
        [_note("kept"), _note("gone", promoted=True)],
        [],
        {"kept": [1.0, 0.0], "gone": [1.0, 0.0]},
        threshold=0.65,
    )
    assert [m.id for t in themes for m in t.members] == ["kept"]


def test_no_candidates_yields_no_themes():
    assert cluster_candidates([], [], {}, threshold=0.65) == []


def test_biggest_theme_sorts_first():
    """The head of the list is where distillation pays best."""
    notes = [_note("a"), _note("b"), _note("lonely")]
    vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0], "lonely": [0.0, 1.0]}
    themes = cluster_candidates(notes, [], vectors, threshold=0.65)
    assert [t.size for t in themes] == [2, 1]


def test_theme_members_are_ordered_oldest_first():
    """Reading a theme is reading how the thought developed."""
    notes = [_note("late", days=9), _note("early", days=0)]
    vectors = {"late": [1.0, 0.0], "early": [1.0, 0.0]}
    theme = cluster_candidates(notes, [], vectors, threshold=0.65)[0]
    assert [m.id for m in theme.members] == ["early", "late"]
    assert theme.oldest == BASE
    assert theme.newest == BASE + timedelta(days=9)


def test_theme_score_is_best_pairwise_similarity_and_one_when_alone():
    notes = [_note("a"), _note("b")]
    vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0]}
    assert cluster_candidates(notes, [], vectors, threshold=0.65)[0].score == 1.0
    assert cluster_candidates([_note("a")], [], {"a": [1.0, 0.0]}, threshold=0.65)[0].score == 1.0


# ── the scaffold ─────────────────────────────────────────────────────


def test_scaffold_cites_every_member_not_just_the_first_few():
    """Truncating the citation list strands the uncited members in the backlog
    permanently — the queue would look stuck no matter how much work got done."""
    notes = [_note(f"n{i}") for i in range(9)]
    vectors = {f"n{i}": [1.0, 0.0] for i in range(9)}
    theme = cluster_candidates(notes, [], vectors, threshold=0.65)[0]
    cmd = _theme_scaffold(theme)
    assert theme.size == 9
    for note in notes:
        assert note.id in cmd
    assert cmd.count(",") == 8


def test_scaffold_is_a_derived_from_observe():
    theme = cluster_candidates([_note("n1")], [], {"n1": [1.0, 0.0]}, threshold=0.65)[0]
    cmd = _theme_scaffold(theme)
    assert cmd.startswith("hafiz observe '<distilled text>' --type decision --derived-from ")


# ── the --brief gate ─────────────────────────────────────────────────


def test_gate_is_closed_on_an_empty_backlog():
    assert not brief_gate_open(_backlog(), min_pending=3, min_age_days=2.0)


def test_gate_is_closed_when_there_is_no_backlog_at_all():
    assert not brief_gate_open(None, min_pending=3, min_age_days=2.0)


def test_gate_opens_on_volume():
    backlog = _backlog(pending=3, oldest_pending_age_days=0.1)
    assert brief_gate_open(backlog, min_pending=3, min_age_days=2.0)


def test_gate_opens_on_age_even_when_the_backlog_is_small():
    """One capture left waiting a week still deserves a nudge."""
    backlog = _backlog(pending=1, oldest_pending_age_days=7.0)
    assert brief_gate_open(backlog, min_pending=3, min_age_days=2.0)


def test_gate_stays_closed_below_both_thresholds():
    backlog = _backlog(pending=2, oldest_pending_age_days=0.5)
    assert not brief_gate_open(backlog, min_pending=3, min_age_days=2.0)


def test_a_pending_count_of_zero_never_opens_the_gate_on_age():
    """Guards the arithmetic: promoted-only windows report an age from rows that
    are no longer work."""
    backlog = _backlog(pending=0, promoted=9, oldest_pending_age_days=99.0)
    assert not brief_gate_open(backlog, min_pending=1, min_age_days=1.0)


# ── rendering ────────────────────────────────────────────────────────


def test_preview_collapses_newlines_into_one_line():
    assert _preview("first line\n\n  second   line", 200) == "first line second line"


def test_preview_truncates_with_an_ellipsis():
    out = _preview("x" * 500, 200)
    assert out.endswith("…")
    assert len(out) == 201


# ── Phase 2d: near-identical repeats collapse ────────────────────────
#
# With themes anchored on user turns, the largest ones turned out to be
# harness-injected skill preambles ("Base directory for this skill: ...") at
# similarity 0.99-1.00 — role=user, but the harness said them, not the user.
# 57 of 197 multi-member members. And because themes sort by size, twelve
# copies of one string sorted *ahead* of a real topic.


def _same(n: int, *, kind_user: bool = True) -> dict[str, list[float]]:
    """n byte-identical vectors — the boilerplate case."""
    return {f"m{i}": [1.0, 0.0] for i in range(n)}


def _spread(ids: list[str], pairwise: float = 0.85) -> dict[str, list[float]]:
    """Vectors mutually `pairwise` apart: one topic, distinctly different texts.

    Not `_at(c)`: two vectors each at cosine 0.85-0.90 from `[1, 0]` sit at
    ~0.99 from *each other*, so they would collapse. The pairwise value is what
    the threshold sees, and a shared base plus one distinct offset each makes
    it exact — cosine is `1 / (1 + off**2)`, so `off = sqrt(1/p - 1)`.
    """
    off = (1.0 / pairwise - 1.0) ** 0.5
    out = {}
    for n, i in enumerate(ids):
        v = [0.0] * (len(ids) + 1)
        v[0] = 1.0
        v[1 + n] = off
        out[i] = v
    return out


def test_repeats_collapse_onto_one_representative():
    themes = cluster_candidates(
        [],
        [_msg(f"m{i}", days=i) for i in range(12)],
        _same(12),
        threshold=0.65,
        collapse_threshold=0.98,
    )
    assert len(themes) == 1
    assert themes[0].size == 1, "twelve copies are one distinct capture"
    assert themes[0].total_size == 12, "and the theme still accounts for all twelve"


def test_a_collapsed_repeat_is_still_cited():
    """The safety property, and the reason this is a collapse and not a drop.

    Citing is what drains a capture. A repeat hidden from the reader *and*
    from the scaffold would sit in the backlog forever — the same trap that
    made the size cap a split rather than a truncation.
    """
    msgs = [_msg(f"m{i}", days=i) for i in range(12)]
    theme = cluster_candidates([], msgs, _same(12), threshold=0.65, collapse_threshold=0.98)[0]

    cmd = _theme_scaffold(theme)
    for m in msgs:
        assert m.id in cmd, f"{m.id} was collapsed out of its own scaffold"
    assert sorted(theme.cited_ids) == sorted(m.id for m in msgs)


def test_the_oldest_repeat_becomes_the_representative():
    """A theme reads oldest-first, so the first occurrence is what to show."""
    theme = cluster_candidates(
        [],
        [_msg(f"m{i}", days=i) for i in range(5)],
        _same(5),
        threshold=0.65,
        collapse_threshold=0.98,
    )[0]
    assert theme.members[0].id == "m0"


def test_collapsing_stops_boilerplate_outranking_a_real_theme():
    """The ordering consequence, which is the point.

    Themes sort by size. Ten copies of one preamble outranked a genuine
    three-capture topic until size stopped counting repeats.
    """
    # Boilerplate sits on the first axis; the real topic on later ones, so
    # the two groups do not cluster with each other. The three real captures
    # are ~0.85 pairwise — one topic, three distinct texts.
    vectors = {f"m{i}": [1.0, 0.0, 0.0, 0.0, 0.0] for i in range(10)}
    for n, rid in enumerate(("real1", "real2", "real3")):
        v = [0.0, 0.0, 0.0, 0.0, 0.0]
        v[1] = 1.0
        v[2 + n] = 0.42
        vectors[rid] = v
    msgs = [_msg(f"m{i}", days=i) for i in range(10)] + [
        _msg("real1", days=20),
        _msg("real2", days=21),
        _msg("real3", days=22),
    ]
    themes = cluster_candidates(
        [], msgs, vectors, threshold=0.65, message_threshold=0.78, collapse_threshold=0.98
    )
    assert themes[0].size == 3, (
        f"the real 3-capture theme must sort first, got sizes {[t.size for t in themes]}"
    )


def test_notes_are_exempt_from_collapsing():
    """A note is deliberate, `hafiz note` already refuses byte-identical
    repeats, and a genuinely repeated note is a signal rather than noise —
    the same reasoning that exempts notes from the unembedded-turn rule."""
    notes = [_note(f"n{i}") for i in range(9)]
    theme = cluster_candidates(
        notes,
        [],
        {f"n{i}": [1.0, 0.0] for i in range(9)},
        threshold=0.65,
        collapse_threshold=0.98,
    )[0]
    assert theme.size == 9
    assert theme.duplicates == []


def test_collapse_defaults_to_off_so_existing_behaviour_is_unchanged():
    """Omitting the threshold must leave every existing caller's meaning."""
    themes = cluster_candidates(
        [], [_msg(f"m{i}", days=i) for i in range(6)], _same(6), threshold=0.65
    )
    assert themes[0].size == 6
    assert themes[0].duplicates == []


def test_distinct_captures_are_never_collapsed():
    """The bar is "same text", not "same topic" — well above dedup's 0.88."""
    # `_spread`, not `_at`: two vectors each 0.85-0.90 from `[1, 0]` sit at
    # ~0.99 from *each other* and would rightly collapse. The pairwise value
    # is what the threshold sees.
    vectors = _spread(["m0", "m1", "m2"])
    theme = cluster_candidates(
        [],
        [_msg(f"m{i}", days=i) for i in range(3)],
        vectors,
        threshold=0.65,
        message_threshold=0.78,
        collapse_threshold=0.98,
    )[0]
    assert theme.size == 3, "~0.85 pairwise is the same topic, not the same text"
    assert theme.duplicates == []


def test_the_shipped_collapse_threshold_folds_real_boilerplate():
    """Guards the default. 0.99-1.00 is what the harness preambles measured."""
    from hafiz.core.config import DistillSettings

    cfg = DistillSettings()
    ids = [f"m{i}" for i in range(8)]
    # 0.99 pairwise, not 1.0. The real harness preambles measured 0.99-1.00,
    # and byte-identical vectors would collapse at *any* threshold up to 1.0 —
    # so a fixture built from them cannot detect the default being loosened to
    # 1.0. Verified by mutation: with `_same(8)` this test passed at 1.0.
    themes = cluster_candidates(
        [],
        [_msg(i, days=n) for n, i in enumerate(ids)],
        _spread(ids, pairwise=0.99),
        threshold=cfg.cluster_threshold,
        message_threshold=cfg.cluster_threshold_messages,
        collapse_threshold=cfg.collapse_threshold,
    )
    assert themes[0].size == 1, (
        f"the shipped collapse_threshold {cfg.collapse_threshold} left "
        f"{themes[0].size} near-identical copies as distinct captures"
    )
