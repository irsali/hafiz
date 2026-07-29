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
