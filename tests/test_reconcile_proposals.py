"""Reconcile's clustering and its resolution proposals.

DB-free: everything here is pure logic over embeddings that are already in
hand. The DB-backed half — scan coverage and the ``doctor`` count — lives in
``test_cli.py``.

The thing under test is a *safety* property as much as a correctness one.
Reconcile proposes destructive commands, and the near-duplicate pairs in a real
store are not all restatements: a 91%-similar pair turned out to be one row
holding a test account's email and another holding its password. Retiring
either would have destroyed a fact the other did not carry. So the rule is that
a proposal may never suggest keeping a row that is materially shorter than one
it would retire.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from hafiz.commands.reconcile import _commands_for, _preview, _shell_quote
from hafiz.core.annotations import _build_cluster, _similarity_matrix, _single_linkage

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


# ── the proposal ─────────────────────────────────────────────────────


def test_newest_row_is_the_primary_when_it_loses_nothing():
    anns = [_ann("old text, quite long here", days=0), _ann("newer text, also long", days=5)]
    cluster = _build_cluster("decision", "p", anns, [0, 1], _sim(2))
    assert cluster.action == "retire"
    assert cluster.primary.id == str(anns[1].id)
    assert [m.id for m in cluster.others] == [str(anns[0].id)]


def test_shorter_newest_row_downgrades_the_proposal_to_merge():
    """The credential/email case: the newer row is a fraction of the older, so
    keeping only it would drop text. Never propose that."""
    anns = [_ann("x" * 1000, days=0), _ann("y" * 200, days=5)]
    cluster = _build_cluster("decision", "p", anns, [0, 1], _sim(2))
    assert cluster.action == "merge"
    # The primary becomes the *longest* row — the one the merged text supersedes.
    assert cluster.primary.id == str(anns[0].id)


@pytest.mark.parametrize(
    ("newest_len", "action"),
    [(800, "retire"), (799, "merge")],
)
def test_merge_boundary_is_eighty_percent(newest_len, action):
    anns = [_ann("x" * 1000, days=0), _ann("y" * newest_len, days=5)]
    assert _build_cluster("decision", "p", anns, [0, 1], _sim(2)).action == action


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
    anns = [_ann("old", days=0), _ann("newer text", days=5), _ann("newest text", days=9)]
    cluster = _build_cluster("decision", "p", anns, [0, 1, 2], _sim(3))
    cmds = _commands_for(cluster)
    assert len(cmds) == 2
    assert all(c.startswith("hafiz forget ") and c.endswith(" --annotation") for c in cmds)
    assert cluster.primary.id not in " ".join(cmds)


def test_merge_emits_exactly_one_observe_then_retires_the_rest():
    """``--supersedes`` takes a single id, so a merge is one new row superseding
    the primary — not one new row per member, which is what a naive
    command-per-member rendering produced and would have tripled the cluster."""
    anns = [_ann("x" * 1000, days=0), _ann("z" * 900, days=1), _ann("y" * 100, days=5)]
    cluster = _build_cluster("learning", "proj", anns, [0, 1, 2], _sim(3))
    cmds = _commands_for(cluster)
    assert cluster.action == "merge"
    assert sum(c.startswith("hafiz observe ") for c in cmds) == 1
    assert cmds[0].startswith("hafiz observe '<merged text>' --type learning")
    assert f"--supersedes {cluster.primary.id}" in cmds[0]
    assert sorted(cmds[1:]) == sorted(f"hafiz forget {m.id} --annotation" for m in cluster.others)


def test_merge_command_carries_project_and_source_so_the_new_row_keeps_them():
    anns = [_ann("x" * 1000, days=0, source="user:anjum"), _ann("y" * 100, days=5)]
    cmd = _commands_for(_build_cluster("fact", "cookie-notice", anns, [0, 1], _sim(2)))[0]
    assert "--source user:anjum" in cmd
    # Always quoted: real project names have spaces in them ("Admin Portal").
    assert "--project 'cookie-notice'" in cmd


def test_project_name_with_a_space_survives_as_one_argument():
    anns = [_ann("x" * 1000, days=0), _ann("y" * 100, days=5)]
    cmd = _commands_for(_build_cluster("fact", "Admin Portal", anns, [0, 1], _sim(2)))[0]
    assert "--project 'Admin Portal'" in cmd


def test_merge_command_omits_project_when_the_rows_have_none():
    anns = [_ann("x" * 1000, days=0), _ann("y" * 100, days=5)]
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


@pytest.mark.parametrize("action", ["retire", "merge"])
def test_human_output_renders_both_actions(action, monkeypatch, capsys):
    """The rich path is not exercised by the ``--json`` tests, and a rename
    once left an undefined name in it that every JSON test still passed over."""
    from hafiz.commands import reconcile as mod

    short, long_ = ("y" * 100, "x" * 1000) if action == "merge" else ("y" * 900, "x" * 1000)
    anns = [_ann(long_, days=0), _ann(short, days=5)]
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
    assert ("KEEP" in out) if action == "retire" else ("MERGE" in out)
    assert "RETIRE" in out
    assert f"hafiz forget {cluster.others[0].id} --annotation" in out
