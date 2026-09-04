"""Tests for "is my index fresh?" — the question `hafiz status` answers.

Two defects, both measured on a live 15-project index:

* ``last_commit_per_project`` was ``max(hash)`` — a *lexicographic* max over
  hex strings, i.e. an arbitrary commit. It reported Admin Portal at
  ``b37cced2`` (2026-04-17) when the true latest was ``13367ea0``
  (2026-07-17), because ``b3`` > ``13`` as text. 5 of 15 projects disagreed.
* Nothing compared the index to the repo at all, so four repos drifted 31-64
  commits behind with hooks installed and firing.

The first test here is the one P1-9 asks for by name: two commits whose hash
ordering and date ordering disagree. Any fix without it regresses.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

from hafiz.commands.maintenance import _staleness_note
from hafiz.core.database import close_engine, get_session_factory
from hafiz.core.git_context import commits_behind_head


async def _db_available() -> bool:
    try:
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text("SELECT 1 FROM commits LIMIT 1"))
        return True
    except Exception:
        return False


async def _wipe() -> None:
    factory = get_session_factory()
    async with factory() as s:
        await s.execute(text("DELETE FROM files WHERE project LIKE 'freshness-test%'"))
        await s.execute(text("DELETE FROM commits WHERE project LIKE 'freshness-test%'"))
        await s.commit()


@pytest.fixture
async def db():
    """Scoped to the DB-backed tests only; the rest of this module is pure."""
    if not await _db_available():
        pytest.skip("Postgres not reachable")
    await _wipe()
    yield
    await _wipe()
    await close_engine()


# ── The P1-9 regression: hash order vs date order ───────────────────────


async def _seed(project: str, rows: list[tuple[str, datetime | None]]) -> None:
    """Insert one file per (sha, committed_at), all under ``project``."""
    import uuid

    from hafiz.core.database import Commit, File, get_session_factory

    factory = get_session_factory()
    async with factory() as s:
        for i, (sha, committed_at) in enumerate(rows):
            if committed_at is not None:
                s.add(Commit(hash=sha, project=project, committed_at=committed_at))
            s.add(
                File(
                    id=uuid.uuid4(),
                    project=project,
                    path=f"/tmp/{project}/f{i}.py",
                    language="python",
                    last_seen_commit=sha,
                    created_at=datetime.now(UTC) + timedelta(seconds=i),
                )
            )
        await s.commit()


async def test_latest_commit_is_by_date_not_by_hash(db):
    """THE regression test for P1-9.

    ``b37cced2`` sorts after ``13367ea0`` as a string but is three months
    older. Ordering by hash reports the stale one; ordering by ``committed_at``
    reports the real HEAD. Uses the exact pair observed in production.
    """
    from hafiz.core.store import last_indexed_commit_per_project

    older_but_higher_hash = "b37cced29297086fb07e1b36dc2733af3dcf6909"
    newer_but_lower_hash = "13367ea04b23e1824e9b376637a4d4da29c40d1d"
    assert older_but_higher_hash > newer_but_lower_hash, "premise: hash order is inverted"

    await _seed(
        "freshness-test-a",
        [
            (older_but_higher_hash, datetime(2026, 4, 17, tzinfo=UTC)),
            (newer_but_lower_hash, datetime(2026, 7, 17, tzinfo=UTC)),
        ],
    )

    latest = await last_indexed_commit_per_project()
    assert latest["freshness-test-a"] == newer_but_lower_hash


async def test_dated_commits_beat_undated_ones(db):
    """A hash predating the `commits` table has no date; it must not win."""
    from hafiz.core.store import last_indexed_commit_per_project

    await _seed(
        "freshness-test-b",
        [
            ("ffffffffffffffffffffffffffffffffffffffff", None),
            ("0000000000000000000000000000000000000000", datetime(2026, 7, 1, tzinfo=UTC)),
        ],
    )
    latest = await last_indexed_commit_per_project()
    assert latest["freshness-test-b"] == "0000000000000000000000000000000000000000"


async def test_undated_commits_still_reported_rather_than_dropped(db):
    """The join must be an outer one — an undated hash is better than nothing."""
    from hafiz.core.store import last_indexed_commit_per_project

    await _seed("freshness-test-c", [("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", None)])
    latest = await last_indexed_commit_per_project()
    assert latest["freshness-test-c"] == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


async def test_undated_ties_break_on_ingest_order(db):
    """With no dates at all, fall back to the most recently ingested file."""
    from hafiz.core.store import last_indexed_commit_per_project

    await _seed(
        "freshness-test-d",
        [
            ("ffffffffffffffffffffffffffffffffffffffff", None),
            ("1111111111111111111111111111111111111111", None),
        ],
    )
    # _seed stamps created_at increasing, so the second row is newest.
    latest = await last_indexed_commit_per_project()
    assert latest["freshness-test-d"] == "1111111111111111111111111111111111111111"


async def test_projects_are_reported_independently(db):
    from hafiz.core.store import last_indexed_commit_per_project

    await _seed("freshness-test-e", [("e" * 40, datetime(2026, 7, 1, tzinfo=UTC))])
    await _seed("freshness-test-f", [("f" * 40, datetime(2026, 7, 2, tzinfo=UTC))])
    latest = await last_indexed_commit_per_project()
    assert latest["freshness-test-e"] == "e" * 40
    assert latest["freshness-test-f"] == "f" * 40


async def test_indexed_root_is_recovered_from_file_paths(db):
    """The repo path isn't stored; it's derived from the common path prefix."""
    from hafiz.core.store import indexed_root_per_project

    await _seed(
        "freshness-test-g",
        [("a" * 40, None), ("b" * 40, None)],
    )
    roots = await indexed_root_per_project()
    assert roots["freshness-test-g"] == "/tmp/freshness-test-g"


async def test_indexed_root_survives_a_path_that_no_longer_exists(db):
    """A project's directory may be gone (repo moved, deleted).

    The root must still be derived from the recorded paths — probing the
    filesystem here would strip a valid directory to its parent and point the
    staleness probe at the wrong repo.
    """
    from hafiz.core.store import indexed_root_per_project

    await _seed("freshness-test-h", [("a" * 40, None), ("b" * 40, None)])
    assert not Path("/tmp/freshness-test-h").exists()
    roots = await indexed_root_per_project()
    assert roots["freshness-test-h"] == "/tmp/freshness-test-h"


async def test_indexed_root_of_a_single_file_is_its_directory(db):
    """commonpath of one path is that path — it must be reduced to a dir."""
    from hafiz.core.store import indexed_root_per_project

    await _seed("freshness-test-i", [("a" * 40, None)])
    roots = await indexed_root_per_project()
    assert roots["freshness-test-i"] == "/tmp/freshness-test-i"


# ── Staleness probe against a real repo ─────────────────────────────────


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin",
        },
    )


@pytest.fixture
def repo(tmp_path) -> Path:
    _git(["init", "-q", "."], tmp_path)
    return tmp_path


def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name, encoding="utf-8")
    _git(["add", "-A"], repo)
    _git(["commit", "-q", "-m", name], repo)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()


def test_head_itself_is_zero_behind(repo):
    sha = _commit(repo, "a")
    assert commits_behind_head(sha, repo) == {
        "head_commit": sha,
        "commits_behind": 0,
        "is_ancestor": True,
    }


def test_counts_commits_behind(repo):
    base = _commit(repo, "a")
    _commit(repo, "b")
    head = _commit(repo, "c")
    got = commits_behind_head(base, repo)
    assert got["commits_behind"] == 2
    assert got["is_ancestor"] is True
    assert got["head_commit"] == head


def test_unreachable_sha_reports_diverged_not_a_count(repo):
    """A rebased-away sha must not be reported as "0 behind".

    `git rev-list A..HEAD` on a missing A yields nothing, which would read as
    "up to date" — the inverse of the truth.
    """
    _commit(repo, "a")
    got = commits_behind_head("0" * 40, repo)
    assert got["is_ancestor"] is False
    assert got["commits_behind"] is None


def test_non_repo_degrades_to_unknown(tmp_path):
    """status must still print when a project's repo has moved or vanished."""
    assert commits_behind_head("a" * 40, tmp_path / "nope") == {
        "head_commit": None,
        "commits_behind": None,
        "is_ancestor": None,
    }


def test_blank_sha_degrades_to_unknown(repo):
    assert commits_behind_head("", repo)["commits_behind"] is None


# ── Status label rendering ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ({"commits_behind": 0, "is_ancestor": True}, "current"),
        ({"commits_behind": 3, "is_ancestor": True}, "3 behind"),
        ({"commits_behind": 64, "is_ancestor": True}, "64 behind"),
        ({"commits_behind": None, "is_ancestor": False}, "diverged"),
        ({"commits_behind": None, "is_ancestor": None}, "unknown"),
        ({}, "unknown"),
    ],
)
def test_staleness_labels(entry, expected):
    assert _staleness_note(entry)[0] == expected


def test_large_drift_is_styled_as_an_error_not_a_warning():
    """31-64 commits behind was the observed real case; it should read as red."""
    assert _staleness_note({"commits_behind": 64, "is_ancestor": True})[1] == "red"
    assert _staleness_note({"commits_behind": 2, "is_ancestor": True})[1] == "yellow"


# ── The untagged bucket is a shadow index, not a stale repo ─────────────
#
# A project-less ingest can't update a project's rows (`files` is unique on
# (project, path)), so it writes a parallel untagged copy. 1,956 such rows
# accumulated unnoticed because nothing counted them — and asking "how far
# behind HEAD is it?" is the wrong question: its files span every repo the
# broken hook ever walked, so the derived root came out as "/".


async def test_untagged_files_are_excluded_from_per_repo_staleness(db):
    from hafiz.commands.maintenance import _index_staleness

    out = await _index_staleness({None: "a" * 40, "freshness-test-j": "b" * 40})
    assert "(none)" not in out
    assert None not in out
    assert "freshness-test-j" in out


# ── Freshness as a signal a caller can read ─────────────────────────────
#
# Search results carried no freshness field at all, so a caller could not tell a
# current result from a 28-commit-stale one. The only safe policy was a blanket
# "never trust the index", which a real integrator adopted — and which throws out
# the 89% of the corpus that is documentation to protect against the 3.9% that is
# code. Measured: 4.4% of live files changed since their indexed commit, 3 deleted
# outright, but the repo being actively edited was 80.5% stale.


def test_stale_projects_keeps_only_what_is_actionable():
    from hafiz.core.freshness import stale_projects

    got = stale_projects(
        {
            "behind": {"commits_behind": 12, "is_ancestor": True},
            "diverged": {"commits_behind": None, "is_ancestor": False},
            "current": {"commits_behind": 0, "is_ancestor": True},
            "unknown": {"commits_behind": None, "is_ancestor": None},
        }
    )
    assert sorted(got) == ["behind", "diverged"]


def test_unknown_is_not_reported_as_stale():
    """ "Unknown" usually means the repo isn't on this machine. Warning about it
    would train the reader to ignore the warning."""
    from hafiz.core.freshness import stale_projects

    assert stale_projects({"p": {"commits_behind": None, "is_ancestor": None}}) == {}


async def test_an_empty_project_scope_does_no_work(monkeypatch):
    """A result set with no project-tagged rows must not sweep the whole store."""
    from hafiz.core import freshness

    async def _boom(*a, **kw):
        raise AssertionError("should not query")

    monkeypatch.setattr("hafiz.core.store.last_indexed_commit_per_project", _boom)
    assert await freshness.index_staleness([]) == {}
    assert await freshness.index_staleness([None]) == {}


@pytest.mark.parametrize(
    ("staleness", "expected"),
    [
        ({"Web": {"commits_behind": 12, "is_ancestor": True}}, "Web 12 behind"),
        ({"Admin": {"commits_behind": None, "is_ancestor": False}}, "Admin diverged"),
        (
            {
                "Web": {"commits_behind": 12, "is_ancestor": True},
                "Admin": {"commits_behind": None, "is_ancestor": False},
            },
            "Web 12 behind, Admin diverged",
        ),
    ],
)
def test_staleness_line_is_short_enough_to_inject(staleness, expected):
    from hafiz.commands.query import _staleness_line

    assert _staleness_line(staleness) == expected


def test_status_counts_untagged_files():
    """The field that would have surfaced the shadow index without an audit.

    Synchronous: ``status`` drives its own ``asyncio.run``.
    """
    import asyncio
    import json

    from typer.testing import CliRunner

    from hafiz.cli import app

    def _run(coro):
        async def _wrapped():
            try:
                return await coro
            finally:
                await close_engine()

        return asyncio.run(_wrapped())

    if not _run(_db_available()):
        pytest.skip("Postgres not reachable")

    result = CliRunner().invoke(app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload["untagged"]["files"], int)


# ---------------------------------------------------------------------------
# Capture freshness — "is anything still arriving?"
# ---------------------------------------------------------------------------
#
# Retention-overdue answers "is the sweep keeping up", which a store
# receiving nothing passes trivially. That gap let transcript capture die
# on 2026-06-30 and go unnoticed until 2026-09-04, with `status` reporting
# `retention.overdue: 0` — healthy-looking — the whole time.


def test_stale_captures_flags_agents_with_transcripts_waiting():
    from hafiz.core.freshness import stale_captures

    flagged = stale_captures(
        {
            "claude-code": {"days_since": 66, "pending_on_disk": 201},
            "cursor": {"days_since": 1, "pending_on_disk": 0},
        }
    )
    assert set(flagged) == {"claude-code"}


def test_stale_captures_ignores_a_quiet_agent_with_nothing_pending():
    """There is no remedy for "you stopped using this agent".

    Warning on every `status` about an agent whose files already rotated
    away teaches the operator to skim past the block — which is exactly
    how the original missing signal became expensive.
    """
    from hafiz.core.freshness import stale_captures

    assert stale_captures({"test-agent": {"days_since": 131, "pending_on_disk": 0}}) == {}
    assert stale_captures({"hermes": {"days_since": 11}}) == {}  # no on-disk store
    assert stale_captures({"never": {"days_since": None, "pending_on_disk": None}}) == {}
