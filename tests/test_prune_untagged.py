"""Tests for ``hafiz prune --untagged`` — the shadow index nothing could reach.

``files`` is unique on ``(project, path)``, so an ingest with no ``--project``
cannot update a project's rows: it writes a *parallel untagged copy*. Those rows
were immortal. ``ingest`` skips vanished-file tombstoning outright when
``project is None``::

    if project is not None and diff_scope is None:
        files_tombstoned = await tombstone_vanished_files(project, seen_paths)

and the paths are still on disk anyway, so no walk would ever call them
vanished. ``prune`` was a no-op stub. Measured: 1,958 untagged rows on a live
15-project deployment, returned by search alongside the properly-tagged copies,
with no command able to remove them.

The safety property under test: untagged paths that no project covers are the
*only* copy of those files, and must not be dropped by default.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from hafiz.core.database import File, Unit, close_engine, get_session_factory
from hafiz.core.store import tombstone_untagged_files

MARK = "/tmp/prune-untagged-test"


async def _db_available() -> bool:
    try:
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text("SELECT 1 FROM files LIMIT 1"))
        return True
    except Exception:
        return False


async def _wipe() -> None:
    factory = get_session_factory()
    async with factory() as s:
        await s.execute(text(f"DELETE FROM files WHERE path LIKE '{MARK}%'"))
        await s.commit()


@pytest.fixture
async def db():
    if not await _db_available():
        pytest.skip("Postgres not reachable")
    await _wipe()
    yield
    await _wipe()
    await close_engine()


async def _seed(path: str, project: str | None, *, units: int = 0) -> uuid.UUID:
    file_id = uuid.uuid4()
    factory = get_session_factory()
    async with factory() as s:
        s.add(File(id=file_id, path=path, project=project))
        for i in range(units):
            s.add(
                Unit(
                    id=uuid.uuid4(),
                    file_id=file_id,
                    kind="doc.paragraph",
                    name=f"u{i}",
                    identity_key=f"{file_id}-{i}",
                )
            )
        await s.commit()
    return file_id


async def _sweep(**kw):
    """Always scoped to this module's paths.

    ``tombstone_untagged_files`` is global by design — an unscoped call here
    would tombstone the developer's own untagged rows as a side effect of
    running the suite.
    """
    return await tombstone_untagged_files(path_prefix=MARK, **kw)


async def _live(file_id: uuid.UUID) -> bool:
    factory = get_session_factory()
    async with factory() as s:
        row = await s.get(File, file_id)
        return row is not None and row.valid_until is None


# ── Partitioning: redundant vs the only copy ────────────────────────────


async def test_an_untagged_duplicate_of_a_tagged_path_is_tombstoned(db):
    await _seed(f"{MARK}/a.md", "prune-test-proj")
    dup = await _seed(f"{MARK}/a.md", None)

    stats = await _sweep()
    assert stats["duplicated"] >= 1
    assert not await _live(dup)


async def test_the_tagged_copy_is_never_touched(db):
    tagged = await _seed(f"{MARK}/a.md", "prune-test-proj")
    await _seed(f"{MARK}/a.md", None)

    await _sweep()
    assert await _live(tagged)


async def test_an_untagged_path_no_project_covers_is_left_alone(db):
    """Dropping this would lose the only index copy of the file."""
    only = await _seed(f"{MARK}/orphan.md", None)

    stats = await _sweep()
    assert stats["unindexed"] >= 1
    assert await _live(only), "the only copy must survive by default"


async def test_include_unindexed_drops_the_only_copy_too(db):
    only = await _seed(f"{MARK}/orphan.md", None)

    await _sweep(include_unindexed=True)
    assert not await _live(only)


async def test_a_tombstoned_tagged_row_does_not_count_as_coverage(db):
    """A dead tagged row is not a copy — the untagged one is still the only one."""
    tagged = await _seed(f"{MARK}/a.md", "prune-test-proj")
    factory = get_session_factory()
    async with factory() as s:
        (await s.get(File, tagged)).valid_until = datetime.now(UTC)
        await s.commit()
    untagged = await _seed(f"{MARK}/a.md", None)

    await _sweep()
    assert await _live(untagged)


# ── Cascade + reversibility ─────────────────────────────────────────────


async def test_units_are_tombstoned_with_their_file(db):
    """Search filters files, but `status` counts units — leave neither behind."""
    await _seed(f"{MARK}/a.md", "prune-test-proj")
    dup = await _seed(f"{MARK}/a.md", None, units=3)

    stats = await _sweep()
    assert stats["units_tombstoned"] >= 3

    factory = get_session_factory()
    async with factory() as s:
        rows = (
            await s.execute(
                text("SELECT count(*) FROM units WHERE file_id = :f AND valid_until IS NULL"),
                {"f": str(dup)},
            )
        ).scalar()
    assert rows == 0


async def test_the_tombstone_is_soft_so_the_row_survives_for_audit(db):
    await _seed(f"{MARK}/a.md", "prune-test-proj")
    dup = await _seed(f"{MARK}/a.md", None)

    await _sweep()
    factory = get_session_factory()
    async with factory() as s:
        assert await s.get(File, dup) is not None


# ── Dry run ─────────────────────────────────────────────────────────────


async def test_dry_run_changes_nothing(db):
    await _seed(f"{MARK}/a.md", "prune-test-proj")
    dup = await _seed(f"{MARK}/a.md", None)

    stats = await _sweep(dry_run=True)
    assert stats["duplicated"] >= 1
    assert stats["files_tombstoned"] == 0
    assert await _live(dup)


async def test_running_twice_is_idempotent(db):
    await _seed(f"{MARK}/a.md", "prune-test-proj")
    await _seed(f"{MARK}/a.md", None)

    first = await _sweep()
    second = await _sweep()
    assert first["files_tombstoned"] >= 1
    assert second["untagged"] < first["untagged"]


# ── CLI surface ─────────────────────────────────────────────────────────


def test_bare_prune_is_still_a_reporting_noop():
    """Installed hooks and scripts call `hafiz prune`; it must not start
    tombstoning anything just because the flag now exists."""
    import json

    from typer.testing import CliRunner

    from hafiz.cli import app

    result = CliRunner().invoke(app, ["prune", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["noop"] is True
    assert payload["reason"] == "handled-on-ingest"


def test_untagged_flag_is_documented_on_the_command():
    from typer.testing import CliRunner

    from hafiz.cli import app

    out = CliRunner().invoke(app, ["prune", "--help"]).output
    assert "--untagged" in out
    assert "--include-unindexed" in out
