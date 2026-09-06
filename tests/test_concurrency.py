"""Concurrent writers on the embedded backend — Open Question 5.

Postgres gives every writer its own snapshot and locks at row granularity,
so two Hafiz processes writing at once is a non-event there. SQLite has
**one** write lock for the whole file. That difference is invisible until
something writes from more than one place at a time — which Hafiz now does:
``ingest`` runs in the foreground while capture hooks write transcripts in
the background, and a warm daemon serves a third caller.

The specific hazard this module exists for
------------------------------------------

``index_file`` *used to* flush rows — taking the WAL write lock — and only
then await the embedder before committing. Embedding is ONNX inference:
hundreds of milliseconds to seconds of CPU that needs no database at all.
So the write lock was held across it, and a concurrent writer waited on
``busy_timeout`` (5s) before failing outright with "database is locked".

``index_file`` now embeds before it writes, and these tests are what keeps
it that way. They fail if the phases are ever reordered back.

The consequence is asymmetric. A failed ``ingest`` is loud and retryable.
A failed **capture** is silent: the transcript is dropped and
``retention.overdue`` still reports ``0``, which reads as health. That is
the same false-health signature that let transcript capture die unnoticed
for two months (see ``capture_freshness`` in ``core/freshness.py``).

Why the test is shaped this way
-------------------------------

The obvious test — start two writers, see if anything breaks — is a coin
flip. It passes on a fast machine, and CI is a fast machine. So instead of
racing and hoping, this pins a *structural* invariant:

    **How long the write lock is held must not depend on how long
    embedding takes.**

That is deterministic, it is the property the fix establishes, and it holds
regardless of host speed or the value of ``busy_timeout``. It also fires
well before the hard failure: the second writer gets *slow* long before it
starts getting *refused*, so the test does not have to sit through a 5s
timeout to detect the defect.

Synchronisation is exact rather than timed. The stub embedder signals an
``asyncio.Event`` at the moment it is called — by which point every flush
in ``index_file`` has already happened, so the write lock is provably held.
The second writer starts on that signal. No "sleep a bit and hope".

Measured on this tree, with a 2.0s stub embedder:

    before the fix   second writer blocked 2.046s   (tracks the embedder 1:1)
    after the fix    second writer blocked 0.011s   (independent of it)

Falsification check, because a timing test can easily measure its own
harness instead of its subject: both tests were run unchanged against
Postgres, where MVCC means there is nothing to find. They pass there and
fail on SQLite, so what they detect is the write lock and not, say, an
artefact of ``asyncio.gather`` ordering.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

import pytest
from sqlalchemy import text

from hafiz.core.annotations import store_annotation
from hafiz.core.database import close_engine, get_session_factory
from hafiz.core.dialect import backend_of
from hafiz.core.store import index_file

#: How long the stub embedder pretends to take. Deliberately *under*
#: ``busy_timeout`` (5s): the point is to catch the defect as added latency,
#: before it becomes a hard "database is locked", and to keep the suite fast.
EMBED_HOLD_SECONDS = 2.0

#: The second writer must be essentially unaffected. Half the hold is a wide
#: margin — pre-fix it blocks for the *entire* hold, so the gap between pass
#: and fail is 100x, not a few percent. Nothing here is timing-sensitive
#: enough to flake on a loaded CI box.
MAX_BLOCKED_SECONDS = EMBED_HOLD_SECONDS / 2

PROJECT = "hafiz-concurrency-test"
SOURCE = "agent:test-concurrency"


async def _fake_vector(text_in: str) -> list[float]:
    h = hashlib.sha256(text_in.encode()).digest()
    return [(b - 128) / 128.0 for b in (h * 24)[:768]]


async def _is_embedded() -> bool:
    factory = get_session_factory()
    async with factory() as s:
        return backend_of(s) == "sqlite"


@pytest.fixture(autouse=True)
async def _embedded_only():
    """Run only on the embedded backend.

    The skip reason is prefixed ``backend-specific:`` on purpose. The
    conftest fails the whole run when a test skips for a *database* reason
    under an explicitly named backend, and that guard matches on reason
    text — so a reason mentioning Postgres would turn this legitimate skip
    into a session failure. The prefix is exempted structurally in
    ``conftest._DB_SKIP_EXEMPT_PREFIXES`` rather than by careful wording.
    """
    if not await _is_embedded():
        pytest.skip(
            "backend-specific: a single database-wide write lock has no "
            "analogue under Postgres MVCC"
        )
    yield
    await _cleanup()
    await close_engine()


async def _cleanup():
    factory = get_session_factory()
    async with factory() as s:
        await s.execute(text("DELETE FROM annotations WHERE source = :s"), {"s": SOURCE})
        await s.execute(
            text(
                "DELETE FROM embeddings WHERE unit_revision_id IN ("
                "  SELECT ur.id FROM unit_revisions ur"
                "  JOIN units u ON u.id = ur.unit_id"
                "  JOIN files f ON f.id = u.file_id WHERE f.project = :p)"
            ),
            {"p": PROJECT},
        )
        await s.execute(
            text(
                "DELETE FROM unit_revisions WHERE unit_id IN ("
                "  SELECT u.id FROM units u JOIN files f ON f.id = u.file_id"
                "  WHERE f.project = :p)"
            ),
            {"p": PROJECT},
        )
        await s.execute(
            text("DELETE FROM units WHERE file_id IN (SELECT id FROM files WHERE project = :p)"),
            {"p": PROJECT},
        )
        await s.execute(text("DELETE FROM files WHERE project = :p"), {"p": PROJECT})
        await s.commit()


def _source_file(tmp_path: Path) -> tuple[Path, str]:
    """A file with enough units that the embedder is given real work."""
    body = "\n\n".join(f"def unit_{i}():\n    return {i}" for i in range(12))
    path = tmp_path / "contended.py"
    path.write_text(body, encoding="utf-8")
    return path, body


async def test_write_lock_is_not_held_across_embedding(monkeypatch, tmp_path):
    """A capture firing mid-ingest must not wait on the embedder.

    Writer A indexes a file with a stub embedder that takes
    ``EMBED_HOLD_SECONDS``. Writer B — a real ``store_annotation``, the same
    call a capture hook makes — starts the instant A's embedder is entered,
    which is after A's flushes and therefore after A holds the write lock.

    If embedding sits inside the write transaction, B blocks for the whole
    hold. If it sits outside, B is unaffected. Nothing else in the two paths
    differs, so the elapsed time of B isolates exactly that question.
    """
    monkeypatch.setattr("hafiz.core.annotations.embed_query", _fake_vector)

    path, body = _source_file(tmp_path)
    embedder_entered = asyncio.Event()

    async def slow_embed(texts: list[str]) -> list[list[float]]:
        # Signalled from *inside* the embedder, which is the exact window
        # under test: whatever locks index_file holds while embedding, it
        # holds them now. Releasing writer B here rather than after a timed
        # sleep is what makes the result deterministic instead of a race.
        embedder_entered.set()
        await asyncio.sleep(EMBED_HOLD_SECONDS)
        return [await _fake_vector(t) for t in texts]

    async def writer_a():
        # Mirrors hafiz/commands/ingest.py exactly: one file, one
        # transaction, the caller owns the commit.
        factory = get_session_factory()
        async with factory() as s:
            async with s.begin():
                await index_file(path, body, project=PROJECT, session=s, embed_fn=slow_embed)

    async def writer_b() -> float:
        await embedder_entered.wait()
        started = time.perf_counter()
        await store_annotation(
            "a capture that fired while an ingest was running",
            kind="note",
            source=SOURCE,
            project=PROJECT,
        )
        return time.perf_counter() - started

    _indexed, blocked_for = await asyncio.gather(writer_a(), writer_b())

    assert blocked_for < MAX_BLOCKED_SECONDS, (
        f"A concurrent write blocked for {blocked_for:.2f}s while an ingest was "
        f"embedding for {EMBED_HOLD_SECONDS:.1f}s. The SQLite write lock is being "
        f"held across the embedder call, so lock-hold time scales with model "
        f"latency instead of with database work.\n\n"
        f"On a slower host, or a larger file, that crosses busy_timeout (5s) and "
        f"the second writer stops merely being slow and starts failing with "
        f"'database is locked'. When the second writer is a capture hook, that "
        f"loss is silent.\n\n"
        f"Fix: in index_file, compute embeddings before the write transaction "
        f"opens, not inside it."
    )


async def test_the_second_writer_actually_lands(monkeypatch, tmp_path):
    """Guard the guard: prove writer B does real, contended work.

    Without this, the timing test above could pass for the wrong reason —
    a writer that silently no-ops, or one that never reaches the database,
    is also very fast. So assert both halves actually committed while
    overlapping: the annotation exists, and so does the indexed file.
    """
    monkeypatch.setattr("hafiz.core.annotations.embed_query", _fake_vector)

    path, body = _source_file(tmp_path)
    embedder_entered = asyncio.Event()
    a_committed = False
    b_finished_before_a_committed = False

    async def slow_embed(texts: list[str]) -> list[list[float]]:
        embedder_entered.set()
        await asyncio.sleep(EMBED_HOLD_SECONDS)
        return [await _fake_vector(t) for t in texts]

    async def writer_a():
        nonlocal a_committed
        factory = get_session_factory()
        async with factory() as s:
            async with s.begin():
                await index_file(path, body, project=PROJECT, session=s, embed_fn=slow_embed)
        a_committed = True

    async def writer_b():
        nonlocal b_finished_before_a_committed
        await embedder_entered.wait()
        await store_annotation(
            "second writer landing check", kind="note", source=SOURCE, project=PROJECT
        )
        # A sleeps for EMBED_HOLD_SECONDS after this point, so a B that
        # committed here provably did so with A's transaction still open.
        b_finished_before_a_committed = not a_committed

    await asyncio.gather(writer_a(), writer_b())

    factory = get_session_factory()
    async with factory() as s:
        annotations = (
            await s.execute(
                text("SELECT COUNT(*) FROM annotations WHERE source = :s"), {"s": SOURCE}
            )
        ).scalar()
        files = (
            await s.execute(text("SELECT COUNT(*) FROM files WHERE project = :p"), {"p": PROJECT})
        ).scalar()

    assert annotations == 1, "the concurrent writer did not commit its annotation"
    assert files == 1, "the ingest side did not commit its file"
    assert b_finished_before_a_committed, (
        "the two writers did not overlap, so the timing test above proves nothing "
        "about contention — it would pass even with a global lock held all run"
    )
