"""The clustering corpus is separate from the readable candidate list.

`distill` hands a reader raw turns *and* groups turns into themes. Those are
two different questions, and they used to share one cap (`message_limit`,
default 50). Because only a small share of source-layer turns carry a vector
— measured at 8.2% of 46,922 turns over a real 30-day window, from a 30-token
floor plus tool-result suppression, both deliberate — a 50-row slice offered
clustering roughly 4 rows. The observed result was 4 "themes", 3 of them
singletons.

The fix is a second query, not a filter on the first. Filtering the candidate
list to embedded rows would have deleted 46 of 50 *readable* turns to buy
theme coverage, which is a regression: an unembedded turn is perfectly
distillable by reading it, and `DistillBundle.messages` is what carries that
text to the agent. `test_reading_list_still_carries_unembedded_turns` pins
that, because it was very nearly shipped the other way.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from hafiz.core.database import (
    Communication,
    CommunicationMessage,
    close_engine,
    get_session_factory,
)
from hafiz.core.distill import find_distill_candidates

AGENT = "theme-corpus-test"
PROJECT = "theme-corpus-test-project"
DIM = 768

# The suite runs against the real configured store, so an assertion that
# assumes an empty table reads the owner's live transcripts instead. Every
# seeded communication is scoped to PROJECT and every query passes
# `project=PROJECT`, which is the same scoping path production uses.


async def _db_available() -> tuple[bool, str]:
    try:
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text("SELECT 1 FROM communications LIMIT 1"))
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


async def _wipe() -> None:
    factory = get_session_factory()
    async with factory() as s:
        await s.execute(text(f"DELETE FROM communications WHERE agent = '{AGENT}'"))
        await s.commit()


@pytest.fixture
async def db():
    ok, why = await _db_available()
    if not ok:
        pytest.skip(f"database not reachable — {why}")
    await _wipe()
    yield
    await _wipe()
    await close_engine()


def _vec(seed: float) -> list[float]:
    """A unit-ish vector that clusters with its neighbours.

    All seeds sit close together on purpose: the point of these tests is
    *which rows reach the clusterer*, not how well it separates them.
    """
    return [seed] + [0.01] * (DIM - 1)


async def _seed(
    *,
    embedded: int = 0,
    unembedded: int = 0,
    retention_until: datetime | None = None,
    tombstoned: bool = False,
    salient_short: bool = False,
) -> str:
    """One communication with the requested mix of turns. Returns its id."""
    now = datetime.now(UTC)
    factory = get_session_factory()
    async with factory() as s:
        comm = Communication(
            id=uuid.uuid4(),
            agent=AGENT,
            external_id=str(uuid.uuid4()),
            started_at=now - timedelta(hours=1),
            scope_kind="project",
            scope_value=PROJECT,
            retention_until=retention_until,
            valid_until=now - timedelta(minutes=1) if tombstoned else None,
        )
        s.add(comm)
        await s.flush()

        seq = 0
        for i in range(embedded):
            s.add(
                CommunicationMessage(
                    id=uuid.uuid4(),
                    communication_id=comm.id,
                    seq=seq,
                    role="user",
                    content=f"an embedded turn about retention policy number {i}",
                    ts=now - timedelta(minutes=60 - i),
                    embedding=_vec(0.5),
                )
            )
            seq += 1
        for i in range(unembedded):
            # Newest, so a recency-ordered candidate query prefers these.
            s.add(
                CommunicationMessage(
                    id=uuid.uuid4(),
                    communication_id=comm.id,
                    seq=seq,
                    role="assistant",
                    content=f"short echo {i}",
                    ts=now - timedelta(seconds=30 - i),
                    embedding=None,
                )
            )
            seq += 1
        if salient_short:
            s.add(
                CommunicationMessage(
                    id=uuid.uuid4(),
                    communication_id=comm.id,
                    seq=seq,
                    role="user",
                    content="ship it",
                    ts=now - timedelta(minutes=5),
                    marked_salient=True,
                    embedding=_vec(0.5),
                )
            )
        await s.commit()
        return str(comm.id)


def _message_members(bundle) -> list:
    return [m for t in bundle.themes for m in t.members if m.kind == "message"]


@pytest.mark.asyncio
async def test_theme_corpus_is_not_bounded_by_the_reading_limit(db):
    """The whole point: a tiny `message_limit` must not starve clustering.

    This is the regression that mattered. With one shared cap, the 12
    unembedded turns are newer and win the recency ordering, so clustering saw
    almost nothing.
    """
    await _seed(embedded=12, unembedded=12)

    bundle = await find_distill_candidates(
        since=timedelta(days=1),
        project=PROJECT,
        include_transcripts=False,
        message_limit=2,
        theme_corpus_limit=100,
    )

    assert len(bundle.messages) <= 2, "the readable list must still honour its own cap"
    assert len(_message_members(bundle)) == 12, (
        "clustering must see the full embedded corpus, not the 2-row reading slice"
    )


@pytest.mark.asyncio
async def test_reading_list_still_carries_unembedded_turns(db):
    """Guard against the fix I nearly shipped.

    Filtering the candidate query on `embedding IS NOT NULL` would make this
    fail. An unembedded turn is readable and therefore distillable; deleting it
    from the handed-over text to improve theme coverage is a net loss.
    """
    await _seed(embedded=1, unembedded=5)

    bundle = await find_distill_candidates(
        since=timedelta(days=1),
        project=PROJECT,
        include_transcripts=False,
        message_limit=50,
    )

    contents = [m.content for m in bundle.messages]
    assert any(c.startswith("short echo") for c in contents), (
        "unembedded turns must still reach the reader"
    )
    assert bundle.backlog.skipped_unembedded == 5


@pytest.mark.asyncio
async def test_theme_corpus_honours_retention(db):
    """Retention is a stated guarantee and applies to *both* queries.

    The corpus query is a second call to the same function precisely so this
    predicate cannot drift between copies. Distilling an expired turn would
    mint an annotation carrying its content with no retention of its own.
    """
    await _seed(embedded=6, retention_until=datetime.now(UTC) - timedelta(days=1))

    bundle = await find_distill_candidates(
        since=timedelta(days=1),
        project=PROJECT,
        include_transcripts=False,
        theme_corpus_limit=100,
    )

    assert _message_members(bundle) == []
    assert bundle.messages == []


@pytest.mark.asyncio
async def test_theme_corpus_honours_a_tombstone(db):
    """`forget` soft-tombstones a communication; clustering must respect it."""
    await _seed(embedded=6, tombstoned=True)

    bundle = await find_distill_candidates(
        since=timedelta(days=1),
        project=PROJECT,
        include_transcripts=False,
        theme_corpus_limit=100,
    )

    assert _message_members(bundle) == []


@pytest.mark.asyncio
async def test_a_salient_short_turn_still_reaches_clustering(db):
    """`marked_salient` short-circuits the embed policy, so it has a vector.

    That interaction is load-bearing: the corpus query selects on
    `embedding IS NOT NULL`, so if `marked_salient` ever stopped forcing an
    embedding, the most important turns would silently leave the grouping set.
    This fails loudly if that happens.
    """
    await _seed(embedded=2, salient_short=True)

    bundle = await find_distill_candidates(
        since=timedelta(days=1),
        project=PROJECT,
        include_transcripts=False,
        theme_corpus_limit=100,
    )

    contents = [m.content for m in _message_members(bundle)]
    assert "ship it" in contents


@pytest.mark.asyncio
async def test_corpus_limit_is_respected(db):
    """The cap exists because clustering is O(n²) in the corpus size."""
    await _seed(embedded=10)

    bundle = await find_distill_candidates(
        since=timedelta(days=1),
        project=PROJECT,
        include_transcripts=False,
        theme_corpus_limit=4,
    )

    assert len(_message_members(bundle)) == 4


@pytest.mark.asyncio
async def test_the_corpus_limit_is_spent_only_on_embedded_rows(db):
    """The SQL predicate has to do the filtering, not `cluster_candidates`.

    `cluster_candidates` already drops vector-less messages in Python, so
    removing `WHERE embedding IS NOT NULL` from the corpus query leaves every
    other test in this file green — it was verified by mutation and found
    exactly that gap. What the predicate actually protects is the LIMIT
    budget: without it the cap is spent on rows that are then discarded, which
    is the original starvation bug simply relocated to a bigger number.

    Here the 20 unembedded turns are newer, so under the recency ordering they
    would consume a corpus cap of 10 entirely and hand clustering nothing.
    """
    await _seed(embedded=5, unembedded=20)

    bundle = await find_distill_candidates(
        since=timedelta(days=1),
        project=PROJECT,
        include_transcripts=False,
        theme_corpus_limit=10,
    )

    assert len(_message_members(bundle)) == 5, (
        "the corpus cap must be filled with embedded rows, not spent on turns "
        "that clustering will discard anyway"
    )
