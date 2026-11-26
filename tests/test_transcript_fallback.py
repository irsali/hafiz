"""Earlier-conversation search — the compliance guarantees behind the hook fallback.

Transcripts are retention-bounded and user-redactable, so the search that feeds
the per-prompt recall hook must never surface:

* a message from a forgotten (tombstoned) communication,
* a message from a communication past ``retention_until`` (even if not yet swept),
* a message from the session in progress.

DB-backed: the filters live in SQL, so a stub would test nothing.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from hafiz.core.communications import (
    MessageInput,
    append_messages,
    forget_communication,
    search_earlier_messages,
    search_messages,
    upsert_communication,
)
from hafiz.core.database import close_engine


@pytest.fixture(autouse=True)
async def _isolate_engine():
    await close_engine()
    yield
    await close_engine()


def _topic() -> str:
    """Text long enough to be embedded, distinctive enough not to collide across tests."""
    tag = uuid.uuid4().hex[:8]
    return (
        f"Investigating why the zebra-{tag} renewal queue stalls when the quokka-{tag} "
        "scheduler restarts during the nightly rollover; the retry budget is exhausted "
        "before the lease is released and the worker never recovers."
    )


async def _seed(
    text: str,
    *,
    external_id: str | None = None,
    retention_until: datetime | None = None,
    forget: bool = False,
    extra: list[str] | None = None,
    agent: str = "claude-code",
) -> uuid.UUID:
    comm, _ = await upsert_communication(
        agent=agent,
        external_id=external_id or f"ext-{uuid.uuid4().hex}",
        retention_until=retention_until,
    )
    texts = [text, *(extra or [])]
    now = datetime.now(UTC)
    await append_messages(
        comm.id,
        [MessageInput(seq=i, role="user", content=t, ts=now) for i, t in enumerate(texts)],
        embed=True,
    )
    if forget:
        await forget_communication(comm.id)
    return comm.id


def _comm_ids(hits) -> set[str]:
    return {h.message.communication_id for h in hits}


async def _earlier(query: str, **kw):
    kw.setdefault("exclude_external_id", None)
    kw.setdefault("min_score", 0.5)
    kw.setdefault("limit", 10)
    return await search_earlier_messages(query, **kw)


async def test_a_live_earlier_conversation_is_found():
    text = _topic()
    live = await _seed(text)

    hits = await _earlier(text)

    assert str(live) in _comm_ids(hits)
    hit = next(h for h in hits if h.message.communication_id == str(live))
    assert hit.agent == "claude-code" and hit.score >= 0.5


async def test_a_forgotten_conversation_is_never_returned():
    text = _topic()
    forgotten = await _seed(text, forget=True)
    live = await _seed(text)

    hits = await _earlier(text)

    assert str(forgotten) not in _comm_ids(hits)
    assert str(live) in _comm_ids(hits)


async def test_a_conversation_past_retention_is_never_returned_even_if_not_swept():
    text = _topic()
    expired = await _seed(text, retention_until=datetime.now(UTC) - timedelta(days=1))
    live = await _seed(text, retention_until=datetime.now(UTC) + timedelta(days=30))

    hits = await _earlier(text)

    assert str(expired) not in _comm_ids(hits)
    assert str(live) in _comm_ids(hits)


async def test_the_current_session_is_excluded():
    text = _topic()
    current_id = f"current-{uuid.uuid4().hex}"
    current = await _seed(text, external_id=current_id)
    earlier = await _seed(text)

    hits = await _earlier(text, exclude_external_id=current_id)
    assert str(current) not in _comm_ids(hits)
    assert str(earlier) in _comm_ids(hits)

    # Without the exclusion the current session is an ordinary match.
    assert str(current) in _comm_ids(await _earlier(text))


async def test_the_score_floor_is_applied():
    relevant = _topic()
    unrelated = (
        "Recipe notes for the sourdough starter: feed it twice a day with equal parts rye "
        "flour and water, keep it warm, and discard half before every feeding to keep the "
        "acidity down."
    )
    await _seed(relevant, extra=[unrelated])
    scored = {h.message.content: h.score for h in await _earlier(relevant, min_score=0.0)}
    assert scored[relevant] > scored[unrelated]
    floor = (scored[relevant] + scored[unrelated]) / 2

    hits = await _earlier(relevant, min_score=floor)

    contents = {h.message.content for h in hits}
    assert relevant in contents and unrelated not in contents
    assert all(h.score >= floor for h in hits)


async def test_the_limit_is_applied():
    text = _topic()
    await _seed(text, extra=[text + " Variant one.", text + " Variant two."])

    hits = await _earlier(text, limit=2)

    assert len(hits) == 2


async def test_a_zero_limit_returns_nothing_without_searching():
    assert await _earlier(_topic(), limit=0) == []


async def test_recall_search_also_hides_forgotten_and_expired_conversations():
    """The same gap existed in ``hafiz recall --query`` / ``--include-transcripts``."""
    text = _topic()
    forgotten = await _seed(text, forget=True)
    expired = await _seed(text, retention_until=datetime.now(UTC) - timedelta(days=1))
    live = await _seed(text)

    found = {m.communication_id for m, _ in await search_messages(text, limit=20)}

    assert str(live) in found
    assert str(forgotten) not in found and str(expired) not in found
