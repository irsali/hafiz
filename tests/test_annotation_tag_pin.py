"""Tests for `--tags` — addressing a curated set instead of ranking for one.

The feature exists because "the rules that always apply" is not a semantic
property. Measured on a 1,528-row store, the best hand-tuned query for a set of
topic-independent hard rules recovered 2 of ~6; the rest sat below any usable
floor. No query text retrieves a set that curation defines, so the set has to be
addressable by tag.

The load-bearing guarantee is **where** the filter runs. It is a `WHERE` clause
on the same SELECT as the vector scan, so `LIMIT` applies to the *filtered* rows.
The plausible regression — filter in Python after truncation — leaves every
CLI-level test green (the flag parses, the guard fires, the help text is right)
while silently making the documented behaviour false: a pin over a rarely-matched
tag would come back empty because the untagged neighbours ate the limit. So the
test that matters seeds decoys that are *closer to the query than the pinned
rows* and asserts the pin still fills the limit.

Hits a real Postgres and cleans up after itself; skips when one isn't reachable.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from hafiz.core.annotations import search_annotations, store_annotation
from hafiz.core.database import close_engine, get_session_factory

# Distinctive so cleanup can't touch a developer's real rows.
MARK = "tagpintest"
TAG_A = f"{MARK}-alpha"
TAG_B = f"{MARK}-beta"


async def _wipe() -> None:
    factory = get_session_factory()
    async with factory() as s:
        await s.execute(text("DELETE FROM annotations WHERE content LIKE :p"), {"p": f"{MARK}%"})
        await s.commit()


@pytest.fixture
async def seeded():
    """Two pinned rows on an off-query topic, plus decoys that match the query.

    The decoys are deliberately the better *semantic* answer. That's the point:
    a pin has to win on membership, not on similarity.
    """
    try:
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text("SELECT 1 FROM annotations LIMIT 1"))
    except Exception:
        pytest.skip("Postgres not reachable")

    await _wipe()
    for i in range(2):
        await store_annotation(
            f"{MARK} pinned rule {i}: commit messages carry no AI attribution",
            kind="learning",
            source="user:test",
            tags=[TAG_A],
        )
    await store_annotation(
        f"{MARK} pinned rule 2: run the test suite locally before proposing a deploy",
        kind="learning",
        source="user:test",
        tags=[TAG_A, TAG_B],
    )
    for i in range(8):
        await store_annotation(
            f"{MARK} decoy {i}: vector search reranking and embedding similarity thresholds",
            kind="learning",
            source="user:test",
        )
    yield
    await _wipe()
    await close_engine()


# The query is about the decoys, never about the pinned rows.
QUERY = "vector search reranking embedding similarity"


async def test_only_tagged_rows_come_back(seeded):
    got = await search_annotations(QUERY, tags=[TAG_A], limit=50, telemetry_command=None)
    assert len(got) == 3
    assert all(TAG_A in (r.tags or []) for r in got)
    assert not any("decoy" in r.content for r in got)


async def test_the_filter_runs_before_the_limit(seeded):
    """The regression this file exists for.

    Eight decoys out-rank the three pinned rows on this query. If the tag filter
    were applied in Python after `LIMIT 3`, the limit would be spent on decoys
    and this would return zero rows. It returns three.
    """
    got = await search_annotations(QUERY, tags=[TAG_A], limit=3, telemetry_command=None)
    assert len(got) == 3
    assert all(TAG_A in (r.tags or []) for r in got)


async def test_the_limit_still_truncates_the_pin(seeded):
    """`--tags` chooses what's eligible, not what survives — and says nothing
    when it cuts. Documented, so pinned here: callers must size `--limit` above
    the tagged set."""
    got = await search_annotations(QUERY, tags=[TAG_A], limit=2, telemetry_command=None)
    assert len(got) == 2  # silently short of the 3 that carry the tag


async def test_tags_are_or_not_and(seeded):
    """Array overlap (`&&`), not containment: one matching tag qualifies."""
    only_b = await search_annotations(QUERY, tags=[TAG_B], limit=50, telemetry_command=None)
    assert len(only_b) == 1
    either = await search_annotations(QUERY, tags=[TAG_A, TAG_B], limit=50, telemetry_command=None)
    assert len(either) == 3  # union, not intersection


async def test_an_unknown_tag_matches_nothing(seeded):
    got = await search_annotations(
        QUERY, tags=[f"{MARK}-nonexistent"], limit=50, telemetry_command=None
    )
    assert got == []


async def test_untagged_rows_are_excluded_not_treated_as_wildcard(seeded):
    """`tags IS NULL` yields NULL under `&&`, so untagged rows drop out. Worth
    pinning: the opposite (NULL as "matches anything") would make every pin a
    plain search wearing a filter's name."""
    got = await search_annotations(QUERY, tags=[TAG_A], limit=50, telemetry_command=None)
    assert all(r.tags for r in got)


async def test_no_tags_argument_searches_everything(seeded):
    """Absent `tags` must not narrow anything — the decoys are reachable."""
    got = await search_annotations(QUERY, limit=50, telemetry_command=None)
    assert any("decoy" in r.content for r in got)
