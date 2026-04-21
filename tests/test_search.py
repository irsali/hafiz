"""Tests for hafiz.core.search.

The dataclass test is a pure unit test. The vector_search test is an
integration test that runs against a live Postgres when one is available and
skips gracefully otherwise.
"""

import pytest


@pytest.mark.asyncio
async def test_search_result_dataclass():
    """SearchResult should be importable and constructable."""
    from hafiz.core.search import SearchResult

    result = SearchResult(
        id="test-id",
        content="def hello(): pass",
        source_file="test.py",
        line_start=1,
        line_end=1,
        chunk_type="code",
        language="python",
        project="test-project",
        score=0.95,
        metadata={},
    )
    assert result.score == 0.95
    assert result.source_file == "test.py"


async def _db_available() -> bool:
    """Return True iff a live Postgres with the hafiz schema is reachable."""
    try:
        from sqlalchemy import text

        from hafiz.core.database import get_session_factory

        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(text("SELECT 1 FROM chunks LIMIT 1"))
        return True
    except Exception:
        return False
    finally:
        try:
            from hafiz.core.database import close_engine

            await close_engine()
        except Exception:
            pass


@pytest.mark.asyncio
async def test_vector_search_live_db_returns_list():
    """vector_search returns a list of SearchResult against a live DB, else skips."""
    if not await _db_available():
        pytest.skip("No live Postgres with hafiz schema available")

    from hafiz.core.database import close_engine
    from hafiz.core.search import SearchResult, vector_search

    try:
        results = await vector_search("test query", limit=5)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, SearchResult)
            assert 0.0 <= r.score <= 1.0
    finally:
        await close_engine()
