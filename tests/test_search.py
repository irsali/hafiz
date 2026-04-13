"""Tests for hafiz.core.search — stubs for integration testing.

These tests require a running PostgreSQL instance with pgvector.
Mark them as integration tests to skip in CI without a DB.
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


@pytest.mark.asyncio
async def test_vector_search_requires_db():
    """vector_search should raise when no DB is available (expected in unit tests)."""
    from hafiz.core.search import vector_search
    from hafiz.core.database import close_engine

    # This should fail without a running DB — that's expected behavior
    with pytest.raises(Exception):
        await vector_search("test query", limit=5)
    await close_engine()
