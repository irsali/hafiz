"""Tests for hafiz.core.search against the structural-grounding schema.

The dataclass test is a pure unit test. The vector_search test is an
integration test that runs against a live Postgres when one is available
and skips gracefully otherwise.
"""

import pytest

from hafiz.core.search import _normalize_domains, _validate_domain_filters


def test_normalize_domains_strips_lowercases_dedupes():
    assert _normalize_domains([" Code ", "DOC", "code", ""]) == ["code", "doc"]


def test_normalize_domains_none_and_empty_return_empty():
    assert _normalize_domains(None) == []
    assert _normalize_domains([]) == []
    assert _normalize_domains(["", "  "]) == []


def test_normalize_domains_rejects_dotted_kinds():
    with pytest.raises(ValueError, match="single token"):
        _normalize_domains(["code.function"])


def test_validate_domain_filters_passes_disjoint():
    _validate_domain_filters(["code"], ["doc"])  # no raise


def test_validate_domain_filters_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        _validate_domain_filters(["code", "doc"], ["doc"])


@pytest.mark.asyncio
async def test_search_result_dataclass():
    """SearchResult should be importable and constructable with the new
    fields populated from the joined embeddings/units/files query."""
    from hafiz.core.search import SearchResult

    result = SearchResult(
        id="emb-id",
        unit_id="unit-id",
        unit_name="foo.bar",
        kind="code.function",
        content="def hello(): pass",
        source_file="test.py",
        line_start=1,
        line_end=1,
        language="python",
        project="test-project",
        part_index=0,
        score=0.95,
    )
    assert result.score == 0.95
    assert result.source_file == "test.py"
    assert result.kind == "code.function"
    assert result.unit_name == "foo.bar"


async def _db_available() -> bool:
    """Return True iff a live Postgres with the new schema is reachable."""
    try:
        from sqlalchemy import text

        from hafiz.core.database import get_session_factory

        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(text("SELECT 1 FROM embeddings LIMIT 1"))
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
    """vector_search returns a list of SearchResult against a live DB,
    else skips. Semantic correctness is covered elsewhere; this guards
    the query shape."""
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
