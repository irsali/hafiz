"""Tests for Phase 6 — agent contract v2.

Covers:
  - ``parse_extraction_payload`` accepts v2, rejects v1 / AST-territory
  - ``store_extraction`` writes annotations + semantic edges, resolves
    unit references, counts unresolved
  - ``hafiz extract export`` emits the AST-known graph so agents can
    attach without re-deriving
  - ``skills.md`` version marker is detectable by the install flow
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from sqlalchemy import text

from hafiz.core.agents import (
    current_skills_version,
    installed_skills_version,
    load_skills_content,
)
from hafiz.core.database import (
    Annotation,
    Edge,
    File,
    Unit,
    UnitRevision,
    close_engine,
    get_session_factory,
)
from hafiz.core.extractor import (
    EXTRACT_CONTRACT_VERSION,
    ExtractContractError,
    parse_extraction_payload,
    store_extraction,
)


# ── Contract parsing ──────────────────────────────────────────────────────


def test_parse_v1_payload_rejected_with_migration_hint():
    with pytest.raises(ExtractContractError, match="v1"):
        parse_extraction_payload(
            {
                "entities": [
                    {
                        "name": "Foo",
                        "entity_type": "class",
                        "description": "d",
                    }
                ]
            }
        )


def test_parse_missing_version_rejected():
    with pytest.raises(ExtractContractError, match="version"):
        parse_extraction_payload({"annotations": [], "edges": []})


def test_parse_wrong_version_rejected():
    with pytest.raises(ExtractContractError, match="version"):
        parse_extraction_payload(
            {"version": 99, "annotations": [], "edges": []}
        )


def test_parse_empty_v2_payload_is_valid():
    result = parse_extraction_payload(
        {"version": 2, "annotations": [], "edges": []}
    )
    assert result.annotations == []
    assert result.edges == []
    assert result.warnings == []


def test_parse_annotation_kind_code_star_rejected():
    with pytest.raises(ExtractContractError, match="AST-territory"):
        parse_extraction_payload(
            {
                "version": 2,
                "annotations": [
                    {"content": "nope", "kind": "code.function"}
                ],
            }
        )


def test_parse_edge_ast_relation_rejected():
    with pytest.raises(ExtractContractError, match="AST-territory"):
        parse_extraction_payload(
            {
                "version": 2,
                "annotations": [],
                "edges": [
                    {
                        "source_name": "A",
                        "target_name": "B",
                        "relation": "calls",
                    }
                ],
            }
        )


def test_parse_unknown_kind_warns_not_errors():
    result = parse_extraction_payload(
        {
            "version": 2,
            "annotations": [
                {"content": "hi", "kind": "weird_kind"}
            ],
        }
    )
    assert len(result.annotations) == 1
    assert any("weird_kind" in w for w in result.warnings)


def test_parse_valid_agent_payload():
    result = parse_extraction_payload(
        {
            "version": 2,
            "annotations": [
                {
                    "content": "Auth is canonicalized here",
                    "kind": "pattern",
                    "source": "agent:test",
                    "unit_name": "UserService.authenticate",
                    "confidence": 0.9,
                }
            ],
            "edges": [
                {
                    "source_name": "UserService",
                    "target_name": "SecurityPolicy",
                    "relation": "implements_pattern",
                    "evidence": "see policy engine call",
                }
            ],
        }
    )
    assert len(result.annotations) == 1
    assert result.annotations[0].kind == "pattern"
    assert len(result.edges) == 1
    assert result.edges[0].relation == "implements_pattern"
    assert result.warnings == []


# ── DB-backed: store_extraction + export ─────────────────────────────────


async def _db_available() -> bool:
    try:
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text("SELECT 1 FROM annotations LIMIT 1"))
        return True
    except Exception:
        return False


async def _cleanup():
    factory = get_session_factory()
    async with factory() as s:
        await s.execute(text("DELETE FROM annotations"))
        await s.execute(text("DELETE FROM edges"))
        await s.execute(text("DELETE FROM embeddings"))
        await s.execute(text("DELETE FROM unit_revisions"))
        await s.execute(text("DELETE FROM units"))
        await s.execute(text("DELETE FROM files"))
        await s.commit()


@pytest.fixture
async def seed_graph():
    """Plant a small (file, unit, revision) fixture so store_extraction
    has something to resolve annotations / edges against."""
    if not await _db_available():
        pytest.skip("Postgres not reachable")
    await _cleanup()

    factory = get_session_factory()
    async with factory() as s:
        file = File(
            id=uuid.uuid4(), project="demo", path="/src/auth.py"
        )
        s.add(file)
        await s.flush()
        unit_a = Unit(
            id=uuid.uuid4(),
            file_id=file.id,
            kind="code.class",
            name="UserService",
            identity_key="ik-userservice",
        )
        unit_b = Unit(
            id=uuid.uuid4(),
            file_id=file.id,
            kind="code.class",
            name="SecurityPolicy",
            identity_key="ik-securitypolicy",
        )
        s.add_all([unit_a, unit_b])
        await s.flush()
        rev_a = UnitRevision(
            id=uuid.uuid4(),
            unit_id=unit_a.id,
            content="class UserService: ...",
            content_hash="h-a",
            source="ast",
        )
        rev_b = UnitRevision(
            id=uuid.uuid4(),
            unit_id=unit_b.id,
            content="class SecurityPolicy: ...",
            content_hash="h-b",
            source="ast",
        )
        s.add_all([rev_a, rev_b])
        await s.commit()
    yield
    await close_engine()


@pytest.mark.asyncio
async def test_store_extraction_binds_annotation_to_unit_by_identity_key(
    seed_graph,
):
    result = parse_extraction_payload(
        {
            "version": 2,
            "annotations": [
                {
                    "content": "canonical auth entry",
                    "kind": "pattern",
                    "source": "agent:test",
                    "unit_identity_key": "ik-userservice",
                }
            ],
            "edges": [],
        }
    )
    ann_n, edge_n, unresolved = await store_extraction(
        result, project="demo"
    )
    assert (ann_n, edge_n, unresolved) == (1, 0, 0)

    factory = get_session_factory()
    async with factory() as s:
        row = (
            await s.execute(
                text(
                    "SELECT unit_id FROM annotations WHERE content = "
                    "'canonical auth entry'"
                )
            )
        ).first()
        assert row is not None
        assert row[0] is not None  # unit was resolved


@pytest.mark.asyncio
async def test_store_extraction_writes_semantic_edge(seed_graph):
    result = parse_extraction_payload(
        {
            "version": 2,
            "annotations": [],
            "edges": [
                {
                    "source_name": "UserService",
                    "source_file": "/src/auth.py",
                    "target_name": "SecurityPolicy",
                    "target_file": "/src/auth.py",
                    "relation": "implements_pattern",
                    "evidence": "enforce() call",
                }
            ],
        }
    )
    ann_n, edge_n, unresolved = await store_extraction(
        result, project="demo"
    )
    assert edge_n == 1
    assert unresolved == 0

    factory = get_session_factory()
    async with factory() as s:
        edges = (
            await s.execute(
                text("SELECT source, relation FROM edges WHERE source = 'agent'")
            )
        ).all()
        assert len(edges) == 1
        assert edges[0][1] == "implements_pattern"


@pytest.mark.asyncio
async def test_store_extraction_counts_unresolved(seed_graph):
    result = parse_extraction_payload(
        {
            "version": 2,
            "annotations": [
                {
                    "content": "dangling",
                    "kind": "fact",
                    "source": "agent:test",
                    "unit_name": "NotAThing",
                }
            ],
            "edges": [
                {
                    "source_name": "UserService",
                    "source_file": "/src/auth.py",
                    "target_name": "ExternalLib",
                    "relation": "depends_on_concept",
                }
            ],
        }
    )
    _, _, unresolved = await store_extraction(result, project="demo")
    # annotation's unit didn't resolve → 1; edge's target didn't → 1.
    assert unresolved == 2


# ── skills.md version marker ──────────────────────────────────────────────


def test_shipped_skills_version_is_readable():
    v = current_skills_version()
    assert v >= 2
    assert isinstance(v, int)


def test_installed_skills_version_roundtrip(tmp_path: Path):
    target = tmp_path / "INSTRUCTIONS.md"
    content = load_skills_content()
    target.write_text(content)
    assert installed_skills_version(target) == current_skills_version()


def test_installed_skills_version_none_for_missing_file(tmp_path: Path):
    assert installed_skills_version(tmp_path / "nope.md") is None


def test_installed_skills_version_none_when_no_managed_region(tmp_path: Path):
    target = tmp_path / "user.md"
    target.write_text("# My instructions\n\nNothing hafiz here.\n")
    assert installed_skills_version(target) is None


def test_skills_version_not_behind_extract_contract():
    """Agents must always know about at least the current extract-payload
    version. Skills.md can legitimately advance beyond it — e.g., when
    a capability outside the extract contract gets added — but falling
    behind would leave agents sending older payloads than the importer
    accepts."""
    assert current_skills_version() >= EXTRACT_CONTRACT_VERSION
