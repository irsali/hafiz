"""End-to-end tests for the structural-grounding ingest pipeline.

Uses a deterministic mock embedder so these tests don't depend on GPU
state or embedding-model availability — what they validate is that the
store/parse/supersession plumbing is correct.

Covers the load-bearing guarantees from Phase 3 of the work item:
  - First ingest creates units + revisions + embeddings.
  - Re-ingesting the same content is a no-op (idempotency).
  - Changing a unit's body adds exactly one new revision and supersedes
    exactly one old revision. Unchanged units are untouched.
  - Partial edit (one paragraph of a ten-paragraph document) re-embeds
    only the affected embedding part, not all of them.
  - Deleting a unit tombstones it (valid_until set; current revision
    superseded) without touching still-present units.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from sqlalchemy import select, text

from hafiz.core.database import (
    Embedding,
    File,
    Unit,
    UnitRevision,
    close_engine,
    get_session_factory,
)
from hafiz.core.store import index_file

# ── Mock embedder ──────────────────────────────────────────────────────


async def mock_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic 768-dim vectors derived from text hashes. Collisions
    are fine — these tests don't assert similarity values."""
    out: list[list[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode()).digest()
        # Repeat the 32-byte hash to fill 768 floats, scaled to [-1, 1].
        v = [(b - 128) / 128.0 for b in (h * 24)[:768]]
        out.append(v)
    return out


# ── Fixtures ────────────────────────────────────────────────────────────


async def _db_available() -> bool:
    try:
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text("SELECT 1 FROM units LIMIT 1"))
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
        await s.execute(text("DELETE FROM commits"))
        await s.commit()


@pytest.fixture(autouse=True)
async def _skip_and_clean():
    if not await _db_available():
        pytest.skip("Postgres not reachable")
    await _cleanup()
    yield
    await close_engine()


# ── Small helpers ──────────────────────────────────────────────────────


async def _count(table: str, *, where: str | None = None) -> int:
    factory = get_session_factory()
    async with factory() as s:
        q = f"SELECT COUNT(*) FROM {table}"
        if where:
            q += f" WHERE {where}"
        return (await s.execute(text(q))).scalar() or 0


def _py(body: str) -> tuple[Path, str]:
    return Path("/tmp/fake.py"), body


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_ingest_creates_units_revisions_embeddings(tmp_path):
    abs_path, content = _py(
        "def hello():\n    return 'world'\n\nclass Foo:\n    def bar(self):\n        return 1\n"
    )
    result = await index_file(
        abs_path,
        content,
        project="t",
        embed_fn=mock_embed,
    )

    # code.module + code.function + code.class + code.method = 4 units
    assert result.units_seen == 4
    assert result.revisions_created == 4
    assert result.embeddings_written >= 4  # ≥1 embedding per revision
    assert result.units_tombstoned == 0

    assert await _count("files") == 1
    assert await _count("units") == 4
    assert await _count("unit_revisions") == 4
    assert await _count("unit_revisions", where="superseded_at IS NULL") == 4


@pytest.mark.asyncio
async def test_reingesting_unchanged_content_is_noop():
    abs_path, content = _py("def foo():\n    return 1\n")

    first = await index_file(abs_path, content, project="t", embed_fn=mock_embed)
    second = await index_file(abs_path, content, project="t", embed_fn=mock_embed)

    assert first.revisions_created == 2  # module + function
    assert second.revisions_created == 0
    assert second.embeddings_written == 0
    assert second.units_tombstoned == 0

    # Still exactly one current revision per unit.
    assert await _count("units") == 2
    assert await _count("unit_revisions", where="superseded_at IS NULL") == 2
    assert await _count("unit_revisions") == 2


@pytest.mark.asyncio
async def test_changing_one_function_body_supersedes_exactly_one_revision():
    abs_path = Path("/tmp/mod.py")
    v1 = "def stable():\n    return 'unchanged'\n\ndef mutating():\n    return 1\n"
    v2 = (
        "def stable():\n    return 'unchanged'\n\n"
        "def mutating():\n    return 2\n"  # body changed
    )

    await index_file(abs_path, v1, project="t", embed_fn=mock_embed)
    result = await index_file(abs_path, v2, project="t", embed_fn=mock_embed)

    # mutating() changed. module's content also changed (file content
    # differs), so module revision bumps too. stable() unchanged.
    assert result.revisions_created == 2  # mutating + module
    assert result.units_tombstoned == 0

    # One current revision per unit (3 units: module, stable, mutating).
    assert await _count("unit_revisions", where="superseded_at IS NULL") == 3
    # Total revisions = 3 original + 2 new = 5.
    assert await _count("unit_revisions") == 5
    # Exactly 2 revisions were superseded (module v1 and mutating v1).
    assert await _count("unit_revisions", where="superseded_at IS NOT NULL") == 2


@pytest.mark.asyncio
async def test_deleted_unit_is_tombstoned():
    abs_path = Path("/tmp/mod.py")
    v1 = "def keeper():\n    return 1\n\ndef goner():\n    return 2\n"
    v2 = "def keeper():\n    return 1\n"  # goner removed

    await index_file(abs_path, v1, project="t", embed_fn=mock_embed)
    result = await index_file(abs_path, v2, project="t", embed_fn=mock_embed)

    # module body changed → new revision. goner vanished → tombstoned.
    # keeper unchanged → no new revision.
    assert result.units_tombstoned == 1

    # goner's unit still exists but tombstoned.
    factory = get_session_factory()
    async with factory() as s:
        stmt = select(Unit).where(Unit.name == "goner")
        goner = (await s.execute(stmt)).scalar_one()
        assert goner.valid_until is not None

        # goner's current revision was superseded.
        stmt = select(UnitRevision).where(UnitRevision.unit_id == goner.id)
        revs = (await s.execute(stmt)).scalars().all()
        assert len(revs) == 1
        assert revs[0].superseded_at is not None


@pytest.mark.asyncio
async def test_partial_edit_in_long_doc_reembeds_only_changed_parts():
    """A Markdown document larger than the per-part max should split
    into multiple embedding parts; editing one section should re-embed
    only the parts overlapping that section."""
    abs_path = Path("/tmp/long.md")

    # Build a 10-section doc well over the default 24k-char part cap.
    sections = [f"## Section {i}\n\n" + ("x" * 4000) + "\n" for i in range(10)]
    v1 = "# Doc\n\n" + "".join(sections)

    first = await index_file(abs_path, v1, project="t", embed_fn=mock_embed)
    first_embeddings = await _count("embeddings")
    assert first.embeddings_written == first_embeddings
    assert first_embeddings > 1  # doc should split across multiple parts

    # Edit only section 7 — a small change deep inside.
    sections[7] = "## Section 7\n\n" + ("y" * 4000) + "\n"
    v2 = "# Doc\n\n" + "".join(sections)

    second = await index_file(abs_path, v2, project="t", embed_fn=mock_embed)

    # Changed units: the root `# Doc` heading unit (which covers the whole
    # file) and the `Section 7` heading + its paragraph. Other sections'
    # units are unchanged.
    # We're validating the *mechanism*: fewer revisions than total units.
    total_units = await _count("units")
    assert 0 < second.revisions_created < total_units
    # The partial-edit guarantee: re-embedding count tracks changed
    # revisions, not the whole doc.
    assert second.embeddings_written < first_embeddings


@pytest.mark.asyncio
async def test_embedding_cascade_delete_on_revision_removal():
    """Removing a unit_revision cascades-deletes its embedding rows —
    required for supersession cleanup downstream."""
    abs_path = Path("/tmp/tiny.py")
    await index_file(abs_path, "x = 1\n", project="t", embed_fn=mock_embed)

    factory = get_session_factory()
    async with factory() as s:
        rev = (await s.execute(select(UnitRevision).limit(1))).scalar_one()
        emb_count_before = (
            await s.execute(
                select(text("COUNT(*)"))
                .select_from(Embedding)
                .where(Embedding.unit_revision_id == rev.id)
            )
        ).scalar()
        assert emb_count_before >= 1

        await s.delete(rev)
        await s.commit()

    assert await _count("embeddings", where=f"unit_revision_id = '{rev.id}'") == 0


@pytest.mark.asyncio
async def test_file_unique_project_path_lets_distinct_projects_share_path():
    """The same absolute path can live under multiple projects; the
    UNIQUE is (project, path), not path alone."""
    abs_path = Path("/tmp/shared.py")
    await index_file(abs_path, "x = 1\n", project="a", embed_fn=mock_embed)
    await index_file(abs_path, "x = 1\n", project="b", embed_fn=mock_embed)

    factory = get_session_factory()
    async with factory() as s:
        files = (await s.execute(select(File))).scalars().all()
        assert len(files) == 2
        assert {f.project for f in files} == {"a", "b"}


@pytest.mark.asyncio
async def test_every_embedding_belongs_to_its_own_revision_body():
    """An embedding part must be a slice of the revision it hangs off.

    Sounds tautological; is not. ``identity_key`` is a hash of
    (project, path, kind, name, parent_name) and is therefore **not unique
    within a file** — two same-named functions, or the far more common case
    of two identical markdown headings like ``## Notes``, collide. Any
    bookkeeping in ``index_file`` that keys per-unit work by identity_key
    silently gives the first occurrence the second occurrence's data.

    Caught exactly that while moving embedding out of the write transaction
    (see tests/test_concurrency.py): the ``pending`` map was keyed by
    identity_key, so the superseded revision was stored with the surviving
    revision's vector. Nothing else in the suite noticed, because every
    other fixture uses distinct unit names.
    """
    abs_path = Path("/tmp/dupe_headings.md")
    body = "## Notes\n\nfirst body here\n\n## Notes\n\nsecond body here\n"
    await index_file(abs_path, body, project="dupes", embed_fn=mock_embed)

    factory = get_session_factory()
    async with factory() as s:
        revisions = (
            (
                await s.execute(
                    select(UnitRevision)
                    .join(Unit, Unit.id == UnitRevision.unit_id)
                    .join(File, File.id == Unit.file_id)
                    .where(File.project == "dupes")
                )
            )
            .scalars()
            .all()
        )
        assert revisions, "the fixture produced no revisions"

        # Two headings sharing an identity means one Unit and two revisions:
        # the second supersedes the first. Both must carry their own text.
        for rev in revisions:
            parts = (
                (
                    await s.execute(
                        select(Embedding)
                        .where(Embedding.unit_revision_id == rev.id)
                        .order_by(Embedding.part_index)
                    )
                )
                .scalars()
                .all()
            )
            assert parts, f"revision {rev.id} has no embedding"
            joined = "".join(p.content for p in parts)
            assert joined in rev.content or rev.content in joined, (
                f"revision body {rev.content!r} does not match its embedded "
                f"text {joined!r} — embedding bookkeeping crossed revisions"
            )
