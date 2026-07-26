"""Tests for exact-content duplicate handling on annotation writes.

Surface-don't-block is the right design for *near*-duplicates: only the author
can judge whether a similar row refines, contradicts, or merely resembles the
old one. An **exact** match on the same kind + source + project admits no such
judgement — it is never a legitimate new annotation. Unenforced, it accumulated:
a real store reached 34 near-duplicate clusters over 80 live annotations,
several character-for-character identical.

Two lanes, deliberately different:

* ``observe`` **refuses** (exit 2, ``existing_id`` returned). The caller may
  have meant ``--supersedes``, and a non-zero exit is what makes them look.
* ``note`` **succeeds idempotently** (``deduped: true``, exit 0, same id, no new
  row). "Raw capture is never gated" protects the caller from friction; it does
  not entitle the store to keep identical rows.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from hafiz.core.annotations import (
    DuplicateAnnotationError,
    ExactDuplicateAnnotationError,
    find_exact_duplicate,
    store_annotation,
    store_annotation_checked,
)
from hafiz.core.database import close_engine, get_session_factory

PROJECT = "exactdup-test"


async def _db_available() -> bool:
    try:
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text("SELECT 1 FROM annotations LIMIT 1"))
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
async def db():
    if not await _db_available():
        pytest.skip("Postgres not reachable")

    async def _wipe():
        factory = get_session_factory()
        async with factory() as s:
            await s.execute(text(f"DELETE FROM annotations WHERE project = '{PROJECT}'"))
            await s.commit()

    await _wipe()
    yield
    await _wipe()
    await close_engine()


# ── find_exact_duplicate ────────────────────────────────────────────────


async def test_finds_an_identical_live_row():
    await store_annotation("same text", kind="decision", source="agent:x", project=PROJECT)
    found = await find_exact_duplicate(
        "same text", kind="decision", source="agent:x", project=PROJECT
    )
    assert found is not None


async def test_different_text_is_not_a_duplicate():
    await store_annotation("text A", kind="decision", source="agent:x", project=PROJECT)
    assert (
        await find_exact_duplicate("text B", kind="decision", source="agent:x", project=PROJECT)
        is None
    )


async def test_whitespace_difference_is_not_an_exact_duplicate():
    """Exact means exact — anything fuzzier is the near-duplicate check's job."""
    await store_annotation("same text", kind="decision", source="agent:x", project=PROJECT)
    assert (
        await find_exact_duplicate("same  text", kind="decision", source="agent:x", project=PROJECT)
        is None
    )


async def test_different_kind_is_not_a_duplicate():
    """The same sentence can be both a decision and a warning."""
    await store_annotation("same text", kind="decision", source="agent:x", project=PROJECT)
    assert (
        await find_exact_duplicate("same text", kind="warning", source="agent:x", project=PROJECT)
        is None
    )


async def test_different_source_is_not_a_duplicate():
    """Two authors independently asserting the same thing is signal, not noise."""
    await store_annotation("same text", kind="decision", source="agent:x", project=PROJECT)
    assert (
        await find_exact_duplicate(
            "same text", kind="decision", source="user:anjum", project=PROJECT
        )
        is None
    )


async def test_null_source_matches_null_source():
    """NULL == NULL must count, or an untagged rewrite slips through."""
    await store_annotation("same text", kind="decision", source=None, project=PROJECT)
    found = await find_exact_duplicate("same text", kind="decision", source=None, project=PROJECT)
    assert found is not None


async def test_null_source_does_not_match_a_named_source():
    await store_annotation("same text", kind="decision", source=None, project=PROJECT)
    assert (
        await find_exact_duplicate("same text", kind="decision", source="agent:x", project=PROJECT)
        is None
    )


async def test_superseded_rows_do_not_count():
    """Re-stating a retired belief is a new assertion, not a duplicate."""
    await store_annotation(
        "same text",
        kind="decision",
        source="agent:x",
        project=PROJECT,
        valid_until=datetime.now(UTC) - timedelta(days=1),
    )
    assert (
        await find_exact_duplicate("same text", kind="decision", source="agent:x", project=PROJECT)
        is None
    )


# ── observe lane: refuse ────────────────────────────────────────────────


async def test_observe_refuses_an_exact_duplicate():
    first = await store_annotation_checked(
        "decision text", kind="decision", source="agent:x", project=PROJECT
    )
    with pytest.raises(ExactDuplicateAnnotationError) as exc:
        await store_annotation_checked(
            "decision text", kind="decision", source="agent:x", project=PROJECT
        )
    assert exc.value.existing_id == str(first.annotation.id)


async def test_refusal_writes_nothing():
    await store_annotation_checked(
        "decision text", kind="decision", source="agent:x", project=PROJECT
    )
    with pytest.raises(ExactDuplicateAnnotationError):
        await store_annotation_checked(
            "decision text", kind="decision", source="agent:x", project=PROJECT
        )
    factory = get_session_factory()
    async with factory() as s:
        count = (
            await s.execute(text(f"SELECT count(*) FROM annotations WHERE project = '{PROJECT}'"))
        ).scalar()
    assert count == 1


async def test_refusal_is_not_governed_by_strict_mode(monkeypatch):
    """Exact duplication is refused regardless of `dedup.strict`.

    `strict` governs the judgement call on *near* duplicates. There is no
    judgement to make about a byte-identical row.
    """
    from hafiz.core import config

    settings = config.load_settings()
    settings.dedup.strict = False
    monkeypatch.setattr(config, "load_settings", lambda: settings)

    await store_annotation_checked(
        "decision text", kind="decision", source="agent:x", project=PROJECT
    )
    with pytest.raises(ExactDuplicateAnnotationError):
        await store_annotation_checked(
            "decision text", kind="decision", source="agent:x", project=PROJECT
        )


async def test_allow_duplicate_forces_the_write():
    await store_annotation_checked(
        "decision text", kind="decision", source="agent:x", project=PROJECT
    )
    result = await store_annotation_checked(
        "decision text",
        kind="decision",
        source="agent:x",
        project=PROJECT,
        allow_duplicate=True,
    )
    assert result.deduped is False


async def test_supersedes_bypasses_the_exact_check():
    """Superseding with identical text is odd but the conflict is already
    resolved explicitly, so the check must not stand in the way."""
    first = await store_annotation_checked(
        "decision text", kind="decision", source="agent:x", project=PROJECT
    )
    result = await store_annotation_checked(
        "decision text",
        kind="decision",
        source="agent:x",
        project=PROJECT,
        supersedes_id=str(first.annotation.id),
    )
    assert result.annotation.id != first.annotation.id


async def test_error_message_names_all_three_ways_out():
    err = ExactDuplicateAnnotationError("abc-123", "decision")
    message = str(err)
    assert "supersede" in message
    assert "edit" in message
    assert "--allow-duplicate" in message


# ── note lane: idempotent success ───────────────────────────────────────


async def test_note_returns_the_existing_row_instead_of_erroring():
    first = await store_annotation_checked(
        "note text",
        kind="note",
        source="agent:x",
        project=PROJECT,
        detect_near=False,
        dedupe_silently=True,
    )
    again = await store_annotation_checked(
        "note text",
        kind="note",
        source="agent:x",
        project=PROJECT,
        detect_near=False,
        dedupe_silently=True,
    )
    assert again.deduped is True
    assert again.annotation.id == first.annotation.id


async def test_note_dedup_writes_no_second_row():
    for _ in range(3):
        await store_annotation_checked(
            "note text",
            kind="note",
            source="agent:x",
            project=PROJECT,
            detect_near=False,
            dedupe_silently=True,
        )
    factory = get_session_factory()
    async with factory() as s:
        count = (
            await s.execute(text(f"SELECT count(*) FROM annotations WHERE project = '{PROJECT}'"))
        ).scalar()
    assert count == 1


async def test_note_lane_still_skips_near_duplicate_detection():
    """The firehose contract: a *similar* note is never gated or surfaced."""
    await store_annotation_checked(
        "wondering if refresh tokens should live in httponly cookies",
        kind="note",
        source="agent:x",
        project=PROJECT,
        detect_near=False,
        dedupe_silently=True,
    )
    result = await store_annotation_checked(
        "wondering whether refresh tokens ought to live in httponly cookies",
        kind="note",
        source="agent:x",
        project=PROJECT,
        detect_near=False,
        dedupe_silently=True,
    )
    assert result.deduped is False
    assert result.near_duplicates == []


async def test_a_fresh_note_is_not_flagged_as_deduped():
    result = await store_annotation_checked(
        f"unique note {uuid.uuid4()}",
        kind="note",
        source="agent:x",
        project=PROJECT,
        detect_near=False,
        dedupe_silently=True,
    )
    assert result.deduped is False


# ── near-duplicate behaviour is unchanged ───────────────────────────────


async def test_near_duplicates_still_only_surface_by_default(monkeypatch):
    from hafiz.core import config

    settings = config.load_settings()
    settings.dedup.strict = False
    monkeypatch.setattr(config, "load_settings", lambda: settings)

    await store_annotation_checked(
        "we decided to store consent receipts in Postgres with a 13-month TTL",
        kind="decision",
        source="agent:x",
        project=PROJECT,
    )
    result = await store_annotation_checked(
        "we decided to store consent receipts in Postgres with a 13 month TTL",
        kind="decision",
        source="agent:x",
        project=PROJECT,
    )
    # Written despite being near-identical; the match rides back for display.
    assert result.deduped is False
    assert result.annotation is not None


async def test_strict_mode_still_blocks_near_duplicates(monkeypatch):
    from hafiz.core import config

    settings = config.load_settings()
    settings.dedup.strict = True
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    try:
        await store_annotation_checked(
            "we decided to store consent receipts in Postgres with a 13-month TTL",
            kind="decision",
            source="agent:x",
            project=PROJECT,
        )
        with pytest.raises(DuplicateAnnotationError):
            await store_annotation_checked(
                "we decided to store consent receipts in Postgres with a 13 month TTL",
                kind="decision",
                source="agent:x",
                project=PROJECT,
            )
    finally:
        settings.dedup.strict = False
