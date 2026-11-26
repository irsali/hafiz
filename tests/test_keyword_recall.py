"""The keyword channel: exact strings an embedding smears, found by full-text.

What these pin, in order of how badly a regression would hurt:

1. **Filter parity.** A keyword hit must never leak a row the vector path would
   have filtered out — a retired belief, another project's note, a row the
   caller's tag/kind/source filter excludes. The channel is a second way *in*,
   and a second way in that skips the checks is a data-leak, not a ranking
   quirk. Each exclusion is tested through the public ``search_annotations`` so
   the assertion is on what a caller sees.
2. **The channel earns its keep.** An exact token buried among on-topic decoys
   is found with the channel on and missed with it off.
3. **One tokenizer on both backends.** ``agents.py``, ``5432``, ``ERR_X`` and
   ``--min-score`` must match however the stored text punctuated them, on
   Postgres and on SQLite alike.
4. **The index stays in sync and survives its migration** — triggers on SQLite,
   a generated column on Postgres; up and down.

Runs on whichever backend ``HAFIZ_TEST_BACKEND`` selects; the SQLite-specific
tests build their own store and run either way.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select, text

from hafiz.core.annotations import (
    _keyword_rows,
    invalidate_annotation,
    search_annotations,
    store_annotation,
)
from hafiz.core.database import Annotation, close_engine, get_session_factory
from hafiz.core.dialect import (
    POSTGRESQL,
    SQLITE,
    backend_of,
    keyword_query_text,
    similarity,
)
from hafiz.core.keyword import MAX_TERMS, match_terms

MARK = "kwrecalltest"
#: A token no embedding model has an opinion about.
TOKEN = "zqx7-4471"
PROBE_VECTOR = [0.1] * 768


# ---------------------------------------------------------------------------
# Term extraction — pure, no database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("what does agents.py do", [("agents", "py")]),
        ("is port 5432 open", [("port",), ("5432",), ("open",)]),
        ("where is ERR_X raised", [("err", "x"), ("raised",)]),
        ("what does --min-score filter", [("min", "score"), ("filter",)]),
        # A path also yields its last segment: a note says "agents.py", a
        # question quotes the path.
        (
            "see src/hafiz/core/agents.py",
            [("see",), ("src", "hafiz", "core", "agents", "py"), ("agents", "py")],
        ),
        # Sentence punctuation falls away; punctuation inside a chunk stays.
        ("ECONNREFUSED?", [("econnrefused",)]),
    ],
)
def test_terms_keep_exact_strings_together(prompt, expected):
    assert match_terms(prompt) == expected


def test_stopwords_and_single_letters_are_not_looked_up():
    assert match_terms("what is the a of it x") == []
    # ...but a single letter inside a phrase is part of the exact string.
    assert ("err", "x") in match_terms("ERR_X")


def test_terms_are_deduped_and_capped():
    assert match_terms("port port PORT") == [("port",)]
    many = " ".join(f"word{n}" for n in range(200))
    assert len(match_terms(many)) == MAX_TERMS


def test_a_blob_is_not_a_lookup():
    assert match_terms("a" * 200 + " ok") == [("ok",)]
    assert match_terms("") == []


def test_query_text_is_phrases_ored_per_backend():
    terms = [("agents", "py"), ("5432",)]
    assert keyword_query_text(POSTGRESQL, terms) == "'agents' <-> 'py' | '5432'"
    assert keyword_query_text(SQLITE, terms) == '"agents py" OR "5432"'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _wipe() -> None:
    async with get_session_factory()() as s:
        await s.execute(text("DELETE FROM annotations WHERE content LIKE :p"), {"p": f"{MARK}%"})
        await s.commit()


@pytest.fixture
async def clean():
    try:
        async with get_session_factory()() as s:
            await s.execute(text("SELECT 1 FROM annotations LIMIT 1"))
    except Exception:
        pytest.skip("database not reachable")
    await _wipe()
    yield
    await _wipe()
    await close_engine()


@pytest.fixture
def keyword_channel(monkeypatch):
    """Switch the channel on or off for one call of the code under test."""
    import hafiz.core.config as config

    real = config.load_settings

    def set_to(enabled: bool) -> None:
        def patched():
            settings = real()
            settings.recall.keyword_channel = enabled
            return settings

        monkeypatch.setattr(config, "load_settings", patched)

    return set_to


async def _decoys(n: int = 24) -> None:
    """On-topic rows for the *prose* of the query, none containing the token."""
    for i in range(n):
        await store_annotation(
            f"{MARK} decoy {i}: how the retry configuration backs off after a failed request",
            kind="learning",
            source="user:test",
        )


QUESTION = f"what is the retry configuration for {TOKEN}"


async def _contents(**filters) -> list[str]:
    got = await search_annotations(QUESTION, limit=3, telemetry_command=None, **filters)
    return [r.content for r in got]


# ---------------------------------------------------------------------------
# The channel earns its keep
# ---------------------------------------------------------------------------


async def test_an_exact_token_is_found_with_the_channel_on_and_missed_off(clean, keyword_channel):
    await _decoys()
    await store_annotation(
        f"{MARK} target: the {TOKEN} gateway is pinned to a single region",
        kind="fact",
        source="user:test",
    )
    keyword_channel(False)
    assert not any(TOKEN in c for c in await _contents()), (
        "the decoys should crowd the target out of a vector-only top 3 — if this "
        "fails the fixture no longer proves the channel does anything"
    )
    keyword_channel(True)
    assert any(TOKEN in c for c in await _contents())


async def test_the_channel_does_nothing_without_reranking(clean, keyword_channel):
    """``--no-rerank`` is the pure-vector, exact-pin mode: no arbiter, no keyword rows."""
    await _decoys()
    await store_annotation(
        f"{MARK} target: the {TOKEN} gateway is pinned to a single region",
        kind="fact",
        source="user:test",
    )
    keyword_channel(True)
    got = await search_annotations(QUESTION, limit=3, rerank=False, telemetry_command=None)
    assert not any(TOKEN in r.content for r in got)


async def test_the_default_is_on():
    from hafiz.core.config import RecallSettings

    assert RecallSettings().keyword_channel is True
    assert 0 < RecallSettings().keyword_candidates <= 20


# ---------------------------------------------------------------------------
# Filter parity — a keyword hit never leaks what the vector path filters out
# ---------------------------------------------------------------------------


async def test_a_retired_row_is_never_returned(clean):
    await _decoys()
    await store_annotation(f"{MARK} live: {TOKEN} is current", kind="fact", source="u:t")
    dead = await store_annotation(f"{MARK} dead: {TOKEN} was retired", kind="fact", source="u:t")
    await invalidate_annotation(str(dead.id))
    got = await _contents()
    assert any("live:" in c for c in got)
    assert not any("dead:" in c for c in got)
    # ...and `include_superseded` reads prior beliefs, exactly as it does for vectors.
    assert any("dead:" in c for c in await _contents(active_only=False))


async def test_an_expired_or_not_yet_valid_row_is_never_returned(clean):
    await _decoys()
    now = datetime.now(UTC)
    await store_annotation(
        f"{MARK} expired: {TOKEN}", kind="fact", source="u:t", valid_until=now - timedelta(days=1)
    )
    await store_annotation(
        f"{MARK} future: {TOKEN}", kind="fact", source="u:t", valid_from=now + timedelta(days=1)
    )
    got = await _contents()
    assert not any("expired:" in c or "future:" in c for c in got)


async def test_another_projects_row_is_never_returned(clean):
    await _decoys()
    for project in ("kw-proj-a", "kw-proj-b"):
        await store_annotation(
            f"{MARK} {project}: {TOKEN}",
            kind="fact",
            source="u:t",
            project=project,
            new_project=True,
        )
    got = await _contents(project="kw-proj-a")
    assert any("kw-proj-a:" in c for c in got)
    assert not any("kw-proj-b:" in c for c in got)


async def test_kind_source_and_tags_filters_apply_to_keyword_hits(clean):
    await _decoys()
    await store_annotation(
        f"{MARK} keep: {TOKEN}", kind="warning", source="agent:x", tags=["kw-keep"]
    )
    await store_annotation(
        f"{MARK} wrong-kind: {TOKEN}", kind="fact", source="agent:x", tags=["kw-keep"]
    )
    await store_annotation(
        f"{MARK} wrong-source: {TOKEN}", kind="warning", source="agent:y", tags=["kw-keep"]
    )
    await store_annotation(
        f"{MARK} wrong-tag: {TOKEN}", kind="warning", source="agent:x", tags=["other"]
    )
    await store_annotation(f"{MARK} no-tag: {TOKEN}", kind="warning", source="agent:x")

    got = await _contents(kind="warning", source="agent:x", tags=["kw-keep"])
    assert any("keep:" in c for c in got)
    assert not any(w in c for c in got for w in ("wrong-", "no-tag"))


async def test_a_row_with_no_embedding_is_never_returned(clean):
    """The vector path requires an embedding; so does the keyword path."""
    await _decoys()
    async with get_session_factory()() as s:
        s.add(Annotation(content=f"{MARK} unembedded: {TOKEN}", kind="fact", embedding=None))
        await s.commit()
    assert not any("unembedded:" in c for c in await _contents())


async def test_a_prompt_with_nothing_to_look_up_is_just_a_vector_search(clean):
    await _decoys(3)
    got = await search_annotations("what is the", limit=3, telemetry_command=None)
    assert all(MARK in r.content for r in got) or got == []


# ---------------------------------------------------------------------------
# One tokenizer, both backends
# ---------------------------------------------------------------------------

STORED = "see src/hafiz/core/agents.py, port 5432, raised ERR_X when --min-score is 0.5 (v1.2.3)"


async def _hits(query: str) -> int:
    """Keyword-only matches for ``query`` among the rows this module wrote."""
    cosine = similarity(Annotation.embedding, PROBE_VECTOR).label("s")

    def only_mine(stmt):
        return stmt.where(Annotation.content.like(f"{MARK}%"))

    rows, _ = await _keyword_rows(query, only_mine, cosine, 10)
    return len(rows)


@pytest.mark.parametrize(
    ("query", "matches"),
    [
        ("agents.py", True),
        ("core/agents.py", True),
        ("5432", True),
        ("ERR_X", True),
        ("err x", True),  # the same words, however punctuated
        ("--min-score", True),
        ("min-score", True),
        ("v1.2.3", True),
        ("src/hafiz/core/agents.py", True),
        ("agents.ts", False),
        ("5433", False),
        # No stemming, by design: the two backends must agree, and SQLite has none.
        ("agent", False),
        ("ports", False),
        # A phrase is adjacent words, not a bag.
        ("score-min", False),
    ],
)
async def test_exact_strings_match_however_the_stored_text_punctuated_them(clean, query, matches):
    await store_annotation(f"{MARK} {STORED}", kind="fact", source="u:t")
    assert (await _hits(query) > 0) is matches


async def test_a_natural_language_prompt_matches_through_some_of_its_words(clean):
    """OR, not AND: requiring every word of a prompt would match nothing."""
    await store_annotation(f"{MARK} {STORED}", kind="fact", source="u:t")
    assert await _hits("how do I get around the ERR_X error that nobody understands") > 0


# ---------------------------------------------------------------------------
# The index stays in sync, and it migrates (up and down)
# ---------------------------------------------------------------------------


def _backend_name() -> str:
    from hafiz.core.config import load_settings
    from hafiz.core.dialect import backend_of_url

    return backend_of_url(load_settings().database.url)


async def test_an_edit_and_a_delete_reach_the_index(clean):
    """A generated column (Postgres) and triggers (SQLite) both promise this."""
    row = await store_annotation(f"{MARK} first {TOKEN}", kind="fact", source="u:t")
    assert await _hits(TOKEN) == 1
    async with get_session_factory()() as s:
        await s.execute(
            text("UPDATE annotations SET content = :c WHERE id = :i"),
            {
                "c": f"{MARK} second wording",
                "i": row.id.hex if _backend_name() == SQLITE else row.id,
            },
        )
        await s.commit()
    assert await _hits(TOKEN) == 0
    assert await _hits("wording") == 1
    async with get_session_factory()() as s:
        await s.execute(text("DELETE FROM annotations WHERE content LIKE :p"), {"p": f"{MARK}%"})
        await s.commit()
    assert await _hits("wording") == 0


async def _alembic(direction: str, revision: str) -> None:
    from alembic import command
    from hafiz.core.database import _alembic_config

    cfg = _alembic_config()
    fn = command.upgrade if direction == "up" else command.downgrade
    # env.py runs its own event loop, so it cannot share this one.
    await asyncio.get_running_loop().run_in_executor(None, fn, cfg, revision)


async def _index_exists() -> bool:
    async with get_session_factory()() as s:
        if backend_of(s) == POSTGRESQL:
            return bool(
                (
                    await s.execute(
                        text(
                            "SELECT count(*) FROM information_schema.columns "
                            "WHERE table_name='annotations' AND column_name='content_tsv'"
                        )
                    )
                ).scalar()
            )
        return bool(
            (
                await s.execute(
                    text("SELECT count(*) FROM sqlite_master WHERE name = 'annotations_fts'")
                )
            ).scalar()
        )


async def test_the_migration_goes_down_and_up_and_backfills(clean):
    """0012 down drops the index; recall degrades to vector-only; 0012 up
    rebuilds it from the rows that are already there."""
    await store_annotation(f"{MARK} survives {TOKEN}", kind="fact", source="u:t")
    assert await _index_exists()
    await close_engine()

    await _alembic("down", "0011")
    assert not await _index_exists()
    # No index: the channel returns nothing and never raises.
    assert await _hits(TOKEN) == 0
    await close_engine()

    await _alembic("up", "head")
    assert await _index_exists()
    assert await _hits(TOKEN) == 1, "the backfill must index rows written while it was absent"


# --- SQLite specifics: a store the test builds itself -----------------------


@pytest.fixture
async def embedded_store(tmp_path):
    """Point the process at a fresh SQLite file, then put everything back."""
    from hafiz.core import config as _config
    from hafiz.core.database import create_tables

    url = f"sqlite:///{tmp_path / 'kw.db'}"
    real = _config.load_settings

    def patched():
        settings = real()
        settings.database.url = url
        return settings

    _config.load_settings = patched  # type: ignore[assignment]
    _config.reset_settings()
    await close_engine()
    try:
        await create_tables(url)
        yield url
    finally:
        await close_engine()
        _config.load_settings = real  # type: ignore[assignment]
        _config.reset_settings()


async def test_a_new_embedded_store_has_the_index_without_replaying_migrations(embedded_store):
    async with get_session_factory()() as s:
        names = (
            (
                await s.execute(
                    text("SELECT name FROM sqlite_master WHERE name LIKE 'annotations_fts%'")
                )
            )
            .scalars()
            .all()
        )
    assert "annotations_fts" in names
    assert {"annotations_fts_ai", "annotations_fts_ad", "annotations_fts_au"} <= set(names)


async def test_the_sqlite_index_survives_a_vacuum(embedded_store):
    """The FTS table is keyed on the annotation id, not its rowid: a rowid-keyed
    external-content table would point at the wrong rows once VACUUM renumbered
    them, and `forget --hard` runs a VACUUM."""
    from hafiz.core.dialect import reclaim_free_pages

    keep = await store_annotation(f"{MARK} keep {TOKEN}", kind="fact", source="u:t")
    for i in range(5):
        gone = await store_annotation(f"{MARK} gone{i}", kind="fact", source="u:t")
        async with get_session_factory()() as s:
            await s.execute(text("DELETE FROM annotations WHERE id = :i"), {"i": gone.id.hex})
            await s.commit()
    async with get_session_factory()() as s:
        assert await reclaim_free_pages(s)
    cosine = similarity(Annotation.embedding, PROBE_VECTOR).label("s")
    rows, _ = await _keyword_rows(TOKEN, lambda s: s, cosine, 10)
    assert [a.id for a, _ in rows] == [keep.id]


async def test_ensure_keyword_index_is_idempotent_and_backfills(embedded_store):
    from hafiz.core.dialect import drop_keyword_index, ensure_keyword_index

    await store_annotation(f"{MARK} before {TOKEN}", kind="fact", source="u:t")
    async with get_session_factory()() as s:
        conn = await s.connection()
        await conn.run_sync(drop_keyword_index)
        await conn.run_sync(drop_keyword_index)  # dropping twice is fine
        await conn.run_sync(ensure_keyword_index)
        await conn.run_sync(ensure_keyword_index)  # creating twice is fine
        await s.commit()
    cosine = similarity(Annotation.embedding, PROBE_VECTOR).label("s")
    rows, _ = await _keyword_rows(TOKEN, lambda s: s, cosine, 10)
    assert len(rows) == 1


async def test_a_store_without_the_index_still_recalls(embedded_store):
    """Recall is read on every prompt: a missing index costs candidates, not the turn."""
    from hafiz.core.dialect import drop_keyword_index

    await store_annotation(f"{MARK} vector only {TOKEN}", kind="fact", source="u:t")
    async with get_session_factory()() as s:
        conn = await s.connection()
        await conn.run_sync(drop_keyword_index)
        await s.commit()
    got = await search_annotations(QUESTION, limit=3, telemetry_command=None)
    assert any("vector only" in r.content for r in got)


async def test_a_long_annotation_does_not_break_the_write(clean):
    """A generated tsvector raises past 1 MiB; the column indexes a bounded prefix."""
    huge = f"{MARK} " + " ".join(f"w{n}" for n in range(250_000))
    async with get_session_factory()() as s:
        row = Annotation(content=huge, kind="fact", embedding=None)
        s.add(row)
        await s.commit()
        found = (await s.execute(select(Annotation.id).where(Annotation.id == row.id))).first()
    assert found is not None
