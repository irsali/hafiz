"""Tests for collapsing byte-identical content in one result set.

The markdown parser attaches each paragraph to **every ancestor heading**, so a
paragraph nested under ``H1 > H2 > H3`` becomes three units with identical
content, differing only in ``name`` / ``parent_name``::

    Web Channel — AI Agent Instructions #p506
    Web Channel — AI Agent Instructions > Integrations #p506
    Web Channel — AI Agent Instructions > Integrations > Geolocation Services #p506

Measured on a 15-project deployment: 8,852 clusters, 17,625 redundant of 33,967
live doc units (51.9%), and 17,701 of 49,334 live embeddings. Every non-annotation
query was spending ~36% of its result set on repeats of text it had already
returned.

Identical content embeds identically, so duplicates tie on score — collapsing
can't reorder anything. The only decision is which copy to keep, and the deepest
heading path carries the most context.
"""

from __future__ import annotations

import uuid

import pytest

from hafiz.core.search import _DEDUP_OVERFETCH, vector_search


class _Row:
    """Stand-in for one ``(Embedding, UnitRevision, Unit, File, similarity)`` row."""

    def __init__(self, *, name, content, path, content_hash=None, sim=0.7, kind="doc.paragraph"):
        self.emb = type(
            "E",
            (),
            {
                "id": uuid.uuid4(),
                "content": content,
                "content_hash": content_hash or content,
                "part_index": 0,
            },
        )()
        self.rev = type("R", (), {"line_start": 1, "line_end": 2})()
        self.unit = type("U", (), {"id": uuid.uuid4(), "name": name, "kind": kind})()
        self.file = type("F", (), {"path": path, "language": "md", "project": "p"})()
        self.sim = sim

    def __iter__(self):
        return iter((self.emb, self.rev, self.unit, self.file, self.sim))


@pytest.fixture
def fake_search(monkeypatch):
    """Drive ``vector_search`` off a fixed row list, no DB and no model."""
    captured: dict = {}

    def _install(rows):
        async def _embed(_q):
            return [0.0] * 768

        class _Result:
            def all(self):
                return list(rows)

        class _Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def execute(self, stmt):
                # Record the SQL LIMIT so over-fetching is observable.
                captured["limit"] = stmt._limit
                return _Result()

        monkeypatch.setattr("hafiz.core.search.embed_query", _embed)
        monkeypatch.setattr("hafiz.core.search.get_session_factory", lambda: _Session)
        return captured

    return _install


# ── Collapsing ──────────────────────────────────────────────────────────

TEXT = "The per-region blocking flag sets opt-in vs opt-out posture."


async def test_the_same_paragraph_under_three_headings_returns_once(fake_search):
    fake_search(
        [
            _Row(name="Doc #p506", content=TEXT, path="/a/AGENTS.md"),
            _Row(name="Doc > Integrations #p506", content=TEXT, path="/a/AGENTS.md"),
            _Row(name="Doc > Integrations > Geo #p506", content=TEXT, path="/a/AGENTS.md"),
        ]
    )
    results = await vector_search("region rules", limit=10)
    assert len(results) == 1


async def test_the_deepest_heading_path_is_the_copy_kept(fake_search):
    """ "Doc > Integrations > Geo" tells the reader where this lives; "Doc" doesn't."""
    fake_search(
        [
            _Row(name="Doc #p506", content=TEXT, path="/a/AGENTS.md"),
            _Row(name="Doc > Integrations > Geo #p506", content=TEXT, path="/a/AGENTS.md"),
            _Row(name="Doc > Integrations #p506", content=TEXT, path="/a/AGENTS.md"),
        ]
    )
    results = await vector_search("region rules", limit=10)
    assert results[0].unit_name == "Doc > Integrations > Geo #p506"


async def test_the_deepest_copy_keeps_the_position_of_the_first_one(fake_search):
    """Replacing must not reorder: duplicates tie on score, so the slot is the
    slot. A swap that moved rows would change ranking as a side effect."""
    fake_search(
        [
            _Row(name="Other", content="unrelated text", path="/a/AGENTS.md", sim=0.9),
            _Row(name="Doc #p1", content=TEXT, path="/a/AGENTS.md", sim=0.7),
            _Row(name="Doc > Deep #p1", content=TEXT, path="/a/AGENTS.md", sim=0.7),
        ]
    )
    results = await vector_search("region rules", limit=10)
    assert [r.unit_name for r in results] == ["Other", "Doc > Deep #p1"]


async def test_identical_text_in_two_files_is_kept_twice(fake_search):
    """Two files saying the same thing are two sources, not a duplicate."""
    fake_search(
        [
            _Row(name="A #p1", content=TEXT, path="/a/AGENTS.md"),
            _Row(name="B #p1", content=TEXT, path="/b/AGENTS.md"),
        ]
    )
    results = await vector_search("region rules", limit=10)
    assert len(results) == 2


async def test_different_parts_of_one_unit_are_not_collapsed(fake_search):
    """A long paragraph splits into parts; those are different content."""
    fake_search(
        [
            _Row(name="A #p1", content="part one", path="/a/x.md"),
            _Row(name="A #p1", content="part two", path="/a/x.md"),
        ]
    )
    results = await vector_search("x", limit=10)
    assert len(results) == 2


# ── Interaction with limit and threshold ────────────────────────────────


async def test_dedup_happens_before_the_limit_is_applied(fake_search):
    """The regression that makes dedup worth doing in core rather than in the
    caller: trimming first spends the caller's limit on copies."""
    rows = []
    for i in range(3):  # 3 distinct paragraphs, 3 copies each
        for depth in range(3):
            rows.append(
                _Row(name=f"Doc{' > H' * depth} #p{i}", content=f"text {i}", path="/a/x.md")
            )
    fake_search(rows)
    results = await vector_search("x", limit=3)
    assert len(results) == 3
    assert len({r.content for r in results}) == 3, "limit spent on duplicates"


async def test_the_sql_limit_is_over_fetched_when_deduping(fake_search):
    captured = fake_search([_Row(name="A", content="a", path="/a/x.md")])
    await vector_search("x", limit=10)
    assert captured["limit"] == 10 * _DEDUP_OVERFETCH


async def test_no_over_fetch_when_dedup_is_off(fake_search):
    captured = fake_search([_Row(name="A", content="a", path="/a/x.md")])
    await vector_search("x", limit=10, dedup=False)
    assert captured["limit"] == 10


async def test_dedup_off_returns_every_copy(fake_search):
    fake_search(
        [
            _Row(name="Doc #p1", content=TEXT, path="/a/x.md"),
            _Row(name="Doc > H #p1", content=TEXT, path="/a/x.md"),
        ]
    )
    results = await vector_search("x", limit=10, dedup=False)
    assert len(results) == 2


async def test_the_similarity_floor_still_applies(fake_search):
    fake_search(
        [
            _Row(name="A", content="a", path="/a/x.md", sim=0.8),
            _Row(name="B", content="b", path="/a/x.md", sim=0.2),
        ]
    )
    results = await vector_search("x", limit=10, similarity_threshold=0.5)
    assert [r.unit_name for r in results] == ["A"]


async def test_scores_are_untouched_by_collapsing(fake_search):
    fake_search(
        [
            _Row(name="Doc #p1", content=TEXT, path="/a/x.md", sim=0.7331),
            _Row(name="Doc > H #p1", content=TEXT, path="/a/x.md", sim=0.7331),
        ]
    )
    results = await vector_search("x", limit=10)
    assert results[0].score == 0.7331


async def test_an_empty_result_set_is_still_empty(fake_search):
    fake_search([])
    assert await vector_search("x", limit=10) == []


async def test_a_duplication_tail_beyond_the_overfetch_shortens_never_corrupts(fake_search):
    """The over-fetch factor is 4 but the measured tail runs to 8 copies. When a
    paragraph is duplicated past the factor, the honest failure is a *shorter*
    result set — not a duplicated or misordered one."""
    rows = [
        _Row(name=f"Doc{' > H' * d} #p1", content=TEXT, path="/a/x.md")
        for d in range(8)  # one paragraph, eight copies
    ]
    fake_search(rows)
    results = await vector_search("x", limit=2)
    assert len(results) == 1  # short, because there is genuinely only one thing
    assert results[0].content == TEXT
