"""Unit tests for the Phase 0 Mermaid builders in ``hafiz.core.view``.

These are pure functions over :class:`JournalEntry` — no DB. The focus is the
maintainer traps called out in the workitem: node-id safety, text escaping of
adversarial content, truncation, dangling supersession links, and dimming of
superseded nodes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hafiz.core.journal import JournalEntry
from hafiz.core.view import (
    supersession_to_mermaid,
    timeline_to_mermaid,
    to_mermaid,
)


def _entry(
    eid: str,
    content: str,
    *,
    kind: str = "decision",
    supersedes_id: str | None = None,
    when: datetime | None = None,
) -> JournalEntry:
    return JournalEntry(
        id=eid,
        content=content,
        kind=kind,
        source="agent:test",
        project="hafiz",
        tags=[],
        confidence=1.0,
        valid_from=when or datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        valid_until=None,
        session_id=None,
        task=None,
        commit_hash=None,
        metadata={},
        supersedes_id=supersedes_id,
    )


def test_empty_renders_valid_skeleton():
    assert supersession_to_mermaid([]) == "graph LR"
    assert timeline_to_mermaid([]).startswith("timeline")


def test_node_ids_are_sanitized_not_content_or_uuid():
    # A UUID-shaped id and content with chars illegal in a mermaid id must
    # never appear as a node identifier — only n0/n1… counters.
    entries = [
        _entry("3f2e-uuid-with-dashes", "first"),
        _entry("another id with spaces", "second"),
    ]
    out = supersession_to_mermaid(entries)
    assert "n0[" in out
    assert "n1[" in out
    # The raw ids must not leak as identifiers (before the label quote).
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("n") and "[" in stripped:
            node_id = stripped.split("[", 1)[0]
            assert node_id.isidentifier(), node_id


def test_adversarial_text_is_escaped():
    nasty = 'use "JWT" in [localStorage] {risky} (per RFC#1) \n second line'
    out = supersession_to_mermaid([_entry("a", nasty)])
    # A raw double-quote inside the label would terminate it early — must be gone.
    label = out.split('["', 1)[1].split('"]', 1)[0]
    assert '"' not in label
    # Newlines collapsed; mermaid-significant brackets swapped out.
    assert "\n" not in label
    assert "[" not in label and "]" not in label
    assert "{" not in label and "}" not in label
    assert "(" not in label and ")" not in label
    assert "#" not in label  # entity-escape hazard


def test_truncation_keeps_diagram_legible():
    long = "x" * 500
    out = supersession_to_mermaid([_entry("a", long)])
    label = out.split('["', 1)[1].split('"]', 1)[0]
    # kind prefix + truncated body + ellipsis — nowhere near 500 chars.
    assert len(label) < 100
    assert "…" in label


def test_supersession_edge_drawn_within_set():
    old = _entry("old", "use localStorage")
    new = _entry("new", "use httponly cookies", supersedes_id="old")
    out = supersession_to_mermaid([old, new])
    assert "-->|superseded by|" in out
    # The superseded node is dimmed via a classDef.
    assert "classDef superseded" in out
    assert "class " in out


def test_dangling_supersedes_renders_ghost_node():
    # supersedes_id points outside the window — keep the edge honest.
    new = _entry("new", "replacement decision", supersedes_id="missing-old")
    out = supersession_to_mermaid([new])
    assert "(outside window)" in out
    assert "-->|superseded by|" in out


def test_standalone_entries_still_appear():
    a = _entry("a", "lonely decision one")
    b = _entry("b", "lonely decision two")
    out = supersession_to_mermaid([a, b])
    assert "n0[" in out and "n1[" in out
    assert "-->|superseded by|" not in out  # no chains


def test_timeline_groups_by_month_oldest_first():
    may = _entry("m", "may decision", when=datetime(2026, 5, 3, tzinfo=UTC))
    jun = _entry("j", "june decision", when=datetime(2026, 6, 9, tzinfo=UTC))
    out = timeline_to_mermaid([jun, may])  # pass newest-first; expect resort
    assert out.index("2026-05") < out.index("2026-06")
    assert "may decision" in out
    assert "june decision" in out


def test_to_mermaid_dispatch_and_unknown_kind():
    e = [_entry("a", "x")]
    assert to_mermaid(e, kind="supersession").startswith("graph LR")
    assert to_mermaid(e, kind="timeline").startswith("timeline")
    with pytest.raises(ValueError, match="unknown mermaid kind"):
        to_mermaid(e, kind="bogus")
