"""Pure-function tests for ``hafiz.core.capture``.

DB-hitting logic (``store_transcript`` → source-layer communications)
is covered by dogfooding; here we pin down the turn splitter, the slug
generator, and the ``--source`` → agent derivation.
"""

from __future__ import annotations

import re

import pytest

from hafiz.core.capture import _agent_from_source, _slugify, split_transcript


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Line 1\n\nLine 2\n\nLine 3", ["Line 1", "Line 2", "Line 3"]),
        ("A\n\n\n\nB", ["A", "B"]),  # multiple blank lines collapse
        ("  hello  \n\n\tworld\t", ["hello", "world"]),  # stripped
        ("just one paragraph", ["just one paragraph"]),
        ("", []),
        ("   \n\n   ", []),  # whitespace-only collapses to nothing
    ],
)
def test_split_transcript(raw, expected):
    assert split_transcript(raw) == expected


def test_split_transcript_preserves_dialogue_prefixes():
    turns = split_transcript("Q: What?\n\nA: An answer.\n\nQ: Again.")
    assert [t[:2] for t in turns] == ["Q:", "A:", "Q:"]


def test_slugify_with_title_produces_kebab_and_suffix():
    slug = _slugify("Hello World — 2026!")
    assert slug.startswith("hello-world-2026-")
    # trailing 6-hex random suffix
    assert re.fullmatch(r"hello-world-2026-[0-9a-f]{6}", slug)


def test_slugify_without_title_is_pure_suffix():
    slug = _slugify(None)
    assert re.fullmatch(r"[0-9a-f]{6}", slug)


def test_slugify_strips_and_caps_long_titles():
    slug = _slugify("a" * 200)
    # kebab base is capped at 40 chars, then -<6hex>
    assert re.fullmatch(r"a{40}-[0-9a-f]{6}", slug)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("agent:hermes", "hermes"),       # agent: prefix stripped
        ("agent:claude-code", "claude-code"),
        ("user:anjum", "user:anjum"),     # non-agent source passes through
        ("capture", "capture"),           # bare value passes through
        (None, "capture"),                # missing → default
        ("", "capture"),                  # empty → default
        ("agent:", "capture"),            # empty after prefix → default
    ],
)
def test_agent_from_source(source, expected):
    assert _agent_from_source(source) == expected
