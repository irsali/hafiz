"""Tests for ``hafiz.core.durations.parse_duration``."""

from datetime import timedelta

import pytest

from hafiz.core.durations import parse_duration


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("6h", timedelta(hours=6)),
        ("30d", timedelta(days=30)),
        ("2w", timedelta(weeks=2)),
        ("3m", timedelta(days=90)),
        ("1y", timedelta(days=365)),
        ("45", timedelta(days=45)),
        (" 7d ", timedelta(days=7)),
    ],
)
def test_parses_common_forms(raw, expected):
    assert parse_duration(raw) == expected


@pytest.mark.parametrize("bad", ["", "d", "abc", "-5d", "5 weeks", "5z"])
def test_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)
