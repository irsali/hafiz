"""The soft size brake on ``observe``.

DB-free. ``kind`` is the only schema an annotation has, so decision +
rationale + scope + rejected-alternatives land in one blob; on a real store the
mean grew 746 → 1,127 chars in eight weeks. This is the cheap brake on that
curve, and its defining property is that it is **advisory** — every test here
that asserts a warning also has to hold that the write still succeeded.
"""

from __future__ import annotations

import pytest

from hafiz.core.annotations import oversized_warning
from hafiz.core.config import AnnotationSettings, HafizSettings

DEFAULT = AnnotationSettings().max_recommended_chars


@pytest.fixture
def limit(monkeypatch):
    """Pin the configured limit, whatever the ambient hafiz.toml says."""

    def _set(value: int) -> None:
        settings = HafizSettings()
        settings.annotations.max_recommended_chars = value
        monkeypatch.setattr("hafiz.core.config.load_settings", lambda: settings)

    return _set


def test_default_limit_is_the_measured_p90_not_the_median():
    """1,500 fires on the 12.7% tail holding 23.5% of all text. A limit near
    the 951-char median would warn on a third of writes and be ignored."""
    assert DEFAULT == 1500


def test_a_record_under_the_limit_is_not_flagged(limit):
    limit(1500)
    assert oversized_warning("x" * 1500, kind="decision") is None


def test_a_record_over_the_limit_reports_its_size_and_what_to_do(limit):
    limit(1500)
    warning = oversized_warning("x" * 1501, kind="decision")
    assert warning is not None
    assert warning["chars"] == 1501
    assert warning["limit"] == 1500
    assert "--derived-from" in warning["hint"]


def test_note_is_never_flagged_however_long(limit):
    """Raw capture is never gated, and a note is below decision-grade anyway."""
    limit(1500)
    assert oversized_warning("x" * 50_000, kind="note") is None


@pytest.mark.parametrize("kind", ["decision", "learning", "warning", "fact", "pattern"])
def test_every_other_kind_is_flagged(kind, limit):
    limit(100)
    assert oversized_warning("x" * 500, kind=kind) is not None


def test_zero_disables_the_brake(limit):
    limit(0)
    assert oversized_warning("x" * 50_000, kind="decision") is None


def test_a_negative_limit_disables_rather_than_warning_on_everything(limit):
    limit(-1)
    assert oversized_warning("", kind="decision") is None


def test_the_check_is_on_characters_not_lines(limit):
    """A long record split over many short lines is just as expensive to
    inject as one paragraph, so line count is the wrong unit."""
    limit(100)
    assert oversized_warning("a\n" * 60, kind="decision") is not None
