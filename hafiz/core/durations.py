"""Human-readable durations and ages.

:func:`parse_duration` accepts forms like ``30d``, ``2w``, ``6h``, ``3m``,
``1y``. A bare integer is interpreted as days. Whitespace around the value is
tolerated. In the observation/expiration context, lowercase ``m`` means
**months** (not minutes) — minutes are not a useful expiration granularity
here.

:func:`age_label` is the inverse: a timestamp rendered as ``"3mo ago"`` for
display. It lives here (core, no Rich/Typer) rather than in the CLI layer
because both the compact output formats and the rich table need it.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

_PATTERN = re.compile(r"^\s*(\d+)\s*([hdwmy]?)\s*$")

_UNIT_SECONDS = {
    "h": 3600,
    "d": 86400,
    "w": 7 * 86400,
    "m": 30 * 86400,
    "y": 365 * 86400,
    "": 86400,  # bare integer → days
}


def parse_duration(s: str) -> timedelta:
    """Parse strings like ``"30d"``, ``"2w"``, ``"6h"``, ``"3m"``, ``"1y"``.

    Raises ``ValueError`` on unparseable input so the caller can surface
    a friendly CLI error.
    """
    m = _PATTERN.match(s or "")
    if not m:
        raise ValueError(f"cannot parse duration: {s!r}")
    n = int(m.group(1))
    unit = m.group(2)
    return timedelta(seconds=n * _UNIT_SECONDS[unit])


STALE_DAYS = 90
"""Annotations older than this are dimmed / flagged ``stale`` on recall."""


def age_label(ts: datetime, *, now: datetime | None = None) -> tuple[str, int, bool]:
    """Render ``ts`` as ``(human label, age in days, stale flag)``.

    e.g. ``"today"``, ``"1d ago"``, ``"12d ago"``, ``"3mo ago"``, ``"2y ago"``.
    Timestamps in the future come back as ``"future"`` with a negative age and
    are never stale.
    """
    now = now or datetime.now(UTC)
    days = (now - ts.astimezone(UTC)).days
    if days < 0:
        return "future", days, False
    if days == 0:
        return "today", 0, False
    if days == 1:
        return "1d ago", 1, False
    if days < 30:
        return f"{days}d ago", days, days > STALE_DAYS
    if days < 365:
        return f"{round(days / 30)}mo ago", days, days > STALE_DAYS
    return f"{round(days / 365)}y ago", days, True
