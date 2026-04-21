"""Parse human-readable duration strings into ``timedelta``.

Accepts forms like ``30d``, ``2w``, ``6h``, ``3m``, ``1y``. A bare integer
is interpreted as days. Whitespace around the value is tolerated.

In the observation/expiration context, lowercase ``m`` means **months**
(not minutes) — minutes are not a useful expiration granularity here.
"""

from __future__ import annotations

import re
from datetime import timedelta

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
