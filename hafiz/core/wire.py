"""Lossless serialization for the daemon protocol.

Distinct from ``to_dict()`` on the same objects, and deliberately so:

* ``to_dict()`` is the **public ``--json`` contract**. It is trimmed on
  purpose — ``AnnotationResult.metadata`` and ``snippet`` are omitted,
  ``SearchResult.is_neighbor`` is omitted — because those are internals
  callers should not depend on.
* ``to_wire()`` is the **internal transport**. It must carry every field,
  because the object is rebuilt on the other side and handed to code that
  expects the real thing.

Using ``to_dict()`` for transport is the trap this module exists to close.
It round-trips *almost* correctly, so the warm path returns objects that
are subtly poorer than the direct path — ``--format compact`` renders
without snippets, ``--include-superseded`` loses its validity dates — and
nothing errors. The daemon is supposed to change latency, not results.

Derived from the dataclass fields rather than a hand-written key list, so
**a field added to one of these types crosses the wire automatically**. The
hand-maintained equivalent in ``daemon.py`` had already silently dropped
``tags`` once; it was caught in review rather than by a test, and
``unit_id``, ``metadata`` and ``snippet`` were still missing when this was
written.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any

#: Marks an ISO-8601 string that was a ``datetime`` before transport. JSON has
#: no datetime, and a bare string would rebuild as a ``str`` — comparisons
#: against ``datetime.now(UTC)`` would then raise on the warm path only.
_DT_PREFIX = "\x00dt:"


def _encode(value: Any) -> Any:
    if isinstance(value, datetime):
        return _DT_PREFIX + value.isoformat()
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_DT_PREFIX):
        return datetime.fromisoformat(value[len(_DT_PREFIX) :])
    if isinstance(value, dict):
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def to_wire(obj: Any) -> dict:
    """Every field of a dataclass instance, JSON-safe."""
    return {f.name: _encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}


def from_wire[T](cls: type[T], payload: dict) -> T:
    """Rebuild a dataclass from :func:`to_wire` output.

    Unknown keys are dropped rather than raising: a newer daemon left
    running against an older client would otherwise turn a forward-compatible
    field into a hard failure on every call. The version handshake already
    replaces genuinely stale daemons; this is the belt for the moment in
    between.
    """
    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: _decode(v) for k, v in payload.items() if k in known})
