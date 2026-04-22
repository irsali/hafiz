"""Sticky per-tunable state — the "remembered recommendations" layer.

Phase 3 of the tunable-registry work item. Sits between the TOML and
the built-in default in the resolution chain:

    environment  →  hafiz.toml  →  **sticky tuning state**  →  default

Populated by ``hafiz doctor --apply`` (or ``hafiz config apply``) after
a successful probe run; consulted on every ``resolve_tunable(key)``
call so the picked-up values show up everywhere the tunable is read.

State file layout (``~/.cache/hafiz/tuning_state.json``):

    {
      "schema": 1,
      "host_fingerprint": "17c96ae6245db6a8",
      "onnxruntime_version": "1.24.4",
      "written_at": "2026-04-22T12:34:56+00:00",
      "entries": {
        "embedding.max_part_chars": {
          "value": 16000,
          "rationale": "...",
          "confidence": "high",
          "probed_at": "2026-04-22T12:34:50+00:00",
          "measured": {...}
        }
      }
    }

A fingerprint that no longer matches the current host is treated as
stale and ignored — we never silently apply recommendations from a
different machine class. Stale state is flagged on load and cleared
the next time a write happens; it's never deleted behind the user's
back without some user action (hafiz config apply / unset / clear).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Bump if the on-disk shape changes incompatibly. Loaders reject older
# schemas rather than silently misinterpreting them.
CURRENT_SCHEMA = 1


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class TuningEntry:
    value: Any
    rationale: str | None = None
    confidence: str | None = None
    probed_at: str | None = None
    measured: dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningState:
    host_fingerprint: str
    onnxruntime_version: str | None
    entries: dict[str, TuningEntry] = field(default_factory=dict)
    written_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    schema: int = CURRENT_SCHEMA

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "host_fingerprint": self.host_fingerprint,
            "onnxruntime_version": self.onnxruntime_version,
            "written_at": self.written_at,
            "entries": {k: asdict(v) for k, v in self.entries.items()},
        }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def cache_file_path() -> Path:
    """XDG-compliant cache file location. Overridable via XDG_CACHE_HOME."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "hafiz" / "tuning_state.json"


def load_state() -> TuningState | None:
    """Return the persisted state, or None if absent / unreadable / wrong schema.

    A corrupt file is logged and removed — we'd rather re-probe than
    propagate garbage into the resolution chain. The only path that
    writes state back is save_state(), so data loss is limited to the
    sticky layer (TOML + env are never touched).
    """
    path = cache_file_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Corrupt tuning-state cache at %s (%s); removing.", path, exc)
        try:
            path.unlink()
        except OSError:
            pass
        return None

    if not isinstance(data, dict) or data.get("schema") != CURRENT_SCHEMA:
        logger.info(
            "Ignoring tuning-state cache at %s — schema %s, expected %s.",
            path,
            data.get("schema") if isinstance(data, dict) else "?",
            CURRENT_SCHEMA,
        )
        return None

    try:
        entries_raw = data.get("entries", {})
        entries = {
            k: TuningEntry(
                value=v["value"],
                rationale=v.get("rationale"),
                confidence=v.get("confidence"),
                probed_at=v.get("probed_at"),
                measured=v.get("measured", {}) or {},
            )
            for k, v in entries_raw.items()
            if isinstance(v, dict) and "value" in v
        }
    except (KeyError, TypeError) as exc:
        logger.warning(
            "Malformed tuning-state entries at %s (%s); ignoring.", path, exc
        )
        return None

    return TuningState(
        host_fingerprint=data.get("host_fingerprint", ""),
        onnxruntime_version=data.get("onnxruntime_version"),
        entries=entries,
        written_at=data.get("written_at", ""),
        schema=CURRENT_SCHEMA,
    )


def save_state(state: TuningState) -> None:
    path = cache_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.as_jsonable(), indent=2))
    except OSError as exc:
        logger.warning("Could not persist tuning state at %s: %s", path, exc)


def clear_state() -> bool:
    """Delete the cache file. Returns True if a file was removed."""
    path = cache_file_path()
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError as exc:
        logger.warning("Could not clear tuning state at %s: %s", path, exc)
        return False


# ---------------------------------------------------------------------------
# Staleness / querying
# ---------------------------------------------------------------------------


def is_stale(state: TuningState, *, fingerprint: str, ort_version: str | None) -> bool:
    """True when the cached state should be ignored.

    Fingerprint mismatch means the user has moved to a materially
    different host (different RAM class, GPU appeared/disappeared,
    different OS/arch) and the old recommendations no longer apply.
    ORT version change is less decisive but worth a re-probe — provider
    availability or allocator behavior can shift between versions.
    """
    if not state.host_fingerprint:
        return True
    if state.host_fingerprint != fingerprint:
        return True
    # Only mark stale on ORT mismatch if both are known. Absent version
    # info (older state file, ORT uninstalled) is not enough to invalidate.
    if (
        state.onnxruntime_version is not None
        and ort_version is not None
        and state.onnxruntime_version != ort_version
    ):
        return True
    return False


def get_value(key: str, *, fingerprint: str, ort_version: str | None) -> Any | None:
    """Look up one sticky entry, honoring staleness. Returns None when
    nothing applies — callers should fall through to the default."""
    state = load_state()
    if state is None:
        return None
    if is_stale(state, fingerprint=fingerprint, ort_version=ort_version):
        return None
    entry = state.entries.get(key)
    return entry.value if entry is not None else None


# ---------------------------------------------------------------------------
# Constructors used by `hafiz config apply` / `hafiz doctor --apply`
# ---------------------------------------------------------------------------


def build_state(
    *,
    fingerprint: str,
    ort_version: str | None,
    entries: dict[str, TuningEntry],
) -> TuningState:
    return TuningState(
        host_fingerprint=fingerprint,
        onnxruntime_version=ort_version,
        entries=entries,
    )


def merge_into_state(
    state: TuningState | None,
    *,
    fingerprint: str,
    ort_version: str | None,
    new_entries: dict[str, TuningEntry],
) -> TuningState:
    """Combine ``new_entries`` into (possibly existing) ``state``.

    If ``state`` is stale for this host, we discard it — otherwise a
    user who re-probed on a different machine would carry forward
    irrelevant values. If ``state`` is None or absent, we start fresh.
    Collisions on ``new_entries`` overwrite silently; the writer is
    responsible for deciding what goes in.
    """
    if state is None or is_stale(
        state, fingerprint=fingerprint, ort_version=ort_version
    ):
        base_entries: dict[str, TuningEntry] = {}
    else:
        base_entries = dict(state.entries)
    base_entries.update(new_entries)
    return build_state(
        fingerprint=fingerprint, ort_version=ort_version, entries=base_entries
    )
