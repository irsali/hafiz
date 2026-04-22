"""Tunable registry — the generalized pattern behind hafiz's self-tuning.

Each ``Tunable`` is a named configuration knob with a typed default, a
description, and (optionally) a prober that measures a recommended value
for the current host. The registry is the single source of truth that
``hafiz doctor`` walks to generate recommendations and ``hafiz config``
walks to list / set / reset values.

Two kinds of tunable live here:

  **Probed tunables** (``prober`` is set)
      Have a measurable cost-function we can evaluate on the host — e.g.
      ``embedding.max_part_chars`` probes by embedding candidate-size
      inputs and watching RSS. These feed recommendations into sticky
      state (phase 3).

  **Policy caps** (``prober`` is None)
      User-judgment knobs with a safe default and no auto-tuning — e.g.
      ``ingest.max_file_bytes``. Listed so ``hafiz config`` can still
      surface and mutate them; skipped by probe passes.

Resolution precedence (mirrors :mod:`hafiz.core.embeddings.probe_device`):

    environment variable  →  hafiz.toml  →  sticky tuning state  →  built-in default

Phase 1 implements env → TOML → default via pydantic-settings (no
sticky layer yet). Phase 3 slots sticky state in between TOML and the
built-in default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from hafiz.core.config import find_config_file, load_toml


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """One prober's verdict for a tunable on this host."""

    recommended_value: Any
    rationale: str
    confidence: str = "medium"  # one of: "high" | "medium" | "low"
    measured: dict[str, Any] = field(default_factory=dict)


# A prober takes the :class:`HostProbe` (phase 2) and returns a
# recommendation, or raises if probing failed hard. Using ``Any`` for the
# host type in phase 1 — the concrete HostProbe lands with phase 2.
ProberFn = Callable[[Any], ProbeResult]
ValidatorFn = Callable[[Any], None]  # raises ValueError on invalid input


@dataclass(frozen=True)
class Tunable:
    """A named, typed configuration knob.

    ``key`` uses dotted notation that mirrors pydantic settings paths:
    ``"embedding.max_part_chars"`` resolves as
    ``settings.embedding.max_part_chars``. This keeps the CLI surface
    (``hafiz config get embedding.max_part_chars``) and the Python
    config surface (``settings.embedding.max_part_chars``) in lockstep.

    ``prober`` is ``None`` for policy-cap tunables that we deliberately
    don't auto-tune. ``validator`` is optional — pydantic already handles
    type coercion, but validators let us enforce per-key invariants
    (positive integers, bounded ranges) before persisting to sticky state.
    """

    key: str
    default: Any
    type_: type
    description: str
    prober: ProberFn | None = None
    validator: ValidatorFn | None = None

    @property
    def is_policy(self) -> bool:
        """True when this tunable has no prober (judgment call, not measurable)."""
        return self.prober is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TUNABLE_REGISTRY: dict[str, Tunable] = {}


def register(tunable: Tunable) -> None:
    """Register a tunable. Duplicate keys raise — the registry is a source
    of truth, not a merge point."""
    if tunable.key in TUNABLE_REGISTRY:
        raise ValueError(
            f"Tunable {tunable.key!r} is already registered; "
            f"tunables must have unique keys."
        )
    TUNABLE_REGISTRY[tunable.key] = tunable


def get(key: str) -> Tunable:
    """Look up a registered tunable by key. Raises KeyError if unknown."""
    if key not in TUNABLE_REGISTRY:
        raise KeyError(f"Unknown tunable: {key!r}")
    return TUNABLE_REGISTRY[key]


def all_tunables() -> list[Tunable]:
    """All registered tunables in registration order. Stable for tests + CLI."""
    return list(TUNABLE_REGISTRY.values())


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


Source = Literal["env", "toml", "sticky", "default"]


def _env_var_name(key: str) -> str:
    """HAFIZ_ prefix + double-underscore nesting (pydantic-settings convention)."""
    return "HAFIZ_" + key.replace(".", "__").upper()


def _coerce(tunable: "Tunable", raw: str) -> Any:
    """Best-effort type coercion for env var strings. Mirrors how
    pydantic-settings would coerce the same value, but we parse it
    explicitly so resolution doesn't depend on re-reading settings."""
    if tunable.type_ is int:
        return int(raw)
    if tunable.type_ is float:
        return float(raw)
    if tunable.type_ is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return raw


def _toml_lookup(key: str) -> tuple[bool, Any]:
    """Check whether ``key`` is explicitly present in the discovered
    hafiz.toml. Returns ``(found, value)`` — ``found`` distinguishes
    "TOML set this to its default value" from "TOML didn't mention it".
    """
    path = find_config_file()
    if path is None:
        return False, None
    try:
        data = load_toml(path)
    except OSError:
        return False, None
    obj: Any = data
    for part in key.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return False, None
        obj = obj[part]
    return True, obj


def _sticky_lookup(key: str) -> tuple[bool, Any]:
    """Check sticky state. Returns ``(found, value)``. Imports are
    deferred so tunables.py doesn't need host_probe at module load."""
    from hafiz.core.host_probe import probe_host
    from hafiz.core.tuning_state import get_value

    host = probe_host()
    value = get_value(
        key,
        fingerprint=host.fingerprint,
        ort_version=host.onnxruntime_version,
    )
    return (value is not None, value)


def resolve(key: str) -> Any:
    """Effective value through the full precedence chain.

    Order: **env → TOML → sticky → default**.

    Callers at site-of-use (chunker, ingest) should always go through
    this function rather than reading pydantic settings directly —
    pydantic doesn't know about the sticky layer and will skip it.
    """
    return resolve_with_source(key)[0]


def resolve_with_source(key: str) -> tuple[Any, Source]:
    """Same resolution as :func:`resolve`, plus the layer the value came from.

    Used by ``hafiz config show`` / ``hafiz config get`` so users can see
    *why* a tunable is set to a given value, not just what the value is.
    """
    t = get(key)

    env_name = _env_var_name(key)
    if env_name in os.environ:
        return _coerce(t, os.environ[env_name]), "env"

    found, value = _toml_lookup(key)
    if found:
        return value, "toml"

    found, value = _sticky_lookup(key)
    if found:
        return value, "sticky"

    return t.default, "default"


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------


def _positive_int(value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"expected int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"expected positive int, got {value}")


def _lazy_probe_max_part_chars(host: Any) -> ProbeResult:
    """Trampoline to keep the heavy fastembed import out of module load."""
    from hafiz.core.probers import probe_max_part_chars

    return probe_max_part_chars(host)


register(
    Tunable(
        key="embedding.max_part_chars",
        default=2_000,
        type_=int,
        description=(
            "Max characters per embedding part. ONNX attention is O(n²) in "
            "sequence length, so long parts blow up memory on CPU. "
            "Raise this on GPU hosts (16_000+) or after a successful "
            "`hafiz doctor --apply`."
        ),
        validator=_positive_int,
        prober=_lazy_probe_max_part_chars,
    )
)


register(
    Tunable(
        key="ingest.max_file_bytes",
        default=2_097_152,  # 2 MB
        type_=int,
        description=(
            "Skip files larger than this during ingest. Guards against "
            "minified bundles, generated output, and accidentally-committed "
            "binaries that slip past the binary-content probe."
        ),
        validator=_positive_int,
        # policy cap — no prober
    )
)
