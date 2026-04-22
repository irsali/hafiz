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

from dataclasses import dataclass, field
from typing import Any, Callable

from hafiz.core.config import get_settings


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


def resolve(key: str) -> Any:
    """Resolve a tunable's effective value, applying the precedence chain.

    Phase 1: env → TOML → default (pydantic-settings handles all three
    via the HAFIZ_*__* env prefix and the loaded hafiz.toml).

    Phase 3 will insert a sticky-state lookup between TOML and default.
    Callers at site-of-use (chunker, ingest) should always go through
    this function rather than reading settings directly, so that insertion
    is invisible to them.
    """
    # Look up the tunable to validate the key is known. This catches typos
    # at the callsite, which matters once we have a dozen tunables.
    get(key)

    settings = get_settings()
    obj: Any = settings
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


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
