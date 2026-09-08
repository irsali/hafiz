"""Sticky device state for embedding model selection.

Caches the outcome of the GPU probe at ~/.cache/hafiz/device_state.json so
subsequent hafiz invocations skip the probe and go straight to the resolved
device. Users inspect via `hafiz embedding status` and force a re-probe via
`hafiz embedding retry`.

State schema:
  device              "cpu" | "gpu"
  reason              human-facing message (None when device=gpu and all clear)
  reason_category     "out_of_memory" | "provider_unavailable" |
                      "unsupported_arch" | "non_finite_output" |
                      "unknown" | None
  probed_at           ISO-8601 UTC timestamp (seconds precision)
  onnxruntime_version ORT version stamped at probe time (displayed; staleness
                      reads the fingerprint below, which subsumes it)
  probe_fingerprint   the host facts that can change a probe's outcome; a
                      mismatch invalidates the verdict. None on states written
                      before fingerprinting existed, which re-probe once.
  gpu_name            first CUDA device name if known, else None

Invalidation is by **cause**, not by clock. Every rung of the GPU remediation
ladder is a package change that leaves the ORT *version* untouched — uninstall
the shadowing CPU wheel, ``pip install tensorrt`` — so version-only staleness
meant hafiz ignored the fix it had just recommended in its own error message.
The fingerprint sees those; a timer would not, and would additionally re-probe
when nothing had changed. That matters because probing is not free or
side-effect-free: it builds a GPU session and runs a real embed, so a timed
re-probe makes an arbitrary later command slow and, in the out-of-memory case,
takes VRAM from whatever process holds it.

The one exception is that case. ``out_of_memory`` means another process held
the VRAM; no fingerprint can observe that, and it resolves on its own. So it —
and only it — also expires on time.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

#: How long a VRAM-contention fallback stands before hafiz tries the GPU again.
#: Long enough that a re-probe is a rare event rather than a recurring tax on
#: whichever command happens to trigger it.
_OOM_RETRY_AFTER = timedelta(hours=24)


@dataclass
class DeviceState:
    device: str
    reason: str | None
    reason_category: str | None
    probed_at: str
    onnxruntime_version: str | None
    gpu_name: str | None
    #: Defaulted so a state file written before this field existed still loads
    #: (``load_state`` would otherwise treat the missing key as corruption and
    #: delete it). Such a state has unknown provenance, so it re-probes once.
    probe_fingerprint: str | None = None


def cache_file_path() -> Path:
    """XDG-compliant cache file location for the device-state JSON."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "hafiz" / "device_state.json"


def load_state() -> DeviceState | None:
    path = cache_file_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return DeviceState(**data)
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        logger.warning("Corrupt device-state cache at %s (%s); removing.", path, exc)
        try:
            path.unlink()
        except OSError:
            pass
        return None


def save_state(state: DeviceState) -> None:
    path = cache_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not persist device state at %s: %s", path, exc)


def clear_state() -> bool:
    """Delete the cache file. Returns True if a file was actually removed."""
    path = cache_file_path()
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError as exc:
        logger.warning("Could not clear device state at %s: %s", path, exc)
        return False


def _ort_version() -> str | None:
    try:
        import onnxruntime as ort

        return ort.__version__
    except ImportError:
        return None


def _provider_set() -> frozenset[str]:
    try:
        import onnxruntime as ort

        return frozenset(ort.get_available_providers())
    except ImportError:
        return frozenset()


def probe_fingerprint() -> str:
    """The host facts that can change the outcome of a device probe.

    Deliberately cheap: a version string and a provider-list lookup, both of
    which the caller has already paid for by importing onnxruntime. This runs
    on the hot path of every ``auto``-device invocation, which is why
    :func:`hafiz.core.embeddings._gpu_name` is **not** part of it — that shells
    out to ``nvidia-smi`` with a two-second timeout, and a GPU being renamed
    under a stable provider list cannot change a probe's outcome anyway.
    """
    providers = _provider_set()
    return "|".join(
        (
            _ort_version() or "?",
            "cuda" if "CUDAExecutionProvider" in providers else "-",
            "trt" if "TensorrtExecutionProvider" in providers else "-",
        )
    )


def _probe_age(state: DeviceState) -> timedelta | None:
    try:
        return datetime.now(UTC) - datetime.fromisoformat(state.probed_at)
    except (TypeError, ValueError):
        return None


def staleness_reason(state: DeviceState) -> str | None:
    """Why ``state`` should be re-probed, in one human phrase — None to keep it.

    The reason is surfaced by ``hafiz embedding status`` rather than kept
    internal: "stale: yes" without a cause tells the user nothing they can act
    on, and the causes here are all things they did (installed a wheel, freed
    VRAM) and would recognise.
    """
    if state.probe_fingerprint is None:
        return "recorded before probe fingerprinting — re-probing once"

    current = probe_fingerprint()
    if current != state.probe_fingerprint:
        return f"host changed since the probe ({state.probe_fingerprint} → {current})"

    if state.reason_category == "out_of_memory":
        age = _probe_age(state)
        if age is not None and age >= _OOM_RETRY_AFTER:
            return (
                f"prior fallback was VRAM contention {age.days * 24 + age.seconds // 3600}h "
                "ago, which is transient — retrying the GPU"
            )

    return None


def is_stale(state: DeviceState) -> bool:
    """True when the cached state should be invalidated. See :func:`staleness_reason`."""
    return staleness_reason(state) is not None


def build_state(
    device: str,
    *,
    reason: str | None,
    category: str | None,
    gpu_name: str | None,
) -> DeviceState:
    return DeviceState(
        device=device,
        reason=reason,
        reason_category=category,
        probed_at=datetime.now(UTC).isoformat(timespec="seconds"),
        onnxruntime_version=_ort_version(),
        gpu_name=gpu_name,
        probe_fingerprint=probe_fingerprint(),
    )


def classify_exception(exc: BaseException) -> tuple[str, str]:
    """Categorize a CUDA/ORT exception. Returns (category, human_message)."""
    text = str(exc) if exc else ""
    low = text.lower()

    if (
        "out of memory" in low
        or "cuda_error_out_of_memory" in low
        or "cudaerrormemoryallocation" in low
    ):
        return (
            "out_of_memory",
            "GPU out of memory (likely VRAM contention with another process).",
        )
    if "cudaexecutionprovider" in low and (
        "not available" in low or "failed to create" in low or "unable to load" in low
    ):
        return (
            "provider_unavailable",
            "CUDA provider not available (driver, runtime, or ORT build mismatch).",
        )
    if "no kernel image" in low or ("unsupported" in low and "compute" in low):
        return (
            "unsupported_arch",
            "GPU architecture not supported by this ORT build.",
        )
    if "non-finite" in low or "nan/inf" in low:
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        return ("non_finite_output", first_line[:400])
    if "cuda driver version" in low and "insufficient" in low:
        return (
            "provider_unavailable",
            "CUDA driver version is insufficient for this runtime.",
        )
    first_line = text.strip().splitlines()[0] if text.strip() else "Unknown CUDA failure."
    return "unknown", first_line[:200]
