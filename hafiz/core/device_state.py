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
  onnxruntime_version ORT version stamped at probe time; drives auto-invalidation
  gpu_name            first CUDA device name if known, else None
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DeviceState:
    device: str
    reason: str | None
    reason_category: str | None
    probed_at: str
    onnxruntime_version: str | None
    gpu_name: str | None


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
        data = json.loads(path.read_text())
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
        path.write_text(json.dumps(asdict(state), indent=2))
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


def is_stale(state: DeviceState) -> bool:
    """True when the cached state should be invalidated (e.g. ORT upgraded)."""
    current = _ort_version()
    if current is None or state.onnxruntime_version is None:
        return False
    return current != state.onnxruntime_version


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
