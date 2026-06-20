"""Host inspection — pure-read, no side effects.

Gathers the facts every tunable prober needs to make recommendations:
total/available RAM, CPU count, onnxruntime providers, GPU name + VRAM.
No heavy deps — Linux /proc parsing, macOS sysctl/vm_stat, and a
subprocess to nvidia-smi, all fail-soft so the probe never blocks hafiz
on an unusual host.

The :class:`HostProbe` is frozen and hashable so it can key cache
invalidation later (phase 3 uses ``fingerprint`` to decide whether
sticky recommendations still apply to the current machine).
"""

from __future__ import annotations

import hashlib
import logging
import platform as _platform
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostProbe:
    """Snapshot of host capabilities relevant to hafiz tuning.

    All sizes are in **MB** to keep JSON output readable and avoid
    int-overflow surprises in downstream consumers. Fields that we
    couldn't measure come back as ``None`` — callers should tolerate
    partial probes (probing on an unusual host shouldn't crash).
    """

    ram_total_mb: int | None
    ram_available_mb: int | None
    swap_total_mb: int | None
    swap_used_mb: int | None
    cpu_count: int | None
    platform: str  # e.g. "linux-x86_64"
    onnx_providers: tuple[str, ...]
    gpu_name: str | None
    gpu_vram_total_mb: int | None
    gpu_vram_free_mb: int | None
    onnxruntime_version: str | None
    # Free-form measurements that don't warrant a typed field yet.
    # Keeps the schema stable while we iterate on what probers actually need.
    extra: dict[str, Any] = field(default_factory=dict)

    # -- fingerprinting -------------------------------------------------

    @property
    def fingerprint(self) -> str:
        """Stable hash keyed to host class, not exact numbers.

        Used to decide whether sticky tuning state is still valid for
        this machine. Uses *classes* of RAM/VRAM rather than exact MB
        so a tiny fluctuation (free RAM drifting between runs) doesn't
        invalidate recommendations. Hash changes when:

          - OS / arch changes
          - RAM class changes (8 / 16 / 32 / 64 / 128+ GB bucket)
          - GPU presence or name changes
          - onnxruntime build providers change (CPU-only → CUDA or back)

        All of those warrant re-probing; anything else doesn't.
        """
        parts = [
            self.platform,
            str(_ram_class(self.ram_total_mb)),
            "|".join(sorted(self.onnx_providers)),
            self.gpu_name or "no-gpu",
            str(_vram_class(self.gpu_vram_total_mb)),
        ]
        blob = "\x1f".join(parts).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready dict. Tuples become lists so ``json.dumps`` works."""
        return {
            "ram_total_mb": self.ram_total_mb,
            "ram_available_mb": self.ram_available_mb,
            "swap_total_mb": self.swap_total_mb,
            "swap_used_mb": self.swap_used_mb,
            "cpu_count": self.cpu_count,
            "platform": self.platform,
            "onnx_providers": list(self.onnx_providers),
            "gpu_name": self.gpu_name,
            "gpu_vram_total_mb": self.gpu_vram_total_mb,
            "gpu_vram_free_mb": self.gpu_vram_free_mb,
            "onnxruntime_version": self.onnxruntime_version,
            "fingerprint": self.fingerprint,
            "extra": dict(self.extra),
        }


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def probe_host() -> HostProbe:
    """Gather host facts. Never raises — unreachable fields come back None."""
    ram = _read_memory()
    cpu = _cpu_count()
    ort_providers, ort_version = _onnxruntime_info()
    gpu_name, gpu_total, gpu_free = _nvidia_smi()

    return HostProbe(
        ram_total_mb=ram.get("mem_total_mb"),
        ram_available_mb=ram.get("mem_available_mb"),
        swap_total_mb=ram.get("swap_total_mb"),
        swap_used_mb=ram.get("swap_used_mb"),
        cpu_count=cpu,
        platform=_platform_tag(),
        onnx_providers=tuple(ort_providers),
        gpu_name=gpu_name,
        gpu_vram_total_mb=gpu_total,
        gpu_vram_free_mb=gpu_free,
        onnxruntime_version=ort_version,
    )


# ---------------------------------------------------------------------------
# Helpers — each one fail-soft
# ---------------------------------------------------------------------------


def _platform_tag() -> str:
    try:
        return f"{_platform.system().lower()}-{_platform.machine().lower()}"
    except Exception:
        return "unknown"


def _cpu_count() -> int | None:
    try:
        import os

        return os.cpu_count()
    except Exception:
        return None


def _read_memory() -> dict[str, int]:
    """Total/available RAM + swap in MB, by platform. Empty dict if unknown."""
    system = _platform.system()
    if system == "Darwin":
        return _read_memory_darwin()
    return _read_meminfo_linux()


def _read_meminfo_linux() -> dict[str, int]:
    """Parse /proc/meminfo on Linux. Returns empty dict on other platforms."""
    out: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
    except OSError:
        return out

    # /proc/meminfo values are in kB. Convert to MB.
    want = {
        "MemTotal": "mem_total_mb",
        "MemAvailable": "mem_available_mb",
        "SwapTotal": "swap_total_mb",
        "SwapFree": "_swap_free_kb",
    }
    parsed: dict[str, int] = {}
    for line in lines:
        key, _, rest = line.partition(":")
        if key in want:
            value = rest.strip().split()
            if value and value[0].isdigit():
                parsed[key] = int(value[0])  # kB

    if "MemTotal" in parsed:
        out["mem_total_mb"] = parsed["MemTotal"] // 1024
    if "MemAvailable" in parsed:
        out["mem_available_mb"] = parsed["MemAvailable"] // 1024
    if "SwapTotal" in parsed:
        out["swap_total_mb"] = parsed["SwapTotal"] // 1024
        if "_swap_free_kb" in parsed:
            # Used = total - free. /proc doesn't publish SwapUsed directly.
            out["swap_used_mb"] = (parsed["SwapTotal"] - parsed["_swap_free_kb"]) // 1024
    return out


def _sysctl_int(key: str) -> int | None:
    """Read an integer sysctl value, or None if unavailable."""
    try:
        out = subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL, timeout=2)
    except (subprocess.SubprocessError, OSError):
        return None
    text = out.decode().strip()
    return int(text) if text.isdigit() else None


def _read_memory_darwin() -> dict[str, int]:
    """RAM + swap on macOS via sysctl / vm_stat. Empty dict if unreadable.

    Total RAM is ``hw.memsize`` (bytes). "Available" has no direct equivalent
    to Linux's MemAvailable; we approximate it as (free + inactive) pages from
    ``vm_stat`` — inactive pages are reclaimable on demand. Swap comes from
    ``vm.swapusage``. Every field is independent and fail-soft.
    """
    out: dict[str, int] = {}

    total_bytes = _sysctl_int("hw.memsize")
    if total_bytes is not None:
        out["mem_total_mb"] = total_bytes // (1024 * 1024)

    page_size = _sysctl_int("hw.pagesize") or 4096
    try:
        vm = subprocess.check_output(["vm_stat"], stderr=subprocess.DEVNULL, timeout=2).decode()
    except (subprocess.SubprocessError, OSError):
        vm = ""
    if vm:
        pages: dict[str, int] = {}
        for line in vm.splitlines():
            key, _, rest = line.partition(":")
            digits = rest.strip().rstrip(".")
            if digits.isdigit():
                pages[key.strip()] = int(digits)
        free = pages.get("Pages free", 0)
        inactive = pages.get("Pages inactive", 0)
        if free or inactive:
            out["mem_available_mb"] = (free + inactive) * page_size // (1024 * 1024)

    # vm.swapusage → "total = 2048.00M  used = 512.00M  free = 1536.00M"
    try:
        swap = subprocess.check_output(
            ["sysctl", "-n", "vm.swapusage"], stderr=subprocess.DEVNULL, timeout=2
        ).decode()
    except (subprocess.SubprocessError, OSError):
        swap = ""
    swap_fields = dict(_parse_swapusage(swap))
    if "total" in swap_fields:
        out["swap_total_mb"] = swap_fields["total"]
        if "used" in swap_fields:
            out["swap_used_mb"] = swap_fields["used"]
    return out


def _parse_swapusage(text: str) -> list[tuple[str, int]]:
    """Pull (label, MB) pairs from ``vm.swapusage`` output. Tolerates K/M/G."""
    multiplier = {"K": 1 / 1024, "M": 1.0, "G": 1024.0}
    pairs: list[tuple[str, int]] = []
    for label in ("total", "used", "free"):
        marker = f"{label} = "
        idx = text.find(marker)
        if idx < 0:
            continue
        token = text[idx + len(marker) :].split()[0]  # e.g. "512.00M"
        unit = token[-1].upper()
        if unit not in multiplier:
            continue
        try:
            mb = float(token[:-1]) * multiplier[unit]
        except ValueError:
            continue
        pairs.append((label, int(mb)))
    return pairs


def _onnxruntime_info() -> tuple[list[str], str | None]:
    """Return (available execution providers, ort version)."""
    try:
        import onnxruntime as ort

        return list(ort.get_available_providers()), getattr(ort, "__version__", None)
    except ImportError:
        return [], None
    except Exception as exc:
        logger.debug("onnxruntime probe failed: %s", exc)
        return [], None


def _nvidia_smi() -> tuple[str | None, int | None, int | None]:
    """Return (gpu_name, vram_total_mb, vram_free_mb) for GPU 0, or Nones."""
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                    "-i",
                    "0",
                ],
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            .decode()
            .strip()
        )
    except (subprocess.SubprocessError, OSError):
        return None, None, None

    first = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in first.split(",")]
    if len(parts) != 3:
        return None, None, None

    name = parts[0] or None
    try:
        total = int(parts[1])
        free = int(parts[2])
    except ValueError:
        return name, None, None
    return name, total, free


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------


def _ram_class(mb: int | None) -> str:
    """Bucket RAM into classes so minor fluctuations don't invalidate cache."""
    if mb is None:
        return "unknown"
    gb = mb / 1024
    for bound, label in ((8, "8"), (16, "16"), (32, "32"), (64, "64"), (128, "128")):
        if gb <= bound:
            return label
    return "256+"


def _vram_class(mb: int | None) -> str:
    if mb is None:
        return "none"
    gb = mb / 1024
    for bound, label in ((4, "4"), (8, "8"), (12, "12"), (16, "16"), (24, "24")):
        if gb <= bound:
            return label
    return "32+"
