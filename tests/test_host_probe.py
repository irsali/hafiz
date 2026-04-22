"""Tests for hafiz.core.host_probe.

The probe is fail-soft (unreachable fields → None), so these tests
exercise the parsing logic directly against synthetic inputs rather
than the host machine.
"""

from __future__ import annotations

from hafiz.core.host_probe import HostProbe, _ram_class, _vram_class, probe_host


# ── end-to-end: probing the test host ──────────────────────────────────


def test_probe_host_returns_hostprobe_instance():
    h = probe_host()
    assert isinstance(h, HostProbe)
    assert isinstance(h.platform, str) and h.platform
    # CPU count should be populated on any normal Linux host.
    assert h.cpu_count is None or h.cpu_count > 0
    # Fingerprint is deterministic for a given field set.
    assert h.fingerprint == probe_host().fingerprint


def test_probe_host_as_dict_is_json_ready():
    import json

    h = probe_host()
    blob = json.dumps(h.as_dict())  # must not raise
    back = json.loads(blob)
    assert back["platform"] == h.platform
    assert back["fingerprint"] == h.fingerprint
    assert isinstance(back["onnx_providers"], list)


# ── fingerprint stability ──────────────────────────────────────────────


def test_fingerprint_is_same_for_same_host():
    a = HostProbe(
        ram_total_mb=64_000,
        ram_available_mb=30_000,
        swap_total_mb=8_000,
        swap_used_mb=100,
        cpu_count=16,
        platform="linux-x86_64",
        onnx_providers=("CPUExecutionProvider",),
        gpu_name=None,
        gpu_vram_total_mb=None,
        gpu_vram_free_mb=None,
        onnxruntime_version="1.24.0",
    )
    b = HostProbe(**{**a.__dict__, "ram_available_mb": 25_000, "swap_used_mb": 500})
    # Changing *available* RAM doesn't change the fingerprint — it's
    # fluctuating state, not host class.
    assert a.fingerprint == b.fingerprint


def test_fingerprint_changes_when_ram_class_changes():
    a = HostProbe(
        ram_total_mb=16_000,  # 16 GB bucket
        ram_available_mb=8_000,
        swap_total_mb=0,
        swap_used_mb=0,
        cpu_count=4,
        platform="linux-x86_64",
        onnx_providers=("CPUExecutionProvider",),
        gpu_name=None,
        gpu_vram_total_mb=None,
        gpu_vram_free_mb=None,
        onnxruntime_version=None,
    )
    b = HostProbe(**{**a.__dict__, "ram_total_mb": 64_000})  # 64 GB bucket
    assert a.fingerprint != b.fingerprint


def test_fingerprint_changes_when_gpu_appears():
    a = HostProbe(
        ram_total_mb=32_000,
        ram_available_mb=16_000,
        swap_total_mb=0,
        swap_used_mb=0,
        cpu_count=8,
        platform="linux-x86_64",
        onnx_providers=("CPUExecutionProvider",),
        gpu_name=None,
        gpu_vram_total_mb=None,
        gpu_vram_free_mb=None,
        onnxruntime_version=None,
    )
    b = HostProbe(
        **{
            **a.__dict__,
            "gpu_name": "RTX 4090",
            "gpu_vram_total_mb": 24_000,
            "gpu_vram_free_mb": 20_000,
            "onnx_providers": ("CUDAExecutionProvider", "CPUExecutionProvider"),
        }
    )
    assert a.fingerprint != b.fingerprint


# ── class bucketing ────────────────────────────────────────────────────


def test_ram_class_buckets():
    assert _ram_class(None) == "unknown"
    assert _ram_class(4_000) == "8"       # 4 GB → bucket "8"
    assert _ram_class(16_000) == "16"
    assert _ram_class(17_000) == "32"     # 16.6 GB → bucket "32"
    assert _ram_class(200_000) == "256+"  # ~195 GB exceeds the 128 bucket


def test_vram_class_handles_absent_gpu():
    assert _vram_class(None) == "none"
    assert _vram_class(16_000) == "16"
    assert _vram_class(80_000) == "32+"
