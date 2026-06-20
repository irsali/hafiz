"""Tests for hafiz.core.host_probe.

The probe is fail-soft (unreachable fields → None), so these tests
exercise the parsing logic directly against synthetic inputs rather
than the host machine.
"""

from __future__ import annotations

from hafiz.core.host_probe import (
    HostProbe,
    _parse_swapusage,
    _ram_class,
    _read_memory_darwin,
    _vram_class,
    probe_host,
)

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
    assert _ram_class(4_000) == "8"  # 4 GB → bucket "8"
    assert _ram_class(16_000) == "16"
    assert _ram_class(17_000) == "32"  # 16.6 GB → bucket "32"
    assert _ram_class(200_000) == "256+"  # ~195 GB exceeds the 128 bucket


def test_vram_class_handles_absent_gpu():
    assert _vram_class(None) == "none"
    assert _vram_class(16_000) == "16"
    assert _vram_class(80_000) == "32+"


# ── macOS memory probe (mocked subprocess) ─────────────────────────────


def test_parse_swapusage_pulls_total_and_used():
    line = "total = 2048.00M  used = 512.00M  free = 1536.00M"
    fields = dict(_parse_swapusage(line))
    assert fields["total"] == 2048
    assert fields["used"] == 512
    assert fields["free"] == 1536


def test_parse_swapusage_handles_units_and_garbage():
    assert dict(_parse_swapusage("total = 1.00G  used = 0.00M  free = 1.00G")) == {
        "total": 1024,
        "used": 0,
        "free": 1024,
    }
    assert _parse_swapusage("") == []
    assert _parse_swapusage("no swap configured") == []


def test_read_memory_darwin_parses_sysctl_and_vm_stat(monkeypatch):
    page_size = 4096
    sysctl = {
        "hw.memsize": str(16 * 1024 * 1024 * 1024),  # 16 GiB in bytes
        "hw.pagesize": str(page_size),
        "vm.swapusage": "total = 1024.00M  used = 256.00M  free = 768.00M",
    }
    vm_stat = (
        "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n"
        "Pages free:                       262144.\n"  # 1 GiB free
        "Pages inactive:                   262144.\n"  # 1 GiB inactive
        "Pages active:                     500000.\n"
    )

    def fake_check_output(cmd, **kwargs):
        if cmd[0] == "sysctl":
            return sysctl[cmd[-1]].encode()
        if cmd[0] == "vm_stat":
            return vm_stat.encode()
        raise AssertionError(f"unexpected command {cmd!r}")

    monkeypatch.setattr("hafiz.core.host_probe.subprocess.check_output", fake_check_output)

    mem = _read_memory_darwin()
    assert mem["mem_total_mb"] == 16 * 1024
    # available ≈ (free + inactive) pages × page size → 2 GiB → 2048 MB
    assert mem["mem_available_mb"] == 2048
    assert mem["swap_total_mb"] == 1024
    assert mem["swap_used_mb"] == 256


def test_read_memory_darwin_fail_soft_when_tools_missing(monkeypatch):
    def boom(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr("hafiz.core.host_probe.subprocess.check_output", boom)
    # Every field unreadable → empty dict, never raises.
    assert _read_memory_darwin() == {}
