"""Tests for accelerator diagnosis.

The motivating case: a host with an RTX 5060 Ti reported only
``['AzureExecutionProvider', 'CPUExecutionProvider']``. The obvious diagnosis —
"the GPU extra isn't installed" — was wrong. ``hafiz[gpu]`` *was* installed, and
both wheels were present:

    onnxruntime-1.27.0.dist-info       <- CPU wheel
    onnxruntime_gpu-1.27.0.dist-info   <- GPU wheel
    onnxruntime/                       <- ONE package directory

They share the import name, so the CPU wheel (a `fastembed` dependency)
overwrote the GPU one's files while its metadata survived. Telling that user to
install the extra would have appeared to succeed and changed nothing — which is
why "shadowed" has to be a distinct state from "missing".
"""

from __future__ import annotations

import pytest

from hafiz.core.accelerators import diagnose_accelerators

CPU_ONLY = ("AzureExecutionProvider", "CPUExecutionProvider")
WITH_CUDA = ("CUDAExecutionProvider", "CPUExecutionProvider")
WITH_OPENVINO = ("OpenVINOExecutionProvider", "CPUExecutionProvider")
GPU = "NVIDIA GeForce RTX 5060 Ti"


def _find(findings, name):
    return next(f for f in findings if f.name == name)


@pytest.fixture
def no_npu(monkeypatch):
    monkeypatch.setattr("hafiz.core.accelerators._intel_npu_present", lambda: False)


@pytest.fixture
def with_npu(monkeypatch):
    monkeypatch.setattr("hafiz.core.accelerators._intel_npu_present", lambda: True)


def _installed(monkeypatch, mapping: dict[str, str]):
    monkeypatch.setattr("hafiz.core.accelerators.installed_ort_distributions", lambda: mapping)


# ── The shadowing case ──────────────────────────────────────────────────


def test_gpu_wheel_installed_but_provider_absent_is_shadowed(monkeypatch, no_npu):
    """THE regression: the extra is installed; the CPU wheel is shadowing it."""
    _installed(monkeypatch, {"onnxruntime": "1.27.0", "onnxruntime-gpu": "1.27.0"})
    cuda = _find(diagnose_accelerators(providers=CPU_ONLY, gpu_name=GPU), "cuda")
    assert cuda.state == "shadowed"
    assert not cuda.ok


def test_shadowed_fix_uninstalls_the_cpu_wheel(monkeypatch, no_npu):
    _installed(monkeypatch, {"onnxruntime": "1.27.0", "onnxruntime-gpu": "1.27.0"})
    cuda = _find(diagnose_accelerators(providers=CPU_ONLY, gpu_name=GPU), "cuda")
    assert "uninstall" in cuda.fix
    assert "onnxruntime" in cuda.fix


def test_shadowed_fix_explicitly_warns_off_reinstalling_the_extra(monkeypatch, no_npu):
    """Installing the extra again would appear to succeed and change nothing."""
    _installed(monkeypatch, {"onnxruntime": "1.27.0", "onnxruntime-gpu": "1.27.0"})
    cuda = _find(diagnose_accelerators(providers=CPU_ONLY, gpu_name=GPU), "cuda")
    assert "do NOT reinstall" in cuda.fix


def test_shadowed_detail_names_both_versions(monkeypatch, no_npu):
    _installed(monkeypatch, {"onnxruntime": "1.27.0", "onnxruntime-gpu": "1.26.0"})
    cuda = _find(diagnose_accelerators(providers=CPU_ONLY, gpu_name=GPU), "cuda")
    assert "1.26.0" in cuda.detail
    assert "1.27.0" in cuda.detail


# ── The genuinely-missing case ───────────────────────────────────────────


def test_gpu_present_with_no_accelerator_wheel_is_missing(monkeypatch, no_npu):
    _installed(monkeypatch, {"onnxruntime": "1.27.0"})
    cuda = _find(diagnose_accelerators(providers=CPU_ONLY, gpu_name=GPU), "cuda")
    assert cuda.state == "missing"
    assert not cuda.ok


def test_missing_fix_installs_the_extra(monkeypatch, no_npu):
    _installed(monkeypatch, {"onnxruntime": "1.27.0"})
    cuda = _find(diagnose_accelerators(providers=CPU_ONLY, gpu_name=GPU), "cuda")
    assert "cuda" in cuda.fix
    assert "uninstall" not in cuda.fix


# ── Nothing to do ───────────────────────────────────────────────────────


def test_cuda_active_is_ok(monkeypatch, no_npu):
    _installed(monkeypatch, {"onnxruntime-gpu": "1.27.0"})
    cuda = _find(diagnose_accelerators(providers=WITH_CUDA, gpu_name=GPU), "cuda")
    assert cuda.state == "active"
    assert cuda.ok
    assert cuda.fix == ""


def test_no_gpu_is_not_a_problem(monkeypatch, no_npu):
    """A CPU-only host must not be nagged about hardware it doesn't have."""
    _installed(monkeypatch, {"onnxruntime": "1.27.0"})
    cuda = _find(diagnose_accelerators(providers=CPU_ONLY, gpu_name=None), "cuda")
    assert cuda.state == "no-hardware"
    assert cuda.ok


def test_cuda_active_reported_even_without_an_nvidia_smi_name(monkeypatch, no_npu):
    """nvidia-smi can be absent in a container while CUDA still works."""
    _installed(monkeypatch, {"onnxruntime-gpu": "1.27.0"})
    cuda = _find(diagnose_accelerators(providers=WITH_CUDA, gpu_name=None), "cuda")
    assert cuda.state == "active"


# ── OpenVINO / Intel NPU ────────────────────────────────────────────────


def test_npu_present_without_openvino_is_missing(monkeypatch, with_npu):
    _installed(monkeypatch, {"onnxruntime": "1.27.0"})
    ov = _find(diagnose_accelerators(providers=CPU_ONLY, gpu_name=None), "openvino")
    assert ov.state == "missing"


def test_openvino_recommendation_is_honestly_labelled_unmeasured(monkeypatch, with_npu):
    """Offered because the hardware is idle, not because it's known to help."""
    _installed(monkeypatch, {"onnxruntime": "1.27.0"})
    ov = _find(diagnose_accelerators(providers=CPU_ONLY, gpu_name=None), "openvino")
    assert "nmeasured" in ov.detail


def test_no_npu_means_no_openvino_advice(monkeypatch, no_npu):
    _installed(monkeypatch, {"onnxruntime": "1.27.0"})
    ov = _find(diagnose_accelerators(providers=CPU_ONLY, gpu_name=None), "openvino")
    assert ov.state == "no-hardware"
    assert ov.ok


def test_openvino_shadowing_is_detected_too(monkeypatch, with_npu):
    _installed(monkeypatch, {"onnxruntime": "1.27.0", "onnxruntime-openvino": "1.20.0"})
    ov = _find(diagnose_accelerators(providers=CPU_ONLY, gpu_name=None), "openvino")
    assert ov.state == "shadowed"
    assert "uninstall" in ov.fix


def test_openvino_active_is_ok(monkeypatch, with_npu):
    _installed(monkeypatch, {"onnxruntime-openvino": "1.20.0"})
    ov = _find(diagnose_accelerators(providers=WITH_OPENVINO, gpu_name=None), "openvino")
    assert ov.state == "active"
    assert ov.ok


# ── Distribution discovery ──────────────────────────────────────────────


def test_installed_distributions_reads_metadata_not_the_import():
    """Metadata is what survives shadowing — the divergence IS the signal."""
    from hafiz.core.accelerators import installed_ort_distributions

    found = installed_ort_distributions()
    assert isinstance(found, dict)
    for name in found:
        assert name in ("onnxruntime", "onnxruntime-gpu", "onnxruntime-openvino")


def test_every_accelerator_is_reported_exactly_once(monkeypatch, with_npu):
    _installed(monkeypatch, {"onnxruntime": "1.27.0"})
    findings = diagnose_accelerators(providers=CPU_ONLY, gpu_name=GPU)
    assert sorted(f.name for f in findings) == ["cuda", "openvino"]
