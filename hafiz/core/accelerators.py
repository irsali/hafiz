"""Diagnose why embedding is running on CPU when the host can do better.

Motivating case, measured: a host with an RTX 5060 Ti reported
``onnx_providers = ['AzureExecutionProvider', 'CPUExecutionProvider']`` — no
CUDA. Each query cost ~2.2s wall but ~23s CPU across cores.

The obvious diagnosis ("the GPU extra isn't installed") was wrong, and acting on
it would not have fixed the host. ``hafiz[gpu]`` *was* installed. Both wheels
were present in the venv:

    onnxruntime-1.27.0.dist-info       <- CPU wheel
    onnxruntime_gpu-1.27.0.dist-info   <- GPU wheel
    onnxruntime/                       <- ONE package directory

They share the import name ``onnxruntime``. ``fastembed`` depends on the CPU
wheel, so installing it after (or alongside) the GPU wheel overwrites the shared
package directory, leaving the GPU distribution's metadata behind and its
providers gone. Recommending an extra install here is worse than useless: it
would appear to succeed and change nothing.

So this module distinguishes three states, and only one of them is fixed by
installing an extra:

* **shadowed** — the accelerator wheel is installed but its provider is absent.
  Uninstall the CPU wheel; do not reinstall the extra.
* **missing** — hardware present, no accelerator wheel. Install the extra.
* **active** / **no hardware** — nothing to do.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import distributions
from pathlib import Path


@dataclass
class AcceleratorFinding:
    """One accelerator's state on this host, and what to do about it."""

    name: str  # "cuda" | "openvino"
    provider: str  # the ONNX Runtime provider that would appear
    hardware: str | None  # what was detected, or None
    #: "active" | "shadowed" | "missing" | "no-hardware"
    state: str
    detail: str
    fix: str

    @property
    def ok(self) -> bool:
        """True when there is nothing actionable."""
        return self.state in ("active", "no-hardware")


# Distribution names that provide the `onnxruntime` import package. Only one can
# win, because they all install into the same directory.
_ORT_DISTRIBUTIONS = {
    "onnxruntime": "cpu",
    "onnxruntime-gpu": "cuda",
    "onnxruntime-openvino": "openvino",
}


def installed_ort_distributions() -> dict[str, str]:
    """Map installed onnxruntime distribution names to their version.

    Reads distribution *metadata*, which is what survives the shadowing: the
    package directory is overwritten but each wheel's ``.dist-info`` remains.
    That divergence is precisely the signal.
    """
    found: dict[str, str] = {}
    for dist in distributions():
        try:
            name = (dist.metadata["Name"] or "").strip().lower()
        except Exception:
            continue
        if name in _ORT_DISTRIBUTIONS:
            found[name] = dist.version or "?"
    return found


def _intel_npu_present() -> bool:
    """True if an Intel NPU accelerator device node exists."""
    return any(Path(p).exists() for p in ("/dev/accel/accel0", "/dev/accel0"))


def diagnose_accelerators(
    *, providers: tuple[str, ...] | list[str], gpu_name: str | None
) -> list[AcceleratorFinding]:
    """Assess each accelerator hafiz can use against what this host has.

    ``providers`` is the ONNX Runtime provider list (from
    :func:`hafiz.core.host_probe.probe_host`); ``gpu_name`` is the nvidia-smi
    name, or None.
    """
    available = set(providers)
    installed = installed_ort_distributions()
    findings: list[AcceleratorFinding] = []

    # ── CUDA ──
    cuda_provider = "CUDAExecutionProvider"
    if cuda_provider in available:
        findings.append(
            AcceleratorFinding(
                name="cuda",
                provider=cuda_provider,
                hardware=gpu_name,
                state="active",
                detail=f"CUDA available ({gpu_name or 'GPU'})",
                fix="",
            )
        )
    elif not gpu_name:
        findings.append(
            AcceleratorFinding(
                name="cuda",
                provider=cuda_provider,
                hardware=None,
                state="no-hardware",
                detail="no NVIDIA GPU detected",
                fix="",
            )
        )
    elif "onnxruntime-gpu" in installed:
        findings.append(
            AcceleratorFinding(
                name="cuda",
                provider=cuda_provider,
                hardware=gpu_name,
                state="shadowed",
                detail=(
                    f"onnxruntime-gpu {installed['onnxruntime-gpu']} is installed but "
                    f"{cuda_provider} is absent — the CPU wheel "
                    f"(onnxruntime {installed.get('onnxruntime', '?')}) shares the same "
                    f"import package and is shadowing it"
                ),
                fix=(
                    "pipx runpip hafiz uninstall -y onnxruntime "
                    "(keep onnxruntime-gpu; do NOT reinstall the extra — it is "
                    "already installed and would be shadowed again)"
                ),
            )
        )
    else:
        findings.append(
            AcceleratorFinding(
                name="cuda",
                provider=cuda_provider,
                hardware=gpu_name,
                state="missing",
                detail=f"{gpu_name} present but no CUDA onnxruntime installed",
                fix="pipx install -e '.[cuda]' --force  (or: pipx inject hafiz onnxruntime-gpu)",
            )
        )

    # ── OpenVINO (Intel iGPU / NPU) ──
    ov_provider = "OpenVINOExecutionProvider"
    npu = _intel_npu_present()
    if ov_provider in available:
        findings.append(
            AcceleratorFinding(
                name="openvino",
                provider=ov_provider,
                hardware="Intel NPU" if npu else "Intel",
                state="active",
                detail="OpenVINO available",
                fix="",
            )
        )
    elif not npu:
        findings.append(
            AcceleratorFinding(
                name="openvino",
                provider=ov_provider,
                hardware=None,
                state="no-hardware",
                detail="no Intel NPU device node found",
                fix="",
            )
        )
    elif "onnxruntime-openvino" in installed:
        findings.append(
            AcceleratorFinding(
                name="openvino",
                provider=ov_provider,
                hardware="Intel NPU",
                state="shadowed",
                detail=(
                    f"onnxruntime-openvino {installed['onnxruntime-openvino']} is "
                    f"installed but {ov_provider} is absent — the CPU wheel is "
                    f"shadowing it"
                ),
                fix="pipx runpip hafiz uninstall -y onnxruntime",
            )
        )
    else:
        findings.append(
            AcceleratorFinding(
                name="openvino",
                provider=ov_provider,
                hardware="Intel NPU",
                state="missing",
                detail=(
                    "Intel NPU device node present but no OpenVINO onnxruntime "
                    "installed. Unmeasured for this embedding model — treat as "
                    "an experiment, not a recommendation"
                ),
                fix="pipx install -e '.[openvino]' --force",
            )
        )

    return findings
