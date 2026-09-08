"""Tests for embedding device selection (config -> sticky -> probe)."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core import config as cfg_mod
from hafiz.core import device_state as dstate
from hafiz.core import embeddings
from hafiz.core.accelerators import AcceleratorFinding

runner = CliRunner()

#: Captured before any fixture stubs it, so the shadowed-accelerator tests can
#: exercise the real filter while still inheriting ``fake_models``.
_REAL_SHADOW_PROBE = embeddings._shadowed_accelerator


@pytest.fixture(autouse=True)
def _isolate_device_state(tmp_path, monkeypatch):
    """Point XDG_CACHE_HOME at a tmp dir and reset module-level state."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("HAFIZ_EMBEDDING__DEVICE", raising=False)
    embeddings.reset_cache()
    cfg_mod.reset_settings()
    yield
    embeddings.reset_cache()
    cfg_mod.reset_settings()


@pytest.fixture
def fake_models(monkeypatch):
    """Swap the heavyweight TextEmbedding builders with sentinels."""
    cpu = MagicMock(name="cpu_model")
    gpu = MagicMock(name="gpu_model")
    monkeypatch.setattr(embeddings, "_build_cpu_model", lambda _m: cpu)
    monkeypatch.setattr(embeddings, "_build_gpu_model", lambda _m: gpu)
    monkeypatch.setattr(embeddings, "_gpu_name", lambda: "FakeGPU 9999")
    monkeypatch.setattr(dstate, "_ort_version", lambda: "1.24.4")
    # The shadow diagnosis reads the *real* venv's distribution metadata, so
    # leaving it live would make these outcomes depend on whether the developer
    # happens to have onnxruntime-gpu installed. TestShadowedAccelerator drives
    # it deliberately instead.
    monkeypatch.setattr(embeddings, "_shadowed_accelerator", lambda _gpu: None)
    return cpu, gpu


def _trap(message: str):
    """Monkeypatch target that blows up if called. Reads better than `lambda:`."""

    def _boom(*_a, **_kw):
        raise AssertionError(message)

    return _boom


# ─── classify_exception ─────────────────────────────────────────────────


class TestClassifyException:
    def test_out_of_memory(self):
        cat, msg = dstate.classify_exception(
            RuntimeError("CUDA_ERROR_OUT_OF_MEMORY: out of memory")
        )
        assert cat == "out_of_memory"
        assert "VRAM" in msg

    def test_provider_unavailable(self):
        cat, _ = dstate.classify_exception(RuntimeError("Failed to create CUDAExecutionProvider"))
        assert cat == "provider_unavailable"

    def test_unsupported_arch(self):
        cat, _ = dstate.classify_exception(
            RuntimeError("no kernel image is available for execution on device")
        )
        assert cat == "unsupported_arch"

    def test_driver_insufficient(self):
        cat, _ = dstate.classify_exception(
            RuntimeError("CUDA driver version is insufficient for runtime")
        )
        assert cat == "provider_unavailable"

    def test_unknown(self):
        cat, msg = dstate.classify_exception(RuntimeError("some weird totally unrelated error"))
        assert cat == "unknown"
        assert "weird" in msg

    def test_non_finite_output(self):
        cat, msg = dstate.classify_exception(
            RuntimeError("GPU probe returned non-finite values (NaN/Inf).")
        )
        assert cat == "non_finite_output"
        assert "non-finite" in msg.lower()


# ─── state file I/O ─────────────────────────────────────────────────────


class TestStateFile:
    def test_missing_cache_returns_none(self):
        assert dstate.load_state() is None

    def test_round_trip(self):
        state = dstate.build_state("gpu", reason=None, category=None, gpu_name="RTX 5060 Ti")
        dstate.save_state(state)
        loaded = dstate.load_state()
        assert loaded is not None
        assert loaded.device == "gpu"
        assert loaded.gpu_name == "RTX 5060 Ti"

    def test_corrupt_cache_is_deleted(self):
        path = dstate.cache_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json")
        assert dstate.load_state() is None
        assert not path.exists()

    def test_clear_state(self):
        dstate.save_state(dstate.build_state("cpu", reason="x", category="unknown", gpu_name=None))
        assert dstate.clear_state() is True
        assert dstate.clear_state() is False

    def test_is_stale_on_version_change(self, monkeypatch):
        monkeypatch.setattr(dstate, "_ort_version", lambda: "1.24.4")
        state = dstate.build_state("gpu", reason=None, category=None, gpu_name=None)
        monkeypatch.setattr(dstate, "_ort_version", lambda: "99.99.99")
        assert dstate.is_stale(state) is True

    def test_is_stale_matches(self, monkeypatch):
        monkeypatch.setattr(dstate, "_ort_version", lambda: "1.24.4")
        state = dstate.build_state("gpu", reason=None, category=None, gpu_name=None)
        assert dstate.is_stale(state) is False


# ─── staleness: cause, not clock ───────────────────────────────────────


class TestStaleness:
    """Invalidation tracks the *cause* of a verdict, not elapsed time.

    The bug this class pins: every rung of the GPU remediation ladder is a
    package change that leaves ``ort.__version__`` untouched. Under the old
    version-only rule the sticky ``cpu`` verdict survived all of them, so hafiz
    ignored the fix its own error message had recommended.
    """

    CUDA = "CUDAExecutionProvider"
    TRT = "TensorrtExecutionProvider"

    def _state(self, monkeypatch, *, providers=(), ort="1.24.4", category=None, age_hours=0.0):
        """A state as it would have been written on a host with these facts."""
        monkeypatch.setattr(dstate, "_ort_version", lambda: ort)
        monkeypatch.setattr(dstate, "_provider_set", lambda: frozenset(providers))
        state = dstate.build_state("cpu", reason="x", category=category, gpu_name=None)
        if age_hours:
            when = datetime.now(UTC) - timedelta(hours=age_hours)
            state.probed_at = when.isoformat(timespec="seconds")
        return state

    @staticmethod
    def _now(monkeypatch, *, providers=(), ort="1.24.4"):
        """Re-point the host facts to what a *later* run would observe."""
        monkeypatch.setattr(dstate, "_ort_version", lambda: ort)
        monkeypatch.setattr(dstate, "_provider_set", lambda: frozenset(providers))

    def test_unchanged_host_is_not_stale(self, monkeypatch):
        state = self._state(monkeypatch, providers=(self.CUDA,))
        assert dstate.staleness_reason(state) is None

    def test_unshadowing_the_gpu_wheel_invalidates_the_cpu_verdict(self, monkeypatch):
        """The exact bug: same ORT version, provider appears, verdict must drop.

        ``pipx runpip hafiz uninstall onnxruntime`` un-shadows onnxruntime-gpu.
        The version is identical before and after — only the provider list
        moves — so version-only staleness said "not stale" and the user stayed
        on CPU after doing precisely what ``hafiz doctor`` told them to.
        """
        state = self._state(monkeypatch, providers=(), ort="1.27.0")
        self._now(monkeypatch, providers=(self.CUDA,), ort="1.27.0")
        reason = dstate.staleness_reason(state)
        assert reason is not None
        assert "host changed" in reason

    def test_installing_tensorrt_invalidates_the_verdict(self, monkeypatch):
        """hafiz's own NaN message says `pip install tensorrt`; it must take effect."""
        state = self._state(monkeypatch, providers=(self.CUDA,), category="non_finite_output")
        self._now(monkeypatch, providers=(self.CUDA, self.TRT))
        assert dstate.staleness_reason(state) is not None

    def test_ort_upgrade_still_invalidates(self, monkeypatch):
        state = self._state(monkeypatch, providers=(self.CUDA,), ort="1.24.4")
        self._now(monkeypatch, providers=(self.CUDA,), ort="1.27.0")
        assert dstate.is_stale(state) is True

    def test_losing_a_provider_invalidates_a_gpu_verdict(self, monkeypatch):
        """Shadowing can also happen *later* — a reinstall that puts the CPU wheel last."""
        state = self._state(monkeypatch, providers=(self.CUDA,))
        self._now(monkeypatch, providers=())
        assert dstate.is_stale(state) is True

    def test_a_state_without_a_fingerprint_reprobes_once(self, monkeypatch):
        """Migration: files written before fingerprinting have unknown provenance."""
        state = self._state(monkeypatch, providers=(self.CUDA,))
        state.probe_fingerprint = None
        reason = dstate.staleness_reason(state)
        assert reason is not None
        assert "before probe fingerprinting" in reason

    def test_a_renamed_gpu_does_not_invalidate(self, monkeypatch):
        """``gpu_name`` is deliberately outside the fingerprint — it costs nvidia-smi.

        A different GPU under an unchanged provider list cannot change what the
        probe would conclude, so paying a subprocess on every auto-device
        invocation to notice it would buy nothing.
        """
        state = self._state(monkeypatch, providers=(self.CUDA,))
        state.gpu_name = "some entirely different card"
        assert dstate.is_stale(state) is False

    # ── the one time-based rule ──

    def test_vram_contention_expires_on_time(self, monkeypatch):
        state = self._state(monkeypatch, category="out_of_memory", age_hours=25)
        reason = dstate.staleness_reason(state)
        assert reason is not None
        assert "VRAM contention" in reason

    def test_vram_contention_stands_inside_the_window(self, monkeypatch):
        state = self._state(monkeypatch, category="out_of_memory", age_hours=23)
        assert dstate.staleness_reason(state) is None

    @pytest.mark.parametrize(
        "category", ["provider_unavailable", "unsupported_arch", "non_finite_output", "unknown"]
    )
    def test_no_other_category_expires_on_time(self, monkeypatch, category):
        """These need a package change, which the fingerprint already sees.

        Re-probing them on a timer would rebuild a GPU session and run a real
        embed to re-learn something that cannot have changed.
        """
        state = self._state(monkeypatch, category=category, age_hours=10_000)
        assert dstate.staleness_reason(state) is None

    def test_a_clock_skewed_future_probe_is_not_stale(self, monkeypatch):
        state = self._state(monkeypatch, category="out_of_memory", age_hours=-500)
        assert dstate.staleness_reason(state) is None

    def test_an_unparseable_timestamp_does_not_crash(self, monkeypatch):
        state = self._state(monkeypatch, category="out_of_memory")
        state.probed_at = "not a timestamp"
        assert dstate.staleness_reason(state) is None

    # ── the fingerprint itself ──

    def test_the_fingerprint_is_stable_across_calls(self, monkeypatch):
        self._now(monkeypatch, providers=(self.CUDA,))
        assert dstate.probe_fingerprint() == dstate.probe_fingerprint()

    def test_the_fingerprint_names_what_it_found(self, monkeypatch):
        self._now(monkeypatch, providers=(self.CUDA, self.TRT), ort="1.27.0")
        assert dstate.probe_fingerprint() == "1.27.0|cuda|trt"
        self._now(monkeypatch, providers=(), ort="1.27.0")
        assert dstate.probe_fingerprint() == "1.27.0|-|-"

    def test_a_round_trip_preserves_the_fingerprint(self, monkeypatch):
        state = self._state(monkeypatch, providers=(self.CUDA,))
        dstate.save_state(state)
        assert dstate.load_state().probe_fingerprint == state.probe_fingerprint

    def test_a_state_file_predating_the_field_still_loads(self):
        """Rather than reading as corrupt and being deleted — see the field default."""
        path = dstate.cache_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "device": "cpu",
                    "reason": "CUDAExecutionProvider not available",
                    "reason_category": "provider_unavailable",
                    "probed_at": "2026-06-23T07:24:24+00:00",
                    "onnxruntime_version": "1.27.0",
                    "gpu_name": None,
                }
            )
        )
        loaded = dstate.load_state()
        assert loaded is not None
        assert loaded.probe_fingerprint is None
        assert dstate.is_stale(loaded) is True


# ─── _build_gpu_model probe validation ─────────────────────────────────


class TestGpuBuilderProbe:
    """The un-mocked _build_gpu_model: verify the NaN guard and TRT preference."""

    def _fake_fastembed(self, vector):
        """Return a class that mimics ``fastembed.TextEmbedding`` for the probe call."""

        class _Fake:
            def __init__(self, **kwargs):
                self.providers = kwargs.get("providers")

            def embed(self, texts):
                return [vector for _ in list(texts)]

        return _Fake

    def test_raises_when_probe_returns_nan(self, monkeypatch):
        monkeypatch.setattr(
            embeddings,
            "TextEmbedding",
            self._fake_fastembed([float("nan")] * 4 + [0.1] * 764),
        )
        monkeypatch.setattr(embeddings, "_tensorrt_available", lambda: False)
        with pytest.raises(RuntimeError, match="non-finite"):
            embeddings._build_gpu_model("any-model")

    def test_raises_when_probe_returns_inf(self, monkeypatch):
        monkeypatch.setattr(
            embeddings,
            "TextEmbedding",
            self._fake_fastembed([float("inf")] + [0.0] * 767),
        )
        monkeypatch.setattr(embeddings, "_tensorrt_available", lambda: False)
        with pytest.raises(RuntimeError, match="non-finite"):
            embeddings._build_gpu_model("any-model")

    def test_passes_through_valid_probe(self, monkeypatch):
        monkeypatch.setattr(
            embeddings,
            "TextEmbedding",
            self._fake_fastembed([0.01] * 768),
        )
        monkeypatch.setattr(embeddings, "_tensorrt_available", lambda: False)
        model = embeddings._build_gpu_model("any-model")
        assert model.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def test_trt_available_uses_trt_and_skips_cuda_ep(self, monkeypatch, tmp_path):
        # When TRT is available, CUDA EP is deliberately omitted from providers.
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        monkeypatch.delenv("ORT_TENSORRT_ENGINE_CACHE_ENABLE", raising=False)
        monkeypatch.delenv("ORT_TENSORRT_CACHE_PATH", raising=False)
        monkeypatch.setattr(
            embeddings,
            "TextEmbedding",
            self._fake_fastembed([0.01] * 768),
        )
        monkeypatch.setattr(embeddings, "_tensorrt_available", lambda: True)

        model = embeddings._build_gpu_model("any-model")

        assert model.providers == ["TensorrtExecutionProvider", "CPUExecutionProvider"]
        assert "CUDAExecutionProvider" not in model.providers
        assert os.environ.get("ORT_TENSORRT_ENGINE_CACHE_ENABLE") == "1"
        assert os.environ.get("ORT_TENSORRT_CACHE_PATH", "").endswith("trt_engines")


# ─── probe_device ──────────────────────────────────────────────────────


class TestProbeDevice:
    def test_cpu_never_touches_cuda(self, fake_models, monkeypatch):
        cpu, _gpu = fake_models
        monkeypatch.setattr(
            embeddings,
            "_cuda_available",
            _trap("CUDA should not be checked for device=cpu"),
        )
        model, state = embeddings.probe_device("cpu", "fake-model")
        assert model is cpu
        assert state.device == "cpu"
        assert state.reason_category is None

    def test_gpu_raises_when_cuda_unavailable(self, fake_models, monkeypatch):
        monkeypatch.setattr(embeddings, "_cuda_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDAExecutionProvider is not available"):
            embeddings.probe_device("gpu", "fake-model")

    def test_gpu_raises_on_probe_failure(self, fake_models, monkeypatch):
        monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)
        monkeypatch.setattr(
            embeddings,
            "_build_gpu_model",
            _trap("CUDA_ERROR_OUT_OF_MEMORY"),
        )
        with pytest.raises(RuntimeError, match="probe failed"):
            embeddings.probe_device("gpu", "fake-model")

    def test_auto_no_cuda_goes_cpu(self, fake_models, monkeypatch):
        cpu, _gpu = fake_models
        monkeypatch.setattr(embeddings, "_cuda_available", lambda: False)
        model, state = embeddings.probe_device("auto", "fake-model")
        assert model is cpu
        assert state.device == "cpu"
        assert state.reason_category == "provider_unavailable"
        assert dstate.load_state().device == "cpu"

    def test_auto_cuda_works(self, fake_models, monkeypatch):
        _cpu, gpu = fake_models
        monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)
        model, state = embeddings.probe_device("auto", "fake-model")
        assert model is gpu
        assert state.device == "gpu"
        assert state.gpu_name == "FakeGPU 9999"
        assert dstate.load_state().device == "gpu"

    def test_auto_cuda_fails_falls_back_and_categorizes(self, fake_models, monkeypatch):
        cpu, _gpu = fake_models
        monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)

        def oom(_):
            raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY: out of memory")

        monkeypatch.setattr(embeddings, "_build_gpu_model", oom)
        model, state = embeddings.probe_device("auto", "fake-model")
        assert model is cpu
        assert state.device == "cpu"
        assert state.reason_category == "out_of_memory"
        assert "VRAM" in state.reason


# ─── shadowed accelerators ─────────────────────────────────────────────


class TestShadowedAccelerator:
    """Hardware bought and installed for, but silently unused, gets a panel.

    ``onnxruntime``, ``onnxruntime-gpu`` and ``onnxruntime-openvino`` install
    the same import package, so the last one installed wins and the others'
    providers vanish while their metadata stays. The result reads exactly like
    "no GPU here" — which is why this branch used to be silent, and why that
    was wrong for the user who had paid for the wheel.
    """

    @staticmethod
    def _finding(state: str) -> AcceleratorFinding:
        return AcceleratorFinding(
            name="cuda",
            provider="CUDAExecutionProvider",
            hardware="RTX 5060 Ti",
            state=state,
            detail="onnxruntime-gpu 1.27.0 is installed but the CPU wheel is shadowing it",
            fix="pipx runpip hafiz uninstall -y onnxruntime",
        )

    @pytest.fixture
    def cpu_host(self, fake_models, monkeypatch):
        """An auto-device host with no CUDA provider — the branch under test."""
        monkeypatch.setattr(embeddings, "_cuda_available", lambda: False)
        return fake_models[0]

    @pytest.fixture
    def real_probe(self, monkeypatch):
        """Undo ``fake_models``' hermetic stub so the real filter runs."""
        monkeypatch.setattr(embeddings, "_shadowed_accelerator", _REAL_SHADOW_PROBE)

    def test_shadowed_cuda_is_announced(self, cpu_host, monkeypatch, capsys):
        monkeypatch.setattr(
            embeddings, "_shadowed_accelerator", lambda _gpu: self._finding("shadowed")
        )
        _model, state = embeddings.probe_device("auto", "fake-model")
        err = capsys.readouterr().err
        assert "shadowing" in err
        assert "uninstall" in err
        assert state.device == "cpu"

    def test_the_shadow_detail_lands_in_the_sticky_reason(self, cpu_host, monkeypatch):
        """So `hafiz embedding status` can explain the CPU verdict days later."""
        monkeypatch.setattr(
            embeddings, "_shadowed_accelerator", lambda _gpu: self._finding("shadowed")
        )
        _model, state = embeddings.probe_device("auto", "fake-model")
        assert "shadowing" in state.reason
        assert state.reason_category == "provider_unavailable"

    def test_the_panel_never_reaches_stdout(self, cpu_host, monkeypatch, capsys):
        """`hafiz mcp` speaks JSON-RPC on stdout; one stray byte corrupts the stream."""
        monkeypatch.setattr(
            embeddings, "_shadowed_accelerator", lambda _gpu: self._finding("shadowed")
        )
        embeddings.probe_device("auto", "fake-model")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err != ""

    @pytest.mark.parametrize("finding_state", ["missing", "active", "no-hardware"])
    def test_only_the_shadowed_state_is_announced(
        self, cpu_host, real_probe, monkeypatch, capsys, finding_state
    ):
        """`missing` is a user who never asked for GPU — telling them is noise.

        This is the deliberate boundary: the panel fires for a broken install,
        not for an unexercised opportunity.
        """
        monkeypatch.setattr(
            embeddings, "diagnose_accelerators", lambda **_kw: [self._finding(finding_state)]
        )
        embeddings.probe_device("auto", "fake-model")
        assert capsys.readouterr().err == ""

    def test_the_filter_picks_shadowed_out_of_a_mixed_list(self, real_probe, monkeypatch):
        findings = [self._finding("no-hardware"), self._finding("shadowed")]
        monkeypatch.setattr(embeddings, "diagnose_accelerators", lambda **_kw: findings)
        found = embeddings._shadowed_accelerator("RTX 5060 Ti")
        assert found is not None
        assert found.state == "shadowed"

    def test_a_failing_diagnosis_does_not_break_embedding(
        self, cpu_host, real_probe, monkeypatch, capsys
    ):
        """A diagnosis is a nicety; embedding is the job."""

        def boom(**_kw):
            raise RuntimeError("importlib.metadata blew up")

        monkeypatch.setattr(embeddings, "diagnose_accelerators", boom)
        model, state = embeddings.probe_device("auto", "fake-model")
        assert model is cpu_host
        assert state.device == "cpu"
        assert capsys.readouterr().err == ""

    def test_a_clean_cpu_only_host_stays_silent(self, cpu_host, real_probe, monkeypatch, capsys):
        """No GPU, no wheels, nothing to say — the common case must not gain noise."""
        monkeypatch.setattr(embeddings, "diagnose_accelerators", lambda **_kw: [])
        _model, state = embeddings.probe_device("auto", "fake-model")
        assert capsys.readouterr().err == ""
        assert "not available in this onnxruntime build" in state.reason


# ─── get_embed_model selection ─────────────────────────────────────────


class TestSelection:
    def test_config_cpu_wins_over_sticky_gpu(self, fake_models, monkeypatch):
        cpu, _gpu = fake_models
        dstate.save_state(dstate.build_state("gpu", reason=None, category=None, gpu_name="Old"))
        monkeypatch.setenv("HAFIZ_EMBEDDING__DEVICE", "cpu")
        cfg_mod.reset_settings()
        monkeypatch.setattr(
            embeddings,
            "_cuda_available",
            _trap("CUDA probe should not run with config=cpu"),
        )
        model = embeddings.get_embed_model()
        assert model is cpu
        # Sticky state is not overwritten by explicit-config resolution.
        assert dstate.load_state().device == "gpu"

    def test_sticky_cpu_reused(self, fake_models, monkeypatch):
        cpu, _gpu = fake_models
        dstate.save_state(
            dstate.build_state(
                "cpu",
                reason="prior OOM",
                category="out_of_memory",
                gpu_name=None,
            )
        )
        monkeypatch.setattr(
            embeddings,
            "_cuda_available",
            _trap("should skip probe when sticky cpu state exists"),
        )
        model = embeddings.get_embed_model()
        assert model is cpu

    def test_sticky_gpu_reused(self, fake_models, monkeypatch):
        _cpu, gpu = fake_models
        dstate.save_state(dstate.build_state("gpu", reason=None, category=None, gpu_name="FakeGPU"))
        monkeypatch.setattr(
            embeddings,
            "_cuda_available",
            _trap("should go direct to GPU builder on sticky gpu"),
        )
        model = embeddings.get_embed_model()
        assert model is gpu

    def test_stale_sticky_triggers_reprobe(self, fake_models, monkeypatch):
        _cpu, gpu = fake_models
        monkeypatch.setattr(dstate, "_ort_version", lambda: "1.24.4")
        dstate.save_state(
            dstate.build_state("cpu", reason="old", category="unknown", gpu_name=None)
        )
        monkeypatch.setattr(dstate, "_ort_version", lambda: "2.0.0")
        monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)
        model = embeddings.get_embed_model()
        assert model is gpu
        assert dstate.load_state().device == "gpu"

    def test_no_cache_triggers_probe(self, fake_models, monkeypatch):
        _cpu, gpu = fake_models
        monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)
        model = embeddings.get_embed_model()
        assert model is gpu
        assert dstate.load_state().device == "gpu"

    def test_cached_gpu_failing_reinit_reprobes(self, fake_models, monkeypatch):
        _cpu, gpu = fake_models
        dstate.save_state(dstate.build_state("gpu", reason=None, category=None, gpu_name="Old"))

        calls = {"gpu": 0}

        def flaky_gpu(_):
            calls["gpu"] += 1
            if calls["gpu"] == 1:
                raise RuntimeError("CUDA_ERROR_OUT_OF_MEMORY")
            return gpu

        monkeypatch.setattr(embeddings, "_build_gpu_model", flaky_gpu)
        monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)

        model = embeddings.get_embed_model()
        assert model is gpu
        assert calls["gpu"] == 2


# ─── CLI commands ───────────────────────────────────────────────────────


class TestCLI:
    def test_status_help(self):
        result = runner.invoke(app, ["embedding", "status", "--help"])
        assert result.exit_code == 0
        assert "--json" in result.output

    def test_retry_help(self):
        result = runner.invoke(app, ["embedding", "retry", "--help"])
        assert result.exit_code == 0

    def test_status_json_no_cache(self):
        result = runner.invoke(app, ["embedding", "status", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["configured"] == "auto"
        assert data["source"] == "not-probed"
        assert data["sticky"] is None

    def test_status_json_with_cache(self, fake_models):
        dstate.save_state(
            dstate.build_state(
                "cpu",
                reason="prior OOM",
                category="out_of_memory",
                gpu_name=None,
            )
        )
        result = runner.invoke(app, ["embedding", "status", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["source"] == "sticky"
        assert data["effective_device"] == "cpu"
        assert data["sticky"]["reason_category"] == "out_of_memory"

    def test_retry_clears_and_reprobes(self, fake_models, monkeypatch):
        dstate.save_state(
            dstate.build_state(
                "cpu",
                reason="prior OOM",
                category="out_of_memory",
                gpu_name=None,
            )
        )
        monkeypatch.setattr(embeddings, "_cuda_available", lambda: True)
        result = runner.invoke(app, ["embedding", "retry", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert data["cleared_prior"] is True
        assert data["device"] == "gpu"
        assert dstate.load_state().device == "gpu"

    def test_retry_with_explicit_config_is_noop(self, fake_models, monkeypatch):
        monkeypatch.setenv("HAFIZ_EMBEDDING__DEVICE", "cpu")
        cfg_mod.reset_settings()
        dstate.save_state(dstate.build_state("gpu", reason=None, category=None, gpu_name=None))
        result = runner.invoke(app, ["embedding", "retry", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert data["cleared_prior"] is True
        assert data["device"] == "cpu"
        assert "explicitly" in data["message"]


class TestModelCachePurge:
    """`_purge_if_incomplete` self-heals a half-downloaded model cache.

    Regression for the interrupted-download failure: a snapshot with config
    blobs but no ``onnx/model.onnx`` (plus a 0-byte ``*.incomplete`` blob) made
    every embed call die with a bare ONNX NO_SUCHFILE.
    """

    MODEL = "nomic-ai/nomic-embed-text-v1.5"
    DIRNAME = "models--nomic-ai--nomic-embed-text-v1.5"

    @staticmethod
    def _abandon(model_dir):
        """Backdate the tree so the repair path reads it as abandoned.

        Purging is now conditional on liveness — a partial cache written moments
        ago belongs to a download in flight, and deleting it was a real bug (see
        tests/test_model_download_notice.py). These two cases are about *shape*,
        so they state the age explicitly rather than depending on wall-clock.
        """
        when = time.time() - 3600
        for path in sorted(model_dir.rglob("*"), reverse=True):
            os.utime(path, (when, when))
        os.utime(model_dir, (when, when))

    def test_incomplete_blob_is_purged(self, tmp_path):
        model_dir = tmp_path / self.DIRNAME
        (model_dir / "blobs").mkdir(parents=True)
        (model_dir / "snapshots" / "abc" / "onnx").mkdir(parents=True)
        (model_dir / "blobs" / "x.incomplete").write_text("")
        self._abandon(model_dir)
        assert embeddings._purge_if_incomplete(tmp_path, self.MODEL) is True
        assert not model_dir.exists()

    def test_missing_onnx_is_purged(self, tmp_path):
        model_dir = tmp_path / self.DIRNAME
        (model_dir / "snapshots" / "abc" / "onnx").mkdir(parents=True)
        self._abandon(model_dir)
        assert embeddings._purge_if_incomplete(tmp_path, self.MODEL) is True
        assert not model_dir.exists()

    def test_healthy_cache_is_kept(self, tmp_path):
        model_dir = tmp_path / self.DIRNAME
        onnx = model_dir / "snapshots" / "abc" / "onnx"
        onnx.mkdir(parents=True)
        (onnx / "model.onnx").write_text("weights")
        assert embeddings._purge_if_incomplete(tmp_path, self.MODEL) is False
        assert model_dir.exists()

    def test_absent_cache_is_noop(self, tmp_path):
        assert embeddings._purge_if_incomplete(tmp_path, self.MODEL) is False

    def test_cache_dir_is_persistent_not_tmp(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        path = embeddings._model_cache_dir()
        assert path == tmp_path / "hafiz" / "models"
        assert path.is_dir()
