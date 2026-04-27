"""Tests for ``probe_max_part_chars`` — recommendation logic only.

The actual measurement subprocess (loads fastembed, runs ONNX) is
exercised implicitly in end-to-end runs. Here we synthesize
``HostProbe`` snapshots and stub ``_run_measurement`` so we can lock
down the recommendation banding rules:

  - GPU shortcut: bands by *total* VRAM, not just free.
  - CPU path: budget anchored to ``min(available × 30%, total × 15%)``.
  - Hard CPU ceiling at 8 000 chars regardless of measured headroom.
  - Conservative fallback when measurement can't run.

These rules are load-bearing — see the 2026-04-27 incident where the
prior probe recommended 16 000 on a 16 GB-VRAM box and the resulting
ingest swap-thrashed VSCode to death.
"""

from __future__ import annotations

import pytest

from hafiz.core import probers
from hafiz.core.host_probe import HostProbe


def _host(
    *,
    ram_total_mb: int = 64_000,
    ram_available_mb: int = 49_000,
    gpu_vram_total_mb: int | None = None,
    gpu_vram_free_mb: int | None = None,
    gpu_name: str | None = None,
) -> HostProbe:
    return HostProbe(
        ram_total_mb=ram_total_mb,
        ram_available_mb=ram_available_mb,
        swap_total_mb=8_000,
        swap_used_mb=0,
        cpu_count=20,
        platform="linux-x86_64",
        onnx_providers=("CPUExecutionProvider",),
        gpu_name=gpu_name,
        gpu_vram_total_mb=gpu_vram_total_mb,
        gpu_vram_free_mb=gpu_vram_free_mb,
        onnxruntime_version="1.24.4",
    )


# ── GPU shortcut bands ─────────────────────────────────────────────────


def test_gpu_shortcut_24gb_recommends_16k(monkeypatch):
    """24 GB+ VRAM with enough free → 16 K, no measurement."""
    called = {"n": 0}

    def fake_run(*a, **kw):
        called["n"] += 1
        return []

    monkeypatch.setattr(probers, "_run_measurement", fake_run)

    host = _host(
        gpu_vram_total_mb=24_000, gpu_vram_free_mb=20_000, gpu_name="RTX 3090"
    )
    result = probers.probe_max_part_chars(host)
    assert result.recommended_value == 16_000
    assert result.measured["path"] == "gpu_shortcut_24gb"
    assert called["n"] == 0  # no fastembed subprocess


def test_gpu_shortcut_16gb_caps_at_8k(monkeypatch):
    """16 GB-class card → 8 K, not 16 K. This is the user's RTX 5060 Ti
    case — the failure mode we're guarding against."""
    monkeypatch.setattr(probers, "_run_measurement", lambda *a, **kw: [])

    host = _host(
        gpu_vram_total_mb=16_311, gpu_vram_free_mb=10_000, gpu_name="RTX 5060 Ti"
    )
    result = probers.probe_max_part_chars(host)
    assert result.recommended_value == 8_000
    assert result.measured["path"] == "gpu_shortcut_16gb"


def test_gpu_shortcut_skipped_when_free_too_low(monkeypatch):
    """A 16 GB card with the user's IDE consuming most of it should fall
    through to the CPU path, not blindly recommend 8 K. Free-VRAM
    floor is the second gate."""
    fake_called = []

    def fake_run(*a, **kw):
        fake_called.append((a, kw))
        return [{"chars": 2_000, "peak_rss_mb": 800, "ok": True, "batch": 8}]

    monkeypatch.setattr(probers, "_run_measurement", fake_run)

    host = _host(
        gpu_vram_total_mb=16_000, gpu_vram_free_mb=2_000, gpu_name="RTX 5060 Ti"
    )
    result = probers.probe_max_part_chars(host)
    # CPU path was taken, not GPU shortcut
    assert result.measured["path"] != "gpu_shortcut_16gb"
    assert fake_called, "expected fall-through to CPU measurement"


def test_gpu_shortcut_skipped_for_small_gpu(monkeypatch):
    """An 8 GB card doesn't qualify for either GPU band."""
    monkeypatch.setattr(
        probers,
        "_run_measurement",
        lambda *a, **kw: [{"chars": 4_000, "peak_rss_mb": 1_500, "ok": True, "batch": 8}],
    )

    host = _host(gpu_vram_total_mb=8_000, gpu_vram_free_mb=7_000)
    result = probers.probe_max_part_chars(host)
    assert result.measured["path"] not in ("gpu_shortcut_16gb", "gpu_shortcut_24gb")


# ── CPU budget banding ─────────────────────────────────────────────────


def test_cpu_budget_caps_at_total_ram_fraction():
    """Budget must be the **min** of (available × 30%, total × 15%).
    A box with 64 GB total but a transiently-empty 60 GB available
    must not license the full 30%-of-available budget."""
    host = _host(ram_total_mb=64_000, ram_available_mb=60_000)
    budget, basis = probers._compute_budget(host)
    # 30% of 60 000 = 18 000; 15% of 64 000 = 9 600 → min wins.
    assert budget == 9_600
    assert "total" in basis


def test_cpu_budget_uses_available_when_lower():
    """Tight memory pressure (low available) must shrink the budget
    even though total is large. Probing under load shouldn't license
    a value that already fits poorly."""
    host = _host(ram_total_mb=64_000, ram_available_mb=10_000)
    budget, basis = probers._compute_budget(host)
    # 30% of 10 000 = 3 000; 15% of 64 000 = 9 600 → available wins.
    assert budget == 3_000
    assert "available" in basis


def test_cpu_budget_floored_at_1500mb():
    """A pathologically low available must not collapse to 0 — the
    floor protects the smallest-candidate path from spurious failure."""
    host = _host(ram_total_mb=2_000, ram_available_mb=500)
    budget, _ = probers._compute_budget(host)
    assert budget >= 1_500


# ── CPU recommendation: hard ceiling ───────────────────────────────────


def test_cpu_recommendation_is_capped_at_ceiling(monkeypatch):
    """Even if 16 K chars fit the budget, the CPU path must cap at
    ``_CPU_CEILING_CHARS`` (8 K). The recommendation is an opinion, not
    a max; a user who wants more sets it explicitly."""
    monkeypatch.setattr(
        probers,
        "_run_measurement",
        lambda *a, **kw: [
            {"chars": 2_000, "peak_rss_mb": 1_000, "ok": True, "batch": 8},
            {"chars": 4_000, "peak_rss_mb": 1_300, "ok": True, "batch": 8},
            {"chars": 8_000, "peak_rss_mb": 2_200, "ok": True, "batch": 8},
            {"chars": 16_000, "peak_rss_mb": 5_500, "ok": True, "batch": 8},
        ],
    )

    # 128 GB box, no GPU → all candidates fit easily.
    host = _host(ram_total_mb=128_000, ram_available_mb=100_000)
    result = probers.probe_max_part_chars(host)
    assert result.recommended_value == 8_000
    assert result.measured["uncapped_best"] == 16_000
    assert result.measured["ceiling"] == 8_000


def test_cpu_recommendation_picks_largest_under_budget(monkeypatch):
    """When the ceiling isn't hit, recommend the largest candidate that
    stayed under the budget — not the smallest, not the default."""
    monkeypatch.setattr(
        probers,
        "_run_measurement",
        lambda *a, **kw: [
            {"chars": 2_000, "peak_rss_mb": 1_000, "ok": True, "batch": 8},
            {"chars": 4_000, "peak_rss_mb": 1_500, "ok": True, "batch": 8},
            # 8 K blew past budget → should not be picked.
            {"chars": 8_000, "peak_rss_mb": 9_999, "ok": True, "batch": 8},
        ],
    )

    host = _host(ram_total_mb=16_000, ram_available_mb=8_000)
    result = probers.probe_max_part_chars(host)
    # Budget = min(8000 * 0.30, 16000 * 0.15) = min(2400, 2400) = 2400 MB
    # 4K peak (1500) fits; 8K peak (9999) doesn't → 4 000.
    assert result.recommended_value == 4_000


# ── Conservative fallback ──────────────────────────────────────────────


def test_fallback_when_measurement_returns_nothing(monkeypatch):
    monkeypatch.setattr(
        probers, "_run_measurement", lambda *a, **kw: [{"_fatal": "no fastembed"}]
    )
    host = _host()
    result = probers.probe_max_part_chars(host)
    assert result.recommended_value == 2_000
    assert result.confidence == "low"
    assert result.measured["path"] == "fallback"


def test_fallback_when_smallest_candidate_exceeds_budget(monkeypatch):
    """If even 2 000 chars peaks above budget, the host is genuinely
    too memory-tight; recommendation must stay at default with a 'low'
    confidence — not silently push something larger."""
    monkeypatch.setattr(
        probers,
        "_run_measurement",
        lambda *a, **kw: [
            {"chars": 2_000, "peak_rss_mb": 999_999, "ok": True, "batch": 8},
        ],
    )
    host = _host(ram_total_mb=16_000, ram_available_mb=4_000)
    result = probers.probe_max_part_chars(host)
    assert result.recommended_value == 2_000
    assert result.measured["path"] == "budget_exceeded"


# ── Probe safety brake ─────────────────────────────────────────────────


def test_probe_passes_safety_ceiling_to_subprocess(monkeypatch):
    """The prober must hand the subprocess a ``safety_ceiling_mb`` equal
    to the recommendation budget. Without it, the subprocess walks all
    candidates ascending and can OOM the host on the way up — the
    failure mode that swap-thrashed a desktop session on 2026-04-27
    even after the *recommendation* logic was correct."""
    captured: dict = {}

    def fake_run(candidates, **kwargs):
        captured["kwargs"] = kwargs
        return [
            {"chars": 2_000, "peak_rss_mb": 1_000, "ok": True, "batch": 8},
        ]

    monkeypatch.setattr(probers, "_run_measurement", fake_run)

    host = _host(ram_total_mb=64_000, ram_available_mb=49_000)
    probers.probe_max_part_chars(host)

    assert "safety_ceiling_mb" in captured["kwargs"]
    expected_budget = min(int(49_000 * 0.30), int(64_000 * 0.15))
    assert captured["kwargs"]["safety_ceiling_mb"] == expected_budget


def test_probe_handles_stopped_sentinel_gracefully(monkeypatch):
    """When the subprocess hits the safety brake, it emits ``_stopped``
    rows alongside the measured ``ok`` rows. The recommendation logic
    must keep working — picking the largest candidate that fit, and
    treating the sentinel as informational."""
    monkeypatch.setattr(
        probers,
        "_run_measurement",
        lambda *a, **kw: [
            {"chars": 2_000, "peak_rss_mb": 1_500, "ok": True, "batch": 8},
            {"chars": 4_000, "peak_rss_mb": 3_500, "ok": True, "batch": 8},
            {"_stopped": "candidate 8000 predicted ~10500 MB > ceiling 9000 MB"},
        ],
    )

    host = _host(ram_total_mb=64_000, ram_available_mb=49_000)
    result = probers.probe_max_part_chars(host)
    # 4 K fits the 9 563 MB budget; 8 K was never measured — we should
    # still recommend 4 K, not fall back to 2 K.
    assert result.recommended_value == 4_000
    assert result.confidence == "high"


# ── Regression: the original failure mode ──────────────────────────────


def test_regression_5060ti_does_not_recommend_16k(monkeypatch):
    """The driving incident: an RTX 5060 Ti (16 GB total VRAM, ~5 GB
    free at probe time, 64 GB host RAM) used to trigger the GPU
    shortcut and recommend 16 000. Real ingest then OOM-killed the
    desktop session. Whatever path runs on this host now, the
    recommendation must NOT be 16 000."""
    monkeypatch.setattr(
        probers,
        "_run_measurement",
        lambda *a, **kw: [
            {"chars": 2_000, "peak_rss_mb": 1_000, "ok": True, "batch": 8},
            {"chars": 4_000, "peak_rss_mb": 1_500, "ok": True, "batch": 8},
            {"chars": 8_000, "peak_rss_mb": 2_500, "ok": True, "batch": 8},
            {"chars": 16_000, "peak_rss_mb": 5_500, "ok": True, "batch": 8},
        ],
    )
    # Probe-time numbers from the user's machine on 2026-04-27.
    host = _host(
        ram_total_mb=63_755,
        ram_available_mb=49_324,
        gpu_vram_total_mb=16_311,
        gpu_vram_free_mb=5_191,  # below the free-VRAM gate → CPU path
        gpu_name="RTX 5060 Ti",
    )
    result = probers.probe_max_part_chars(host)
    assert result.recommended_value <= 8_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
