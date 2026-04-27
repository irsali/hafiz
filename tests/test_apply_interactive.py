"""Tests for the interactive ``hafiz config apply`` flow.

The interactive prompt was added after a silent ``config apply`` set
``embedding.max_part_chars = 16000`` on a 16 GB-VRAM box and the
resulting ingest swap-thrashed VSCode to death (incident 2026-04-27).
The contract codified here:

  - ``--json`` is non-interactive (machine consumers don't get prompts).
  - ``--yes`` skips prompts and persists everything (CI escape hatch).
  - When stdin/stdout aren't a TTY (CliRunner, pipes), ``_is_interactive``
    returns False and we degrade to ``--yes`` semantics so runs don't
    hang waiting for an answer.
  - ``_interactive_filter`` honors the user's per-row choice — accept,
    skip (clears recommended), custom (replaces recommended after
    validating against the tunable's coercer + validator).
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.commands import maintenance
from hafiz.commands.maintenance import _interactive_filter
from hafiz.core.config import reset_settings
from hafiz.core.tuning_state import cache_file_path, load_state

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Redirect XDG_CACHE_HOME + HOME so writes stay in tmp."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    reset_settings()
    import os as _os

    for key in list(_os.environ):
        if key.startswith("HAFIZ_"):
            monkeypatch.delenv(key, raising=False)
    yield
    reset_settings()


def _row(
    *,
    key: str = "embedding.max_part_chars",
    current: int = 2_000,
    recommended: int | None = 8_000,
    confidence: str = "high",
    rationale: str = "fake probe rationale",
    probe_error: str | None = None,
) -> dict:
    return {
        "key": key,
        "current": current,
        "default": 2_000,
        "description": "fake",
        "is_policy": False,
        "recommended": recommended,
        "rationale": rationale,
        "confidence": confidence,
        "measured": {"path": "test"},
        "probe_error": probe_error,
    }


# ── _interactive_filter unit tests ─────────────────────────────────────


def test_filter_accept_keeps_recommendation(monkeypatch):
    """Default-blank answer means accept — the row passes through with
    recommended unchanged and ``user_choice='accept'`` set."""
    answers = iter(["y"])
    monkeypatch.setattr(
        "hafiz.commands.maintenance.Prompt.ask", lambda *a, **kw: next(answers)
    )

    out = _interactive_filter([_row()])
    assert len(out) == 1
    assert out[0]["recommended"] == 8_000
    assert out[0]["user_choice"] == "accept"


def test_filter_skip_clears_recommendation(monkeypatch):
    """A 'no' answer must clear ``recommended`` so ``_apply_tuning``
    skips the row entirely — sticky state must NOT be touched."""
    answers = iter(["n"])
    monkeypatch.setattr(
        "hafiz.commands.maintenance.Prompt.ask", lambda *a, **kw: next(answers)
    )

    out = _interactive_filter([_row()])
    assert out[0]["recommended"] is None
    assert out[0]["user_choice"] == "skip"


def test_filter_custom_validates_and_replaces(monkeypatch):
    """A 'custom' answer prompts for a number; the value must be
    validated through the tunable's coercer + validator before
    replacing the recommendation."""
    answers = iter(["c", "4096"])
    monkeypatch.setattr(
        "hafiz.commands.maintenance.Prompt.ask", lambda *a, **kw: next(answers)
    )

    out = _interactive_filter([_row()])
    assert out[0]["recommended"] == 4_096
    assert out[0]["confidence"] == "user"
    assert out[0]["user_choice"] == "custom"
    assert "4096" not in out[0]["rationale"]  # rationale references original
    assert "8000" in out[0]["rationale"]


def test_filter_custom_rejects_invalid_then_retries(monkeypatch):
    """Invalid custom values (negative, non-int) re-prompt; only a
    passing value advances. Guards against the user typo'ing a number
    that would later blow up at use time."""
    answers = iter(["c", "not-a-number", "0", "4000"])
    monkeypatch.setattr(
        "hafiz.commands.maintenance.Prompt.ask", lambda *a, **kw: next(answers)
    )

    out = _interactive_filter([_row()])
    assert out[0]["recommended"] == 4_000


def test_filter_passes_through_no_op_rows(monkeypatch):
    """Rows with no actionable change (recommended==current, probe
    error, or recommended is None) must NOT prompt — only real
    proposals get the user's attention."""
    asked = []
    monkeypatch.setattr(
        "hafiz.commands.maintenance.Prompt.ask",
        lambda *a, **kw: asked.append((a, kw)) or "y",
    )

    rows = [
        _row(recommended=2_000),  # equals current
        _row(recommended=None),   # no proposal
        _row(probe_error="boom"),
    ]
    out = _interactive_filter(rows)
    assert len(out) == 3
    assert asked == []


# ── End-to-end via CliRunner: --yes / --json bypass prompts ────────────


def test_config_apply_yes_persists_without_prompts(monkeypatch):
    """``--yes`` must skip the prompt path entirely. The probe is
    stubbed so we don't actually load fastembed here."""
    from hafiz.core.host_probe import HostProbe

    fake_host = HostProbe(
        ram_total_mb=64_000,
        ram_available_mb=49_000,
        swap_total_mb=8_000,
        swap_used_mb=0,
        cpu_count=20,
        platform="linux-x86_64",
        onnx_providers=("CPUExecutionProvider",),
        gpu_name=None,
        gpu_vram_total_mb=None,
        gpu_vram_free_mb=None,
        onnxruntime_version="1.24.4",
    )
    monkeypatch.setattr("hafiz.core.host_probe.probe_host", lambda: fake_host)
    # Stub the actual collection so we don't run probers.
    monkeypatch.setattr(
        maintenance,
        "_collect_tuning",
        lambda host, *, probe: [_row(recommended=8_000)],
    )
    # Sentinel: prompt path must NOT be entered.
    monkeypatch.setattr(
        "hafiz.commands.maintenance.Prompt.ask",
        lambda *a, **kw: pytest.fail("Prompt called despite --yes"),
    )

    result = runner.invoke(app, ["config", "apply", "--yes", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert any(a["key"] == "embedding.max_part_chars" for a in payload["applied"])

    state = load_state()
    assert state is not None
    assert state.entries["embedding.max_part_chars"].value == 8_000


def test_config_apply_json_is_non_interactive_by_default(monkeypatch):
    """Without --yes but with --json, we must still skip the prompt —
    machine consumers piping the JSON shouldn't hang."""
    from hafiz.core.host_probe import HostProbe

    fake_host = HostProbe(
        ram_total_mb=64_000,
        ram_available_mb=49_000,
        swap_total_mb=8_000,
        swap_used_mb=0,
        cpu_count=20,
        platform="linux-x86_64",
        onnx_providers=("CPUExecutionProvider",),
        gpu_name=None,
        gpu_vram_total_mb=None,
        gpu_vram_free_mb=None,
        onnxruntime_version="1.24.4",
    )
    monkeypatch.setattr("hafiz.core.host_probe.probe_host", lambda: fake_host)
    monkeypatch.setattr(
        maintenance,
        "_collect_tuning",
        lambda host, *, probe: [_row(recommended=4_000)],
    )
    monkeypatch.setattr(
        "hafiz.commands.maintenance.Prompt.ask",
        lambda *a, **kw: pytest.fail("Prompt called in --json mode"),
    )

    result = runner.invoke(app, ["config", "apply", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["interactive"] is False

    state = load_state()
    assert state.entries["embedding.max_part_chars"].value == 4_000


def test_config_apply_help_documents_yes_flag():
    result = runner.invoke(app, ["config", "apply", "--help"])
    assert result.exit_code == 0
    assert "--yes" in result.output
    assert "interactive" in result.output.lower()


def test_doctor_help_documents_yes_flag():
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "--yes" in result.output


def test_config_apply_with_no_recommendations_is_noop(monkeypatch):
    """When every probed row equals current, nothing is applied and the
    cache file stays unwritten."""
    from hafiz.core.host_probe import HostProbe

    fake_host = HostProbe(
        ram_total_mb=64_000,
        ram_available_mb=49_000,
        swap_total_mb=None,
        swap_used_mb=None,
        cpu_count=20,
        platform="linux-x86_64",
        onnx_providers=("CPUExecutionProvider",),
        gpu_name=None,
        gpu_vram_total_mb=None,
        gpu_vram_free_mb=None,
        onnxruntime_version="1.24.4",
    )
    monkeypatch.setattr("hafiz.core.host_probe.probe_host", lambda: fake_host)
    monkeypatch.setattr(
        maintenance,
        "_collect_tuning",
        # recommended == current → nothing to apply
        lambda host, *, probe: [_row(current=4_000, recommended=4_000)],
    )

    result = runner.invoke(app, ["config", "apply", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["applied"] == []
    assert not cache_file_path().exists()
