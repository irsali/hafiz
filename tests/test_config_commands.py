"""End-to-end tests for `hafiz config get/set/unset/apply/clear-sticky`.

These tests exercise the full write path — TOML round-trip, sticky
state round-trip, and the resolution chain that ties them together.
XDG_CACHE_HOME and the user-scope config path are redirected per-test
so we never touch the real user environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core.config import reset_settings
from hafiz.core.tuning_state import (
    TuningEntry,
    build_state,
    cache_file_path,
    save_state,
)

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Redirect XDG_CACHE_HOME, HOME, and cwd so config writes stay in tmp."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    # `_resolve_config_target(local=False)` uses Path.home() / ".config" /
    # "hafiz" / "hafiz.toml". Redirect home.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # Work in a clean cwd so --local writes don't collide with the repo.
    monkeypatch.chdir(tmp_path)
    reset_settings()
    # Clear any pre-existing HAFIZ_* env vars that would short-circuit
    # resolution (we want TOML / sticky / default to be testable).
    for key in list(__import__("os").environ):
        if key.startswith("HAFIZ_"):
            monkeypatch.delenv(key, raising=False)
    yield
    reset_settings()


# ── config get ─────────────────────────────────────────────────────────


def test_config_get_reports_default_when_nothing_set():
    result = runner.invoke(app, ["config", "get", "embedding.max_part_chars", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["key"] == "embedding.max_part_chars"
    assert payload["value"] == 2000
    assert payload["source"] == "default"


def test_config_get_rejects_unknown_key():
    result = runner.invoke(app, ["config", "get", "nope.whatever", "--json"])
    assert result.exit_code != 0
    # JSON mode emits a well-formed error payload on stdout.
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "unknown_tunable"


# ── config set / unset roundtrip (user scope) ──────────────────────────


def test_config_set_writes_user_scope_toml_and_get_sees_it(tmp_path):
    result = runner.invoke(
        app,
        ["config", "set", "embedding.max_part_chars", "4096", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["key"] == "embedding.max_part_chars"
    assert payload["value"] == 4096
    assert payload["scope"] == "user"

    target = Path(payload["target"])
    assert target.is_file(), f"{target} not created"
    content = target.read_text()
    assert "max_part_chars = 4096" in content

    # A subsequent `config get` must see the TOML-sourced value.
    got = runner.invoke(app, ["config", "get", "embedding.max_part_chars", "--json"])
    assert got.exit_code == 0
    payload2 = json.loads(got.stdout)
    assert payload2["value"] == 4096
    assert payload2["source"] == "toml"


def test_config_set_rejects_invalid_value():
    # `0` violates the positive-int validator; using 0 instead of -5
    # avoids click's `-5`-is-an-option parse. Negative values work from
    # a real shell; test the validator path with a boundary-violating
    # positive-looking value.
    result = runner.invoke(
        app,
        ["config", "set", "embedding.max_part_chars", "0", "--json"],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "validation_failed"


def test_config_set_rejects_uncoercible_value():
    result = runner.invoke(
        app,
        ["config", "set", "embedding.max_part_chars", "not-a-number", "--json"],
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["error"] == "coerce_failed"


def test_config_unset_removes_key_and_prunes_empty_tables():
    runner.invoke(app, ["config", "set", "embedding.max_part_chars", "8000"])
    target = Path.home() / ".config" / "hafiz" / "hafiz.toml"
    assert "max_part_chars" in target.read_text()

    result = runner.invoke(app, ["config", "unset", "embedding.max_part_chars", "--json"])
    assert result.exit_code == 0, result.output

    # After unset the key should be gone — and, since `[embedding]` now
    # has no children, the empty table should have been pruned.
    content = target.read_text()
    assert "max_part_chars" not in content
    assert "[embedding]" not in content


def test_config_unset_noop_when_file_absent():
    result = runner.invoke(app, ["config", "unset", "embedding.max_part_chars", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload.get("no_op") is True


# ── resolution chain (env > toml > sticky > default) ───────────────────


def test_resolution_env_wins_over_toml(monkeypatch):
    runner.invoke(app, ["config", "set", "embedding.max_part_chars", "4000"])
    monkeypatch.setenv("HAFIZ_EMBEDDING__MAX_PART_CHARS", "9999")

    got = runner.invoke(app, ["config", "get", "embedding.max_part_chars", "--json"])
    payload = json.loads(got.stdout)
    assert payload["value"] == 9999
    assert payload["source"] == "env"


def test_resolution_toml_wins_over_sticky():
    # Lay down a sticky value for the current host fingerprint.
    from hafiz.core.host_probe import probe_host

    host = probe_host()
    save_state(
        build_state(
            fingerprint=host.fingerprint,
            ort_version=host.onnxruntime_version,
            entries={"embedding.max_part_chars": TuningEntry(value=16_000)},
        )
    )
    # TOML-set a different value.
    runner.invoke(app, ["config", "set", "embedding.max_part_chars", "4096"])

    got = runner.invoke(app, ["config", "get", "embedding.max_part_chars", "--json"])
    payload = json.loads(got.stdout)
    assert payload["source"] == "toml"
    assert payload["value"] == 4096


def test_resolution_sticky_wins_over_default():
    from hafiz.core.host_probe import probe_host

    host = probe_host()
    save_state(
        build_state(
            fingerprint=host.fingerprint,
            ort_version=host.onnxruntime_version,
            entries={"embedding.max_part_chars": TuningEntry(value=16_000)},
        )
    )

    got = runner.invoke(app, ["config", "get", "embedding.max_part_chars", "--json"])
    payload = json.loads(got.stdout)
    assert payload["source"] == "sticky"
    assert payload["value"] == 16_000


# ── clear-sticky ───────────────────────────────────────────────────────


def test_clear_sticky_removes_cache_file():
    from hafiz.core.host_probe import probe_host

    host = probe_host()
    save_state(
        build_state(
            fingerprint=host.fingerprint,
            ort_version=host.onnxruntime_version,
            entries={"embedding.max_part_chars": TuningEntry(value=8_000)},
        )
    )
    assert cache_file_path().is_file()

    result = runner.invoke(app, ["config", "clear-sticky", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["removed"] is True
    assert not cache_file_path().exists()


# ── config show surfaces the resolution source ─────────────────────────


def test_config_show_includes_tunables_with_sources():
    result = runner.invoke(app, ["config", "show", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "tunables" in payload
    keys = {r["key"] for r in payload["tunables"]}
    assert "embedding.max_part_chars" in keys
    for r in payload["tunables"]:
        assert r["source"] in ("env", "toml", "sticky", "default")
