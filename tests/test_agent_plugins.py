"""Hermes plugin install/uninstall + the provider's testable core.

Like the hooks installer, this edits a file hafiz does not own: Hermes'
``config.yaml`` is hand-written YAML full of comments. Every test here is
ultimately about touching exactly one line of it, and leaving no trace when
removed. All filesystem work is in tmp dirs; the real ``~/.hermes`` is never
read or written.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core.agent_plugins import (
    MARKER_NAME,
    install_hermes,
    load_plugin_file,
    read_memory_provider,
    uninstall_hermes,
    write_memory_provider,
)
from hafiz.core.agents import AGENTS, resolve_target

CONFIG = """\
# my hermes config
model:
  default: some-model   # keep me
memory:
  # memory section
  memory_enabled: true
  provider: ''
  nudge_interval: 10
delegation:
  model: ''
"""


@pytest.fixture
def home(tmp_path: Path) -> Path:
    h = tmp_path / "hermes-home"
    h.mkdir()
    (h / "config.yaml").write_text(CONFIG, encoding="utf-8")
    return h


def _plugin(home: Path) -> Path:
    return home / "plugins" / "hafiz"


# ── config line editing ───────────────────────────────────────────────────


def test_set_provider_changes_exactly_one_line():
    new, undo = write_memory_provider(CONFIG, "hafiz")
    old_lines, new_lines = CONFIG.splitlines(), new.splitlines()
    changed = [(a, b) for a, b in zip(old_lines, new_lines, strict=True) if a != b]
    assert changed == [("  provider: ''", "  provider: hafiz")]
    assert undo == {"previous_raw": "''", "created_block": False}


def test_set_provider_keeps_trailing_comment():
    text = "memory:\n  provider: ''  # none yet\n"
    new, _ = write_memory_provider(text, "hafiz")
    assert new == "memory:\n  provider: hafiz  # none yet\n"


def test_set_provider_inserts_when_block_has_no_key():
    text = "a: 1\nmemory:\n    memory_enabled: true\nb: 2\n"
    new, undo = write_memory_provider(text, "hafiz")
    assert new == "a: 1\nmemory:\n    provider: hafiz\n    memory_enabled: true\nb: 2\n"
    assert undo["previous_raw"] is None


def test_set_provider_appends_block_when_missing():
    new, undo = write_memory_provider("a: 1", "hafiz")
    assert new == "a: 1\nmemory:\n  provider: hafiz\n"
    assert undo["created_block"] is True


def test_nested_provider_keys_elsewhere_are_not_touched():
    text = "other:\n  provider: x\nmemory:\n  provider: ''\n"
    new, _ = write_memory_provider(text, "hafiz")
    assert "other:\n  provider: x\n" in new


def test_flow_style_memory_is_refused():
    assert write_memory_provider("memory: {}\n", "hafiz") is None
    assert read_memory_provider("memory: {provider: x}\n")[0] == "unsupported"


# ── install / uninstall ───────────────────────────────────────────────────


def test_install_writes_plugin_and_sets_provider(home: Path):
    result = install_hermes(home)
    plugin = _plugin(home)
    assert result["plugin_action"] == "created"
    assert result["config"]["action"] == "set"
    assert (plugin / "__init__.py").read_text() == load_plugin_file("provider.py")
    assert (plugin / "plugin.yaml").exists()
    assert (plugin / MARKER_NAME).exists()
    text = (home / "config.yaml").read_text()
    assert "  provider: hafiz\n" in text
    assert "# keep me" in text and "# memory section" in text
    assert (home / "config.yaml.hafiz-backup").read_text() == CONFIG


def test_install_is_idempotent(home: Path):
    install_hermes(home)
    after_first = (home / "config.yaml").read_text()
    again = install_hermes(home)
    assert again["plugin_action"] == "updated"
    assert again["config"]["action"] == "already"
    assert (home / "config.yaml").read_text() == after_first


def test_uninstall_restores_config_exactly(home: Path):
    install_hermes(home)
    result = uninstall_hermes(home)
    assert result["action"] == "removed"
    assert result["config"]["action"] == "restored"
    assert (home / "config.yaml").read_text() == CONFIG
    assert not _plugin(home).exists()


def test_uninstall_after_reinstall_still_restores(home: Path):
    install_hermes(home)
    install_hermes(home)  # second run must not lose the undo record
    uninstall_hermes(home)
    assert (home / "config.yaml").read_text() == CONFIG


def test_uninstall_removes_created_memory_block(tmp_path: Path):
    h = tmp_path / "h"
    h.mkdir()
    (h / "config.yaml").write_text("a: 1\n", encoding="utf-8")
    install_hermes(h)
    assert "memory:" in (h / "config.yaml").read_text()
    uninstall_hermes(h)
    assert (h / "config.yaml").read_text() == "a: 1\n"


def test_another_provider_is_never_replaced(home: Path):
    (home / "config.yaml").write_text(CONFIG.replace("provider: ''", "provider: honcho"))
    result = install_hermes(home)
    assert result["config"] == {
        "path": str(home / "config.yaml"),
        "action": "conflict",
        "current": "honcho",
    }
    assert "provider: honcho" in (home / "config.yaml").read_text()
    # ...and uninstall must not touch the user's choice either.
    assert uninstall_hermes(home)["config"]["action"] == "none"
    assert "provider: honcho" in (home / "config.yaml").read_text()


def test_missing_config_prints_lines_instead_of_creating_one(tmp_path: Path):
    h = tmp_path / "h"
    h.mkdir()
    result = install_hermes(h)
    assert result["config"]["action"] == "no_config"
    assert result["config"]["lines"] == "memory:\n  provider: hafiz"
    assert not (h / "config.yaml").exists()


def test_unsupported_config_shape_is_left_untouched(home: Path):
    (home / "config.yaml").write_text("memory: {}\n")
    result = install_hermes(home)
    assert result["config"]["action"] == "manual"
    assert (home / "config.yaml").read_text() == "memory: {}\n"


def test_provider_already_hafiz_is_left_after_uninstall(home: Path):
    """hafiz did not set it, so uninstall must not unset it (it says so instead)."""
    (home / "config.yaml").write_text(CONFIG.replace("provider: ''", "provider: hafiz"))
    install_hermes(home)
    result = uninstall_hermes(home)
    assert result["config"]["action"] == "left"
    assert "provider: hafiz" in (home / "config.yaml").read_text()


def test_hand_installed_plugin_is_backed_up_and_left_by_uninstall_without_marker(home: Path):
    plugin = _plugin(home)
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text("# hand written\n")
    install_hermes(home)
    assert (plugin / "__init__.py.hafiz-backup").read_text() == "# hand written\n"
    uninstall_hermes(home)
    # the displaced plugin is put back, so nothing the user had is lost
    assert (plugin / "__init__.py").read_text() == "# hand written\n"


def test_uninstall_skips_a_plugin_hafiz_did_not_write(home: Path):
    plugin = _plugin(home)
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text("# hand written\n")
    assert uninstall_hermes(home)["action"] == "skipped"
    assert (plugin / "__init__.py").exists()


def test_uninstall_keeps_unrelated_user_files_in_the_plugin_dir(home: Path):
    install_hermes(home)
    (_plugin(home) / "hafiz.json").write_text("{}")
    uninstall_hermes(home)
    assert (_plugin(home) / "hafiz.json").exists()
    assert not (_plugin(home) / "plugin.yaml").exists()


def test_uninstall_when_absent_is_a_no_op(home: Path):
    assert uninstall_hermes(home)["action"] == "not_found"


def test_install_refuses_a_missing_default_home_but_accepts_an_explicit_one(tmp_path: Path):
    with pytest.raises(ValueError, match="Hermes home not found"):
        install_hermes(tmp_path / "nope")
    result = install_hermes(tmp_path / "made", explicit_home=True)
    assert result["plugin_action"] == "created"


# ── CLI ───────────────────────────────────────────────────────────────────


def test_cli_install_and_uninstall_hermes_with_path(home: Path):
    runner = CliRunner()
    r = runner.invoke(app, ["agent", "install", "hermes", "--path", str(home)])
    assert r.exit_code == 0, r.output
    assert (_plugin(home) / "__init__.py").exists()
    r = runner.invoke(app, ["agent", "uninstall", "hermes", "--path", str(home)])
    assert r.exit_code == 0, r.output
    assert not _plugin(home).exists()
    assert (home / "config.yaml").read_text() == CONFIG


def test_cli_install_hermes_honours_hermes_home_env(home: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    r = CliRunner().invoke(app, ["agent", "install", "hermes"])
    assert r.exit_code == 0, r.output
    assert (_plugin(home) / "plugin.yaml").exists()


def test_cli_install_hermes_without_hermes_fails_clearly(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "absent"))
    r = CliRunner().invoke(app, ["agent", "install", "hermes"])
    assert r.exit_code == 1
    assert "Hermes home not found" in r.output


# ── shipped plugin hygiene ────────────────────────────────────────────────


def test_shipped_plugin_files_carry_no_machine_specific_strings():
    blob = load_plugin_file("provider.py") + load_plugin_file("plugin.yaml")
    for needle in ("/home/", "/Users/", "irshad"):
        assert needle not in blob, needle
    assert not re.search(r"\w@\w+\.\w", blob), "looks like an email address"


# ── Codex registry ────────────────────────────────────────────────────────


def test_codex_targets_agents_md(tmp_path: Path, monkeypatch):
    assert "codex" in AGENTS
    monkeypatch.delenv("CODEX_HOME", raising=False)
    target, _ = resolve_target("codex", local=False, path_override=None, file_override=None)
    assert target == Path.home() / ".codex" / "AGENTS.md"
    local, _ = resolve_target("codex", local=True, path_override=None, file_override=None)
    assert local.name == "AGENTS.md" and local.parent == Path.cwd()


def test_codex_global_path_follows_codex_home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "cx"))
    target, _ = resolve_target("codex", local=False, path_override=None, file_override=None)
    assert target == tmp_path / "cx" / "AGENTS.md"


def test_cli_install_codex_into_a_tmp_path(tmp_path: Path):
    r = CliRunner().invoke(app, ["agent", "install", "codex", "--path", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "hafiz" in (tmp_path / "AGENTS.md").read_text().lower()


# ── the provider's testable core ──────────────────────────────────────────


@pytest.fixture
def provider_module(monkeypatch):
    """Load the shipped provider with Hermes' own modules stubbed out."""
    agent_pkg = types.ModuleType("agent")
    mp = types.ModuleType("agent.memory_provider")

    class MemoryProvider:  # minimal stand-in for Hermes' ABC
        pass

    mp.MemoryProvider = MemoryProvider
    tools_pkg = types.ModuleType("tools")
    registry = types.ModuleType("tools.registry")
    registry.tool_error = lambda msg: f'{{"error": "{msg}"}}'
    for name, mod in {
        "agent": agent_pkg,
        "agent.memory_provider": mp,
        "tools": tools_pkg,
        "tools.registry": registry,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    path = Path(__file__).parent.parent / "hafiz" / "data" / "hermes" / "provider.py"
    spec = importlib.util.spec_from_file_location("hermes_hafiz_provider", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_relevance_gate_drops_noise_and_keeps_the_top_cluster(provider_module):
    rows = [{"score": 0.70}, {"score": 0.68}, {"score": 0.50}, {"score": "bad"}]
    kept = provider_module._gate_by_relevance(rows, floor=0.6)
    assert [r["score"] for r in kept] == [0.70, 0.68]


def test_relevance_gate_returns_nothing_when_best_hit_misses_the_floor(provider_module):
    assert provider_module._gate_by_relevance([{"score": 0.4}], floor=0.6) == []
    assert provider_module._gate_by_relevance([], floor=0.6) == []


def test_score_floor_env_override(provider_module, monkeypatch):
    monkeypatch.setenv("HAFIZ_RECALL_MIN_SCORE", "0.9")
    assert provider_module._score_floor(0.5) == 0.9
    monkeypatch.setenv("HAFIZ_RECALL_MIN_SCORE", "junk")
    assert provider_module._score_floor(0.5) == 0.5


def test_socket_path_matches_the_daemon(provider_module, monkeypatch, tmp_path):
    from hafiz.core.daemon import socket_path

    monkeypatch.delenv("HAFIZ_DAEMON_SOCKET", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert provider_module.HafizMemoryProvider()._socket_path() == str(socket_path())
    monkeypatch.setenv("HAFIZ_DAEMON_SOCKET", "/x/y.sock")
    assert provider_module.HafizMemoryProvider()._socket_path() == "/x/y.sock"


def test_project_resolution_order(provider_module, monkeypatch, tmp_path):
    p = provider_module.HafizMemoryProvider()
    p._config = {"project": "forced"}
    assert p._resolve_project(agent_workspace="ws") == "forced"
    p._config = {}
    assert p._resolve_project(agent_workspace="ws") == "ws"
    monkeypatch.chdir(tmp_path)
    assert p._resolve_project(agent_workspace="hermes") == tmp_path.name


def test_provider_config_file_overrides_env(provider_module, monkeypatch, tmp_path):
    hermes_constants = types.ModuleType("hermes_constants")
    hermes_constants.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", hermes_constants)
    monkeypatch.setenv("HAFIZ_BIN", "from-env")
    (tmp_path / "hafiz.json").write_text('{"bin": "/opt/hafiz", "project": ""}')
    cfg = provider_module._load_config()
    assert cfg["bin"] == "/opt/hafiz"  # file wins; blank values do not clobber
    assert cfg["project"] == ""
