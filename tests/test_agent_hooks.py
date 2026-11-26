"""Agent hooks (recall + transcript capture) — `hafiz agent install --hooks`.

These tests guard a file hafiz does not own. An agent's settings.json is
hand-edited configuration containing the user's own hooks, permissions and
env; the capture wiring has to land inside it without disturbing anything
else, survive being re-run, and leave no trace when removed. Every test
here is ultimately about that boundary.

DB-free by construction: the installer is pure filesystem work, and the
`--from-hook` safety tests assert exit codes only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core.agents import (
    AGENTS,
    HOOK_SENTINEL,
    install_hooks,
    uninstall_hooks,
)

RECALL_EVENTS = ("SessionStart", "UserPromptSubmit")
CAPTURE_EVENTS = ("PreCompact", "SessionEnd")
ALL_EVENTS = RECALL_EVENTS + CAPTURE_EVENTS


# A realistic pre-existing settings.json: user hooks on two events (one of
# them an event hafiz also targets, to prove coexistence), plus unrelated
# top-level keys that must survive untouched.
USER_SETTINGS = {
    "model": "opus",
    "permissions": {"allow": ["Bash(git:*)"]},
    "env": {"FOO": "bar"},
    "hooks": {
        "SessionStart": [
            {"hooks": [{"type": "command", "command": "echo mine", "timeout": 5}]},
        ],
        "SessionEnd": [
            {"hooks": [{"type": "command", "command": "echo my-session-end"}]},
        ],
    },
}


@pytest.fixture
def settings(tmp_path: Path) -> Path:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(USER_SETTINGS, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def hooks_spec():
    return AGENTS["claude-code"].hooks


def _entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        entry
        for groups in data.get("hooks", {}).values()
        for group in groups
        for entry in group.get("hooks", [])
    ]


def _ours(path: Path) -> list[dict]:
    return [e for e in _entries(path) if HOOK_SENTINEL in e.get("command", "")]


def _theirs(path: Path) -> list[dict]:
    return [e for e in _entries(path) if HOOK_SENTINEL not in e.get("command", "")]


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def test_install_adds_one_entry_per_event(settings: Path, hooks_spec):
    result = install_hooks(settings, hooks_spec)

    assert result["action"] == "updated"
    assert result["events"] == list(ALL_EVENTS)
    data = json.loads(settings.read_text(encoding="utf-8"))
    for event in ALL_EVENTS:
        ours = [
            e
            for g in data["hooks"][event]
            for e in g["hooks"]
            if HOOK_SENTINEL in e.get("command", "")
        ]
        assert len(ours) == 1, event


def test_install_preserves_user_hooks_and_other_keys(settings: Path, hooks_spec):
    install_hooks(settings, hooks_spec)

    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["model"] == "opus"
    assert data["permissions"] == {"allow": ["Bash(git:*)"]}
    assert data["env"] == {"FOO": "bar"}

    theirs = {e["command"] for e in _theirs(settings)}
    assert theirs == {"echo mine", "echo my-session-end"}


def test_install_coexists_with_a_user_hook_on_the_same_event(settings: Path, hooks_spec):
    """SessionEnd already has a user hook — ours must join, not replace."""
    install_hooks(settings, hooks_spec)

    data = json.loads(settings.read_text(encoding="utf-8"))
    commands = [e["command"] for g in data["hooks"]["SessionEnd"] for e in g["hooks"]]
    assert "echo my-session-end" in commands
    assert any(HOOK_SENTINEL in c for c in commands)


def test_install_creates_the_file_when_absent(tmp_path: Path, hooks_spec):
    target = tmp_path / "nested" / "settings.json"
    result = install_hooks(target, hooks_spec)

    assert result["action"] == "created"
    assert target.is_file()
    assert len(_ours(target)) == len(ALL_EVENTS)


def test_installed_commands_are_hook_safe(settings: Path, hooks_spec):
    """The house rule for anything in a hook: bounded, quiet, exit 0."""
    install_hooks(settings, hooks_spec)

    for entry in _ours(settings):
        command = entry["command"]
        assert command.startswith("timeout "), command
        assert "2>" in command and "/dev/null" in command, command
        assert command.rstrip().endswith(HOOK_SENTINEL), command
        assert "|| true" in command, command
        # The harness kill timer must outlast the shell guard, or the
        # harness reports a hung hook instead of the guard exiting quietly.
        guard = int(command.split()[1].rstrip("s"))
        assert entry["timeout"] > guard, command


def test_capture_hooks_discard_stdout_recall_hooks_keep_it(settings: Path, hooks_spec):
    """Recall's stdout IS the injected context; capture's must never leak into it."""
    install_hooks(settings, hooks_spec)
    data = json.loads(settings.read_text(encoding="utf-8"))

    def ours(event: str) -> str:
        return next(
            e["command"]
            for g in data["hooks"][event]
            for e in g["hooks"]
            if HOOK_SENTINEL in e["command"]
        )

    for event in CAPTURE_EVENTS:
        assert "--from-hook" in ours(event)
        assert ">/dev/null" in ours(event)
    assert "hafiz hook session" in ours("SessionStart")
    assert "hafiz hook prompt" in ours("UserPromptSubmit")
    for event in RECALL_EVENTS:
        assert " >/dev/null" not in ours(event)
        assert "2>/dev/null" in ours(event)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_reinstall_replaces_rather_than_stacks(settings: Path, hooks_spec):
    install_hooks(settings, hooks_spec)
    second = install_hooks(settings, hooks_spec)
    third = install_hooks(settings, hooks_spec)

    assert second["replaced"] == len(ALL_EVENTS)
    assert third["replaced"] == len(ALL_EVENTS)
    assert len(_ours(settings)) == len(ALL_EVENTS)
    assert len(_theirs(settings)) == 2


def test_reinstall_updates_a_stale_command_in_place(settings: Path, hooks_spec):
    install_hooks(settings, hooks_spec)
    data = json.loads(settings.read_text(encoding="utf-8"))
    for group in data["hooks"]["SessionEnd"] + data["hooks"]["UserPromptSubmit"]:
        for entry in group["hooks"]:
            if HOOK_SENTINEL in entry["command"]:
                entry["command"] = f"old-hafiz-command  {HOOK_SENTINEL}"
    settings.write_text(json.dumps(data), encoding="utf-8")

    install_hooks(settings, hooks_spec)

    commands = [e["command"] for e in _ours(settings)]
    assert not any(c.startswith("old-hafiz-command") for c in commands)
    assert len(commands) == len(ALL_EVENTS)


def test_install_keeps_one_backup_and_never_overwrites_it(settings: Path, hooks_spec):
    original = settings.read_text(encoding="utf-8")
    backup = settings.with_suffix(settings.suffix + ".hafiz-backup")

    install_hooks(settings, hooks_spec)
    assert backup.read_text(encoding="utf-8") == original

    # A second run must not replace the last known-good copy with an
    # already-modified one.
    install_hooks(settings, hooks_spec)
    assert backup.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------


def test_uninstall_restores_the_original_file(settings: Path, hooks_spec):
    before = json.loads(settings.read_text(encoding="utf-8"))

    install_hooks(settings, hooks_spec)
    result = uninstall_hooks(settings)

    assert result["action"] == "removed"
    assert result["removed"] == len(ALL_EVENTS)
    assert json.loads(settings.read_text(encoding="utf-8")) == before


def test_uninstall_drops_events_it_emptied(settings: Path, hooks_spec):
    """PreCompact existed only because we added it — don't leave a husk."""
    install_hooks(settings, hooks_spec)
    uninstall_hooks(settings)

    data = json.loads(settings.read_text(encoding="utf-8"))
    assert "PreCompact" not in data["hooks"]
    assert "UserPromptSubmit" not in data["hooks"]
    assert "SessionEnd" in data["hooks"]  # user's own hook keeps it alive
    assert "SessionStart" in data["hooks"]


def test_uninstall_is_a_noop_when_nothing_of_ours_is_present(settings: Path):
    before = settings.read_text(encoding="utf-8")
    result = uninstall_hooks(settings)

    assert result["action"] == "skipped"
    assert result["removed"] == 0
    assert settings.read_text(encoding="utf-8") == before


def test_uninstall_reports_missing_file(tmp_path: Path):
    result = uninstall_hooks(tmp_path / "absent.json")
    assert result["action"] == "not_found"


# ---------------------------------------------------------------------------
# Refusals — never destroy configuration to install a convenience
# ---------------------------------------------------------------------------


def test_install_refuses_unparseable_settings(tmp_path: Path, hooks_spec):
    target = tmp_path / "settings.json"
    target.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        install_hooks(target, hooks_spec)

    assert target.read_text(encoding="utf-8") == "{ this is not json"


def test_install_refuses_a_non_object_settings_file(tmp_path: Path, hooks_spec):
    target = tmp_path / "settings.json"
    target.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="does not contain a JSON object"):
        install_hooks(target, hooks_spec)


def test_install_refuses_a_non_object_hooks_key(tmp_path: Path, hooks_spec):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"hooks": "nope"}), encoding="utf-8")

    with pytest.raises(ValueError, match="not an object"):
        install_hooks(target, hooks_spec)


def test_install_treats_an_empty_file_as_no_settings(tmp_path: Path, hooks_spec):
    target = tmp_path / "settings.json"
    target.write_text("   \n", encoding="utf-8")

    install_hooks(target, hooks_spec)
    assert len(_ours(target)) == len(ALL_EVENTS)


# ---------------------------------------------------------------------------
# Registry + CLI surface
# ---------------------------------------------------------------------------


def test_only_claude_code_declares_a_hook_surface():
    assert AGENTS["claude-code"].supports_hooks
    assert not AGENTS["cursor"].supports_hooks
    assert not AGENTS["github-copilot"].supports_hooks


def test_cli_rejects_hooks_for_an_agent_without_a_hook_surface():
    result = CliRunner().invoke(app, ["agent", "install", "cursor", "--hooks"])
    assert result.exit_code == 1
    assert "no hook surface" in result.output


def test_cli_requires_an_agent_name_for_hooks():
    result = CliRunner().invoke(app, ["agent", "install", "--hooks"])
    assert result.exit_code == 1
    assert "Specify which agent" in result.output


def test_cli_rejects_an_unknown_agent():
    result = CliRunner().invoke(app, ["agent", "install", "nope", "--hooks"])
    assert result.exit_code == 1
    assert "Unknown agent" in result.output


def test_agent_install_help_documents_hooks():
    result = CliRunner().invoke(app, ["agent", "install", "--help"])
    assert result.exit_code == 0
    assert "--hooks" in result.output


def test_import_help_documents_from_hook():
    result = CliRunner().invoke(app, ["import", "claude-code", "--help"])
    assert result.exit_code == 0
    assert "--from-hook" in result.output


# ---------------------------------------------------------------------------
# `--from-hook` never fails the turn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("", id="empty"),
        pytest.param("   \n ", id="whitespace"),
        pytest.param("not json at all", id="garbage"),
        pytest.param("[1,2,3]", id="json-array"),
        pytest.param('"a string"', id="json-string"),
        pytest.param("{}", id="empty-object"),
        pytest.param('{"cwd": "/tmp"}', id="no-transcript-path"),
        pytest.param('{"transcript_path": null}', id="null-transcript-path"),
        pytest.param('{"transcript_path": ""}', id="blank-transcript-path"),
        pytest.param('{"transcript_path": "/nonexistent/x.jsonl"}', id="missing-file"),
    ],
)
def test_from_hook_always_exits_zero(payload: str):
    """A memory layer that can break the conversation gets uninstalled.

    Every one of these is a plausible payload from a harness — a hook
    event firing before a transcript exists, a truncated pipe, a future
    schema change — and none of them may surface as a failed hook.
    """
    result = CliRunner().invoke(app, ["import", "claude-code", "--from-hook"], input=payload)
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Recall hooks: coexistence, redirection, the hand-rolled-script warning
# ---------------------------------------------------------------------------


def test_recall_hook_joins_a_user_hook_on_session_start(settings: Path, hooks_spec):
    install_hooks(settings, hooks_spec)

    data = json.loads(settings.read_text(encoding="utf-8"))
    commands = [e["command"] for g in data["hooks"]["SessionStart"] for e in g["hooks"]]
    assert "echo mine" in commands
    assert any("hafiz hook session" in c for c in commands)


def test_install_flags_a_hand_rolled_recall_hook_but_leaves_it(tmp_path: Path, hooks_spec):
    """Two injectors would print every pinned rule twice — say so, never edit."""
    mine = "python3 ~/.claude/hooks/hafiz-recall.py session"
    target = tmp_path / "settings.json"
    target.write_text(
        json.dumps(
            {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": mine}]}]}}
        ),
        encoding="utf-8",
    )

    result = install_hooks(target, hooks_spec)

    assert result["foreign_recall"] == [mine]
    assert mine in {e["command"] for e in _theirs(target)}


def test_install_reports_no_foreign_recall_on_a_clean_file(settings: Path, hooks_spec):
    assert install_hooks(settings, hooks_spec)["foreign_recall"] == []


def test_cli_install_and_uninstall_honour_path_and_file(tmp_path: Path):
    """`--path`/`--file` redirect the settings file, so a trial never touches the real one."""
    runner = CliRunner()
    args = ["--hooks", "--path", str(tmp_path), "--file", "trial.json"]

    result = runner.invoke(app, ["agent", "install", "claude-code", *args])
    assert result.exit_code == 0, result.output
    target = tmp_path / "trial.json"
    assert len(_ours(target)) == len(ALL_EVENTS)
    assert "--tags always" in result.output

    result = runner.invoke(app, ["agent", "uninstall", "claude-code", *args])
    assert result.exit_code == 0, result.output
    assert _ours(target) == []
