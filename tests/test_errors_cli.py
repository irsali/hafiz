"""End-to-end tests for `hafiz errors` via CliRunner."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core import error_log

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    yield


def _inject(exc_type: type, message: str, argv: list[str]) -> str:
    """Write one record and return its id."""
    try:
        raise exc_type(message)
    except exc_type as e:
        rec = error_log.log_exception(e, argv=argv)
    return rec.id


# ── help discovery ────────────────────────────────────────────────────


def test_errors_help_lists_subcommands():
    result = runner.invoke(app, ["errors", "--help"])
    assert result.exit_code == 0
    out = result.output
    for sub in ("list", "show", "clear"):
        assert sub in out


# ── list ───────────────────────────────────────────────────────────────


def test_list_empty_json():
    result = runner.invoke(app, ["errors", "list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["count"] == 0
    assert payload["errors"] == []


def test_list_shows_recent_json():
    _inject(RuntimeError, "first", ["ingest"])
    rid = _inject(ValueError, "second", ["graph", "stats"])

    result = runner.invoke(app, ["errors", "list", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["count"] == 2
    # Newest first
    assert payload["errors"][0]["id"] == rid
    assert payload["errors"][0]["exception_type"] == "ValueError"
    assert payload["errors"][1]["exception_type"] == "RuntimeError"


def test_list_limit():
    for i in range(5):
        _inject(RuntimeError, f"msg {i}", ["x"])
    result = runner.invoke(app, ["errors", "list", "--limit", "2", "--json"])
    payload = json.loads(result.stdout)
    assert payload["count"] == 2


# ── show ───────────────────────────────────────────────────────────────


def test_show_existing_record_json_has_traceback():
    rid = _inject(RuntimeError, "boom", ["ingest", "."])
    result = runner.invoke(app, ["errors", "show", rid, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["id"] == rid
    assert payload["exception_type"] == "RuntimeError"
    assert "RuntimeError: boom" in payload["traceback"]


def test_show_accepts_short_prefix():
    rid = _inject(RuntimeError, "boom", ["x"])
    prefix = rid[:8]
    result = runner.invoke(app, ["errors", "show", prefix, "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["id"] == rid


def test_show_unknown_id_returns_error():
    result = runner.invoke(app, ["errors", "show", "deadbeef", "--json"])
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "not_found"


# ── clear ──────────────────────────────────────────────────────────────


def test_clear_reports_count():
    _inject(RuntimeError, "a", ["x"])
    _inject(RuntimeError, "b", ["x"])
    result = runner.invoke(app, ["errors", "clear", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["cleared"] == 2

    # Log file is gone now.
    result2 = runner.invoke(app, ["errors", "list", "--json"])
    assert json.loads(result2.stdout)["count"] == 0


def test_clear_noop_when_empty():
    result = runner.invoke(app, ["errors", "clear", "--json"])
    payload = json.loads(result.stdout)
    assert payload["cleared"] == 0


# ── scipy suggestion end-to-end ───────────────────────────────────────


def test_modulenotfound_scipy_gets_pipx_inject_suggestion():
    exc = ModuleNotFoundError("No module named 'scipy'")
    exc.name = "scipy"
    try:
        raise exc
    except ModuleNotFoundError as e:
        rec = error_log.log_exception(e, argv=["graph", "stats"])

    result = runner.invoke(app, ["errors", "show", rec.id, "--json"])
    payload = json.loads(result.stdout)
    assert payload["suggested_action"] is not None
    assert "pipx inject" in payload["suggested_action"]
    assert payload["context"]["missing_module"] == "scipy"
    assert payload["context"]["is_declared_dep"] is True


# ── --group-by exception_type ──────────────────────────────────────────


def test_list_group_by_exception_type_json_shape():
    """Grouped JSON shape must carry the agent-consumable summary
    keys at the top level (total, with_suggestions, most_recent) plus
    a `groups` list — distinct from the flat `errors` list shape."""
    _inject(RuntimeError, "a", ["ingest"])
    _inject(RuntimeError, "b", ["ingest"])
    _inject(ValueError, "c", ["query", "x"])

    result = runner.invoke(
        app, ["errors", "list", "--group-by", "exception_type", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    # New shape — distinct from the flat one.
    assert payload["grouped_by"] == "exception_type"
    assert "groups" in payload
    assert "errors" not in payload  # not the flat shape

    assert payload["total"] == 3
    assert payload["with_suggestions"] == 0
    assert payload["most_recent"]["exception_type"] == "ValueError"

    # Two groups; RuntimeError first (count 2 > 1).
    assert len(payload["groups"]) == 2
    assert payload["groups"][0]["exception_type"] == "RuntimeError"
    assert payload["groups"][0]["count"] == 2
    assert payload["groups"][1]["exception_type"] == "ValueError"
    assert payload["groups"][1]["count"] == 1
    # Each group carries the sample fields agents need to act without
    # a second `errors show` round-trip.
    for g in payload["groups"]:
        assert "most_recent_id" in g
        assert "most_recent_timestamp" in g
        assert "sample_command" in g
        assert "sample_message" in g
        assert "with_suggestions" in g


def test_list_group_by_with_suggestions_count():
    """A recognized error contributes to with_suggestions."""
    exc = ModuleNotFoundError("No module named 'scipy'")
    exc.name = "scipy"
    try:
        raise exc
    except ModuleNotFoundError as e:
        error_log.log_exception(e, argv=["graph", "stats"])
    _inject(RuntimeError, "no advice for me", ["ingest"])

    result = runner.invoke(
        app, ["errors", "list", "--group-by", "exception_type", "--json"]
    )
    payload = json.loads(result.stdout)
    assert payload["total"] == 2
    assert payload["with_suggestions"] == 1
    by_type = {g["exception_type"]: g for g in payload["groups"]}
    assert by_type["ModuleNotFoundError"]["with_suggestions"] == 1
    assert by_type["RuntimeError"]["with_suggestions"] == 0


def test_list_group_by_empty_log():
    result = runner.invoke(
        app, ["errors", "list", "--group-by", "exception_type", "--json"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["total"] == 0
    assert payload["groups"] == []
    assert payload["most_recent"] is None


def test_list_group_by_invalid_value_errors():
    result = runner.invoke(
        app, ["errors", "list", "--group-by", "command", "--json"]
    )
    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "bad_group_by"


def test_list_group_by_ignores_limit():
    """In grouped mode --limit must not silently truncate the input
    set — counts would lie. We examine the full matching window."""
    for i in range(15):
        _inject(RuntimeError, f"msg {i}", ["x"])
    result = runner.invoke(
        app,
        [
            "errors",
            "list",
            "--group-by",
            "exception_type",
            "--limit",
            "2",
            "--json",
        ],
    )
    payload = json.loads(result.stdout)
    assert payload["total"] == 15
    assert payload["groups"][0]["count"] == 15
