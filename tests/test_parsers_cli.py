"""Tests for Phase 7 — ``hafiz parsers list`` and status --diagnose
parser coverage."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from hafiz.cli import app


runner = CliRunner()


def test_parsers_list_help():
    result = runner.invoke(app, ["parsers", "list", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output


def test_parsers_list_rich_output_shows_builtins():
    result = runner.invoke(app, ["parsers", "list"])
    assert result.exit_code == 0
    # In-tree parsers must always show up.
    for name in ("python_ast", "prose", "whole_file"):
        assert name in result.output


def test_parsers_list_json_output_is_parseable():
    result = runner.invoke(app, ["parsers", "list", "--json"])
    assert result.exit_code == 0
    # Output may include trailing whitespace; isolate the JSON object.
    first_brace = result.output.find("{")
    payload = json.loads(result.output[first_brace:])
    assert "parsers" in payload
    names = [p["name"] for p in payload["parsers"]]
    assert "python_ast" in names
    assert "prose" in names
    assert "whole_file" in names
    for p in payload["parsers"]:
        assert isinstance(p["languages"], list)
        assert "module" in p
        assert "class" in p
