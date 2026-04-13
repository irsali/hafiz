"""Tests for hafiz.cli — CLI command registration and basic invocation."""

from typer.testing import CliRunner

from hafiz.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "hafiz" in result.output.lower()


def test_init_help():
    result = runner.invoke(app, ["init", "--help"])
    assert result.exit_code == 0
    assert "Initialize" in result.output


def test_ingest_help():
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "path" in result.output.lower()


def test_query_help():
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    assert "json" in result.output.lower()


def test_status_help():
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0


def test_config_show_help():
    result = runner.invoke(app, ["config", "show", "--help"])
    assert result.exit_code == 0
    assert "json" in result.output.lower()
