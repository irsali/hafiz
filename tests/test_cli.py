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
    assert "--recall" in result.output


def test_query_mutual_exclusion():
    result = runner.invoke(app, ["query", "test", "--project", "x", "--workspace"])
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_status_help():
    result = runner.invoke(app, ["status", "--help"])
    assert result.exit_code == 0
    assert "--diagnose" in result.output


def test_config_show_help():
    result = runner.invoke(app, ["config", "show", "--help"])
    assert result.exit_code == 0
    assert "json" in result.output.lower()


def test_context_help():
    result = runner.invoke(app, ["context", "--help"])
    assert result.exit_code == 0
    assert "--workspace" in result.output
    assert "--project" in result.output


def test_context_mutual_exclusion():
    result = runner.invoke(app, ["context", "test", "--project", "x", "--workspace"])
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_review_help():
    result = runner.invoke(app, ["review", "--help"])
    assert result.exit_code == 0
    assert "--project" in result.output
    assert "--json" in result.output


def test_extract_export_help():
    result = runner.invoke(app, ["extract", "export", "--help"])
    assert result.exit_code == 0
    # v2 export surfaces AST-known units/edges; the v1 chunk-oriented
    # flags (--unextracted, --path, --offset) are gone.
    assert "--limit" in result.output
    assert "--pretty" in result.output
    assert "--project" in result.output


def test_extract_import_help():
    result = runner.invoke(app, ["extract", "import", "--help"])
    assert result.exit_code == 0
    assert "--file" in result.output


def test_removed_recall_command():
    """recall is now query --recall, standalone recall should not exist."""
    result = runner.invoke(app, ["recall", "--help"])
    assert result.exit_code != 0


def test_removed_doctor_command():
    """doctor is now status --diagnose, standalone doctor should not exist."""
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code != 0


def test_removed_chunks_command():
    """chunks export is now extract export, standalone chunks should not exist."""
    result = runner.invoke(app, ["chunks", "export", "--help"])
    assert result.exit_code != 0


def test_observe_help_has_expiry_flags():
    result = runner.invoke(app, ["observe", "--help"])
    assert result.exit_code == 0
    assert "--expires-in" in result.output
    assert "--expires" in result.output


def test_note_help():
    result = runner.invoke(app, ["note", "--help"])
    assert result.exit_code == 0
    assert "raw thought" in result.output.lower() or "note" in result.output.lower()
    assert "--expires-in" in result.output


def test_journal_help():
    result = runner.invoke(app, ["journal", "--help"])
    assert result.exit_code == 0
    assert "--since" in result.output
    assert "--day" in result.output
    assert "--json" in result.output


def test_journal_since_and_day_mutually_exclusive():
    result = runner.invoke(
        app, ["journal", "--since", "7d", "--day", "2026-04-20"]
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_journal_project_and_workspace_mutually_exclusive():
    result = runner.invoke(
        app, ["journal", "--project", "x", "--workspace"]
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_observe_rejects_both_expiry_flags():
    result = runner.invoke(
        app,
        [
            "observe",
            "test",
            "--expires-in",
            "30d",
            "--expires",
            "2026-06-01",
        ],
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_observe_rejects_garbage_duration():
    result = runner.invoke(
        app, ["observe", "test", "--expires-in", "banana"]
    )
    assert result.exit_code == 1
    assert "duration" in result.output.lower()


def test_capture_help():
    result = runner.invoke(app, ["capture", "--help"])
    assert result.exit_code == 0
    assert "--file" in result.output
    assert "--title" in result.output
    assert "--json" in result.output


def test_capture_rejects_both_text_and_file():
    result = runner.invoke(
        app, ["capture", "some text", "--file", "/tmp/does-not-matter.md"]
    )
    assert result.exit_code == 1
    assert "not both" in result.output.lower()


def test_capture_rejects_empty_input():
    # CliRunner's stdin is a non-tty empty stream — we accept it and then
    # bail because it's empty. Either "no input" or "empty" message is fine.
    result = runner.invoke(app, ["capture"])
    assert result.exit_code == 1
    assert "empty" in result.output.lower() or "no input" in result.output.lower()


def test_capture_rejects_missing_file():
    result = runner.invoke(
        app, ["capture", "--file", "/tmp/does-not-exist-hafiz-test.md"]
    )
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_session_help_lists_subcommands():
    result = runner.invoke(app, ["session", "--help"])
    assert result.exit_code == 0
    assert "start" in result.output
    assert "end" in result.output
    assert "show" in result.output


def test_session_start_help_has_task_and_project():
    result = runner.invoke(app, ["session", "start", "--help"])
    assert result.exit_code == 0
    assert "--task" in result.output
    assert "--project" in result.output
    assert "NAME" in result.output or "name" in result.output.lower()


def test_session_end_help():
    result = runner.invoke(app, ["session", "end", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output


def test_session_show_help():
    result = runner.invoke(app, ["session", "show", "--help"])
    assert result.exit_code == 0
    assert "--json" in result.output


def test_observe_help_has_session_and_task():
    result = runner.invoke(app, ["observe", "--help"])
    assert result.exit_code == 0
    assert "--session" in result.output
    assert "--task" in result.output


def test_note_help_has_session_and_task():
    result = runner.invoke(app, ["note", "--help"])
    assert result.exit_code == 0
    assert "--session" in result.output
    assert "--task" in result.output


def test_capture_help_has_session_and_task():
    result = runner.invoke(app, ["capture", "--help"])
    assert result.exit_code == 0
    assert "--session" in result.output
    assert "--task" in result.output


def test_journal_help_has_session_and_task_filters():
    result = runner.invoke(app, ["journal", "--help"])
    assert result.exit_code == 0
    assert "--session" in result.output
    assert "--task" in result.output


def test_observe_help_has_supersedes_and_derived_from():
    result = runner.invoke(app, ["observe", "--help"])
    assert result.exit_code == 0
    assert "--supersedes" in result.output
    assert "--derived-from" in result.output


def test_note_help_has_supersedes_and_derived_from():
    result = runner.invoke(app, ["note", "--help"])
    assert result.exit_code == 0
    assert "--supersedes" in result.output
    assert "--derived-from" in result.output


def test_query_help_has_include_superseded():
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    assert "--include-superseded" in result.output


def test_observe_rejects_bad_supersedes_uuid():
    result = runner.invoke(
        app, ["observe", "test", "--supersedes", "not-a-uuid"]
    )
    assert result.exit_code == 1
    assert "uuid" in result.output.lower()


def test_observe_rejects_bad_derived_from_uuid():
    result = runner.invoke(
        app, ["observe", "test", "--derived-from", "abc,not-a-uuid,def"]
    )
    assert result.exit_code == 1
    assert "uuid" in result.output.lower()


def test_distill_help():
    result = runner.invoke(app, ["distill", "--help"])
    assert result.exit_code == 0
    assert "--since" in result.output
    assert "--no-transcripts" in result.output
    assert "--session" in result.output
    assert "--task" in result.output


def test_distill_project_and_workspace_mutually_exclusive():
    result = runner.invoke(
        app, ["distill", "--project", "x", "--workspace"]
    )
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output
