"""Tests for hafiz.cli — CLI command registration and basic invocation."""

import json

import pytest
from typer.testing import CliRunner

from hafiz.cli import app

runner = CliRunner()


async def _db_available() -> bool:
    """Return True iff a live Postgres with the v5 schema is reachable."""
    try:
        from sqlalchemy import text

        from hafiz.core.database import close_engine, get_session_factory

        session_factory = get_session_factory()
        async with session_factory() as session:
            await session.execute(text("SELECT 1 FROM annotations LIMIT 1"))
        return True
    except Exception:
        return False
    finally:
        try:
            from hafiz.core.database import close_engine

            await close_engine()
        except Exception:
            pass


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
    assert "--observations" in result.output
    assert "--source" in result.output
    # --recall is the deprecated alias; it works but is hidden from help.
    assert "--recall" not in result.output


def test_serve_status_help():
    result = runner.invoke(app, ["serve", "status", "--help"])
    assert result.exit_code == 0
    assert "json" in result.output.lower()


def test_serve_status_json_shape(monkeypatch, tmp_path):
    """`serve status --json` reports a stable shape even with no daemon."""
    # Point at a socket that doesn't exist so status reports not-running
    # without spawning anything.
    monkeypatch.setenv("HAFIZ_DAEMON_SOCKET", str(tmp_path / "absent.sock"))
    result = runner.invoke(app, ["serve", "status", "--json"])
    assert result.exit_code == 0
    import json

    data = json.loads(result.output)
    assert data["ok"] is True
    assert data["running"] is False
    assert "socket" in data


def test_query_recall_has_no_rerank_flag():
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    assert "--no-rerank" in result.output


def test_query_tags_flag_is_documented():
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    assert "--tags" in result.output


def test_query_tags_empty_after_trimming_is_refused():
    """`--tags ""` / `--tags ","` / an unset shell variable. Falling through
    would run an ordinary recall while the caller believes it holds a curated
    pin — the same silent-wrong-answer as the blank query that sat unnoticed in
    a hook for 3.5 weeks, so it exits the same way."""
    for value in ("", ",", " , "):
        result = runner.invoke(app, ["query", "anything", "--observations", "--tags", value])
        assert result.exit_code == 2, f"--tags {value!r} should refuse"
        assert "no tag" in result.output


def test_query_tags_without_observations_is_refused():
    """Only annotations carry tags. Silently ignoring the filter would hand
    back an unfiltered set the caller then trusts, so refuse instead."""
    result = runner.invoke(app, ["query", "anything", "--tags", "always"])
    assert result.exit_code == 1
    assert "--observations" in result.output


def test_query_tags_are_split_and_trimmed(monkeypatch):
    """`--tags a, b` reaches the search layer as ["a", "b"] — comma-split,
    whitespace-trimmed, empties dropped."""
    captured = {}

    def fake_run_recall(text, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("hafiz.commands.observe.run_recall", fake_run_recall)
    result = runner.invoke(
        app, ["query", "standing rules", "--observations", "--tags", "always, pinned ,"]
    )
    assert result.exit_code == 0
    assert captured["tags"] == ["always", "pinned"]


def test_query_without_tags_passes_none(monkeypatch):
    """Absent --tags must be None, not an empty list — an empty list would
    read as "match no tags" if a future caller truth-tests it differently."""
    captured = {}

    def fake_run_recall(text, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("hafiz.commands.observe.run_recall", fake_run_recall)
    result = runner.invoke(app, ["query", "standing rules", "--observations"])
    assert result.exit_code == 0
    assert captured["tags"] is None


def test_forget_annotation_rejects_hard_combo():
    result = runner.invoke(app, ["forget", "some-id", "--annotation", "--hard"])
    assert result.exit_code == 1
    assert "can't combine" in result.output


def test_forget_help_documents_annotation_mode():
    result = runner.invoke(app, ["forget", "--help"])
    assert result.exit_code == 0
    assert "--annotation" in result.output


def test_export_help_lists_flags():
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "--format" in result.output
    assert "--include-transcripts" in result.output
    assert "--out" in result.output


def test_export_help_distinguishes_from_extract_export():
    """The contract collision with `extract export` must be documented."""
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "extract export" in result.output


def test_export_rejects_bad_format(tmp_path):
    result = runner.invoke(
        app, ["export", "--format", "xml", "--out", str(tmp_path / "e"), "--json"]
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert "unknown format" in payload["error"]


def test_query_mutual_exclusion():
    result = runner.invoke(app, ["query", "test", "--project", "x", "--workspace"])
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_query_help_lists_domain_flags():
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    assert "--include-domain" in result.output
    assert "--exclude-domain" in result.output


def test_context_help_lists_domain_flags():
    result = runner.invoke(app, ["context", "--help"])
    assert result.exit_code == 0
    assert "--include-domain" in result.output
    assert "--exclude-domain" in result.output


def test_query_rejects_overlapping_domain_filters():
    """include-domain and exclude-domain sharing a value should error
    before any DB call — the predicate would be unsatisfiable."""
    result = runner.invoke(
        app,
        [
            "query",
            "test",
            "--include-domain",
            "code,doc",
            "--exclude-domain",
            "doc",
        ],
    )
    assert result.exit_code == 2
    assert "overlap" in result.output.lower()


def test_query_rejects_dotted_domain():
    result = runner.invoke(app, ["query", "test", "--include-domain", "code.function"])
    assert result.exit_code == 2
    assert "single token" in result.output.lower()


def test_context_rejects_overlapping_domain_filters():
    result = runner.invoke(
        app,
        [
            "context",
            "test",
            "--include-domain",
            "code",
            "--exclude-domain",
            "code",
        ],
    )
    assert result.exit_code == 2
    assert "overlap" in result.output.lower()


def test_session_start_help_lists_domain_flags():
    result = runner.invoke(app, ["session", "start", "--help"])
    assert result.exit_code == 0
    assert "--include-domain" in result.output
    assert "--exclude-domain" in result.output


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


def test_recall_command_is_source_layer_recall():
    """``hafiz recall`` is the source-layer entry point (messages from a
    session/communication). Annotation search lives under
    ``hafiz query --observations`` — the two commands cover different
    layers. The word "recall" now belongs solely to this command; the old
    ``query --recall`` flag was renamed to resolve the collision."""
    result = runner.invoke(app, ["recall", "--help"])
    assert result.exit_code == 0
    assert "session" in result.output.lower()
    assert "--query" in result.output
    assert "--include-transcripts" not in result.output  # belongs on query/context


def test_query_observations_flag_replaces_recall():
    """`query --observations` searches annotations (the wisdom layer). The
    old `--recall` is a hidden, deprecation-warned alias for one release;
    the top-level `hafiz recall` now owns the word."""
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    assert "--observations" in result.output
    assert "--recall" not in result.output  # alias is hidden


def test_query_recall_alias_still_works_with_warning():
    """The deprecated `--recall` alias still dispatches to annotation search
    and prints a one-line deprecation warning. Output stream is merged in
    this Typer/Click version, so we assert on the combined output."""
    result = runner.invoke(app, ["query", "anything", "--recall", "--json"])
    # The command may exit non-zero if no DB is reachable, but the
    # deprecation notice fires before dispatch regardless.
    assert "--recall is deprecated" in result.output


def test_doctor_command_exists():
    """`hafiz doctor` is the entry point for host probing + tuning
    recommendations (phase 2 of the tunable-registry work item).
    `status --diagnose` remains as a shortcut for the health-check
    subset."""
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "--probe" in result.output
    assert "--json" in result.output


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
    result = runner.invoke(app, ["journal", "--since", "7d", "--day", "2026-04-20"])
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_journal_project_and_workspace_mutually_exclusive():
    result = runner.invoke(app, ["journal", "--project", "x", "--workspace"])
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
    result = runner.invoke(app, ["observe", "test", "--expires-in", "banana"])
    assert result.exit_code == 1
    assert "duration" in result.output.lower()


def test_capture_help():
    result = runner.invoke(app, ["capture", "--help"])
    assert result.exit_code == 0
    assert "--file" in result.output
    assert "--title" in result.output
    assert "--json" in result.output


def test_capture_rejects_both_text_and_file():
    result = runner.invoke(app, ["capture", "some text", "--file", "/tmp/does-not-matter.md"])
    assert result.exit_code == 1
    assert "not both" in result.output.lower()


def test_capture_rejects_empty_input():
    # CliRunner's stdin is a non-tty empty stream — we accept it and then
    # bail because it's empty. Either "no input" or "empty" message is fine.
    result = runner.invoke(app, ["capture"])
    assert result.exit_code == 1
    assert "empty" in result.output.lower() or "no input" in result.output.lower()


def test_capture_rejects_missing_file():
    result = runner.invoke(app, ["capture", "--file", "/tmp/does-not-exist-hafiz-test.md"])
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
    result = runner.invoke(app, ["observe", "test", "--supersedes", "not-a-uuid"])
    assert result.exit_code == 1
    assert "uuid" in result.output.lower()


def test_observe_rejects_bad_derived_from_uuid():
    result = runner.invoke(app, ["observe", "test", "--derived-from", "abc,not-a-uuid,def"])
    assert result.exit_code == 1
    assert "uuid" in result.output.lower()


def test_distill_help():
    result = runner.invoke(app, ["distill", "--help"])
    assert result.exit_code == 0
    assert "--since" in result.output
    assert "--no-transcripts" in result.output
    assert "--session" in result.output
    assert "--task" in result.output


# ─── near-duplicate detection / reconcile ────────────────────────────


def test_observe_help_has_allow_duplicate():
    result = runner.invoke(app, ["observe", "--help"])
    assert result.exit_code == 0
    assert "--allow-duplicate" in result.output


def test_reconcile_registered_and_help():
    result = runner.invoke(app, ["reconcile", "--help"])
    assert result.exit_code == 0
    assert "--threshold" in result.output
    assert "--project" in result.output


def test_observe_surfaces_near_duplicates_and_reconcile_clusters_them():
    """End-to-end: a near-duplicate observe surfaces the prior row in
    ``near_duplicates``, ``reconcile`` clusters the pair, and a genuinely
    distinct row of the same kind is neither surfaced nor clustered.

    Synchronous on purpose — ``runner.invoke`` drives the CLI's own
    ``asyncio.run``, which cannot nest inside a running loop.
    """
    import asyncio

    if not asyncio.run(_db_available()):
        pytest.skip("No live Postgres with hafiz schema available")

    proj = "_dedup_test"
    written_ids: list[str] = []

    def _observe(text: str, extra: list[str] | None = None) -> dict:
        args = ["observe", text, "--type", "decision", "--project", proj, "--json"]
        if extra:
            args += extra
        res = runner.invoke(app, args)
        assert res.exit_code == 0, res.output
        return json.loads(res.output)

    try:
        first = _observe("DEDUPTEST alpha: refresh tokens go in httponly cookies for web")
        written_ids.append(first["annotation"]["id"])
        # First write has nothing to collide with.
        assert first["near_duplicates"] == []

        second = _observe("DEDUPTEST alpha: refresh tokens belong in httponly cookies for web")
        written_ids.append(second["annotation"]["id"])
        # Near-restatement → the first row is surfaced.
        dup_ids = [d["id"] for d in second["near_duplicates"]]
        assert first["annotation"]["id"] in dup_ids
        assert all(d["score"] >= 0.88 for d in second["near_duplicates"])

        # A distinct-but-related decision must NOT be flagged as a duplicate.
        distinct = _observe("DEDUPTEST beta: session tokens go in localStorage for mobile")
        written_ids.append(distinct["annotation"]["id"])
        assert distinct["near_duplicates"] == []

        # reconcile clusters exactly the two near-duplicates, not the distinct one.
        res = runner.invoke(app, ["reconcile", "--project", proj, "--json"])
        assert res.exit_code == 0, res.output
        report = json.loads(res.output)
        clustered = {m["id"] for c in report["clusters"] for m in c["members"]}
        assert first["annotation"]["id"] in clustered
        assert second["annotation"]["id"] in clustered
        assert distinct["annotation"]["id"] not in clustered

        # Coverage is reported, and a full sweep is the default.
        assert report["scanned"] == report["total_live"] == 3
        assert report["truncated"] is False

        # Every cluster carries a runnable proposal that spares its primary.
        cluster = report["clusters"][0]
        assert cluster["suggested_action"] in {"retire", "review", "merge"}
        assert cluster["primary_id"] in {m["id"] for m in cluster["members"] if m["primary"]}
        assert cluster["commands"]
        for other in (m for m in cluster["members"] if not m["primary"]):
            assert f"hafiz forget {other['id']} --annotation" in cluster["commands"]
    finally:
        for ann_id in written_ids:
            runner.invoke(app, ["forget", ann_id, "--annotation"])


def test_oversized_observe_warns_but_still_stores():
    """The brake is advisory. If it ever starts failing writes, agents will
    route around it and the store loses the record entirely — which is worse
    than a long record."""
    import asyncio

    if not asyncio.run(_db_available()):
        pytest.skip("No live Postgres with hafiz schema available")

    proj = "_oversize_test"
    written_ids: list[str] = []
    try:
        long_res = runner.invoke(
            app,
            [
                "observe",
                "OVERSIZE " + "x" * 2000,
                "--type",
                "decision",
                "--project",
                proj,
                "--json",
            ],
        )
        assert long_res.exit_code == 0, long_res.output
        long_data = json.loads(long_res.output)
        written_ids.append(long_data["annotation"]["id"])
        assert long_data["oversized"]["chars"] == 2009
        assert long_data["oversized"]["limit"] == 1500
        # The row is really there, warning notwithstanding.
        assert long_data["annotation"]["content"].startswith("OVERSIZE ")

        short_res = runner.invoke(
            app,
            ["observe", "OVERSIZE short one", "--type", "decision", "--project", proj, "--json"],
        )
        assert short_res.exit_code == 0, short_res.output
        short_data = json.loads(short_res.output)
        written_ids.append(short_data["annotation"]["id"])
        assert short_data["oversized"] is None

        # The note firehose is never gated, and never nagged.
        note_res = runner.invoke(
            app, ["note", "OVERSIZE note " + "y" * 2000, "--project", proj, "--json"]
        )
        assert note_res.exit_code == 0, note_res.output
        note_data = json.loads(note_res.output)
        written_ids.append(note_data["annotation"]["id"])
        assert note_data["oversized"] is None
    finally:
        for ann_id in written_ids:
            runner.invoke(app, ["forget", ann_id, "--annotation"])


def test_reconcile_reports_a_truncated_scan_instead_of_hiding_it():
    """A ``--limit`` that bites must be visible in the output.

    The previous default capped the sweep at the newest 500 live annotations
    and reported the clusters it found with nothing to indicate it had looked
    at part of the store — 34 clusters on a deployment where a full sweep finds
    65. An undercount that reads as a complete answer is worse than a slow one.
    """
    import asyncio

    if not asyncio.run(_db_available()):
        pytest.skip("No live Postgres with hafiz schema available")

    proj = "_dedup_coverage_test"
    written_ids: list[str] = []
    try:
        for i in range(3):
            res = runner.invoke(
                app,
                ["observe", f"COVERAGE row {i}", "--type", "fact", "--project", proj, "--json"],
            )
            assert res.exit_code == 0, res.output
            written_ids.append(json.loads(res.output)["annotation"]["id"])

        res = runner.invoke(app, ["reconcile", "--project", proj, "--limit", "2", "--json"])
        assert res.exit_code == 0, res.output
        capped = json.loads(res.output)
        assert capped["scanned"] == 2
        assert capped["total_live"] == 3
        assert capped["truncated"] is True

        res = runner.invoke(app, ["reconcile", "--project", proj, "--limit", "0", "--json"])
        assert res.exit_code == 0, res.output
        full = json.loads(res.output)
        assert full["scanned"] == full["total_live"] == 3
        assert full["truncated"] is False
    finally:
        for ann_id in written_ids:
            runner.invoke(app, ["forget", ann_id, "--annotation"])


def test_doctor_counts_near_duplicate_annotations():
    """Merge pressure has to surface somewhere nobody has to remember to look.

    ``reconcile`` is read-only and opt-in, so drift accumulated unseen — 146
    rows across 65 clusters on a real store. The count rides on ``doctor``
    (reachable as ``status --diagnose``) rather than ``status``, because it is a
    quadratic scan with no vector index behind it.
    """
    import asyncio

    if not asyncio.run(_db_available()):
        pytest.skip("No live Postgres with hafiz schema available")

    async def _clustered() -> int:
        """Close the engine afterwards: the next ``runner.invoke`` opens its own
        loop, and a pooled connection bound to this one fails there."""
        from hafiz.core.annotations import count_clustered_annotations
        from hafiz.core.database import close_engine

        try:
            return await count_clustered_annotations()
        finally:
            await close_engine()

    proj = "_dedup_doctor_test"
    written_ids: list[str] = []
    try:
        baseline = asyncio.run(_clustered())
        for text in (
            "DOCTORDUP: refresh tokens go in httponly cookies for web clients",
            "DOCTORDUP: refresh tokens belong in httponly cookies for web clients",
        ):
            res = runner.invoke(
                app, ["observe", text, "--type", "decision", "--project", proj, "--json"]
            )
            assert res.exit_code == 0, res.output
            written_ids.append(json.loads(res.output)["annotation"]["id"])

        assert asyncio.run(_clustered()) >= baseline + 2

        res = runner.invoke(app, ["doctor", "--json"])
        assert res.exit_code in (0, 1), res.output
        names = {c["name"] for c in json.loads(res.output)["checks"]}
        assert "Knowledge base deduplicated" in names
    finally:
        for ann_id in written_ids:
            runner.invoke(app, ["forget", ann_id, "--annotation"])


def test_observe_strict_mode_blocks_then_allow_duplicate_overrides(monkeypatch):
    """Strict dedup refuses a *near*-duplicate write (exit 2) until
    ``--allow-duplicate`` is passed.

    The second row is deliberately similar-but-not-identical: byte-identical
    text hits the exact-duplicate check first (which is unconditional and not
    governed by ``strict``), so reusing the same string would exercise that
    path instead of this one.
    """
    import asyncio

    if not asyncio.run(_db_available()):
        pytest.skip("No live Postgres with hafiz schema available")

    monkeypatch.setenv("HAFIZ_DEDUP__STRICT", "true")
    proj = "_dedup_strict_test"
    baseline = "DEDUPTEST strict baseline row about consent receipt retention"
    similar = "DEDUPTEST strict baseline row concerning consent receipt retention"
    written_ids: list[str] = []
    try:
        base = runner.invoke(
            app,
            ["observe", baseline, "--type", "fact", "--project", proj, "--json"],
        )
        assert base.exit_code == 0, base.output
        written_ids.append(json.loads(base.output)["annotation"]["id"])

        blocked = runner.invoke(
            app,
            ["observe", similar, "--type", "fact", "--project", proj, "--json"],
        )
        assert blocked.exit_code == 2, blocked.output
        payload = json.loads(blocked.output)
        assert payload["ok"] is False
        assert payload["near_duplicates"]

        forced = runner.invoke(
            app,
            [
                "observe",
                similar,
                "--type",
                "fact",
                "--project",
                proj,
                "--allow-duplicate",
                "--json",
            ],
        )
        assert forced.exit_code == 0, forced.output
        written_ids.append(json.loads(forced.output)["annotation"]["id"])
    finally:
        for ann_id in written_ids:
            runner.invoke(app, ["forget", ann_id, "--annotation"])


def test_observe_refuses_byte_identical_row_regardless_of_strict(monkeypatch):
    """An exact repeat is refused with ``existing_id``, even in surface-only mode.

    Surface-don't-block is right for *near* duplicates — only the author can
    judge whether a similar row refines or contradicts the old one. A
    byte-identical row admits no such judgement.
    """
    import asyncio

    if not asyncio.run(_db_available()):
        pytest.skip("No live Postgres with hafiz schema available")

    monkeypatch.setenv("HAFIZ_DEDUP__STRICT", "false")
    proj = "_dedup_exact_test"
    text = "DEDUPTEST byte-identical row"
    written_ids: list[str] = []
    try:
        first = runner.invoke(app, ["observe", text, "--type", "fact", "--project", proj, "--json"])
        assert first.exit_code == 0, first.output
        existing = json.loads(first.output)["annotation"]["id"]
        written_ids.append(existing)

        repeat = runner.invoke(
            app, ["observe", text, "--type", "fact", "--project", proj, "--json"]
        )
        assert repeat.exit_code == 2, repeat.output
        payload = json.loads(repeat.output)
        assert payload["ok"] is False
        assert payload["existing_id"] == existing
    finally:
        for ann_id in written_ids:
            runner.invoke(app, ["forget", ann_id, "--annotation"])


def test_note_dedupes_byte_identical_writes_as_success(monkeypatch):
    """The note firehose stays open: an exact repeat returns the existing row
    with ``deduped: true`` and exit 0, rather than being refused."""
    import asyncio

    if not asyncio.run(_db_available()):
        pytest.skip("No live Postgres with hafiz schema available")

    proj = "_dedup_note_test"
    text = "DEDUPTEST identical note"
    written_ids: list[str] = []
    try:
        first = runner.invoke(app, ["note", text, "--project", proj, "--json"])
        assert first.exit_code == 0, first.output
        first_id = json.loads(first.output)["annotation"]["id"]
        written_ids.append(first_id)

        again = runner.invoke(app, ["note", text, "--project", proj, "--json"])
        assert again.exit_code == 0, again.output
        payload = json.loads(again.output)
        assert payload["deduped"] is True
        assert payload["annotation"]["id"] == first_id
    finally:
        for ann_id in written_ids:
            runner.invoke(app, ["forget", ann_id, "--annotation"])


def test_distill_project_and_workspace_mutually_exclusive():
    result = runner.invoke(app, ["distill", "--project", "x", "--workspace"])
    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


# ── review — must run clean against the v5 schema (regression for the
#    pre-v5 Chunk/Entity/Relation/Observation crash) ─────────────────────────


def test_review_runs_clean_on_v5_schema():
    """`hafiz review --json` must produce a well-formed report, not crash.

    Regression guard: review.py used to query removed ORM tables and died
    with ``ArgumentError: ...got Chunk``. It now reads units/edges/
    embeddings/annotations.

    Synchronous on purpose: ``runner.invoke`` drives the CLI's own
    ``asyncio.run``, which cannot nest inside a running event loop.
    """
    import asyncio

    if not asyncio.run(_db_available()):
        pytest.skip("No live Postgres with hafiz schema available")

    result = runner.invoke(app, ["review", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    # Shape contract: stats counts the four v5 knowledge tables; findings
    # and summary are always present.
    for key in ("units", "edges", "embeddings", "annotations"):
        assert key in payload["stats"], payload["stats"]
        assert isinstance(payload["stats"][key], int)
    assert isinstance(payload["findings"], list)
    assert payload["summary"]["total"] == len(payload["findings"])
    # None of the removed-table sentinels should leak into output.
    assert "Chunk" not in result.output
    assert "Entity" not in result.output


def test_distill_backlog_drains_when_a_note_is_promoted_or_declined():
    """The backlog has to converge, or it stops being believable.

    A candidate leaves the queue two ways, and neither needs a new column:
    ``observe --derived-from`` cites it (promoted), or ``forget --annotation``
    retires it (declined). Before this, ``distill`` re-offered the same notes
    on every run forever — across the real store's whole life, 0 of 16 notes
    had ever been drained while 52 ``derived_from`` citations existed between
    decision-grade rows.
    """
    import asyncio

    if not asyncio.run(_db_available()):
        pytest.skip("No live Postgres with hafiz schema available")

    proj = "_distill_drain_test"
    written_ids: list[str] = []
    try:
        for i in range(3):
            res = runner.invoke(
                app, ["note", f"DRAIN candidate number {i}", "--project", proj, "--json"]
            )
            assert res.exit_code == 0, res.output
            written_ids.append(json.loads(res.output)["annotation"]["id"])
        promoted_note, declined_note, remaining_note = written_ids

        def backlog(*extra: str) -> dict:
            res = runner.invoke(app, ["distill", "--project", proj, "--json", *extra])
            assert res.exit_code == 0, res.output
            return json.loads(res.output)

        start = backlog()
        assert start["backlog"]["pending"] == 3
        assert start["backlog"]["promoted"] == 0

        # Three pending clears brief_min_pending, so the gate opens and the
        # scaffold names every candidate.
        brief = runner.invoke(app, ["distill", "--project", proj, "--brief"])
        assert brief.exit_code == 0, brief.output
        assert "Distillation backlog" in brief.output

        # ── promoted: cited via --derived-from ──────────────────────────
        res = runner.invoke(
            app,
            [
                "observe",
                "DRAIN distilled decision",
                "--type",
                "decision",
                "--project",
                proj,
                "--derived-from",
                promoted_note,
                "--json",
            ],
        )
        assert res.exit_code == 0, res.output
        written_ids.append(json.loads(res.output)["annotation"]["id"])

        after_promote = backlog()
        assert after_promote["backlog"]["pending"] == 2
        assert after_promote["backlog"]["promoted"] == 1
        assert promoted_note not in [n["id"] for n in after_promote["notes"]]

        # Still auditable — the drain is visible, not just a shorter list.
        audited = backlog("--include-promoted")
        by_id = {n["id"]: n for n in audited["notes"]}
        assert by_id[promoted_note]["promoted"] is True
        assert by_id[remaining_note]["promoted"] is False

        # ── declined: retired with no replacement ──────────────────────
        res = runner.invoke(app, ["forget", declined_note, "--annotation"])
        assert res.exit_code == 0, res.output

        after_decline = backlog()
        assert after_decline["backlog"]["pending"] == 1
        assert [n["id"] for n in after_decline["notes"]] == [remaining_note]

        # One young capture is below both gates, so brief goes quiet again.
        brief = runner.invoke(app, ["distill", "--project", proj, "--brief"])
        assert brief.exit_code == 0, brief.output
        assert brief.output.strip() == ""
    finally:
        for ann_id in written_ids:
            runner.invoke(app, ["forget", ann_id, "--annotation"])


def test_distill_limit_is_spent_on_candidates_not_on_promoted_rows():
    """`--limit` must bite on work, not on rows that get discarded after.

    Filtering promoted notes in Python meant the cap was consumed by already-
    distilled rows: a window of 3 notes with 2 promoted returned 1 candidate
    under `--limit 2` and looked like a truncated scan of real work. And
    `promoted` is the one metric that says whether the loop is working, so it
    is counted un-capped.
    """
    import asyncio

    if not asyncio.run(_db_available()):
        pytest.skip("No live Postgres with hafiz schema available")

    proj = "_distill_limit_test"
    written_ids: list[str] = []
    try:
        notes: list[str] = []
        for i in range(3):
            res = runner.invoke(app, ["note", f"LIMIT candidate {i}", "--project", proj, "--json"])
            assert res.exit_code == 0, res.output
            notes.append(json.loads(res.output)["annotation"]["id"])
        written_ids.extend(notes)

        # Promote two of the three.
        for note_id in notes[:2]:
            res = runner.invoke(
                app,
                [
                    "observe",
                    f"LIMIT distilled {note_id[:8]}",
                    "--type",
                    "decision",
                    "--project",
                    proj,
                    "--derived-from",
                    note_id,
                    "--json",
                ],
            )
            assert res.exit_code == 0, res.output
            written_ids.append(json.loads(res.output)["annotation"]["id"])

        res = runner.invoke(app, ["distill", "--project", proj, "--limit", "2", "--json"])
        assert res.exit_code == 0, res.output
        data = json.loads(res.output)
        # The one pending note survives the cap; the promoted pair never
        # competed for a slot.
        assert [n["id"] for n in data["notes"]] == [notes[2]]
        assert data["backlog"]["pending"] == 1
        assert data["backlog"]["promoted"] == 2
    finally:
        for ann_id in written_ids:
            runner.invoke(app, ["forget", ann_id, "--annotation"])
