"""Tests for the ingest command's policy guards — the checks that run
before any per-file DB or embedding work, so they don't require a
reachable Postgres.

Right now the only guard is ``ingest.max_file_bytes``: files bigger
than the cap get recorded in ``failures`` and skipped. This is the
safety net for minified / generated / accidentally-committed blobs
that slip past the binary probe.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from hafiz.commands.ingest import _do_ingest
from hafiz.core.config import reset_settings


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    yield
    reset_settings()


def _capture_json_events(target: Path) -> list[dict]:
    """Run ``_do_ingest(--json)`` and return parsed stdout events."""
    import asyncio

    buf = StringIO()
    saved = sys.stdout
    sys.stdout = buf
    try:
        asyncio.run(_do_ingest(str(target), project=None, output_json=True))
    finally:
        sys.stdout = saved

    events: list[dict] = []
    for line in buf.getvalue().splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def test_oversize_file_is_skipped_and_reported(tmp_path, monkeypatch):
    """A file over the cap must appear in `failures`, with the cap
    visible in the message, and must not reach parse/embed."""
    monkeypatch.setenv("HAFIZ_INGEST__MAX_FILE_BYTES", "1024")  # 1 KB cap

    big = tmp_path / "huge.md"
    big.write_text("x" * 5000)  # 5 KB — well over the cap

    events = _capture_json_events(tmp_path)
    complete = [e for e in events if e.get("event") == "complete"]
    assert complete, f"no 'complete' event in {events}"
    summary = complete[-1]

    # The file must be reported as a failure, with a recognizable reason.
    paths = [f["path"] for f in summary["failures"]]
    assert str(big) in paths
    reason = next(f["error"] for f in summary["failures"] if f["path"] == str(big))
    assert "skipped" in reason
    assert "1,024" in reason or "1024" in reason  # cap echoed in message

    # And the guard must have prevented it from being counted as processed.
    assert summary["files_processed"] == 0


def test_undersize_file_is_not_skipped_by_guard(tmp_path, monkeypatch):
    """Regression guard: the size check must be `>`, not `>=` — a file
    exactly at the cap should still pass through (and then get handled
    by the rest of the pipeline)."""
    monkeypatch.setenv("HAFIZ_INGEST__MAX_FILE_BYTES", "1024")

    ok = tmp_path / "ok.md"
    ok.write_text("x" * 1024)  # exactly at cap

    events = _capture_json_events(tmp_path)
    complete = [e for e in events if e.get("event") == "complete"][-1]

    # Whatever happens downstream (DB may not be reachable in this test
    # env), the size guard must not be why this file failed.
    size_skips = [
        f for f in complete["failures"] if str(ok) == f["path"] and "skipped" in f["error"]
    ]
    assert size_skips == [], f"file at cap should not be size-skipped: {complete['failures']}"
