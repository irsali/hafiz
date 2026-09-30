"""A daemon-served recall stays light on the client side.

The per-prompt recall hook runs ``hafiz query --observations --format md`` on
every prompt. With the daemon warm, the search itself took 0.29s, yet the
command took 1.35s and 320 MB: the client loaded fastembed and SQLAlchemy at
import time, and then loaded the cross-encoder *again* to pick excerpts for
``md``/``compact`` — a model the daemon was already holding.

Two guarantees keep that from coming back:

* excerpts are picked where the model is warm (the daemon), with the client
  doing it only when there is no daemon, or one too old to;
* a daemon-served recall imports neither the DB layer nor fastembed.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime

from hafiz.core import daemon, daemon_client
from hafiz.core.annotation_result import AnnotationResult
from hafiz.core.wire import to_wire


def _result(content: str = "body") -> AnnotationResult:
    return AnnotationResult(
        id="a1",
        content=content,
        kind="fact",
        source="agent:test",
        project=None,
        tags=None,
        confidence=1.0,
        valid_from=datetime.now(UTC),
        valid_until=None,
        unit_id=None,
        metadata={},
        score=0.9,
    )


def _track_snippets(monkeypatch) -> list:
    calls = []

    async def fake_attach(query, results, budget):
        calls.append(budget)

    monkeypatch.setattr(daemon_client, "_attach_snippets", fake_attach)
    return calls


async def test_the_client_trusts_a_daemon_that_picked_snippets(monkeypatch):
    calls = _track_snippets(monkeypatch)
    sent = {}

    async def fake_request(req):
        sent.update(req)
        return {"ok": True, "results": [to_wire(_result())], "snippets": True}

    monkeypatch.setattr(daemon_client, "request", fake_request)
    results = await daemon_client.query_recall("q", snippet_budget=300, limit=2)

    assert sent["snippet_budget"] == 300
    assert sent["kwargs"] == {"limit": 2}  # not forwarded to search_annotations
    assert [r.id for r in results] == ["a1"]
    assert calls == []


async def test_an_older_daemon_is_backfilled_client_side(monkeypatch):
    """A daemon from before this change ignores the field and doesn't echo
    ``snippets`` — the output must not silently lose its excerpts."""
    calls = _track_snippets(monkeypatch)

    async def fake_request(req):
        return {"ok": True, "results": [to_wire(_result())]}

    monkeypatch.setattr(daemon_client, "request", fake_request)
    await daemon_client.query_recall("q", snippet_budget=300)
    assert calls == [300]


async def test_without_a_daemon_the_client_does_it_all(monkeypatch):
    from hafiz.core import annotations

    calls = _track_snippets(monkeypatch)

    async def no_daemon(req):
        return None

    async def fake_search(query, **kw):
        return [_result()]

    monkeypatch.setattr(daemon_client, "request", no_daemon)
    monkeypatch.setattr(annotations, "search_annotations", fake_search)
    results = await daemon_client.query_recall("q", snippet_budget=300)
    assert len(results) == 1
    assert calls == [300]


async def test_the_daemon_picks_snippets_when_asked(monkeypatch):
    import asyncio

    from hafiz.core import annotations, snippets

    long = _result("word " * 500)
    seen = {}

    async def fake_search(query, **kw):
        return [long]

    async def fake_attach(query, results, budget):
        seen["budget"] = budget
        results[0].snippet = "excerpt"

    monkeypatch.setattr(annotations, "search_annotations", fake_search)
    monkeypatch.setattr(snippets, "attach_snippets", fake_attach)
    server = daemon._Server(idle_timeout=0, _embed_lock=asyncio.Lock())
    resp = await server._op_query_recall({"query": "q", "kwargs": {}, "snippet_budget": 120})

    assert resp["snippets"] is True
    assert seen["budget"] == 120
    assert resp["results"][0]["snippet"] == "excerpt"


def test_a_daemon_served_recall_loads_no_heavy_modules():
    """End to end through the real command, in a fresh interpreter."""
    script = """
import sys
from datetime import UTC, datetime
from hafiz.core import daemon_client
from hafiz.core.annotation_result import AnnotationResult
from hafiz.core.formats import OutputFormat
from hafiz.core.wire import to_wire

row = AnnotationResult(id="a1", content="body", kind="fact", source="agent:t",
    project=None, tags=None, confidence=1.0, valid_from=datetime.now(UTC),
    valid_until=None, unit_id=None, metadata={}, score=0.9)

async def fake_request(req):
    return {"ok": True, "results": [to_wire(row)], "snippets": True}

daemon_client.request = fake_request
from hafiz.commands.observe import run_recall
run_recall("probe", output_format=OutputFormat.MD)
heavy = sorted(m for m in ("sqlalchemy", "fastembed", "onnxruntime", "hafiz.core.database")
               if m in sys.modules)
print("HEAVY:", ",".join(heavy))
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.rstrip().endswith("HEAVY:"), out.stdout  # md output precedes it
