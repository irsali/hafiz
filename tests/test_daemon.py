"""Tests for the warm daemon: protocol helpers, fallback contract, and wiring.

The original version of this file said DB-hitting ops were "covered by
dogfooding (the daemon is exercised live via the Hermes provider)". That
assumption is what let the feature rot: no *hafiz* command ever called the
daemon, so from this repo's side it was unreachable for three months while
``serve status`` advertised auto-spawn. An external consumer exercising the
socket says nothing about whether hafiz's own CLI reaches it.

So the file now covers two layers. The parts that hold regardless of a
running daemon — socket-path resolution, version handshake, serialization,
the ``HAFIZ_NO_DAEMON`` escape hatch — and, below, the wiring itself.
"""

from __future__ import annotations

import os
from pathlib import Path

from hafiz import __version__
from hafiz.core import daemon, daemon_client

# ---------------------------------------------------------------------------
# Socket path resolution
# ---------------------------------------------------------------------------


def test_socket_path_honors_explicit_override(monkeypatch):
    monkeypatch.setenv("HAFIZ_DAEMON_SOCKET", "/tmp/custom-hafiz.sock")
    assert daemon.socket_path() == Path("/tmp/custom-hafiz.sock")


def test_socket_path_uses_xdg_runtime_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("HAFIZ_DAEMON_SOCKET", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    sock = daemon.socket_path()
    assert sock == tmp_path / "hafiz" / "daemon.sock"
    # The parent runtime dir is created 0700.
    assert (tmp_path / "hafiz").is_dir()
    assert oct(os.stat(tmp_path / "hafiz").st_mode & 0o777) == "0o700"


def test_socket_path_prefers_tmpdir_over_tmp(monkeypatch, tmp_path):
    # macOS sets a per-user $TMPDIR; prefer it over the world-shared /tmp.
    monkeypatch.delenv("HAFIZ_DAEMON_SOCKET", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    sock = daemon.socket_path()
    assert sock == tmp_path / f"hafiz-{os.getuid()}" / "daemon.sock"
    assert oct(os.stat(tmp_path / f"hafiz-{os.getuid()}").st_mode & 0o777) == "0o700"


def test_socket_path_falls_back_to_tmp(monkeypatch):
    monkeypatch.delenv("HAFIZ_DAEMON_SOCKET", raising=False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("TMPDIR", raising=False)
    sock = daemon.socket_path()
    assert str(sock).startswith(f"/tmp/hafiz-{os.getuid()}")


# ---------------------------------------------------------------------------
# Protocol version
# ---------------------------------------------------------------------------


def test_protocol_version_tracks_package_version():
    assert daemon.PROTOCOL_VERSION == __version__


# ---------------------------------------------------------------------------
# Annotation serializer — tolerant of partial attribute sets
# ---------------------------------------------------------------------------


def test_annotation_to_dict_pulls_present_attrs():
    class FakeRow:
        id = "abc-123"
        content = "a learning"
        kind = "learning"
        source = "agent:hermes"
        score = 0.87

    out = daemon._annotation_to_dict(FakeRow())
    assert out["id"] == "abc-123"  # stringified
    assert out["content"] == "a learning"
    assert out["kind"] == "learning"
    assert out["score"] == 0.87
    # Absent attrs are simply omitted, not None.
    assert "valid_until" not in out


def test_annotation_to_dict_stringifies_uuid_like_id():
    import uuid

    class FakeRow:
        id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        content = "x"

    out = daemon._annotation_to_dict(FakeRow())
    assert out["id"] == "12345678-1234-5678-1234-567812345678"
    assert isinstance(out["id"], str)


# ---------------------------------------------------------------------------
# Client escape hatch + fallback contract
# ---------------------------------------------------------------------------


def test_daemon_disabled_env(monkeypatch):
    for val in ("1", "true", "yes", "anything"):
        monkeypatch.setenv("HAFIZ_NO_DAEMON", val)
        assert daemon_client._daemon_disabled() is True
    for val in ("", "0", "false"):
        monkeypatch.setenv("HAFIZ_NO_DAEMON", val)
        assert daemon_client._daemon_disabled() is False
    monkeypatch.delenv("HAFIZ_NO_DAEMON", raising=False)
    assert daemon_client._daemon_disabled() is False


async def test_request_returns_none_when_disabled(monkeypatch):
    """With the escape hatch on, request() never touches the socket."""
    monkeypatch.setenv("HAFIZ_NO_DAEMON", "1")
    resp = await daemon_client.request({"op": "ping"})
    assert resp is None


async def test_request_to_dead_socket_does_not_raise(monkeypatch, tmp_path):
    """A missing socket + spawn disabled degrades to None, never raises.

    We point at a nonexistent socket and stub the spawn to a no-op so the test
    is hermetic (no real daemon process). The contract: callers get None and
    fall back to direct execution.
    """
    monkeypatch.delenv("HAFIZ_NO_DAEMON", raising=False)
    monkeypatch.setenv("HAFIZ_DAEMON_SOCKET", str(tmp_path / "nope.sock"))

    async def _no_spawn():
        return None

    monkeypatch.setattr(daemon_client, "_spawn_daemon", _no_spawn)
    resp = await daemon_client.request({"op": "ping"})
    assert resp is None


# ---------------------------------------------------------------------------
# Wiring — the tests whose absence let the daemon rot
# ---------------------------------------------------------------------------
# `hafiz serve` shipped in f285dbf with a client, a protocol, and ten tests,
# and no command ever called it. It was unreachable for three months while
# `serve status` printed "Auto-spawns on the next client request." Every test
# above exercises the client in isolation, which cannot tell a wired feature
# from a deleted one. These assert the connection itself.


def test_recall_command_goes_through_the_daemon_client(monkeypatch):
    """`hafiz query --observations` must call the daemon-first helper.

    If someone reverts this to `search_annotations`, the daemon silently
    stops being used and only a latency regression would show it.
    """
    called = {}

    async def _fake_query_recall(query, **kwargs):
        called["query"] = query
        called["kwargs"] = kwargs
        return []

    monkeypatch.setattr(daemon_client, "query_recall", _fake_query_recall)

    from hafiz.commands.observe import run_recall

    run_recall("probe query", limit=3)
    assert called["query"] == "probe query"
    assert called["kwargs"]["limit"] == 3


def test_recall_forwards_include_superseded_as_active_only(monkeypatch):
    """The filter the old hand-maintained forwarding dropped.

    `active_only` was absent from the daemon's key list, so once wired,
    `--include-superseded` would have returned only active rows — the user
    asks for prior beliefs and silently gets none.
    """
    seen = {}

    async def _fake_query_recall(query, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(daemon_client, "query_recall", _fake_query_recall)

    from hafiz.commands.observe import run_recall

    run_recall("probe", include_superseded=True)
    assert seen["active_only"] is False


def test_context_command_goes_through_the_daemon_client(monkeypatch):
    from hafiz.core.context import ContextBundle

    called = {}

    async def _fake_context(query, **kwargs):
        called["query"] = query
        return ContextBundle(query=query)

    monkeypatch.setattr(daemon_client, "context", _fake_context)

    from hafiz.commands.context import run_context
    from hafiz.core.formats import OutputFormat

    run_context("probe context", output_format=OutputFormat.JSON)
    assert called["query"] == "probe context"


# ---------------------------------------------------------------------------
# Wire format — lossless, unlike to_dict()
# ---------------------------------------------------------------------------


def test_wire_round_trip_keeps_every_field():
    """`to_dict()` is the trimmed public contract; the wire format is not.

    Using `to_dict()` for transport round-trips *almost* correctly, which is
    the trap: the warm path returns objects missing `metadata` and `snippet`,
    nothing raises, and `--format compact` quietly renders differently.
    """
    import dataclasses
    from datetime import UTC, datetime

    from hafiz.core.annotations import AnnotationResult
    from hafiz.core.wire import from_wire, to_wire

    original = AnnotationResult(
        id="abc",
        content="body",
        kind="fact",
        source="agent:test",
        project="p",
        tags=["x"],
        confidence=0.9,
        valid_from=datetime(2026, 9, 5, 12, tzinfo=UTC),
        valid_until=None,
        unit_id="u1",
        metadata={"k": "v"},
        score=0.8,
        rerank_score=0.95,
        snippet="excerpt",
    )
    rebuilt = from_wire(AnnotationResult, to_wire(original))
    assert rebuilt == original
    # Belt and braces: every declared field made the trip, so adding one
    # later cannot silently fail to cross the wire.
    for f in dataclasses.fields(AnnotationResult):
        assert getattr(rebuilt, f.name) == getattr(original, f.name), f.name


def test_wire_preserves_datetime_type_not_just_value():
    """A bare ISO string would rebuild as `str`, and comparisons against
    `datetime.now(UTC)` would then raise on the warm path only."""
    from datetime import UTC, datetime

    from hafiz.core.annotations import AnnotationResult
    from hafiz.core.wire import from_wire, to_wire

    when = datetime(2026, 9, 5, 12, tzinfo=UTC)
    rebuilt = from_wire(
        AnnotationResult,
        to_wire(
            AnnotationResult(
                id="a",
                content="c",
                kind="fact",
                source=None,
                project=None,
                tags=None,
                confidence=1.0,
                valid_from=when,
                valid_until=None,
                unit_id=None,
                metadata={},
                score=0.5,
            )
        ),
    )
    assert isinstance(rebuilt.valid_from, datetime)
    assert rebuilt.valid_from == when


def test_wire_is_a_superset_of_the_public_json_contract():
    """Anything `--json` exposes must survive transport."""
    from hafiz.core.annotations import AnnotationResult
    from hafiz.core.wire import to_wire

    sample = AnnotationResult(
        id="a",
        content="c",
        kind="fact",
        source=None,
        project=None,
        tags=None,
        confidence=1.0,
        valid_from=None,
        valid_until=None,
        unit_id="u",
        metadata={"m": 1},
        score=0.5,
        snippet="s",
    )
    wire = to_wire(sample)
    for key in ("unit_id", "metadata", "snippet"):
        assert key in wire, f"{key} was dropped by the old hand-written serializer"


def test_context_bundle_round_trips():
    from hafiz.core.context import ContextBundle

    bundle = ContextBundle(query="q", entities=[{"name": "E"}], project_distribution={"p": 2})
    rebuilt = ContextBundle.from_wire(bundle.to_wire())
    assert rebuilt.query == "q"
    assert rebuilt.entities == [{"name": "E"}]
    assert rebuilt.project_distribution == {"p": 2}
    # A real bundle, not a dict — `to_compact()` is called on the result.
    assert hasattr(rebuilt, "to_compact")


def test_unknown_kwargs_are_dropped_not_raised():
    """A newer client against an older daemon must degrade, not crash."""

    def target(query, *, limit=10):
        return query, limit

    assert daemon._supported_kwargs(target, {"limit": 5, "brand_new_filter": True}) == {"limit": 5}


# ---------------------------------------------------------------------------
# Staying hot
# ---------------------------------------------------------------------------


def test_idle_timeout_env_beats_config(monkeypatch):
    monkeypatch.setenv("HAFIZ_DAEMON_IDLE", "42")
    assert daemon.configured_idle_timeout() == 42.0


def test_idle_timeout_zero_means_never(monkeypatch):
    """The setting a continuously-consuming product needs: without it the
    ~0.9s model load is repaid after every quiet spell."""
    monkeypatch.setenv("HAFIZ_DAEMON_IDLE", "0")
    assert daemon.configured_idle_timeout() == 0.0


def test_idle_timeout_ignores_a_typo_rather_than_dying(monkeypatch):
    monkeypatch.setenv("HAFIZ_DAEMON_IDLE", "half an hour")
    assert daemon.configured_idle_timeout() > 0
