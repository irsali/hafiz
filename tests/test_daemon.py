"""Tests for the warm-daemon protocol helpers and client fallback contract.

Socket round-trips and DB-hitting ops are covered by dogfooding (the daemon
is exercised live via the Hermes provider). Here we pin the parts that must
hold regardless of whether a daemon is running: socket-path resolution,
version handshake, request serialization, the annotation serializer, and the
``HAFIZ_NO_DAEMON`` escape hatch.
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
