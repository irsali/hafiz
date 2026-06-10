"""Client for the hafiz warm daemon, with transparent direct-exec fallback.

The contract: callers ask for a memory op (``context`` / ``query_recall`` /
``observe`` / ``capture``) and get back the same data the daemon would
return — but if the daemon is absent, unreachable, version-skewed, or errors,
the client silently runs the **same core function in-process** instead. So
turning the daemon on can only make things faster; it can never make a caller
fail in a way the plain CLI wouldn't.

Lifecycle: ``request()`` connects to the socket; on connection refusal it
auto-spawns ``hafiz serve`` (detached) and retries once. A version mismatch
kills the stale daemon and respawns. Every socket op has a hard timeout so a
wedged daemon degrades to the fallback rather than hanging the caller.

Opt out entirely with ``HAFIZ_NO_DAEMON=1`` — every call then runs direct.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

from hafiz.core.daemon import PROTOCOL_VERSION, socket_path

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT = 1.0   # connecting to a live daemon is instant; cap it low
_REQUEST_TIMEOUT = 30.0  # a warm op is ~tens of ms; this is the wedged-daemon cap
_SPAWN_WARMUP_TIMEOUT = 20.0  # cold start loads the model — allow it to come up


def _daemon_disabled() -> bool:
    return os.environ.get("HAFIZ_NO_DAEMON", "").strip() not in ("", "0", "false")


# ---------------------------------------------------------------------------
# Low-level socket round-trip
# ---------------------------------------------------------------------------


async def _send_one(req: dict, *, timeout: float) -> dict | None:
    """Open a connection, send one request, read one response. None on failure."""
    sock = socket_path()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(sock)), timeout=_CONNECT_TIMEOUT
        )
    except (TimeoutError, OSError):
        return None
    try:
        req.setdefault("version", PROTOCOL_VERSION)
        writer.write((json.dumps(req, default=str) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not line:
            return None
        return json.loads(line)
    except (TimeoutError, OSError, json.JSONDecodeError):
        return None
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (TimeoutError, OSError):
            pass


async def _spawn_daemon() -> None:
    """Spawn the daemon as a detached background process, then poll for ready.

    Runs ``hafiz.core.daemon`` as a module (it has a ``__main__`` entry) under
    the same interpreter — sharing our venv exactly — with ``start_new_session``
    so it survives our exit. No double-fork inside a Typer callback (that path
    interacts badly with Click's exit handling); a clean detached subprocess is
    simpler and reliable.
    """
    import subprocess

    try:
        # Plain Popen, NOT asyncio.create_subprocess_exec: asyncio ties the
        # child to the event loop's child-watcher, which can reap the daemon
        # when our loop closes. Popen + start_new_session fully detaches it.
        subprocess.Popen(
            [sys.executable, "-m", "hafiz.core.daemon"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as e:
        logger.debug("daemon spawn failed: %s", e)
        return

    # Poll for readiness — the daemon warms the model before binding.
    deadline = _SPAWN_WARMUP_TIMEOUT
    waited = 0.0
    step = 0.2
    while waited < deadline:
        await asyncio.sleep(step)
        waited += step
        resp = await _send_one({"op": "ping"}, timeout=2.0)
        if resp and resp.get("pong"):
            return


async def _kill_stale_daemon() -> None:
    """Best-effort: remove the socket so a respawn binds cleanly.

    A version-skewed daemon will exit when its socket disappears on the next
    idle tick; unlinking the socket also forces fresh connections to spawn a
    new one. We don't SIGKILL by PID in v1 — the socket-unlink + respawn path
    is enough and avoids tracking pids.
    """
    sock = socket_path()
    try:
        sock.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public: request with auto-spawn + version handshake + fallback
# ---------------------------------------------------------------------------


async def request(req: dict) -> dict | None:
    """Send ``req`` to the daemon, spawning/respawning as needed.

    Returns the daemon's response dict, or ``None`` if the daemon could not
    serve it (caller should fall back to direct execution). Never raises for
    daemon-side problems.
    """
    if _daemon_disabled():
        return None

    resp = await _send_one(req, timeout=_REQUEST_TIMEOUT)
    if resp is None:
        # No live daemon. A leftover socket file from a crashed daemon would
        # make a fresh daemon fail to bind, so clear it before spawning.
        if socket_path().exists():
            await _kill_stale_daemon()
        await _spawn_daemon()
        resp = await _send_one(req, timeout=_REQUEST_TIMEOUT)
        if resp is None:
            return None

    if resp.get("error") == "version mismatch":
        # Stale daemon from a prior hafiz version — replace it.
        await _kill_stale_daemon()
        await _spawn_daemon()
        resp = await _send_one(req, timeout=_REQUEST_TIMEOUT)
        if resp is None:
            return None

    return resp


# ---------------------------------------------------------------------------
# Typed helpers — daemon-first with direct in-process fallback
# ---------------------------------------------------------------------------


async def context(query: str, **kwargs) -> dict:
    """Return a context bundle dict (daemon-first, direct fallback)."""
    resp = await request({"op": "context", "query": query, **kwargs})
    if resp and resp.get("ok"):
        return resp["bundle"]
    from hafiz.core.context import build_context

    bundle = await build_context(query, **kwargs)
    return bundle.to_dict()


async def query_recall(query: str, **kwargs) -> list[dict]:
    """Return annotation recall results (daemon-first, direct fallback)."""
    resp = await request({"op": "query_recall", "query": query, **kwargs})
    if resp and resp.get("ok"):
        return resp["results"]
    from hafiz.core.annotations import search_annotations
    from hafiz.core.daemon import _annotation_to_dict

    results = await search_annotations(query, **kwargs)
    return [_annotation_to_dict(r) for r in results]


async def observe(content: str, **kwargs) -> dict:
    """Store an annotation (daemon-first, direct fallback)."""
    resp = await request({"op": "observe", "content": content, **kwargs})
    if resp and resp.get("ok"):
        return resp["annotation"]
    from hafiz.core.annotations import store_annotation
    from hafiz.core.daemon import _annotation_to_dict

    ann = await store_annotation(content, **kwargs)
    return _annotation_to_dict(ann)


async def capture(text: str, **kwargs) -> dict:
    """Capture a transcript (daemon-first, direct fallback)."""
    resp = await request({"op": "capture", "text": text, **kwargs})
    if resp and resp.get("ok"):
        return resp
    from hafiz.core.capture import store_transcript

    summary = await store_transcript(text, **kwargs)
    return {
        "ok": True,
        "communication_id": summary.communication_id,
        "title": summary.title,
        "turn_count": summary.turn_count,
        "messages_embedded": summary.messages_embedded,
    }
