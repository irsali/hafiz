"""hafiz serve — on-demand warm daemon over a Unix domain socket.

Every plain ``hafiz`` CLI call re-pays ~1.3–1.7s of cold start: process
launch + fastembed model load + DB connect, before any vector search runs.
A warm daemon loads the embedding model and the (pooled) DB engine **once**
and answers many requests, dropping per-call cost to the actual vector op
plus cheap local IPC.

Design (see workitems/active/hafiz-serve-daemon.md):

  * **Transport: Unix domain socket, 0600**, under
    ``$XDG_RUNTIME_DIR/hafiz/daemon.sock`` (falls back to a user-scoped
    temp dir). Never TCP — a sovereign personal store must not open a
    network port. Filesystem permissions gate access, so no auth token.
  * **Protocol: newline-framed JSON.** One request object per line, one
    response object per line. Every message carries ``version`` (the hafiz
    version); the client respawns the daemon on a mismatch.
  * **Ops (read + write):** ``ping``, ``context``, ``query_recall``,
    ``observe``, ``capture`` — each dispatches to the same core function
    the CLI uses, so shapes never drift.
  * **Idle auto-shutdown:** the daemon exits after ``idle_timeout`` seconds
    with no requests, so it never lingers forever.

The client (``hafiz.core.daemon_client``) auto-spawns this daemon on demand
and falls back to direct in-process execution on any error, so behavior is
never worse than the plain CLI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from hafiz import __version__

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = __version__
DEFAULT_IDLE_TIMEOUT = 1800  # 30 minutes
_RECV_LIMIT = 16 * 1024 * 1024  # 16 MiB per line — generous for capture payloads


# ---------------------------------------------------------------------------
# Socket location
# ---------------------------------------------------------------------------


def runtime_dir() -> Path:
    """User-scoped runtime dir for the socket.

    Resolution order: ``$XDG_RUNTIME_DIR`` (Linux), then ``$TMPDIR`` (set
    per-user on macOS, e.g. ``/var/folders/.../T/`` — preferred over the
    world-shared ``/tmp``), then ``/tmp/hafiz-<uid>`` as a last resort
    (minimal containers). Always created 0700 so only the owner can reach
    the socket inside it.
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    tmpdir = os.environ.get("TMPDIR")
    if xdg and Path(xdg).is_dir():
        d = Path(xdg) / "hafiz"
    elif tmpdir and Path(tmpdir).is_dir():
        d = Path(tmpdir) / f"hafiz-{os.getuid()}"
    else:
        d = Path(f"/tmp/hafiz-{os.getuid()}")  # noqa: S108 — uid-scoped, 0700 below
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Re-assert perms in case the dir pre-existed with looser bits.
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def socket_path() -> Path:
    """Absolute path to the daemon's Unix socket."""
    override = os.environ.get("HAFIZ_DAEMON_SOCKET")
    if override:
        return Path(override)
    return runtime_dir() / "daemon.sock"


# ---------------------------------------------------------------------------
# Request dispatch — each op maps to the same core fn the CLI uses
# ---------------------------------------------------------------------------


@dataclass
class _Server:
    idle_timeout: float
    _embed_lock: asyncio.Lock
    _idle_handle: asyncio.TimerHandle | None = None
    _loop: asyncio.AbstractEventLoop | None = None
    _server: asyncio.AbstractServer | None = None

    async def dispatch(self, req: dict) -> dict:
        """Route one request to its handler. Returns a JSON-serializable dict.

        Errors are returned as ``{"ok": False, "error": ...}`` rather than
        raised, so a single bad request never tears the daemon down.
        """
        op = req.get("op")
        try:
            if op == "ping":
                return {"ok": True, "version": PROTOCOL_VERSION, "pong": True}
            if op == "context":
                return await self._op_context(req)
            if op == "query_recall":
                return await self._op_query_recall(req)
            if op == "observe":
                return await self._op_observe(req)
            if op == "capture":
                return await self._op_capture(req)
            return {"ok": False, "error": f"unknown op: {op!r}"}
        except Exception as e:  # noqa: BLE001 — daemon must survive any handler error
            logger.exception("daemon op %r failed", op)
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    async def _op_context(self, req: dict) -> dict:
        from hafiz.core.context import build_context

        async with self._embed_lock:
            bundle = await build_context(
                req["query"],
                project=req.get("project"),
                limit_chunks=req.get("limit_chunks", 5),
                limit_annotations=req.get("limit_annotations", 5),
                include_domains=req.get("include_domains"),
                exclude_domains=req.get("exclude_domains"),
                min_score=req.get("min_score"),
            )
        return {"ok": True, "bundle": bundle.to_dict()}

    async def _op_query_recall(self, req: dict) -> dict:
        from hafiz.core.annotations import search_annotations

        async with self._embed_lock:
            results = await search_annotations(
                req["query"],
                limit=req.get("limit", 8),
                project=req.get("project"),
                kind=req.get("kind"),
                source=req.get("source"),
                tags=req.get("tags"),
                rerank=req.get("rerank"),  # None → honor config default
                min_score=req.get("min_score"),
            )
        return {"ok": True, "results": [_annotation_to_dict(r) for r in results]}

    async def _op_observe(self, req: dict) -> dict:
        from hafiz.core.annotations import store_annotation

        async with self._embed_lock:
            ann = await store_annotation(
                req["content"],
                kind=req.get("kind", "note"),
                source=req.get("source"),
                project=req.get("project"),
                tags=req.get("tags"),
                confidence=req.get("confidence", 1.0),
                session_id=req.get("session_id"),
                task=req.get("task"),
                supersedes_id=req.get("supersedes_id"),
                derived_from=req.get("derived_from"),
            )
        return {"ok": True, "annotation": _annotation_to_dict(ann)}

    async def _op_capture(self, req: dict) -> dict:
        from hafiz.core.capture import store_transcript

        async with self._embed_lock:
            summary = await store_transcript(
                req["text"],
                title=req.get("title"),
                project=req.get("project"),
                source=req.get("source"),
                tags=req.get("tags"),
                session_id=req.get("session_id"),
                task=req.get("task"),
            )
        return {
            "ok": True,
            "communication_id": summary.communication_id,
            "title": summary.title,
            "turn_count": summary.turn_count,
            "messages_embedded": summary.messages_embedded,
        }

    # -- connection handling -------------------------------------------------

    def _bump_idle(self) -> None:
        """Reset the idle-shutdown timer; called on every request."""
        if self._idle_handle is not None:
            self._idle_handle.cancel()
        if self._loop is not None and self.idle_timeout > 0:
            self._idle_handle = self._loop.call_later(self.idle_timeout, self._shutdown)

    def _shutdown(self) -> None:
        logger.info("idle for %.0fs — shutting down", self.idle_timeout)
        if self._server is not None:
            self._server.close()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve one connection: read newline-framed requests, reply per line."""
        try:
            while not reader.at_eof():
                try:
                    line = await reader.readline()
                except (asyncio.LimitOverrunError, ValueError):
                    # Oversized frame — refuse this connection cleanly.
                    self._write(writer, {"ok": False, "error": "request too large"})
                    break
                if not line:
                    break
                self._bump_idle()
                try:
                    req = json.loads(line)
                except json.JSONDecodeError:
                    self._write(writer, {"ok": False, "error": "invalid json"})
                    continue
                if req.get("version") and req["version"] != PROTOCOL_VERSION:
                    self._write(
                        writer,
                        {
                            "ok": False,
                            "error": "version mismatch",
                            "version": PROTOCOL_VERSION,
                        },
                    )
                    continue
                resp = await self.dispatch(req)
                resp.setdefault("version", PROTOCOL_VERSION)
                self._write(writer, resp)
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    @staticmethod
    def _write(writer: asyncio.StreamWriter, obj: dict) -> None:
        writer.write((json.dumps(obj, default=str) + "\n").encode("utf-8"))


def _annotation_to_dict(ann) -> dict:
    """Serialize an annotation row / search result to the recall shape.

    Tolerant of both ORM ``Annotation`` rows (from ``store_annotation``) and
    search-result objects (from ``search_annotations``), pulling whatever
    attributes are present.
    """
    out: dict = {}
    for key in (
        "id",
        "content",
        "kind",
        "source",
        "project",
        "tags",
        "confidence",
        "score",
        "rerank_score",
        "age_days",
        "stale",
        "valid_from",
        "valid_until",
    ):
        if hasattr(ann, key):
            out[key] = getattr(ann, key)
    if "id" in out:
        out["id"] = str(out["id"])
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def serve(*, idle_timeout: float = DEFAULT_IDLE_TIMEOUT) -> None:
    """Run the daemon until idle-shutdown or the socket server is closed.

    Warms the embedding model up front so the first real request is fast,
    binds the Unix socket with 0600 perms, and serves until idle.
    """
    from hafiz.core.embeddings import get_embed_model

    sock = socket_path()
    # Clean up a stale socket from a previous (crashed) daemon.
    if sock.exists():
        try:
            sock.unlink()
        except OSError:
            pass

    # Warm the model before binding so the first client request doesn't pay
    # the cold-start cost we built this to avoid.
    await asyncio.to_thread(get_embed_model)

    # Warm the reranker too when enabled — otherwise the first recall pays the
    # cross-encoder's cold load. Skip silently if it can't load (recall then
    # falls back to vector order per the reranker's own contract).
    from hafiz.core.reranker import rerank_enabled, warm_reranker

    if rerank_enabled():
        try:
            await warm_reranker()
        except Exception:  # noqa: BLE001 — degrade to vector-only, never block startup
            logger.warning("reranker warm-up failed; recall will use vector order")

    server = _Server(idle_timeout=idle_timeout, _embed_lock=asyncio.Lock())
    server._loop = asyncio.get_running_loop()

    aio_server = await asyncio.start_unix_server(server.handle, path=str(sock), limit=_RECV_LIMIT)
    server._server = aio_server

    # Lock the socket to owner-only (0600). start_unix_server honors umask,
    # so set the bits explicitly rather than trusting the ambient umask.
    os.chmod(sock, stat.S_IRUSR | stat.S_IWUSR)

    server._bump_idle()
    logger.info("hafiz daemon listening on %s (v%s)", sock, PROTOCOL_VERSION)

    try:
        async with aio_server:
            await aio_server.wait_closed()
    finally:
        if sock.exists():
            try:
                sock.unlink()
            except OSError:
                pass
        from hafiz.core.database import close_engine

        await close_engine()


def main() -> None:
    """Module entry point: ``python -m hafiz.core.daemon``.

    The client spawns the daemon this way (detached, shared interpreter).
    Reads the idle timeout from ``HAFIZ_DAEMON_IDLE`` if set.
    """
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)
    raw = os.environ.get("HAFIZ_DAEMON_IDLE", "").strip()
    try:
        idle = float(raw) if raw else DEFAULT_IDLE_TIMEOUT
    except ValueError:
        idle = DEFAULT_IDLE_TIMEOUT
    try:
        asyncio.run(serve(idle_timeout=idle))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
