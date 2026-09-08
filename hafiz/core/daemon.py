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
import contextlib
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


def lock_path() -> Path:
    """Absolute path to the daemon's singleton lock, beside its socket."""
    return socket_path().with_name(socket_path().name + ".lock")


# ---------------------------------------------------------------------------
# Singleton lock
# ---------------------------------------------------------------------------
#
# Observed 2026-09-08: four daemons on one host, 4.1 GB resident between them,
# all started within a second, all bound to the same socket path — so three
# were unreachable and had each paid a ~1 GB model load to become so. Nothing
# serialized startup, and three separate places assumed sole ownership:
#
#   * the client's spawn path (fail to connect → unlink → spawn) has no lock,
#     so N concurrent callers spawn N daemons;
#   * ``serve`` unlinks the socket before binding, so a late daemon steals the
#     path from a working one;
#   * ``serve``'s shutdown unlinks *the path*, so a loser's idle timeout
#     deletes the **winner's** socket and leaves a live daemon nothing can
#     reach.
#
# One exclusive lock, held for the process lifetime, fixes all three: losers
# exit in milliseconds instead of after a model load, and the two unlinks
# become correct because only the owner can reach them.
#
# ``flock`` rather than a pidfile deliberately. The kernel releases it when the
# holder exits for any reason, including SIGKILL, so there is no stale-lock
# recovery path to get wrong — which is the objection the original "v1 doesn't
# track a pid" note was really making. The pid is written *inside* the lock for
# diagnosis, but ownership is the lock, never the file's contents.


def _read_lock_pid() -> int | None:
    """The pid recorded in the lock file, or None if unreadable/absent.

    Advisory only — a pid here does **not** mean a daemon is live (that is
    what the lock itself answers). Used for reporting and for ``serve stop``.
    """
    try:
        raw = lock_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(raw.splitlines()[0])
    except (ValueError, IndexError):
        return None


def acquire_singleton(*, _path: Path | None = None) -> int | None:
    """Take the daemon lock, or return None when another daemon holds it.

    Returns the open file descriptor on success; the caller must keep it open
    for as long as it serves, because closing it releases the lock. Records
    our pid in the file for diagnosis.
    """
    import fcntl

    path = _path or lock_path()
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.fsync(fd)
    except OSError:
        pass  # the lock is what matters; the pid is a convenience
    return fd


def release_singleton(fd: int | None) -> None:
    """Drop the daemon lock. Closing the descriptor is what releases it."""
    if fd is None:
        return
    try:
        os.ftruncate(fd, 0)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


async def _another_daemon_answers(sock: Path) -> bool:
    """True when something is already serving on ``sock``.

    Belt and braces for the one case the lock alone cannot cover. ``flock``
    guards an *inode*, not a name, so deleting the lock file while a daemon
    holds it leaves that daemon locking an orphan — and a newcomer then
    creates a fresh file at the same path and locks that quite happily. Two
    daemons, one socket, exactly the bug the lock exists to prevent. Found by
    deleting the lock file during verification of this very change, which is
    how we know it is reachable rather than theoretical.

    A live socket is the ground truth for "someone is already serving", so
    ask it directly. Costs one failed connect on the normal path, because
    there is no socket there to connect to.

    Deliberately does not go through :mod:`hafiz.core.daemon_client` — that
    module imports this one, and a lazy import to dodge the cycle would put
    the client's auto-spawn logic on the daemon's own startup path.
    """
    if not sock.exists():
        return False
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(sock)), timeout=1.0
        )
    except (TimeoutError, OSError):
        return False  # stale socket file from a crashed daemon
    try:
        writer.write((json.dumps({"op": "ping", "version": PROTOCOL_VERSION}) + "\n").encode())
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=1.0)
        return bool(line) and bool(json.loads(line).get("pong"))
    except (TimeoutError, OSError, json.JSONDecodeError):
        return False
    finally:
        writer.close()
        with contextlib.suppress(OSError, TimeoutError):
            await writer.wait_closed()


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
                # The pid makes "which daemon is answering?" answerable. With
                # four of them bound to one path and only one reachable, there
                # was no way to tell them apart from the outside.
                return {
                    "ok": True,
                    "version": PROTOCOL_VERSION,
                    "pong": True,
                    "pid": os.getpid(),
                }
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

        kwargs = _supported_kwargs(build_context, req.get("kwargs") or {})
        async with self._embed_lock:
            bundle = await build_context(req["query"], **kwargs)
        return {"ok": True, "bundle": bundle.to_wire()}

    async def _op_query_recall(self, req: dict) -> dict:
        """Recall, with every caller argument forwarded.

        Arguments are taken from ``req["kwargs"]`` wholesale rather than
        copied key by key. The previous hand-maintained list was a rot
        generator: a filter added to ``search_annotations`` and not mirrored
        here was silently ignored on the warm path — the caller got an
        unfiltered result set and no error. ``tags`` had already gone missing
        that way (caught in review, not by a test), and ``active_only`` was
        still missing, which would have made ``--include-superseded`` return
        nothing at all once the daemon was actually wired in.

        Unknown keys are dropped against the real signature, so a newer
        client talking to an older daemon degrades to "that filter was
        ignored" rather than ``TypeError`` — and :func:`_supported_kwargs`
        makes that visible in the response.
        """
        from hafiz.core.annotations import search_annotations
        from hafiz.core.wire import to_wire

        kwargs = _supported_kwargs(search_annotations, req.get("kwargs") or {})
        async with self._embed_lock:
            results = await search_annotations(req["query"], **kwargs)
        return {"ok": True, "results": [to_wire(r) for r in results]}

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


def configured_idle_timeout() -> float:
    """Seconds of inactivity before the daemon exits. ``0`` means never.

    ``HAFIZ_DAEMON_IDLE`` wins so a single detached invocation can override
    without editing config; otherwise ``[daemon] idle_timeout`` from
    ``hafiz.toml``. A machine whose daemon backs another process wants
    ``0``, so the ~0.9s model load isn't repaid after every quiet spell.
    """
    raw = os.environ.get("HAFIZ_DAEMON_IDLE", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass  # fall through to config rather than dying on a typo
    try:
        from hafiz.core.config import get_settings

        return float(get_settings().daemon.idle_timeout)
    except Exception:  # noqa: BLE001 — a config problem must not stop the daemon
        return DEFAULT_IDLE_TIMEOUT


def _supported_kwargs(fn, kwargs: dict) -> dict:
    """Keep only the keys ``fn`` actually accepts.

    The daemon and the client are separate processes and can be different
    versions of Hafiz for the window between an upgrade and the next idle
    shutdown. Passing a caller's kwargs straight through would make that
    window a hard ``TypeError`` on every request; filtering makes it a
    degraded-but-working call, which the version handshake then repairs.
    """
    import inspect

    accepted = set(inspect.signature(fn).parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}


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


async def serve(*, idle_timeout: float = DEFAULT_IDLE_TIMEOUT) -> bool:
    """Run the daemon until idle-shutdown or the socket server is closed.

    Takes the singleton lock, warms the embedding model, binds the Unix socket
    with 0600 perms, and serves until idle.

    Returns True if we served, False if another daemon already owns the
    socket — the caller reports that rather than treating it as an error.
    """
    from hafiz.core.embeddings import get_embed_model

    sock = socket_path()

    # Before anything expensive. A daemon that loses this race must cost
    # milliseconds, not the ~1 GB embedding-model load it used to pay before
    # discovering it was unreachable.
    lock_fd = acquire_singleton()
    if lock_fd is None:
        logger.info("another hafiz daemon owns %s (pid %s) — exiting", sock, _read_lock_pid())
        return False

    try:
        # The lock says no peer *started* after us; this says no peer is
        # serving right now. Both are needed — see _another_daemon_answers.
        if await _another_daemon_answers(sock):
            logger.info("a daemon is already serving %s — exiting", sock)
            return False

        # Safe to clear now: we hold the lock and nothing answered on the
        # socket, so a file here is a crashed daemon's leftover rather than a
        # live one whose path we would be stealing.
        if sock.exists():
            try:
                sock.unlink()
            except OSError:
                pass

        # Warm the model before binding so the first client request doesn't pay
        # the cold-start cost we built this to avoid.
        await asyncio.to_thread(get_embed_model)

        # Warm the reranker too when enabled — otherwise the first recall pays
        # the cross-encoder's cold load. Skip silently if it can't load (recall
        # then falls back to vector order per the reranker's own contract).
        from hafiz.core.reranker import rerank_enabled, warm_reranker

        if rerank_enabled():
            try:
                await warm_reranker()
            except Exception:  # noqa: BLE001 — degrade to vector-only, never block startup
                logger.warning("reranker warm-up failed; recall will use vector order")

        server = _Server(idle_timeout=idle_timeout, _embed_lock=asyncio.Lock())
        server._loop = asyncio.get_running_loop()

        aio_server = await asyncio.start_unix_server(
            server.handle, path=str(sock), limit=_RECV_LIMIT
        )
        server._server = aio_server

        # Lock the socket to owner-only (0600). start_unix_server honors umask,
        # so set the bits explicitly rather than trusting the ambient umask.
        os.chmod(sock, stat.S_IRUSR | stat.S_IWUSR)

        # SIGTERM must reach the same shutdown path as the idle timer, so
        # `serve stop` can end the process cleanly: close the server, let the
        # `finally` below unlink the socket and close the DB engine. Without
        # this, `stop` had nothing to ask for and fell back to unlinking the
        # socket, which left the daemon alive and unreachable until it idled.
        import signal

        with contextlib.suppress(NotImplementedError, ValueError):
            for sig in (signal.SIGTERM, signal.SIGINT):
                server._loop.add_signal_handler(sig, server._shutdown)

        server._bump_idle()
        logger.info(
            "hafiz daemon listening on %s (v%s, pid %d)", sock, PROTOCOL_VERSION, os.getpid()
        )

        try:
            async with aio_server:
                await aio_server.wait_closed()
        finally:
            # Correct only because we hold the singleton lock: the path is ours,
            # so unlinking it cannot delete a peer's live socket. It used to —
            # a daemon that lost the startup race would idle out and take the
            # winner's socket with it. Do not remove the lock and leave this.
            if sock.exists():
                try:
                    sock.unlink()
                except OSError:
                    pass
            from hafiz.core.database import close_engine

            await close_engine()
    finally:
        release_singleton(lock_fd)
    return True


def main() -> None:
    """Module entry point: ``python -m hafiz.core.daemon``.

    The client spawns the daemon this way (detached, shared interpreter).
    Reads the idle timeout from ``HAFIZ_DAEMON_IDLE`` if set.
    """
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO)
    idle = configured_idle_timeout()
    try:
        asyncio.run(serve(idle_timeout=idle))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
