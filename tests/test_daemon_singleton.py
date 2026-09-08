"""One daemon per socket, and losing the race must be cheap.

The state this pins was live on the author's machine: **four** hafiz daemons,
4.1 GB resident between them, all started within one second, all bound to the
same socket path — so three were unreachable, and each had paid a ~1 GB
embedding-model load to become so. Nothing serialized startup, and three
separate places assumed sole ownership:

  A. the client's spawn path has no lock, so N concurrent callers spawn N
     daemons;
  B. `serve` unlinked the socket before binding, stealing the path from a
     working daemon;
  C. `serve`'s shutdown unlinked *the path*, so a loser's idle timeout deleted
     the **winner's** socket, leaving a live daemon nothing could reach;
  D. `serve stop` unlinked the socket and trusted the daemon to notice. Nothing
     watched for that, so it left a live 1 GB daemon unreachable for up to 30
     minutes and reported success.

One lifetime `flock` fixes A–C: losing costs milliseconds instead of a model
load, and the two unlinks become correct because only the owner reaches them.
The pid it records fixes D.

`flock` is per open-file-description, not per process, so a second `os.open` +
`flock` in this same process blocks exactly as another process would — which
is what makes these tests possible without spawning daemons.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core import daemon

runner = CliRunner()


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """Point the socket (and therefore the lock) inside a tmp dir."""
    sock = tmp_path / "daemon.sock"
    monkeypatch.setenv("HAFIZ_DAEMON_SOCKET", str(sock))
    return sock


# ─── the lock itself ────────────────────────────────────────────────────


class TestSingletonLock:
    def test_the_first_caller_gets_it(self, runtime):
        fd = daemon.acquire_singleton()
        assert fd is not None
        daemon.release_singleton(fd)

    def test_the_second_caller_is_refused(self, runtime):
        first = daemon.acquire_singleton()
        try:
            assert daemon.acquire_singleton() is None
        finally:
            daemon.release_singleton(first)

    def test_releasing_hands_it_on(self, runtime):
        first = daemon.acquire_singleton()
        daemon.release_singleton(first)
        second = daemon.acquire_singleton()
        assert second is not None
        daemon.release_singleton(second)

    def test_it_records_the_holder_pid(self, runtime):
        fd = daemon.acquire_singleton()
        try:
            assert daemon._read_lock_pid() == os.getpid()
        finally:
            daemon.release_singleton(fd)

    def test_the_lock_lives_beside_the_socket(self, runtime):
        assert daemon.lock_path().parent == runtime.parent
        assert daemon.lock_path().name.startswith(runtime.name)

    def test_the_lock_file_is_owner_only(self, runtime):
        fd = daemon.acquire_singleton()
        try:
            assert (daemon.lock_path().stat().st_mode & 0o777) == 0o600
        finally:
            daemon.release_singleton(fd)

    def test_a_killed_holder_releases_it(self, runtime, tmp_path):
        """Why `flock` and not a pidfile: there is no stale-lock recovery path.

        A SIGKILLed daemon cannot clean up after itself, and a pidfile would
        strand every future daemon behind a pid that no longer exists.
        """
        script = (
            "import fcntl,os,sys,time\n"
            f"fd=os.open({str(daemon.lock_path())!r}, os.O_CREAT|os.O_RDWR, 0o600)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "os.write(fd, str(os.getpid()).encode())\n"
            "sys.stdout.write('held\\n'); sys.stdout.flush()\n"
            "time.sleep(30)\n"
        )
        holder = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "held"
            assert daemon.acquire_singleton() is None  # genuinely cross-process
        finally:
            holder.kill()
            holder.wait(timeout=10)
        fd = daemon.acquire_singleton()
        assert fd is not None, "a killed holder's lock must not strand the next daemon"
        daemon.release_singleton(fd)

    def test_release_clears_the_recorded_pid(self, runtime):
        """A clean shutdown must not leave a pid `serve stop` would signal.

        The lock file outlives the daemon by design — deleting it is what
        reopens the orphaned-lock hole — so its *contents* have to be the
        thing that goes.
        """
        fd = daemon.acquire_singleton()
        assert daemon._read_lock_pid() == os.getpid()
        daemon.release_singleton(fd)
        assert daemon._read_lock_pid() is None

    def test_an_unreadable_pid_is_reported_as_none(self, runtime):
        daemon.lock_path().write_text("not a pid")
        assert daemon._read_lock_pid() is None

    def test_a_missing_lock_file_is_reported_as_none(self, runtime):
        assert daemon._read_lock_pid() is None


# ─── serve() under contention ───────────────────────────────────────────


class TestServeUnderContention:
    @pytest.fixture
    def model_spy(self, monkeypatch):
        """Records whether the embedding model was loaded."""
        from hafiz.core import embeddings

        calls = {"n": 0}

        def loader():
            calls["n"] += 1
            return object()

        monkeypatch.setattr(embeddings, "get_embed_model", loader)
        return calls

    def test_a_loser_returns_false(self, runtime):
        held = daemon.acquire_singleton()
        try:
            assert asyncio.run(daemon.serve(idle_timeout=1)) is False
        finally:
            daemon.release_singleton(held)

    def test_a_loser_never_loads_the_model(self, runtime, model_spy):
        """The whole point. Each of the four wasted daemons paid ~1 GB first."""
        held = daemon.acquire_singleton()
        try:
            asyncio.run(daemon.serve(idle_timeout=1))
        finally:
            daemon.release_singleton(held)
        assert model_spy["n"] == 0

    def test_a_loser_does_not_bind_a_socket(self, runtime, model_spy):
        held = daemon.acquire_singleton()
        try:
            asyncio.run(daemon.serve(idle_timeout=1))
        finally:
            daemon.release_singleton(held)
        assert not runtime.exists()

    def test_a_live_socket_wins_even_when_the_lock_is_free(self, runtime, model_spy, monkeypatch):
        """The orphaned-lock case: `flock` guards an inode, not a name.

        Delete the lock file while a daemon holds it and the holder is locking
        an orphan, so a newcomer creates a fresh file and locks it happily —
        two daemons, one socket. Found by deleting the lock during manual
        verification of this change. A live socket is the ground truth, so
        `serve` asks it even after winning the lock.
        """

        async def answering(_sock):
            return True

        monkeypatch.setattr(daemon, "_another_daemon_answers", answering)
        assert asyncio.run(daemon.serve(idle_timeout=1)) is False
        assert model_spy["n"] == 0, "must bail before the model load, like any other loser"

    def test_bailing_on_a_live_socket_does_not_delete_it(self, runtime, model_spy, monkeypatch):
        """Bug C from the probe side: exiting must not take the socket with it.

        The lock-loser case is covered below; this is the same hazard reached
        through the other early return, where we *hold* the lock but find
        someone already serving.
        """

        async def answering(_sock):
            return True

        runtime.write_text("")  # the live daemon's socket
        monkeypatch.setattr(daemon, "_another_daemon_answers", answering)
        asyncio.run(daemon.serve(idle_timeout=1))
        assert runtime.exists(), "bailing out deleted the serving daemon's socket"

    def test_the_probe_ignores_a_stale_socket_file(self, runtime):
        """A crashed daemon's leftover socket must not block a fresh start."""
        runtime.write_text("")  # a file at the path, but nothing listening
        assert asyncio.run(daemon._another_daemon_answers(runtime)) is False

    def test_the_probe_is_false_when_there_is_no_socket(self, runtime):
        assert asyncio.run(daemon._another_daemon_answers(runtime)) is False

    def test_a_loser_does_not_unlink_the_owners_socket(self, runtime, model_spy):
        """Bug C, from the other side: the loser must not touch the path.

        Both the pre-bind cleanup and the shutdown unlink now sit inside the
        lock, so a process that never holds it cannot delete a live socket.
        """
        runtime.write_text("")  # stand-in for the owner's bound socket
        held = daemon.acquire_singleton()
        try:
            asyncio.run(daemon.serve(idle_timeout=1))
        finally:
            daemon.release_singleton(held)
        assert runtime.exists(), "the winner's socket was deleted by a loser"


# ─── ping carries the pid ───────────────────────────────────────────────


def test_ping_reports_the_pid():
    """With four daemons on one path there was no way to tell which answered."""
    server = daemon._Server(idle_timeout=1, _embed_lock=asyncio.Lock())
    resp = asyncio.run(server.dispatch({"op": "ping"}))
    assert resp["pong"] is True
    assert resp["pid"] == os.getpid()
    assert resp["version"] == daemon.PROTOCOL_VERSION


# ─── the CLI surfaces ───────────────────────────────────────────────────


class TestServeCommand:
    def test_foreground_serve_reports_already_running(self, runtime):
        held = daemon.acquire_singleton()
        try:
            result = runner.invoke(app, ["serve", "--json"])
        finally:
            daemon.release_singleton(held)
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["already_running"] is True
        assert payload["started"] is False

    def test_it_names_the_holder(self, runtime):
        held = daemon.acquire_singleton()
        try:
            result = runner.invoke(app, ["serve", "--json"])
        finally:
            daemon.release_singleton(held)
        assert json.loads(result.output)["pid"] == os.getpid()


class TestStatusCommand:
    def test_status_surfaces_the_serving_pid(self, runtime, monkeypatch):
        async def live(req, *, timeout):
            return {"ok": True, "pong": True, "pid": 4242, "version": "9.9.9"}

        monkeypatch.setattr("hafiz.core.daemon_client._send_one", live)
        payload = json.loads(runner.invoke(app, ["serve", "status", "--json"]).output)
        assert payload["running"] is True
        assert payload["pid"] == 4242

    def test_status_pid_is_null_when_nothing_answers(self, runtime, monkeypatch):
        async def dead(req, *, timeout):
            return None

        monkeypatch.setattr("hafiz.core.daemon_client._send_one", dead)
        payload = json.loads(runner.invoke(app, ["serve", "status", "--json"]).output)
        assert payload["running"] is False
        assert payload["pid"] is None


class TestStopCommand:
    def test_stop_with_no_daemon_says_so(self, runtime):
        result = runner.invoke(app, ["serve", "stop", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["was_running"] is False
        assert payload["signalled"] is False

    def test_stop_signals_the_reported_pid(self, runtime, monkeypatch):
        """The pid comes from the daemon's own ping, not from a guess."""
        from hafiz.commands import serve as serve_cmd

        pings = iter([{"ok": True, "pong": True, "pid": 4242}, None])
        killed = {}

        async def fake_send_one(req, *, timeout):
            return next(pings, None)

        monkeypatch.setattr("hafiz.core.daemon_client._send_one", fake_send_one)
        monkeypatch.setattr(
            serve_cmd.os, "kill", lambda pid, sig: killed.update(pid=pid, sig=sig), raising=False
        )
        result = runner.invoke(app, ["serve", "stop", "--json"])
        payload = json.loads(result.output)
        assert killed == {"pid": 4242, "sig": 15}
        assert payload["signalled"] is True
        assert payload["stopped"] is True

    def test_a_daemon_that_ignores_sigterm_falls_back_to_the_socket(self, runtime, monkeypatch):
        """Never report a stop that did not happen — that was the original bug."""
        from hafiz.commands import serve as serve_cmd

        runtime.write_text("")

        async def always_live(req, *, timeout):
            return {"ok": True, "pong": True, "pid": 4242}

        monkeypatch.setattr("hafiz.core.daemon_client._send_one", always_live)
        monkeypatch.setattr(serve_cmd.os, "kill", lambda pid, sig: None, raising=False)
        monkeypatch.setattr(serve_cmd.time, "sleep", lambda _s: None, raising=False)

        payload = json.loads(runner.invoke(app, ["serve", "stop", "--json"]).output)
        assert payload["signalled"] is True
        assert payload["stopped"] is False
        assert payload["socket_removed"] is True
        assert not runtime.exists()

    def test_a_wedged_daemon_pid_comes_from_the_lock(self, runtime, monkeypatch):
        """A daemon too stuck to answer a ping can still be identified."""
        from hafiz.commands import serve as serve_cmd

        async def no_answer(req, *, timeout):
            return None

        monkeypatch.setattr("hafiz.core.daemon_client._send_one", no_answer)
        monkeypatch.setattr(serve_cmd.os, "kill", lambda pid, sig: None, raising=False)
        held = daemon.acquire_singleton()
        try:
            payload = json.loads(runner.invoke(app, ["serve", "stop", "--json"]).output)
        finally:
            daemon.release_singleton(held)
        assert payload["pid"] == os.getpid()
