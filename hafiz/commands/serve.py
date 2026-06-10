"""hafiz serve — manage the on-demand warm daemon.

Thin presentation layer over :mod:`hafiz.core.daemon` and
:mod:`hafiz.core.daemon_client`: ``serve`` runs the daemon, ``serve status``
pings it, ``serve stop`` shuts it down. The daemon normally auto-spawns on
demand from the client, so running ``serve`` by hand is optional (useful for
debugging or pre-warming).
"""

from __future__ import annotations

import asyncio
import json

from rich.console import Console
from rich.panel import Panel

console = Console()


def run_serve(*, idle_timeout: float, detach: bool, output_json: bool) -> None:
    """Start the daemon. Foreground by default; ``--detach`` backgrounds it."""
    from hafiz.core.daemon import DEFAULT_IDLE_TIMEOUT, serve, socket_path

    timeout = idle_timeout if idle_timeout >= 0 else DEFAULT_IDLE_TIMEOUT

    if detach:
        _spawn_detached(timeout)
        if output_json:
            console.print_json(
                json.dumps({"ok": True, "detached": True, "socket": str(socket_path())})
            )
        else:
            console.print(f"[green]daemon starting[/green] (detached) on {socket_path()}")
        return

    if not output_json:
        console.print(
            Panel(
                f"[bold green]hafiz daemon[/bold green]\n\n"
                f"  Socket:   {socket_path()}\n"
                f"  Idle:     {timeout:.0f}s\n\n"
                f"  [dim]Warming embedding model… serving until idle. Ctrl-C to stop.[/dim]",
                border_style="cyan",
            )
        )

    try:
        asyncio.run(serve(idle_timeout=timeout))
    except KeyboardInterrupt:
        if not output_json:
            console.print("\n[dim]daemon stopped.[/dim]")


def _spawn_detached(idle_timeout: float) -> None:
    """Launch the daemon as a clean detached subprocess (no double-fork).

    Mirrors the client's auto-spawn: run ``hafiz.core.daemon`` under the same
    interpreter with ``start_new_session`` so it outlives this command.
    """
    import os
    import subprocess
    import sys

    env = {**os.environ, "HAFIZ_DAEMON_IDLE": str(idle_timeout)}
    subprocess.Popen(
        [sys.executable, "-m", "hafiz.core.daemon"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )


def run_status(*, output_json: bool) -> None:
    """Ping the daemon and report whether it's live + its version."""
    from hafiz.core.daemon import socket_path
    from hafiz.core.daemon_client import _send_one

    async def _ping():
        return await _send_one({"op": "ping"}, timeout=2.0)

    resp = asyncio.run(_ping())
    live = bool(resp and resp.get("pong"))
    sock = socket_path()

    if output_json:
        console.print_json(
            json.dumps(
                {
                    "ok": True,
                    "running": live,
                    "socket": str(sock),
                    "version": resp.get("version") if resp else None,
                }
            )
        )
        return

    if live:
        console.print(
            Panel(
                f"[bold green]running[/bold green]\n\n"
                f"  Socket:   {sock}\n"
                f"  Version:  {resp.get('version')}",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[yellow]not running[/yellow]\n\n"
                f"  Socket:   {sock}\n"
                f"  [dim]Auto-spawns on the next client request.[/dim]",
                border_style="yellow",
            )
        )


def run_stop(*, output_json: bool) -> None:
    """Shut the daemon down by removing its socket; it exits on next idle tick.

    v1 doesn't track a pid, so ``stop`` unlinks the socket (forcing new
    clients to spawn a fresh daemon) and the running daemon self-exits when
    its idle timer fires or its socket vanishes. This is best-effort.
    """
    from hafiz.core.daemon import socket_path
    from hafiz.core.daemon_client import _send_one

    sock = socket_path()
    was_live = bool(asyncio.run(_send_one({"op": "ping"}, timeout=2.0)))
    removed = False
    try:
        sock.unlink()
        removed = True
    except OSError:
        pass

    if output_json:
        console.print_json(
            json.dumps({"ok": True, "was_running": was_live, "socket_removed": removed})
        )
        return

    if was_live:
        console.print("[green]daemon stopping[/green] (socket removed).")
    else:
        console.print("[dim]no daemon running.[/dim]")
