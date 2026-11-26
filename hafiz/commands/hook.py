"""hafiz hook — harness entry points for every-turn memory recall.

Not for typing by hand: ``hafiz agent install claude-code --hooks`` wires these
into the agent's settings. Output is markdown for the model, not a ``--json``
shape, and the contract is the opposite of the rest of the CLI — these never
fail: any error prints nothing and exits 0.
"""

from __future__ import annotations

import os
import sys
import threading

# Slack beyond the in-process timeout before the watchdog gives up on a wedged
# native call (an ONNX run that ignores task cancellation).
_WATCHDOG_GRACE = 3.0


def _read_stdin() -> str:
    try:
        return sys.stdin.read() if not sys.stdin.isatty() else ""
    except (OSError, ValueError):
        return ""


def run_hook_cmd(mode: str) -> None:
    """Run one hook mode: harness JSON on stdin, markdown on stdout, always exit 0."""
    from hafiz.core.recall_hook import _settings, run_hook

    watchdog = None
    try:
        # Cancellation cannot interrupt a thread stuck inside a native call, and
        # `asyncio.run` waits for such threads on shutdown. Past the time budget
        # nothing has been printed yet, so exiting silently is the correct failure.
        watchdog = threading.Timer(
            _settings().timeout_seconds + _WATCHDOG_GRACE, lambda: os._exit(0)
        )
        watchdog.daemon = True
        watchdog.start()
    except Exception:  # noqa: BLE001
        pass
    try:
        body = run_hook(mode, _read_stdin())
        if body:
            sys.stdout.write(body)
            sys.stdout.flush()
    except Exception:  # noqa: BLE001 — never fail a turn
        pass
    finally:
        if watchdog is not None:
            watchdog.cancel()
