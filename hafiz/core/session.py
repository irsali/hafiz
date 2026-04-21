"""Per-TTY session state for hafiz.

A "session" is a named thread of work the user / agent wants to tag
subsequent captures with. State is stored in a small JSON file keyed by
the controlling TTY (``/dev/pts/N``), so two terminals on the same
machine don't clobber each other's sessions.

No TTY (piped, CI, non-interactive) = no session. The ``observe`` /
``note`` / ``capture`` commands still work — they just don't auto-tag.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

SESSION_DIR = Path.home() / ".cache" / "hafiz"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _tty_key() -> str | None:
    """Stable filename-safe identifier for the current TTY, or None if no TTY.

    ``/dev/pts/3`` → ``pts-3``. Returns None when stdin is not a tty
    (piped, redirected, running under CI), in which case sessions are
    not applicable for this invocation.
    """
    try:
        tty = os.ttyname(0)
    except OSError:
        return None
    return tty.lstrip("/").replace("/", "-")


def _session_path() -> Path | None:
    key = _tty_key()
    if not key:
        return None
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_DIR / f"session-{key}.json"


def make_session_id(name: str) -> str:
    """Build a human-readable + unique session id from a display name.

    ``"Phase 3 migration"`` → ``"phase-3-migration-a3f19c"``. The 6-char
    hex suffix avoids collisions when the same name is reused.
    """
    base = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    base = base[:40] if base else "session"
    return f"{base}-{secrets.token_hex(3)}"


def current_session() -> dict | None:
    """Return the active session dict for this TTY, or None if none."""
    path = _session_path()
    if not path or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def start_session(
    name: str,
    *,
    task: str | None = None,
    project: str | None = None,
) -> dict:
    """Start (or replace) the session for this TTY. Returns the session dict.

    Raises :class:`RuntimeError` if there is no controlling terminal.
    """
    path = _session_path()
    if not path:
        raise RuntimeError("No controlling terminal — `hafiz session` requires a TTY.")

    data = {
        "session_id": make_session_id(name),
        "name": name,
        "task": task,
        "project": project,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "tty": _tty_key(),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def end_session() -> dict | None:
    """Clear the session state for this TTY. Returns the ended session, or None."""
    path = _session_path()
    if not path or not path.exists():
        return None
    data = current_session()
    try:
        path.unlink()
    except OSError:
        pass
    return data


def resolve_session_tag(
    *,
    session_override: str | None,
    task_override: str | None,
) -> tuple[str | None, str | None]:
    """Resolve (session_id, task) for an outgoing write.

    Explicit ``--session`` / ``--task`` flags win over any active session;
    otherwise inherit from :func:`current_session`. Returns (None, None)
    when there's no active session and no overrides.
    """
    active = current_session() or {}
    session_id = session_override or active.get("session_id")
    task = task_override or active.get("task")
    return session_id, task
