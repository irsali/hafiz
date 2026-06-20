"""Per-TTY session cursor for hafiz.

A "session" is a named thread of work the user / agent wants to tag
subsequent captures with. Phase 2 splits responsibility:

* :mod:`hafiz.core.sessions` (plural) owns the persistent ``sessions``
  table — slug ↔ uuid mapping, started_at / ended_at, scope, agent.
* This module (:mod:`hafiz.core.session`, singular) keeps a small
  per-TTY JSON file pointing at *which* session this terminal is in,
  so two terminals on the same machine don't clobber each other's
  active sessions. The JSON file is a cursor, not a record.

JSON shape (current):

    {
      "session_uuid": "...uuid...",
      "session_id":   "phase-3-a3f19c",   # slug, kept for human display
      "name":         "Phase 3 migration",
      "task":         "import-claude-code",
      "project":      "hafiz",
      "started_at":   "2026-04-25T...",
      "tty":          "pts-3"
    }

Pre-Phase-2 cursors (only ``session_id`` slug, no uuid) are upgraded
in place on first read — a real DB row is created and the cursor is
rewritten. No data loss; users don't notice.

No TTY (piped, CI, non-interactive) = no session. The ``observe`` /
``note`` / ``capture`` commands still work — they just don't auto-tag.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import uuid
from datetime import UTC, datetime
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
    """Build a human-readable + unique session slug from a display name.

    ``"Phase 3 migration"`` → ``"phase-3-migration-a3f19c"``. The 6-char
    hex suffix avoids collisions when the same name is reused.
    """
    base = _SLUG_RE.sub("-", (name or "").strip().lower()).strip("-")
    base = base[:40] if base else "session"
    return f"{base}-{secrets.token_hex(3)}"


def _run_async(coro):
    """Run an async coroutine from sync code, even when an event loop
    is already running on another thread (rare: tests, unusual CLIs)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        return asyncio.run(coro)
    # If we *are* inside a running loop, schedule and wait.
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def _read_cursor(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def current_session() -> dict | None:
    """Return the active session dict for this TTY, or None if none.

    For pre-Phase-2 cursors (slug only, no uuid), this triggers an
    in-place upgrade: a DB row is created and the cursor file is
    rewritten in the new shape. Subsequent reads see a uuid.
    """
    path = _session_path()
    if not path or not path.exists():
        return None
    data = _read_cursor(path)
    if not data:
        return None

    # New-shape cursors return as-is.
    if "session_uuid" in data:
        return data

    # Legacy upgrade path: cursor predates Phase 2.
    slug = data.get("session_id")
    if not slug:
        return data
    try:
        upgraded = _run_async(_upgrade_legacy_cursor(slug, data, path))
    except Exception:
        # Don't fail reads — fall back to the raw legacy dict.
        return data
    return upgraded or data


async def _upgrade_legacy_cursor(slug: str, data: dict, path: Path) -> dict | None:
    """Create or look up a sessions row for this slug and rewrite the cursor."""
    from hafiz.core.database import close_engine
    from hafiz.core.sessions import create_session, get_session_by_slug

    try:
        existing = await get_session_by_slug(slug)
        if existing is not None:
            stored = existing
        else:
            stored = await create_session(
                slug=slug,
                name=data.get("name"),
                task=data.get("task"),
                scope_kind="project" if data.get("project") else None,
                scope_value=data.get("project"),
                tty=data.get("tty") or _tty_key(),
                started_at=_parse_iso(data.get("started_at")),
            )
        new_data = {
            "session_uuid": str(stored.id),
            "session_id": stored.slug,
            "name": stored.name,
            "task": stored.task,
            "project": stored.scope_value
            if stored.scope_kind == "project"
            else data.get("project"),
            "started_at": stored.started_at.isoformat(),
            "tty": stored.tty,
        }
        try:
            path.write_text(json.dumps(new_data, indent=2), encoding="utf-8")
        except OSError:
            pass
        return new_data
    finally:
        await close_engine()


def _parse_iso(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def start_session(
    name: str,
    *,
    task: str | None = None,
    project: str | None = None,
    agent: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
) -> dict:
    """Start (or replace) the session for this TTY. Returns the cursor dict.

    Creates a real ``sessions`` row in the DB; the on-disk cursor JSON
    holds both the uuid (for joins) and the slug (for human display).
    ``include_domains`` / ``exclude_domains`` are persisted in the
    cursor (not the DB) and inherited by subsequent ``hafiz query`` /
    ``hafiz context`` calls in this terminal.
    Raises :class:`RuntimeError` if there is no controlling terminal.
    """
    path = _session_path()
    if not path:
        raise RuntimeError("No controlling terminal — `hafiz session` requires a TTY.")

    from hafiz.core.search import _normalize_domains, _validate_domain_filters

    inc = _normalize_domains(include_domains)
    exc = _normalize_domains(exclude_domains)
    _validate_domain_filters(inc, exc)

    slug = make_session_id(name)
    started = datetime.now(UTC)
    tty = _tty_key()
    stored = _run_async(
        _create_session_db(
            slug=slug,
            name=name,
            agent=agent,
            scope_kind="project" if project else None,
            scope_value=project,
            task=task,
            tty=tty,
            started_at=started,
        )
    )

    data = {
        "session_uuid": str(stored.id),
        "session_id": stored.slug,
        "name": stored.name,
        "task": stored.task,
        "project": project,
        "started_at": stored.started_at.isoformat(),
        "tty": stored.tty,
    }
    if inc:
        data["include_domains"] = inc
    if exc:
        data["exclude_domains"] = exc
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


async def _create_session_db(**kwargs):
    from hafiz.core.database import close_engine
    from hafiz.core.sessions import create_session

    try:
        return await create_session(**kwargs)
    finally:
        await close_engine()


def end_session() -> dict | None:
    """Clear the cursor for this TTY and mark the DB row ended.

    Returns the dict of the session that was active, or None.
    """
    path = _session_path()
    if not path or not path.exists():
        return None
    data = current_session()
    try:
        path.unlink()
    except OSError:
        pass
    sid = (data or {}).get("session_uuid")
    if sid:
        try:
            _run_async(_end_session_db(uuid.UUID(sid)))
        except Exception:
            # End-of-session shouldn't fail loudly if DB is briefly
            # unreachable; the cursor is the user-visible state.
            pass
    return data


async def _end_session_db(session_uuid: uuid.UUID):
    from hafiz.core.database import close_engine
    from hafiz.core.sessions import end_session_db

    try:
        return await end_session_db(session_uuid)
    finally:
        await close_engine()


def resolve_session_tag(
    *,
    session_override: str | None,
    task_override: str | None,
) -> tuple[str | None, str | None]:
    """Resolve (session_slug, task) for an outgoing write.

    Returns the *user-facing slug* (string), not a uuid — back-compat
    with the existing call sites in ``hafiz observe`` / ``note`` /
    ``capture``. The slug is then passed through to ``store_annotation``
    where it gets resolved to a uuid (if a sessions row exists) and
    populated on both ``annotations.session_id`` (uuid FK) and
    ``annotations.legacy_session_id`` (text, for human display).

    Explicit ``--session`` / ``--task`` flags win over any active session;
    otherwise inherit from :func:`current_session`. Returns (None, None)
    when there's no active session and no overrides.
    """
    active = current_session() or {}
    session_id = session_override or active.get("session_id")
    task = task_override or active.get("task")
    return session_id, task


def resolve_domain_defaults(
    *,
    include_override: list[str] | None,
    exclude_override: list[str] | None,
) -> tuple[list[str] | None, list[str] | None]:
    """Resolve (include_domains, exclude_domains) for a query/context call.

    Explicit overrides win — if the caller passed *either* flag the
    session defaults are ignored entirely (so a user can flip the
    filter without first ending their session). Otherwise inherit from
    the active session cursor. Returns ``(None, None)`` when neither
    is set.
    """
    if include_override is not None or exclude_override is not None:
        return include_override, exclude_override
    active = current_session() or {}
    inc = active.get("include_domains") or None
    exc = active.get("exclude_domains") or None
    return inc, exc


def resolve_session_uuid_sync() -> uuid.UUID | None:
    """Sync helper: the uuid of the current TTY session, or None.

    Used by sync entry points (CLI command bodies) that want to attach
    annotations to the active session via FK rather than slug.
    """
    active = current_session() or {}
    raw = active.get("session_uuid")
    if not raw:
        return None
    try:
        return uuid.UUID(raw)
    except ValueError:
        return None
