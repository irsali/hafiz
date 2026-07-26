"""Tests for reaching sessions from a non-interactive caller.

The session cursor was keyed by TTY name alone, which made the feature
unreachable from exactly the places it matters — agent-harness hooks and CI
steps have no controlling terminal. Measured in a real deployment: 0 of 1,223
annotations carried a ``session_id`` despite 610 rows in ``sessions``, so
``journal --session``, ``distill --session`` and task tagging were all dead.

DB-free: only cursor-key resolution and the on-disk cursor are exercised.
"""

from __future__ import annotations

import json

import pytest

from hafiz.core import session as sess


@pytest.fixture(autouse=True)
def _isolated_cursor_dir(tmp_path, monkeypatch):
    """Point SESSION_DIR at a temp dir and clear both env vars."""
    monkeypatch.setattr(sess, "SESSION_DIR", tmp_path / "hafiz")
    monkeypatch.delenv(sess.SESSION_ENV, raising=False)
    monkeypatch.delenv(sess.SESSION_KEY_ENV, raising=False)
    return tmp_path


# ── Key resolution precedence ───────────────────────────────────────────


def test_explicit_key_wins_over_env(monkeypatch):
    monkeypatch.setenv(sess.SESSION_KEY_ENV, "from-env")
    assert sess._session_key("explicit") == "key-explicit"


def test_env_key_used_when_no_explicit_key(monkeypatch):
    monkeypatch.setenv(sess.SESSION_KEY_ENV, "from-env")
    assert sess._session_key() == "key-from-env"


def test_falls_back_to_tty(monkeypatch):
    monkeypatch.setattr(sess, "_tty_key", lambda: "pts-7")
    assert sess._session_key() == "pts-7"


def test_no_key_and_no_tty_means_no_session(monkeypatch):
    monkeypatch.setattr(sess, "_tty_key", lambda: None)
    assert sess._session_key() is None
    assert sess._session_path() is None


def test_blank_env_key_is_ignored(monkeypatch):
    """An exported-but-empty var must not shadow the TTY fallback."""
    monkeypatch.setenv(sess.SESSION_KEY_ENV, "   ")
    monkeypatch.setattr(sess, "_tty_key", lambda: "pts-7")
    assert sess._session_key() == "pts-7"


def test_keyed_cursor_is_distinct_from_tty_cursor(monkeypatch):
    """A hook's session must not clobber the human's terminal session."""
    monkeypatch.setattr(sess, "_tty_key", lambda: "pts-7")
    assert sess._session_path("hook-1") != sess._session_path()


# ── Key sanitization (untrusted harness-supplied ids) ───────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cc-abc123", "cc-abc123"),
        ("ok-key_1.2", "ok-key_1.2"),
        ("  padded  ", "padded"),
        ("a/b/c", "a-b-c"),
        ("../../etc/passwd", "etc-passwd"),
        ("....//..//x", "x"),
        ("", "key"),
        ("///", "key"),
    ],
)
def test_sanitize_session_key(raw, expected):
    assert sess.sanitize_session_key(raw) == expected


def test_sanitized_key_cannot_escape_the_cursor_dir(monkeypatch, tmp_path):
    """Path traversal in a harness-supplied id must not write outside SESSION_DIR."""
    monkeypatch.setenv(sess.SESSION_KEY_ENV, "../../../../etc/passwd")
    path = sess._session_path()
    assert path is not None
    assert path.parent == sess.SESSION_DIR
    assert ".." not in str(path)


def test_sanitize_caps_key_length():
    assert len(sess.sanitize_session_key("x" * 500)) == 64


# ── Cursor round-trip without a TTY ─────────────────────────────────────


def _write_cursor(key: str, slug: str, task: str | None = None) -> None:
    path = sess._session_path(key)
    path.write_text(
        json.dumps(
            {
                "session_uuid": "f8dbf1a6-5079-47a8-a0de-29cc125bfed3",
                "session_id": slug,
                "name": "hook probe",
                "task": task,
                "project": "hafiz",
                "started_at": "2026-07-26T13:21:44+00:00",
                "tty": f"key-{key}",
            }
        ),
        encoding="utf-8",
    )


def test_keyed_cursor_round_trips_without_a_tty(monkeypatch):
    monkeypatch.setattr(sess, "_tty_key", lambda: None)
    _write_cursor("cc-abc123", "hook-probe-cf57f0", task="retrieval-fix")

    assert sess.current_session("cc-abc123")["session_id"] == "hook-probe-cf57f0"
    assert sess.resolve_session_tag(
        session_override=None, task_override=None, session_key="cc-abc123"
    ) == ("hook-probe-cf57f0", "retrieval-fix")


def test_env_key_makes_later_processes_find_the_same_session(monkeypatch):
    """The load-bearing case: `session start` then `observe` in separate
    processes, neither with a terminal, both landing on one session."""
    monkeypatch.setattr(sess, "_tty_key", lambda: None)
    monkeypatch.setenv(sess.SESSION_KEY_ENV, "cc-abc123")
    _write_cursor("cc-abc123", "hook-probe-cf57f0", task="retrieval-fix")

    slug, task = sess.resolve_session_tag(session_override=None, task_override=None)
    assert (slug, task) == ("hook-probe-cf57f0", "retrieval-fix")


def test_session_uuid_resolves_from_a_keyed_cursor(monkeypatch):
    monkeypatch.setattr(sess, "_tty_key", lambda: None)
    _write_cursor("cc-abc123", "hook-probe-cf57f0")
    got = sess.resolve_session_uuid_sync("cc-abc123")
    assert str(got) == "f8dbf1a6-5079-47a8-a0de-29cc125bfed3"


def test_domain_defaults_inherit_from_a_keyed_cursor(monkeypatch):
    monkeypatch.setattr(sess, "_tty_key", lambda: None)
    path = sess._session_path("cc-abc123")
    path.write_text(
        json.dumps(
            {
                "session_uuid": "f8dbf1a6-5079-47a8-a0de-29cc125bfed3",
                "session_id": "s-1",
                "include_domains": ["code"],
            }
        ),
        encoding="utf-8",
    )
    inc, exc = sess.resolve_domain_defaults(
        include_override=None, exclude_override=None, session_key="cc-abc123"
    )
    assert (inc, exc) == (["code"], None)


# ── HAFIZ_SESSION direct override ───────────────────────────────────────


def test_hafiz_session_env_tags_without_any_cursor(monkeypatch):
    """For callers that already hold the id — no cursor file involved."""
    monkeypatch.setattr(sess, "_tty_key", lambda: None)
    monkeypatch.setenv(sess.SESSION_ENV, "hook-probe-cf57f0")
    slug, _ = sess.resolve_session_tag(session_override=None, task_override=None)
    assert slug == "hook-probe-cf57f0"


def test_explicit_session_flag_beats_the_env_var(monkeypatch):
    monkeypatch.setenv(sess.SESSION_ENV, "from-env")
    slug, _ = sess.resolve_session_tag(session_override="from-flag", task_override=None)
    assert slug == "from-flag"


def test_hafiz_session_env_beats_the_cursor(monkeypatch):
    monkeypatch.setattr(sess, "_tty_key", lambda: None)
    _write_cursor("cc-abc123", "from-cursor")
    monkeypatch.setenv(sess.SESSION_ENV, "from-env")
    slug, _ = sess.resolve_session_tag(
        session_override=None, task_override=None, session_key="cc-abc123"
    )
    assert slug == "from-env"


def test_blank_hafiz_session_falls_through_to_the_cursor(monkeypatch):
    monkeypatch.setattr(sess, "_tty_key", lambda: None)
    _write_cursor("cc-abc123", "from-cursor")
    monkeypatch.setenv(sess.SESSION_ENV, "  ")
    slug, _ = sess.resolve_session_tag(
        session_override=None, task_override=None, session_key="cc-abc123"
    )
    assert slug == "from-cursor"


# ── Failure mode ────────────────────────────────────────────────────────


def test_start_session_error_names_both_escape_hatches(monkeypatch):
    """The old message just said "requires a TTY" — a dead end for a hook."""
    monkeypatch.setattr(sess, "_tty_key", lambda: None)
    with pytest.raises(RuntimeError) as exc:
        sess.start_session("nope")
    message = str(exc.value)
    assert "--session-key" in message
    assert sess.SESSION_KEY_ENV in message
