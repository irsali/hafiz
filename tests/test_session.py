"""Tests for ``hafiz.core.session`` — pure helpers only.

File-I/O paths (``start_session`` / ``end_session``) are covered by the
dogfood run; here we verify slug shape and resolution precedence.
"""

from __future__ import annotations

import re
from unittest.mock import patch

from hafiz.core.session import (
    make_session_id,
    resolve_domain_defaults,
    resolve_session_tag,
)


def test_make_session_id_with_name():
    sid = make_session_id("Phase 3 migration")
    assert sid.startswith("phase-3-migration-")
    assert re.fullmatch(r"phase-3-migration-[0-9a-f]{6}", sid)


def test_make_session_id_empty_name_falls_back():
    sid = make_session_id("")
    assert re.fullmatch(r"session-[0-9a-f]{6}", sid)


def test_make_session_id_caps_long_names():
    sid = make_session_id("x" * 200)
    assert re.fullmatch(r"x{40}-[0-9a-f]{6}", sid)


def test_resolve_session_tag_without_active_or_override():
    with patch("hafiz.core.session.current_session", return_value=None):
        assert resolve_session_tag(session_override=None, task_override=None) == (None, None)


def test_resolve_session_tag_override_wins_over_active():
    active = {"session_id": "active-sess-1", "task": "active-task"}
    with patch("hafiz.core.session.current_session", return_value=active):
        sid, task = resolve_session_tag(
            session_override="override-sess", task_override="override-task"
        )
        assert sid == "override-sess"
        assert task == "override-task"


def test_resolve_session_tag_inherits_active_when_no_override():
    active = {"session_id": "active-sess-1", "task": "active-task"}
    with patch("hafiz.core.session.current_session", return_value=active):
        assert resolve_session_tag(session_override=None, task_override=None) == (
            "active-sess-1",
            "active-task",
        )


def test_resolve_session_tag_mixed_override():
    """Flag override should work per-field: pass session, inherit task (or vice versa)."""
    active = {"session_id": "active-sess-1", "task": "active-task"}
    with patch("hafiz.core.session.current_session", return_value=active):
        # Override session only; task should inherit.
        assert resolve_session_tag(session_override="override-sess", task_override=None) == (
            "override-sess",
            "active-task",
        )
        # Override task only; session should inherit.
        assert resolve_session_tag(session_override=None, task_override="override-task") == (
            "active-sess-1",
            "override-task",
        )


def test_resolve_domain_defaults_no_session_no_override():
    with patch("hafiz.core.session.current_session", return_value=None):
        assert resolve_domain_defaults(include_override=None, exclude_override=None) == (None, None)


def test_resolve_domain_defaults_inherits_from_session():
    active = {"include_domains": ["code"], "exclude_domains": ["chat"]}
    with patch("hafiz.core.session.current_session", return_value=active):
        assert resolve_domain_defaults(include_override=None, exclude_override=None) == (
            ["code"],
            ["chat"],
        )


def test_resolve_domain_defaults_explicit_override_wins_and_disables_inherit():
    """Passing *either* override skips session inheritance entirely —
    so a user can flip on `--exclude-domain code` without first
    clearing a session-default `--include-domain doc`."""
    active = {"include_domains": ["doc"], "exclude_domains": None}
    with patch("hafiz.core.session.current_session", return_value=active):
        # Only --exclude-domain passed: include should be None (not "doc"
        # from the session), exclude should be the override.
        assert resolve_domain_defaults(include_override=None, exclude_override=["code"]) == (
            None,
            ["code"],
        )


def test_resolve_domain_defaults_session_without_filters_is_none():
    active = {"session_id": "x"}  # no domain keys at all
    with patch("hafiz.core.session.current_session", return_value=active):
        assert resolve_domain_defaults(include_override=None, exclude_override=None) == (None, None)
