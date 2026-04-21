"""Tests for the UUID-list CLI parser used by --derived-from."""

from __future__ import annotations

import pytest

from hafiz.commands.observe import _parse_uuid_list


def test_none_returns_none():
    assert _parse_uuid_list(None) is None


def test_empty_string_returns_none():
    assert _parse_uuid_list("") is None


def test_whitespace_only_returns_none():
    assert _parse_uuid_list("  ,  ,  ") is None


def test_single_uuid():
    sid = "11111111-2222-3333-4444-555555555555"
    assert _parse_uuid_list(sid) == [sid]


def test_multiple_uuids_with_whitespace():
    ids = [
        "11111111-2222-3333-4444-555555555555",
        "66666666-7777-8888-9999-aaaaaaaaaaaa",
    ]
    assert _parse_uuid_list(f"{ids[0]} , {ids[1]}") == ids


def test_bad_uuid_in_list_raises_system_exit():
    with pytest.raises(SystemExit):
        _parse_uuid_list("11111111-2222-3333-4444-555555555555,not-a-uuid")
