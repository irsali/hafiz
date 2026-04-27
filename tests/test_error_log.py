"""Tests for hafiz.core.error_log.

Every test redirects XDG_CACHE_HOME to tmp_path so nothing touches the
user's real error log.
"""

from __future__ import annotations

import json
import os

import pytest

from hafiz.core import error_log
from hafiz.core.error_log import (
    ErrorRecord,
    _exc_inherits_from,
    _recognize_config_validation,
    _recognize_db_connectivity,
    _recognize_pgvector_missing,
    _suggest_action,
    append,
    build_record,
    clear,
    count_recent,
    get,
    group_by_exception_type,
    log_exception,
    log_file_path,
    tail,
)


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    yield


def _make_record(**overrides) -> ErrorRecord:
    defaults = dict(
        id="11111111-1111-1111-1111-111111111111",
        timestamp="2026-04-22T12:00:00+00:00",
        command="ingest",
        argv=["ingest", "."],
        exception_type="RuntimeError",
        message="boom",
        traceback="Traceback...\nRuntimeError: boom",
        cwd="/tmp",
        hafiz_version="0.1.0",
        git_branch=None,
        git_dirty=None,
        host_fingerprint=None,
        suggested_action=None,
        context={},
    )
    defaults.update(overrides)
    return ErrorRecord(**defaults)


# ── path + basic I/O ───────────────────────────────────────────────────


def test_log_path_honors_xdg(tmp_path):
    p = log_file_path()
    assert str(p).startswith(str(tmp_path))
    assert p.name == "errors.log"


def test_append_and_tail_roundtrip():
    append(_make_record(id="a" * 36))
    append(_make_record(id="b" * 36, timestamp="2026-04-22T12:05:00+00:00"))
    rs = tail()
    # Newest first
    assert len(rs) == 2
    assert rs[0].id == "b" * 36
    assert rs[1].id == "a" * 36


def test_tail_limit():
    for i in range(5):
        append(
            _make_record(
                id=f"{i:036d}", timestamp=f"2026-04-22T12:0{i}:00+00:00"
            )
        )
    rs = tail(limit=2)
    assert len(rs) == 2


def test_get_exact_and_prefix():
    append(_make_record(id="abcdef0000000000000000000000000000000001"))
    append(_make_record(id="abcdef0000000000000000000000000000000002"))
    # exact
    assert get("abcdef0000000000000000000000000000000001") is not None
    # unique prefix too long to overlap
    assert get("abcdef00000000000000000000000000000000001") is None  # no such id
    # ambiguous prefix → None (to avoid returning a wrong match)
    assert get("abcdef") is None
    # unique shorter prefix that actually disambiguates
    assert get("abcdef0000000000000000000000000000000001") is not None


def test_clear_returns_count_and_removes_file():
    append(_make_record(id="1" * 36))
    append(_make_record(id="2" * 36))
    assert clear() == 2
    assert not log_file_path().exists()
    assert clear() == 0


def test_corrupt_lines_are_skipped():
    path = log_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("not json\n")
        f.write(
            json.dumps(
                {
                    "id": "aaaa",
                    "timestamp": "2026-04-22T12:00:00+00:00",
                    "command": "x",
                    "argv": [],
                    "exception_type": "RuntimeError",
                    "message": "ok",
                    "traceback": "",
                    "cwd": "/",
                }
            )
            + "\n"
        )
        f.write("also not json\n")
    rs = tail()
    assert len(rs) == 1
    assert rs[0].id == "aaaa"


# ── rotation ──────────────────────────────────────────────────────────


def test_rotation_on_entries_cap(monkeypatch):
    # Shrink the caps for the test so we don't need 1000 entries.
    monkeypatch.setattr(error_log, "MAX_ENTRIES", 3)

    for i in range(5):
        append(
            _make_record(
                id=f"{i:036d}", timestamp=f"2026-04-22T12:0{i}:00+00:00"
            )
        )

    rs = tail()
    # Only the last 3 should remain (newest → oldest).
    assert len(rs) == 3
    assert rs[0].id == f"{4:036d}"
    assert rs[-1].id == f"{2:036d}"


# ── build_record ──────────────────────────────────────────────────────


def test_build_record_captures_traceback_and_argv():
    try:
        raise ValueError("test boom")
    except ValueError as e:
        rec = build_record(e, argv=["graph", "stats"])
    assert rec.exception_type == "ValueError"
    assert rec.message == "test boom"
    assert "ValueError: test boom" in rec.traceback
    assert rec.command == "graph stats"
    assert rec.argv == ["graph", "stats"]


def test_build_record_extracts_command_before_flags():
    try:
        raise RuntimeError("x")
    except RuntimeError as e:
        rec = build_record(e, argv=["ingest", "--project", "hafiz"])
    assert rec.command == "ingest"


# ── suggested_action classifier ───────────────────────────────────────


def test_suggest_action_recognizes_declared_dep():
    exc = ModuleNotFoundError("No module named 'scipy'")
    exc.name = "scipy"
    suggestion, ctx = _suggest_action(exc, argv=[])
    assert suggestion is not None
    assert "pipx inject" in suggestion
    assert "scipy" in suggestion
    assert ctx["missing_module"] == "scipy"
    assert ctx["is_declared_dep"] is True


def test_suggest_action_undeclared_dep():
    exc = ModuleNotFoundError("No module named 'some_random_pkg'")
    exc.name = "some_random_pkg"
    suggestion, ctx = _suggest_action(exc, argv=[])
    # Still offers generic guidance but flags as not declared.
    assert suggestion is not None
    assert ctx["is_declared_dep"] is False


def test_suggest_action_unknown_class_is_none():
    suggestion, ctx = _suggest_action(ValueError("meh"), argv=[])
    assert suggestion is None
    assert ctx == {}


# ── log_exception (the convenience entry point) ────────────────────────


def test_log_exception_persists_and_returns_record():
    try:
        raise RuntimeError("boom")
    except RuntimeError as e:
        rec = log_exception(e, argv=["status"])
    assert rec.id
    rs = tail()
    assert rs[0].id == rec.id
    assert rs[0].exception_type == "RuntimeError"


def test_count_recent_since_filters_by_duration():
    # Append one old, one new. "old" is 10 days ago (ISO string).
    append(
        _make_record(id="old" + "0" * 33, timestamp="2020-01-01T00:00:00+00:00")
    )
    # "new" uses now-ish (whatever tail interprets as recent).
    from datetime import UTC, datetime

    now_iso = datetime.now(UTC).isoformat(timespec="seconds")
    append(_make_record(id="new" + "0" * 33, timestamp=now_iso))

    # Ask for only last 1 hour — excludes the 2020 entry.
    assert count_recent(since="1h") == 1


# ── no secrets: env var values never get into records ─────────────────


def test_record_does_not_include_environment_variables(monkeypatch):
    monkeypatch.setenv("TOTALLY_SECRET_TOKEN", "sk-leak-me")
    try:
        raise RuntimeError("x")
    except RuntimeError as e:
        rec = build_record(e, argv=["anything"])
    blob = json.dumps(rec.as_jsonable())
    assert "sk-leak-me" not in blob
    assert "TOTALLY_SECRET_TOKEN" not in blob


# ── _exc_inherits_from helper ─────────────────────────────────────────


def test_exc_inherits_from_walks_mro():
    class _CustomError(ValueError):
        pass

    exc = _CustomError("x")
    assert _exc_inherits_from(exc, "builtins.ValueError") is True
    assert _exc_inherits_from(exc, "builtins.RuntimeError") is False


# ── pgvector recognizer ───────────────────────────────────────────────


def _make_sa_programming_error(message: str):
    """Build a sqlalchemy ProgrammingError carrying ``message`` in str(exc)."""
    from sqlalchemy.exc import ProgrammingError as SAProgError

    return SAProgError("SELECT 1", {}, Exception(message))


def _make_sa_operational_error(message: str):
    from sqlalchemy.exc import OperationalError as SAOpError

    return SAOpError("SELECT 1", {}, Exception(message))


def test_recognize_pgvector_missing_extension():
    exc = _make_sa_programming_error('extension "vector" does not exist')
    result = _recognize_pgvector_missing(
        exc, argv=["init"], traceback_text=""
    )
    assert result is not None
    suggestion, ctx = result
    assert "pgvector" in suggestion.lower()
    assert "hafiz init" in suggestion
    assert ctx["missing_extension"] == "vector"


def test_recognize_pgvector_missing_type():
    exc = _make_sa_programming_error('type "vector" does not exist')
    result = _recognize_pgvector_missing(
        exc, argv=["status"], traceback_text=""
    )
    assert result is not None


def test_pgvector_recognizer_skips_unrelated_programming_error():
    exc = _make_sa_programming_error('relation "users" does not exist')
    result = _recognize_pgvector_missing(
        exc, argv=["status"], traceback_text=""
    )
    assert result is None


def test_pgvector_recognizer_ignores_non_sqlalchemy_classes():
    exc = ValueError('extension "vector" does not exist')
    result = _recognize_pgvector_missing(exc, argv=[], traceback_text="")
    assert result is None


# ── DB connectivity recognizer ────────────────────────────────────────


def test_recognize_db_connectivity_on_operational_error():
    exc = _make_sa_operational_error("connection refused")
    result = _recognize_db_connectivity(exc, argv=["status"], traceback_text="")
    assert result is not None
    suggestion, ctx = result
    assert "hafiz status --diagnose" in suggestion
    assert ctx["db_error_class"] == "OperationalError"


def test_db_connectivity_recognizer_yields_to_pgvector():
    """An OperationalError carrying a pgvector message should not be
    claimed by the connectivity recognizer — the more specific one wins
    via registry order, but the connectivity recognizer also
    self-yields as belt-and-braces."""
    exc = _make_sa_operational_error('extension "vector" does not exist')
    result = _recognize_db_connectivity(exc, argv=[], traceback_text="")
    assert result is None


def test_db_connectivity_ignores_unrelated_errors():
    exc = RuntimeError("nothing to do with the DB")
    result = _recognize_db_connectivity(exc, argv=[], traceback_text="")
    assert result is None


# ── config validation recognizer ──────────────────────────────────────


def _make_pydantic_validation_error():
    from pydantic import BaseModel, ValidationError

    class _M(BaseModel):
        n: int

    try:
        _M(n="not-an-int")  # type: ignore[arg-type]
    except ValidationError as e:
        return e
    raise AssertionError("expected ValidationError")


def test_recognize_config_validation_when_traceback_points_at_config_loader():
    exc = _make_pydantic_validation_error()
    fake_tb = (
        'Traceback (most recent call last):\n'
        '  File "/path/to/hafiz/core/config.py", line 143, in load_settings\n'
        '    return HafizSettings(**toml_data)\n'
        'pydantic_core._pydantic_core.ValidationError: 1 validation error\n'
    )
    result = _recognize_config_validation(
        exc, argv=["status"], traceback_text=fake_tb
    )
    assert result is not None
    suggestion, ctx = result
    assert "hafiz config show" in suggestion
    # invalid_keys should be populated from .errors()
    assert isinstance(ctx["invalid_keys"], list)
    assert any("n" == k or k.endswith(".n") for k in ctx["invalid_keys"])


def test_config_validation_no_false_positive_outside_config_loader():
    """A ValidationError raised by arbitrary downstream code with no
    hafiz/core/config.py in the traceback must NOT produce a config
    suggestion."""
    exc = _make_pydantic_validation_error()
    result = _recognize_config_validation(
        exc, argv=["query", "x"], traceback_text="some unrelated traceback"
    )
    assert result is None


# ── recognizer registry: ordering + first-match-wins ──────────────────


def test_suggest_action_walks_registry_and_returns_first_match():
    # ModuleNotFoundError is the first recognizer; should still match.
    exc = ModuleNotFoundError("No module named 'scipy'")
    exc.name = "scipy"
    suggestion, ctx = _suggest_action(exc, argv=[], traceback_text="")
    assert suggestion is not None
    assert ctx["missing_module"] == "scipy"


def test_suggest_action_routes_programming_error_to_pgvector():
    """A pgvector-shaped ProgrammingError reaches the pgvector
    recognizer through the registry walk, not the connectivity one."""
    exc = _make_sa_programming_error('extension "vector" does not exist')
    suggestion, ctx = _suggest_action(exc, argv=["init"], traceback_text="")
    assert suggestion is not None
    assert "pgvector" in suggestion.lower()
    assert ctx.get("missing_extension") == "vector"


def test_suggest_action_routes_plain_operational_error_to_connectivity():
    """A plain OperationalError without pgvector wording matches the
    connectivity recognizer."""
    exc = _make_sa_operational_error("connection refused")
    suggestion, ctx = _suggest_action(exc, argv=["status"], traceback_text="")
    assert suggestion is not None
    assert "hafiz status --diagnose" in suggestion
    assert ctx.get("db_error_class") == "OperationalError"


def test_suggest_action_buggy_recognizer_does_not_break_logging(monkeypatch):
    """A recognizer that raises must be skipped silently — the logger's
    job is to never crash."""

    def _explosive(exc, *, argv, traceback_text):
        raise RuntimeError("recognizer bug")

    monkeypatch.setattr(error_log, "_RECOGNIZERS", (_explosive,))
    suggestion, ctx = _suggest_action(
        ValueError("x"), argv=[], traceback_text=""
    )
    assert suggestion is None
    assert ctx == {}


# ── group_by_exception_type ──────────────────────────────────────────


def test_group_by_exception_type_aggregates_and_sorts():
    # newest-first input (the shape tail() returns)
    records = [
        _make_record(
            id=f"{i:036d}",
            timestamp=f"2026-04-22T12:0{i}:00+00:00",
            exception_type=cls,
            message=f"msg {i}",
            command=cmd,
            suggested_action=("fix it" if cls == "ModuleNotFoundError" else None),
        )
        for i, (cls, cmd) in enumerate(
            reversed(
                [
                    ("RuntimeError", "ingest"),
                    ("ModuleNotFoundError", "graph stats"),
                    ("ModuleNotFoundError", "query x"),
                    ("ModuleNotFoundError", "context y"),
                    ("RuntimeError", "ingest"),
                ]
            )
        )
    ]
    groups = group_by_exception_type(records)
    # Two distinct types
    assert {g["exception_type"] for g in groups} == {
        "ModuleNotFoundError",
        "RuntimeError",
    }
    # Sorted by count desc → ModuleNotFoundError (3) first
    assert groups[0]["exception_type"] == "ModuleNotFoundError"
    assert groups[0]["count"] == 3
    assert groups[0]["with_suggestions"] == 3
    assert groups[1]["exception_type"] == "RuntimeError"
    assert groups[1]["count"] == 2
    assert groups[1]["with_suggestions"] == 0


def test_group_by_exception_type_empty():
    assert group_by_exception_type([]) == []


def test_group_by_exception_type_truncates_long_sample_message():
    long_msg = "x" * 500
    rec = _make_record(message=long_msg)
    groups = group_by_exception_type([rec])
    assert len(groups[0]["sample_message"]) <= 201  # 200 + ellipsis
    assert groups[0]["sample_message"].endswith("…")
