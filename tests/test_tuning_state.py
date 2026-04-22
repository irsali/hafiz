"""Tests for hafiz.core.tuning_state — sticky state persistence.

Every test overrides ``XDG_CACHE_HOME`` to a tmp_path so we never
touch the user's real ~/.cache/hafiz/tuning_state.json.
"""

from __future__ import annotations

import json

import pytest

from hafiz.core.tuning_state import (
    CURRENT_SCHEMA,
    TuningEntry,
    TuningState,
    build_state,
    cache_file_path,
    clear_state,
    get_value,
    is_stale,
    load_state,
    merge_into_state,
    save_state,
)


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Redirect XDG_CACHE_HOME to a tmp dir per test."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    yield


# ── path ────────────────────────────────────────────────────────────────


def test_cache_file_path_honors_xdg(tmp_path):
    p = cache_file_path()
    assert str(p).startswith(str(tmp_path))
    assert p.name == "tuning_state.json"


# ── load/save roundtrip ────────────────────────────────────────────────


def test_save_and_load_roundtrip():
    state = build_state(
        fingerprint="abc123",
        ort_version="1.24.4",
        entries={
            "embedding.max_part_chars": TuningEntry(
                value=16_000,
                rationale="measured on this host",
                confidence="high",
                probed_at="2026-04-22T12:00:00+00:00",
                measured={"budget_mb": 12000},
            )
        },
    )
    save_state(state)

    loaded = load_state()
    assert loaded is not None
    assert loaded.host_fingerprint == "abc123"
    assert loaded.onnxruntime_version == "1.24.4"
    assert "embedding.max_part_chars" in loaded.entries
    e = loaded.entries["embedding.max_part_chars"]
    assert e.value == 16_000
    assert e.confidence == "high"
    assert e.measured == {"budget_mb": 12000}


def test_load_returns_none_when_absent():
    assert load_state() is None


def test_corrupt_file_is_removed_and_load_returns_none():
    path = cache_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json at all")
    assert load_state() is None
    assert not path.exists()


def test_wrong_schema_is_ignored_but_not_deleted():
    path = cache_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 999, "entries": {}}))
    assert load_state() is None
    assert path.exists()  # not the user's fault — leave file alone


def test_clear_state_removes_file():
    save_state(build_state(fingerprint="x", ort_version=None, entries={}))
    assert cache_file_path().is_file()
    assert clear_state() is True
    assert not cache_file_path().exists()
    assert clear_state() is False  # idempotent — no file to remove


# ── staleness ──────────────────────────────────────────────────────────


def test_is_stale_on_fingerprint_mismatch():
    state = build_state(
        fingerprint="old_host",
        ort_version="1.24.4",
        entries={},
    )
    assert is_stale(state, fingerprint="new_host", ort_version="1.24.4") is True


def test_is_stale_on_ort_version_change():
    state = build_state(
        fingerprint="same_host",
        ort_version="1.24.4",
        entries={},
    )
    assert is_stale(state, fingerprint="same_host", ort_version="1.25.0") is True


def test_is_stale_false_when_match():
    state = build_state(
        fingerprint="same_host",
        ort_version="1.24.4",
        entries={},
    )
    assert is_stale(state, fingerprint="same_host", ort_version="1.24.4") is False


def test_is_stale_tolerates_missing_ort_version():
    """Older state without ORT stamp shouldn't be invalidated just
    because we've since learned the ORT version."""
    state = build_state(fingerprint="x", ort_version=None, entries={})
    assert is_stale(state, fingerprint="x", ort_version="1.24.4") is False


# ── get_value ──────────────────────────────────────────────────────────


def test_get_value_returns_none_when_absent():
    assert get_value("foo.bar", fingerprint="x", ort_version=None) is None


def test_get_value_returns_value_when_match():
    save_state(
        build_state(
            fingerprint="host1",
            ort_version=None,
            entries={"embedding.max_part_chars": TuningEntry(value=8_000)},
        )
    )
    assert (
        get_value("embedding.max_part_chars", fingerprint="host1", ort_version=None)
        == 8_000
    )


def test_get_value_ignores_stale_entries():
    """When the fingerprint no longer matches, we don't apply the value
    — re-probing is required to move forward on a new host."""
    save_state(
        build_state(
            fingerprint="old_host",
            ort_version=None,
            entries={"embedding.max_part_chars": TuningEntry(value=8_000)},
        )
    )
    assert (
        get_value("embedding.max_part_chars", fingerprint="new_host", ort_version=None)
        is None
    )


# ── merge ──────────────────────────────────────────────────────────────


def test_merge_into_state_creates_fresh_when_none():
    merged = merge_into_state(
        None,
        fingerprint="x",
        ort_version="1.24.4",
        new_entries={"k": TuningEntry(value=1)},
    )
    assert merged.entries["k"].value == 1
    assert merged.host_fingerprint == "x"


def test_merge_into_state_keeps_existing_when_matching_host():
    existing = build_state(
        fingerprint="x",
        ort_version="1.24.4",
        entries={"a": TuningEntry(value=1)},
    )
    merged = merge_into_state(
        existing,
        fingerprint="x",
        ort_version="1.24.4",
        new_entries={"b": TuningEntry(value=2)},
    )
    assert set(merged.entries) == {"a", "b"}


def test_merge_into_state_discards_stale_existing():
    """If the cached state is from a different host, don't carry its
    entries forward — they'd be about a different environment."""
    existing = build_state(
        fingerprint="old_host",
        ort_version="1.24.4",
        entries={"a": TuningEntry(value=1)},
    )
    merged = merge_into_state(
        existing,
        fingerprint="new_host",
        ort_version="1.24.4",
        new_entries={"b": TuningEntry(value=2)},
    )
    assert set(merged.entries) == {"b"}


# ── schema version ─────────────────────────────────────────────────────


def test_schema_constant_matches_persisted_shape():
    save_state(build_state(fingerprint="x", ort_version=None, entries={}))
    raw = json.loads(cache_file_path().read_text())
    assert raw["schema"] == CURRENT_SCHEMA
