"""Tests for hafiz.core.tunables — registry + resolution precedence.

Phase 1 coverage: the registry exists, has the two built-in tunables,
rejects duplicates, and ``resolve()`` honors env > TOML > default via
pydantic-settings. Sticky state + prober integration land with phases
2 and 3 and get their own tests there.
"""

from __future__ import annotations

import pytest

from hafiz.core import tunables
from hafiz.core.config import reset_settings
from hafiz.core.tunables import (
    Tunable,
    TUNABLE_REGISTRY,
    all_tunables,
    get,
    register,
    resolve,
)


@pytest.fixture(autouse=True)
def _reset_settings_between_tests(tmp_path, monkeypatch):
    """Env var overrides only take effect on the next ``get_settings()``
    call, so drop the cached instance around every test. Also redirect
    XDG_CACHE_HOME so the sticky tuning cache (written by a real
    ``hafiz config apply`` on the dev machine) doesn't leak into tests
    that expect resolve() to return the built-in default."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    reset_settings()
    yield
    reset_settings()


# ── built-in registrations ─────────────────────────────────────────────


def test_builtin_tunables_registered():
    keys = {t.key for t in all_tunables()}
    assert "embedding.max_part_chars" in keys
    assert "ingest.max_file_bytes" in keys


def test_max_part_chars_is_probed():
    """Probed tunables are those we auto-tune via ``hafiz doctor``.
    ``embedding.max_part_chars`` is the reference probed tunable for
    phase 1 — phase 2 will wire its prober; phase 1 just declares it
    as a probe target by being a non-policy tunable shape."""
    t = get("embedding.max_part_chars")
    # Phase 1: prober is None (added in phase 2). Still a probed-tunable
    # shape — not a policy cap. Distinguish via intent, not implementation.
    assert t.default == 2_000
    assert t.type_ is int


def test_max_file_bytes_is_policy_cap():
    t = get("ingest.max_file_bytes")
    assert t.is_policy is True
    assert t.prober is None
    assert t.default == 2_097_152


# ── resolution ──────────────────────────────────────────────────────────


def test_resolve_returns_default_when_unset():
    assert resolve("embedding.max_part_chars") == 2_000
    assert resolve("ingest.max_file_bytes") == 2_097_152


def test_resolve_unknown_key_raises():
    with pytest.raises(KeyError):
        resolve("made.up.key")


def test_resolve_honors_env_override(monkeypatch):
    monkeypatch.setenv("HAFIZ_EMBEDDING__MAX_PART_CHARS", "4096")
    reset_settings()
    assert resolve("embedding.max_part_chars") == 4096


def test_resolve_honors_env_override_for_policy_cap(monkeypatch):
    """Policy caps read through the same pydantic-settings chain even
    though they have no prober."""
    monkeypatch.setenv("HAFIZ_INGEST__MAX_FILE_BYTES", "8388608")
    reset_settings()
    assert resolve("ingest.max_file_bytes") == 8_388_608


# ── registry hygiene ───────────────────────────────────────────────────


def test_register_rejects_duplicates():
    original = dict(TUNABLE_REGISTRY)
    try:
        dup = Tunable(
            key="embedding.max_part_chars",
            default=999,
            type_=int,
            description="duplicate",
        )
        with pytest.raises(ValueError, match="already registered"):
            register(dup)
    finally:
        # Guard against leaking state into sibling tests.
        TUNABLE_REGISTRY.clear()
        TUNABLE_REGISTRY.update(original)


def test_register_and_get_roundtrip():
    original = dict(TUNABLE_REGISTRY)
    try:
        t = Tunable(
            key="test.something",
            default=42,
            type_=int,
            description="for testing",
        )
        register(t)
        assert get("test.something") is t
    finally:
        TUNABLE_REGISTRY.clear()
        TUNABLE_REGISTRY.update(original)


def test_unknown_key_raises_keyerror():
    with pytest.raises(KeyError):
        get("does.not.exist")


# ── validator ───────────────────────────────────────────────────────────


def test_validator_rejects_zero_and_negative():
    t = get("embedding.max_part_chars")
    assert t.validator is not None
    with pytest.raises(ValueError):
        t.validator(0)
    with pytest.raises(ValueError):
        t.validator(-1)


def test_validator_rejects_wrong_type():
    t = get("embedding.max_part_chars")
    with pytest.raises(ValueError):
        t.validator("2000")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        t.validator(True)  # bools are sneaky ints in Python
