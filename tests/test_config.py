"""Tests for hafiz.core.config."""

import os
from unittest.mock import patch

import pytest

from hafiz.core.config import (
    DatabaseSettings,
    EmbeddingSettings,
    HafizSettings,
    WorkspaceSettings,
    reset_settings,
)


def test_default_settings():
    """Default settings should have sane defaults."""
    settings = HafizSettings()
    assert "hafiz" in settings.database.url
    assert settings.embedding.model == "nomic-ai/nomic-embed-text-v1.5"
    assert settings.embedding.dimensions == 768
    assert settings.embedding.provider == "fastembed"
    assert settings.llm.provider == "anthropic"


def test_env_override():
    """Environment variables should override defaults."""
    with patch.dict(
        os.environ, {"HAFIZ_DATABASE__URL": "postgresql+asyncpg://test:test@db:5432/test"}
    ):
        reset_settings()
        settings = HafizSettings()
        assert settings.database.url == "postgresql+asyncpg://test:test@db:5432/test"


def test_database_settings_defaults():
    db = DatabaseSettings()
    assert "postgresql" in db.url
    assert "asyncpg" in db.url


def test_embedding_settings_defaults():
    emb = EmbeddingSettings()
    assert emb.model == "nomic-ai/nomic-embed-text-v1.5"
    assert emb.dimensions == 768


def test_workspace_settings_defaults():
    ws = WorkspaceSettings()
    assert ".git" in ws.ignore
    assert "node_modules" in ws.ignore


def test_settings_serialization():
    """Settings should serialize to dict/JSON cleanly."""
    settings = HafizSettings()
    data = settings.model_dump()
    assert "database" in data
    assert "embedding" in data
    assert "llm" in data
    assert "workspace" in data


# ---------------------------------------------------------------------------
# First-run config bootstrap + resolution order
# ---------------------------------------------------------------------------


def test_write_default_config_creates_a_usable_file(tmp_path):
    """`cp hafiz.toml.example hafiz.toml` assumed a cloned repo, which a
    `pipx install hafiz` user does not have. init writes one instead."""
    import tomllib

    from hafiz.core.config import write_default_config

    target = tmp_path / "cfg" / "hafiz.toml"
    written = write_default_config(target, workspace_root=tmp_path / "work")

    assert written == target
    data = tomllib.loads(target.read_text(encoding="utf-8"))
    assert data["database"]["url"].startswith("postgresql+asyncpg://")
    assert data["embedding"]["dimensions"] == 768
    assert data["workspace"]["root"] == (tmp_path / "work").as_posix()


def test_write_default_config_refuses_to_clobber(tmp_path):
    from hafiz.core.config import write_default_config

    target = tmp_path / "hafiz.toml"
    target.write_text("# mine\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_default_config(target)
    assert target.read_text(encoding="utf-8") == "# mine\n"


def test_write_default_config_ignores_env_overrides(tmp_path, monkeypatch):
    """A throwaway env var must not be frozen into the user's config."""
    import tomllib

    from hafiz.core.config import write_default_config

    monkeypatch.setenv("HAFIZ_DATABASE__URL", "postgresql+asyncpg://tmp@localhost:59999/scratch")
    target = tmp_path / "hafiz.toml"
    write_default_config(target)

    data = tomllib.loads(target.read_text(encoding="utf-8"))
    assert "59999" not in data["database"]["url"]
    assert data["database"]["url"].endswith(":5432/hafiz")


def test_env_overridden_keys_parses_two_level_names():
    from hafiz.core.config import env_overridden_keys

    assert env_overridden_keys(
        {
            "HAFIZ_DATABASE__URL": "x",
            "HAFIZ_EMBEDDING__DEVICE": "cpu",
            "HAFIZ_SESSION_KEY": "not-a-setting",  # single level
            "PATH": "/usr/bin",
            "HAFIZ_A__B__C": "too deep",
        }
    ) == {("database", "url"), ("embedding", "device")}


def test_env_beats_toml_as_documented(tmp_path, monkeypatch):
    """The documented order is env -> hafiz.toml -> sticky -> default.

    TOML values are passed as init kwargs, and pydantic-settings ranks init
    above env — so without the fix, `HAFIZ_EMBEDDING__DEVICE=cpu hafiz
    ingest .` (advertised in the README) was silently ignored whenever the
    config file set that key.
    """
    from hafiz.core.config import load_settings

    cfg = tmp_path / "hafiz.toml"
    cfg.write_text(
        '[embedding]\ndevice = "gpu"\nmax_part_chars = 1234\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HAFIZ_EMBEDDING__DEVICE", "cpu")

    settings = load_settings()
    assert settings.embedding.device == "cpu", "env must win over the config file"
    # A key the env does not address still comes from the file. Asserted on
    # `embedding`, not `database`: conftest patches `load_settings` to force
    # the test-DB url — a workaround whose own docstring names this very bug
    # ("toml values arrive as pydantic-settings init args, which beat env
    # vars"), so `database.url` cannot measure anything here.
    assert settings.embedding.max_part_chars == 1234
