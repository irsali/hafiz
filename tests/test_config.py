"""Tests for hafiz.core.config."""

import os
from pathlib import Path
from unittest.mock import patch

from hafiz.core.config import (
    HafizSettings,
    DatabaseSettings,
    EmbeddingSettings,
    LLMSettings,
    WorkspaceSettings,
    load_settings,
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
    with patch.dict(os.environ, {"HAFIZ_DATABASE__URL": "postgresql+asyncpg://test:test@db:5432/test"}):
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
