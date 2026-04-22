"""Configuration management for Hafiz.

Loads settings from hafiz.toml with environment variable overrides.
Env vars use HAFIZ_ prefix with double-underscore nesting:
  HAFIZ_DATABASE__URL=postgresql+asyncpg://...
  HAFIZ_EMBEDDING__MODEL=nomic-ai/nomic-embed-text-v1.5
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


CONFIG_FILENAME = "hafiz.toml"

# Search order for config file
CONFIG_SEARCH_PATHS = [
    Path.cwd() / CONFIG_FILENAME,
    Path.home() / ".config" / "hafiz" / CONFIG_FILENAME,
    Path("/etc/hafiz") / CONFIG_FILENAME,
]


def find_config_file() -> Path | None:
    """Find the first existing hafiz.toml in search paths."""
    for path in CONFIG_SEARCH_PATHS:
        if path.is_file():
            return path
    return None


def load_toml(path: Path) -> dict:
    """Load and parse a TOML file."""
    with open(path, "rb") as f:
        return tomllib.load(f)


class DatabaseSettings(BaseModel):
    url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/hafiz"


class EmbeddingSettings(BaseModel):
    model: str = "nomic-ai/nomic-embed-text-v1.5"
    provider: str = "fastembed"
    dimensions: int = 768
    device: Literal["auto", "cpu", "gpu"] = "auto"


class LLMSettings(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"


class WorkspaceSettings(BaseModel):
    root: str = "."
    projects: list[str] = Field(default_factory=list)
    ignore: list[str] = Field(
        default_factory=lambda: [
            ".git",
            "node_modules",
            "__pycache__",
            ".venv",
            "dist",
            "build",
        ]
    )


class GraphSettings(BaseModel):
    """Tuning knobs for knowledge-graph expansion inside `hafiz context`.

    context_depth        — how many hops to walk out from entities found in
                           the retrieved chunks (undirected). 3 is a sensible
                           default: includes neighbors-of-neighbors without
                           blowing up the bundle.
    context_max_entities — hard cap on how many entities land in the bundle
                           after walking. Results are sorted by (distance,
                           -pagerank) so nearest + most central win the cap.
    """

    context_depth: int = 3
    context_max_entities: int = 25


class HafizSettings(BaseSettings):
    """Main settings object. Loaded from hafiz.toml + env var overrides."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="HAFIZ_",
        env_nested_delimiter="__",
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)


def load_settings() -> HafizSettings:
    """Load settings from hafiz.toml (if found), with env var overrides."""
    config_path = find_config_file()
    if config_path:
        toml_data = load_toml(config_path)
        return HafizSettings(**toml_data)
    return HafizSettings()


# Singleton for convenience
_settings: HafizSettings | None = None


def get_settings() -> HafizSettings:
    """Get the global settings instance (lazy-loaded)."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """Reset the cached settings (useful for testing)."""
    global _settings
    _settings = None
