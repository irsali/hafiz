"""Configuration management for Hafiz.

Loads settings from hafiz.toml with environment variable overrides.
Env vars use HAFIZ_ prefix with double-underscore nesting:
  HAFIZ_DATABASE__URL=postgresql+asyncpg://...
  HAFIZ_EMBEDDING__MODEL=nomic-ai/nomic-embed-text-v1.5
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_FILENAME = "hafiz.toml"


def config_search_paths() -> list[Path]:
    """Search order for the config file. Computed fresh each call so that
    tests (and cwd-changing users) see a consistent view — the module-
    level captured values would bake in whatever ``Path.home()`` /
    ``Path.cwd()`` were at import time."""
    paths = [
        Path.cwd() / CONFIG_FILENAME,
        Path.home() / ".config" / "hafiz" / CONFIG_FILENAME,
    ]
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            paths.append(Path(appdata) / "hafiz" / CONFIG_FILENAME)
    else:
        paths.append(Path("/etc/hafiz") / CONFIG_FILENAME)
    return paths


# Back-compat alias for any external readers that were importing this
# directly. The list is a snapshot at import time — callers wanting a
# live view should use ``config_search_paths()`` instead.
CONFIG_SEARCH_PATHS = config_search_paths()


def find_config_file() -> Path | None:
    """Find the first existing hafiz.toml in search paths."""
    for path in config_search_paths():
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
    # Conservative CPU-safe default. ONNX attention is O(n²) in sequence
    # length; a ~2 KB part is ~512 tokens and keeps peak RSS bounded on
    # a 16 GB laptop. GPU hosts can safely raise this via `hafiz config set`
    # or `hafiz doctor --apply`. See: workitems/active/tunable-registry.md.
    max_part_chars: int = 2_000


class RerankSettings(BaseModel):
    """Cross-encoder reranking of recall results (second-stage precision).

    Vector similarity compresses signal and noise into a narrow band; a
    cross-encoder re-scores the top-K candidates against the query and
    separates them sharply. Applied to annotation recall only (not code
    search or per-turn prefetch). The model ships with fastembed — no extra
    dependency — and loads lazily on first use.
    """

    enabled: bool = True
    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # ~80 MB ONNX
    # Over-fetch this multiple of the requested limit before reranking, so the
    # cross-encoder has real candidates to reorder (it can only reorder what
    # vector recall surfaced).
    candidate_multiplier: int = 3


class DedupSettings(BaseModel):
    """Near-duplicate detection for annotation writes — write hygiene.

    The "supersede on contradiction" contract is a behavioral guarantee
    agents can silently skip. Detection makes the conflict *visible* on
    write: ``store_annotation`` runs one cosine query against live
    annotations of the same kind/project and surfaces any that clear
    ``threshold``. Hafiz detects *similarity*, never *contradiction* — the
    semantic call (supersede? refine? unrelated?) stays with the agent/user.

    Default is surface-only: the write always succeeds and the duplicates
    ride back in the result. ``strict`` flips it to fail-closed — the write
    is refused unless the caller passes ``--supersedes`` or
    ``--allow-duplicate`` — for users who want a hard rail. Never applies to
    ``note`` (the firehose capture path).
    """

    enabled: bool = True
    # Cosine similarity at/above which an existing live annotation counts as a
    # near-duplicate. 0.88 catches genuine restatements while leaving room for
    # distinct-but-related facts (which routinely sit in the 0.80–0.87 band).
    threshold: float = 0.88
    # Fail-closed instead of surface-only. Off by default — a blocking gate
    # that can't tell "contradicts" from "merely similar" trains callers to
    # reflexively bypass it.
    strict: bool = False
    # Cap on how many duplicates to surface per write.
    max_candidates: int = 5


class IngestSettings(BaseModel):
    """Policy caps for the ingest pipeline — hard guards, not probed."""

    # Skip any file larger than this (bytes). Minified bundles, vendored
    # build output, and accidentally-committed binaries can sneak past the
    # binary-content probe; this stops them from OOM-ing the embedder.
    # 2 MB comfortably covers every hand-authored file we care about while
    # rejecting the classes of file that blow memory.
    max_file_bytes: int = 2_097_152  # 2 MB


class TelemetrySettings(BaseModel):
    """Local-only record of what was retrieved, so the store can be evaluated.

    Hafiz could not answer the most basic question about itself: which
    annotations have ever been recalled, which surfaced and were useful, and
    which have never come up once. Answering "is this earning its keep?" for a
    3.5-week deployment required parsing 169 Claude Code transcripts, because
    hafiz kept no record of its own reads. Every quality mechanism that might
    grow from here — decaying dead knowledge, promoting proven knowledge,
    noticing that recall quality regressed — needs data it wasn't collecting.

    What's stored is a new *category* of data for this store: the query text is
    what you were **looking for**, not what you concluded. It never leaves the
    machine, and it inherits the source layer's guarantees — bounded retention,
    reachable by ``forget``, included in ``export``. Set ``retrieval = false``
    to record nothing.
    """

    retrieval: bool = True
    # Matches the communications default. Long enough to see a month-over-month
    # trend, short enough that a query log doesn't become a permanent archive.
    retention_days: int = 90
    # Queries below this length are navigational noise ("yes", "continue") and
    # tell you nothing about what the store was asked for.
    min_query_chars: int = 3


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
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    dedup: DedupSettings = Field(default_factory=DedupSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
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
