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
    """A fresh install needs no server.

    The default flipped from Postgres to embedded on 2026-09-06. It is
    derived from ``dialect.default_db_path`` rather than written out here, so
    there is one definition of where the store lives; a second literal would
    drift and would ignore XDG_DATA_HOME.
    """
    from hafiz.core.dialect import default_db_path, is_embedded

    db = DatabaseSettings()
    assert is_embedded(db.url)
    assert str(default_db_path()) in db.url


def test_the_default_honours_xdg_data_home(monkeypatch, tmp_path):
    """Two definitions of "where the DB lives" is how they start disagreeing."""
    from hafiz.core.config import _default_database_url

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert str(tmp_path) in _default_database_url()


def test_an_existing_config_still_outranks_the_default(tmp_path, monkeypatch):
    """The guarantee the whole default flip rests on.

    Switching the built-in default is only safe because a config file beats
    it — otherwise upgrading hafiz would silently move an existing user's
    brain to an empty file. Asserted rather than reasoned about, because the
    failure looks to a user like their data was deleted.
    """
    import tomllib

    from hafiz.core.config import write_default_config

    target = tmp_path / "hafiz.toml"
    target.write_text(
        '[database]\nurl = "postgresql+asyncpg://someone@elsewhere:5432/theirs"\n',
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError):
        write_default_config(target)

    data = tomllib.loads(target.read_text(encoding="utf-8"))
    assert data["database"]["url"] == "postgresql+asyncpg://someone@elsewhere:5432/theirs", (
        "write_default_config overwrote an existing config; upgrading hafiz would "
        "point an existing user at an empty store"
    )


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
    # The written file must be immediately usable, which since 2026-09-06
    # means an embedded store: a user who has just run `hafiz init` on a
    # machine with no Postgres should get a working config, not one pointing
    # at a server they never installed.
    assert data["database"]["url"].startswith("sqlite:///")
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
    assert "scratch" not in data["database"]["url"]
    # …and it is the real default, not merely "not the env value".
    assert data["database"]["url"] == DatabaseSettings().url


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


# ── Plain config keys, not just tunables ─────────────────────────────────


def test_database_url_is_settable_from_the_cli(tmp_path, monkeypatch):
    """`hafiz config set database.url` must work, because we tell people to run it.

    It did not. `config set` accepted only tunable-registry keys and answered
    "No tunable registered for key 'database.url'" — while the README, the
    `init` failure message and `migrate-backend`'s success message all told
    the user to run exactly that. Found by running the command in the
    fresh-install smoke test rather than by reading the docs.
    """
    import tomllib

    from hafiz.commands import maintenance

    target = tmp_path / "hafiz.toml"
    monkeypatch.setattr(maintenance, "_resolve_config_target", lambda *, local: target)

    maintenance.run_config_set(
        "database.url", "postgresql+asyncpg://u:p@localhost:5432/hafiz", output_json=True
    )
    data = tomllib.loads(target.read_text(encoding="utf-8"))
    assert data["database"]["url"] == "postgresql+asyncpg://u:p@localhost:5432/hafiz"


def test_settings_keys_are_derived_from_the_models_not_an_allowlist():
    """A field added to a settings model should be settable without registration.

    An allowlist would be a second list to keep in sync, and the failure mode
    is silent: the new key simply reports "unknown" forever.
    """
    from hafiz.commands.maintenance import _settings_field

    assert _settings_field("database.url") is not None
    assert _settings_field("embedding.device") is not None
    assert _settings_field("daemon.idle_timeout") is not None

    # And it stays a gate, not a rubber stamp.
    assert _settings_field("database.nonsense") is None
    assert _settings_field("nosuchsection.url") is None
    assert _settings_field("database") is None
    assert _settings_field("a.b.c") is None


def test_an_invalid_value_is_refused_before_it_reaches_the_file(tmp_path, monkeypatch):
    """Validation comes from the settings model, so a bad value never lands.

    Writing first and failing later would leave a config file that breaks
    every subsequent command, including the one needed to fix it.
    """
    import typer

    from hafiz.commands import maintenance

    target = tmp_path / "hafiz.toml"
    monkeypatch.setattr(maintenance, "_resolve_config_target", lambda *, local: target)

    # typer.Exit, which is what every other config error raises — not
    # SystemExit; typer's is a click exception and does not subclass it.
    with pytest.raises(typer.Exit):
        maintenance.run_config_set("embedding.device", "not-a-device", output_json=True)
    assert not target.exists(), "an invalid value created a config file anyway"
