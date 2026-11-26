"""Alembic environment configuration for Hafiz.

Async migrations via asyncpg (Postgres) or aiosqlite (embedded).

The URL is passed through ``normalize_url`` because users write
``sqlite:///path`` in ``hafiz.toml`` — the sync spelling — while
``async_engine_from_config`` requires an async driver. Without this,
Alembic fails on an embedded install with "the asyncio extension requires
an async driver", which reads as a Hafiz bug rather than a URL to fix.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from hafiz.core.database import Base

# Alembic Config object
config = context.config

# Override alembic.ini URL with hafiz config (hafiz.toml / env vars)
try:
    from hafiz.core.config import load_settings
    from hafiz.core.dialect import normalize_url

    config.set_main_option("sqlalchemy.url", normalize_url(load_settings().database.url))
except Exception:
    pass  # Fall back to alembic.ini value

# Logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# MetaData for autogenerate
target_metadata = Base.metadata


def include_object(obj, name, type_, reflected, compare_to):
    """Hide the full-text index from autogenerate.

    The ORM does not model it (a generated ``tsvector`` column on Postgres, an
    FTS5 table and its shadow tables on SQLite), so a comparison would propose
    dropping all of it. Revision 0012 owns those objects.
    """
    if not reflected:
        return True
    from hafiz.core.dialect import KEYWORD_PG_COLUMN, KEYWORD_PG_INDEX, KEYWORD_SQLITE_TABLE

    if type_ == "table":
        return not name.startswith(KEYWORD_SQLITE_TABLE)
    if type_ == "column":
        return name != KEYWORD_PG_COLUMN
    if type_ == "index":
        return name != KEYWORD_PG_INDEX
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generate SQL without connecting)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Run migrations with a given connection."""
    context.configure(
        connection=connection, target_metadata=target_metadata, include_object=include_object
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
