"""Pytest configuration.

Two responsibilities:

1. **Quarantine** test files whose modules are not yet rewired onto the
   new schema (see ``collect_ignore``).

2. **Test-DB isolation** — at module-load time (before any fixture or
   test runs), redirect ``HAFIZ_DATABASE__URL`` to a dedicated
   ``<base-db>_test`` database. The test DB is created on demand via
   ``psql`` (already an environment dependency) and migrated to head
   via ``alembic``. Real production data is never touched, even by
   pre-existing tests that do unscoped ``DELETE FROM <table>`` for
   fixture setup.

   The test DB is **kept across runs** (truncated by individual tests
   as they need, never dropped) so the migration cost amortizes.
   Override behavior with the env vars:

     ``HAFIZ_TEST_DB``           — alternative test DB URL.
     ``HAFIZ_TEST_DB_RECREATE``  — set to ``1`` to drop + recreate
                                   at session start (after destructive
                                   migration changes).
     ``HAFIZ_TEST_DB_DISABLE``   — set to ``1`` to opt out and run
                                   tests against the real DB. Use only
                                   when inspecting test residue.
     ``HAFIZ_TEST_NO_DB``        — set to ``1`` to skip DB setup entirely
                                   and run only the DB-free subset (the
                                   macOS CI leg uses this; Postgres isn't
                                   provisioned there). DB-dependent test
                                   modules are dropped at collection.
     ``HAFIZ_TEST_BACKEND``      — ``postgresql`` (default) or ``sqlite``.
                                   Runs the whole DB-touching suite against
                                   that backend. See "Dual-backend matrix".

Remove an entry from ``collect_ignore`` when its module is rewired.

Dual-backend matrix
-------------------

Hafiz serves one ``--json`` contract from two backends, so both have to be
exercised by the same tests rather than by a separate SQLite-only module.
``HAFIZ_TEST_BACKEND=sqlite pytest`` points the whole session at a
throwaway SQLite file; CI runs both legs.

**A skipped backend fails the run.** When a backend is named explicitly,
any test that skips for a database reason marks the session failed — see
:func:`pytest_sessionfinish`. This is not pedantry: this suite has twice
reported green while silently skipping DB-backed tests (once from a closed
event loop, once from an unreachable Postgres), and a matrix that skips
itself manufactures confidence about a backend nobody exercised.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path
from urllib.parse import urlparse, urlunparse

# Force a no-color terminal for the whole test session. Many tests assert on
# raw flag substrings in ``--help`` output (``assert "--json" in result.output``).
# Typer renders help via Rich; in a CI environment Rich detects the
# ``GITHUB_ACTIONS`` / ``CI`` / ``FORCE_COLOR`` markers and *forces* color mode
# even though ``CliRunner`` captures through a non-tty pipe. Forced color wraps
# each flag literal in ANSI style codes (``\x1b[1m--json\x1b[0m``), so the plain
# substring is no longer present and the assertions fail in CI while passing on
# a local non-tty run. ``TERM=dumb`` makes Rich emit plain text; ``NO_COLOR``
# alone is insufficient (Typer's rich_utils ignores it for tty detection).
# ``setdefault`` respects an explicit local override.
os.environ.setdefault("TERM", "dumb")

# Tests never talk to a warm daemon unless they ask to.
#
# Two reasons, both learned the hard way the moment the daemon was actually
# wired into the read path:
#
# 1. **Monkeypatching stops working.** A test that patches
#    `hafiz.core.annotations.search_annotations` patches *this* process. If a
#    daemon is up, the request crosses a socket into a different process that
#    never saw the patch, and the test asserts against real data. One test
#    flipped from pass to fail purely because a developer had `hafiz serve`
#    running.
# 2. **Tests would spawn daemons.** `request()` auto-spawns when no socket
#    answers, so a suite run on a clean machine would fork background
#    processes that outlive it.
#
# `setdefault`, so a test that genuinely wants the daemon path can override.
os.environ.setdefault("HAFIZ_NO_DAEMON", "1")

collect_ignore = [
    # Uses the old chunker API (ChunkResult, chunk_file, LANGUAGE_MAP). The
    # new chunker is walk_files + prepare_embedding_parts; a fresh test
    # module will replace this.
    "test_chunker.py",
    # hafiz/core/capture.py still imports ChunkResult from chunker and
    # Chunk from database. Un-quarantine when capture is rewired.
    "test_capture.py",
]


# Test modules that require a live Postgres + pgvector. In ``HAFIZ_TEST_NO_DB``
# mode (the macOS CI leg, where Postgres isn't provisioned) these are dropped
# at collection so the DB-free subset can still run. Keep in sync as DB-touching
# modules are added — a missing entry surfaces as a connection error, not a
# silent skip.
_DB_DEPENDENT = [
    "test_annotation_tag_pin.py",
    "test_cli.py",
    "test_communications_schema.py",
    "test_concurrency.py",
    "test_config.py",
    "test_exact_duplicates.py",
    "test_extract_v2.py",
    "test_forget.py",
    "test_git_axis.py",
    "test_graph_analysis.py",
    "test_importer_claude_code.py",
    "test_index_freshness.py",
    "test_mcp.py",
    "test_ingest_flow.py",
    "test_polymorphic_lineage.py",
    "test_prune_untagged.py",
    "test_recall_and_transcripts.py",
    "test_retention_visibility.py",
    "test_retrieval_telemetry.py",
    "test_rewrite_resilience.py",
    "test_search.py",
    "test_session_list_cli.py",
    "test_sessions_db.py",
    "test_structural_schema.py",
]

DEAD_DB_URL = "postgresql+asyncpg://nobody:nobody@127.0.0.1:1/hafiz_no_db"


def _force_dead_db() -> None:
    """In NO_DB mode, point every settings read at a DB that refuses instantly.

    Without this, NO_DB mode was the *least* isolated leg, not the most: it
    skips ``_setup_test_db_once`` entirely, so nothing redirects the URL and any
    module not listed in ``_DB_DEPENDENT`` resolves the production ``hafiz.toml``
    and writes to the real store. Setting ``HAFIZ_DATABASE__URL`` is not
    sufficient — toml values arrive as pydantic-settings *init args*, which beat
    env vars — so ``load_settings`` is patched the same way the test-DB path
    patches it. Measured: ``test_ingest_guards`` silently ingested its
    ``tmp_path`` into a production index on this leg.
    """
    os.environ["HAFIZ_DATABASE__URL"] = DEAD_DB_URL

    from hafiz.core import config as _config

    _real_load_settings = _config.load_settings

    def _patched():
        s = _real_load_settings()
        s.database.url = DEAD_DB_URL
        return s

    _config.load_settings = _patched  # type: ignore[assignment]
    _config.reset_settings()


if os.environ.get("HAFIZ_TEST_NO_DB") == "1":
    collect_ignore += _DB_DEPENDENT
    _force_dead_db()


# ---------------------------------------------------------------------------
# Test DB setup — runs at module-load time so env overrides land before
# any hafiz.core.config import.
# ---------------------------------------------------------------------------


def _read_base_url() -> str:
    """Pick up the production DB URL the same way the app does, but
    without importing hafiz.core.config (which would pin the URL
    before we override it).
    """
    env_url = os.environ.get("HAFIZ_DATABASE__URL")
    if env_url:
        return env_url
    for candidate in (
        Path.cwd() / "hafiz.toml",
        Path.home() / ".config" / "hafiz" / "hafiz.toml",
        Path("/etc/hafiz/hafiz.toml"),
    ):
        if candidate.exists():
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore
            cfg = tomllib.loads(candidate.read_text(encoding="utf-8"))
            url = cfg.get("database", {}).get("url")
            if url:
                return url
    return "postgresql+asyncpg://postgres:postgres@localhost:5432/hafiz"


def _to_psql_url(async_url: str) -> str:
    """Strip the ``+asyncpg`` dialect tag for the psql command line."""
    return re.sub(r"^postgresql\+asyncpg://", "postgresql://", async_url)


def _swap_db(url: str, new_db: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(path="/" + new_db.lstrip("/")))


def _psql(url: str, sql: str) -> tuple[int, str, str]:
    """Run a single SQL command via psql. Returns (rc, stdout, stderr)."""
    result = subprocess.run(
        ["psql", url, "-v", "ON_ERROR_STOP=1", "-c", sql],
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def _ensure_test_db(test_async_url: str, base_async_url: str, recreate: bool) -> None:
    """Idempotently create the test DB if it doesn't exist; optionally
    drop+recreate. ``CREATE DATABASE`` cannot run inside a transaction
    block, so we issue it directly via psql against the ``postgres``
    admin DB."""
    test_db_name = urlparse(test_async_url).path.lstrip("/")
    if not test_db_name:
        raise RuntimeError(f"Could not extract DB name from {test_async_url!r}")

    admin_url = _swap_db(_to_psql_url(base_async_url), "postgres")

    if recreate:
        rc, _out, err = _psql(admin_url, f'DROP DATABASE IF EXISTS "{test_db_name}"')
        if rc != 0:
            raise RuntimeError(f"DROP DATABASE failed: {err}")

    # CREATE DATABASE doesn't have IF NOT EXISTS in PostgreSQL, so we
    # branch on a probe.
    probe = f"SELECT 1 FROM pg_database WHERE datname = '{test_db_name}'"
    rc, out, err = _psql(admin_url, probe)
    if rc != 0:
        raise RuntimeError(f"DB existence probe failed: {err}")
    exists = "1" in out
    if not exists:
        rc, _out, err = _psql(admin_url, f'CREATE DATABASE "{test_db_name}"')
        if rc != 0:
            raise RuntimeError(f"CREATE DATABASE failed: {err}")


def _ensure_pgvector(test_async_url: str) -> None:
    """The hafiz schema needs the ``vector`` extension; a fresh DB
    doesn't have it until we ask. Idempotent."""
    rc, _out, err = _psql(_to_psql_url(test_async_url), "CREATE EXTENSION IF NOT EXISTS vector")
    if rc != 0:
        raise RuntimeError(f"CREATE EXTENSION vector failed: {err}")


def _migrate_test_db(test_async_url: str) -> None:
    """Run ``alembic upgrade head`` against the test DB in-process.

    Uses ``hafiz.core.database.create_tables`` which builds an Alembic
    Config explicitly. The conftest has already patched
    ``hafiz.core.config._settings`` and ``load_settings`` so alembic's
    ``env.py`` (which calls ``load_settings()``) honors the test URL.
    """
    import asyncio

    from hafiz.core.database import close_engine, create_tables

    async def _run():
        try:
            await create_tables(test_async_url)
        finally:
            await close_engine()

    asyncio.run(_run())


def _setup_test_db_once() -> str:
    """Create + migrate the test DB. Returns the URL we redirected to.

    Two redirection mechanisms — both needed:

    1. ``HAFIZ_DATABASE__URL`` env var, so subprocess invocations
       (alembic, the CLI runners in tests) pick up the test URL.
    2. Direct injection into ``hafiz.core.config._settings``, because
       the in-process settings loader prefers ``hafiz.toml`` over env
       vars (toml kwargs become "init args" to pydantic-settings,
       which beat env). Without (2), tests running ``get_settings()``
       would still read the production ``hafiz.toml``.
    """
    if os.environ.get("HAFIZ_TEST_DB_DISABLE") == "1":
        return os.environ.get("HAFIZ_DATABASE__URL", _read_base_url())

    base_url = _read_base_url()
    test_url = os.environ.get("HAFIZ_TEST_DB") or _swap_db(
        base_url, urlparse(base_url).path.lstrip("/") + "_test"
    )
    recreate = os.environ.get("HAFIZ_TEST_DB_RECREATE") == "1"

    _ensure_test_db(test_url, base_url, recreate=recreate)
    _ensure_pgvector(test_url)

    # (1) For subprocesses that re-load config from scratch.
    os.environ["HAFIZ_DATABASE__URL"] = test_url

    # (2) For in-process readers — inject the cached singleton AND
    # patch ``load_settings`` so alembic's env.py (which calls
    # ``load_settings()`` directly) also sees the test URL. Without
    # patching the function, env.py would re-read hafiz.toml and the
    # migration would land on the production DB.
    from hafiz.core import config as _config

    settings = _config.load_settings()
    settings.database.url = test_url
    _config._settings = settings  # type: ignore[attr-defined]

    _real_load_settings = _config.load_settings

    def _patched_load_settings():
        s = _real_load_settings()
        s.database.url = test_url
        return s

    _config.load_settings = _patched_load_settings  # type: ignore[assignment]

    _migrate_test_db(test_url)
    return test_url


# Silence the unused-import lint; shlex stays imported for future use
# (planned: invoking psql with arg lists rather than full URL strings).
_ = shlex


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

#: Which backend this session runs against. Read from the environment rather
#: than a pytest CLI option because the redirection below happens at module
#: load — before pytest hands anyone a parsed config object.
_BACKEND = (os.environ.get("HAFIZ_TEST_BACKEND") or "postgresql").strip().lower()
_BACKEND_WAS_NAMED = bool(os.environ.get("HAFIZ_TEST_BACKEND"))

if _BACKEND not in {"postgresql", "sqlite"}:
    raise RuntimeError(
        f"HAFIZ_TEST_BACKEND={_BACKEND!r} is not a backend. Use 'postgresql' or 'sqlite'."
    )


def _setup_sqlite_test_db_once() -> str:
    """Point the session at a throwaway SQLite file and build the schema.

    Redirected the same two ways as the Postgres path — env var for
    subprocesses, patched ``load_settings`` for in-process readers — because
    ``hafiz.toml`` values arrive as pydantic-settings *init args* and beat env
    vars. Patching only the env var would leave every in-process test writing
    to the developer's real Postgres store while the run claimed to be
    exercising SQLite.

    Deliberately allowed to raise. A backend that cannot be set up must stop
    the session, not degrade into a run full of skips.
    """
    import asyncio
    import tempfile

    path = Path(tempfile.mkdtemp(prefix="hafiz-test-sqlite-")) / "hafiz.db"
    url = f"sqlite:///{path}"

    os.environ["HAFIZ_DATABASE__URL"] = url

    from hafiz.core import config as _config

    settings = _config.load_settings()
    settings.database.url = url
    _config._settings = settings  # type: ignore[attr-defined]

    _real_load_settings = _config.load_settings

    def _patched_load_settings():
        s = _real_load_settings()
        s.database.url = url
        return s

    _config.load_settings = _patched_load_settings  # type: ignore[assignment]

    from hafiz.core.database import close_engine, create_tables

    async def _run():
        try:
            await create_tables(url)
        finally:
            await close_engine()

    asyncio.run(_run())
    return url


def _setup_backend() -> str | None:
    if os.environ.get("HAFIZ_TEST_NO_DB") == "1":
        return None
    if _BACKEND == "sqlite":
        return _setup_sqlite_test_db_once()
    return _setup_test_db_once()


# Eagerly evaluated at module load — establishes the test DB before
# any test's imports resolve hafiz.core.config. Skipped in NO_DB mode,
# where the DB-dependent modules have already been pruned from collection.
_TEST_DB_URL = _setup_backend()


# ---------------------------------------------------------------------------
# "A skipped backend is a failed run"
# ---------------------------------------------------------------------------

#: Substrings that mark a skip as "the database wasn't there". Matched against
#: the skip reason, because the skips themselves live in ~15 test modules and
#: rewriting each one would be a far larger change than gating them centrally.
_DB_SKIP_MARKERS = (
    "postgres",
    "not reachable",
    "no live db",
    "no live postgres",
    "database",
    "migration 0007",
    "pgvector",
)

#: Skips that are *deliberately* backend-conditional rather than symptoms of an
#: absent database. A test of SQLite's single write lock has nothing to assert
#: under Postgres MVCC, and must be able to say so without tripping the guard
#: below — which matches on reason text and would otherwise fire on the word
#: "postgres". Exempted by prefix so the exemption is structural: a test opts in
#: explicitly, rather than by wording its reason carefully enough to slip past.
_DB_SKIP_EXEMPT_PREFIXES = ("backend-specific:",)

_db_skips: list[str] = []


def pytest_runtest_logreport(report) -> None:
    """Record any test that skipped because a database was missing."""
    if not (_BACKEND_WAS_NAMED and report.skipped):
        return
    reason = ""
    if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
        reason = str(report.longrepr[2])
    normalized = reason.lower().removeprefix("skipped: ").lstrip()
    if normalized.startswith(_DB_SKIP_EXEMPT_PREFIXES):
        return
    if any(marker in reason.lower() for marker in _DB_SKIP_MARKERS):
        _db_skips.append(f"{report.nodeid} — {reason}")


def pytest_sessionfinish(session, exitstatus) -> None:  # noqa: ARG001
    """Fail the session if the named backend quietly skipped its own tests.

    The whole point of naming a backend is to assert it works. Letting the
    run go green while its tests skipped is the exact failure this matrix
    exists to prevent.
    """
    if not _db_skips:
        return
    session.exitstatus = 1
    print(f"\n\nERROR: {len(_db_skips)} test(s) skipped for a database reason, but")
    print(f"HAFIZ_TEST_BACKEND={_BACKEND!r} was named explicitly — so they had to run.\n")
    for entry in _db_skips[:20]:
        print(f"  - {entry}")
    if len(_db_skips) > 20:
        print(f"  … and {len(_db_skips) - 20} more")
