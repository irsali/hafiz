"""Sovereign error log — every error hafiz hits gets captured here.

Append-only NDJSON at ``~/.cache/hafiz/errors.log``. Intended to feed
two consumers:

  1. **Humans** — ``hafiz errors show <id>`` surfaces the full
     traceback when something cryptic happened during a previous run.
  2. **Agents** — ``hafiz errors list --json`` returns structured
     records (command, exception class, message, suggested action,
     context) so an agent can recognize patterns and propose fixes
     without re-parsing tracebacks.

Design invariants:

- **The logger must never itself crash.** A failure writing the log
  becomes a best-effort stderr warning; the original exception still
  bubbles up through the caller. Losing an error record is better
  than masking the real exception.
- **No secrets.** We record command + argv + traceback + cwd + git
  branch + hafiz version + host fingerprint. We do **not** record
  environment variables, file contents, or arbitrary argument values
  that might embed tokens (same hygiene as annotations).
- **Rotation at write time.** When a fresh append would push the log
  past its caps, we rewrite keeping the newest ``MAX_ENTRIES`` and
  discard older records. FIFO, no journaling. Simple.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback as _traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Caps. Tuned for "debug log that never grows scary":
#   1000 entries × typical line ~1-3 KB ≈ 1-3 MB worst case.
#   Hard byte cap as belt and braces — rotates on either trigger.
MAX_ENTRIES = 1000
MAX_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ErrorRecord:
    """One captured failure. Shape is the durable JSON contract; add
    fields freely, never rename or remove — agents parse this."""

    id: str
    timestamp: str
    command: str
    argv: list[str]
    exception_type: str
    message: str
    traceback: str
    cwd: str
    hafiz_version: str | None
    git_branch: str | None
    git_dirty: bool | None
    host_fingerprint: str | None
    suggested_action: str | None = None
    # Free-form extension point. Recognizers can stash structured hints
    # here (e.g., ``{"missing_module": "scipy"}``) without requiring a
    # schema bump.
    context: dict[str, Any] = field(default_factory=dict)

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# File location
# ---------------------------------------------------------------------------


def log_file_path() -> Path:
    """XDG-compliant log file location. Override with ``XDG_CACHE_HOME``."""
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / "hafiz" / "errors.log"


# ---------------------------------------------------------------------------
# Recognizers → suggested_action
# ---------------------------------------------------------------------------
#
# Each recognizer inspects an exception and returns ``(suggestion, context)``
# when it has high-confidence advice, else ``None``. ``_suggest_action``
# walks the registry in order; first non-None wins.
#
# Conservative by policy: only classes with a well-understood, safe
# remediation get a suggestion. Everything else gets ``None`` and the
# user sees the traceback. Misfiring with a wrong suggestion is worse
# than staying silent.
#
# The error logger MUST stay independently importable — recognizers
# avoid hard imports of optional or heavyweight modules and identify
# foreign exception classes by walking ``__mro__`` and matching the
# fully-qualified class name. That way a corrupted sqlalchemy or
# pydantic install can still be *logged* by hafiz before the user
# sees a traceback.


_Suggestion = tuple[str, dict[str, Any]]


def _exc_inherits_from(exc: BaseException, *fully_qualified: str) -> bool:
    """Return True if any class in ``exc``'s MRO matches one of the given
    ``module.qualname`` strings. Lets recognizers detect foreign
    exception types without importing them."""
    targets = set(fully_qualified)
    for cls in type(exc).__mro__:
        fq = f"{cls.__module__}.{cls.__qualname__}"
        if fq in targets:
            return True
    return False


def _recognize_module_not_found(
    exc: BaseException, *, argv: list[str], traceback_text: str
) -> _Suggestion | None:
    if not isinstance(exc, ModuleNotFoundError):
        return None
    missing = getattr(exc, "name", None) or str(exc)
    declared = _is_declared_runtime_dep(missing)
    ctx = {"missing_module": missing, "is_declared_dep": declared}
    if declared:
        return (
            f"hafiz declares `{missing}` as a runtime dep, but your install "
            f"is missing it. Fix: `pipx inject hafiz {missing}` or "
            f"`pipx reinstall hafiz`.",
            ctx,
        )
    return (
        f"Module `{missing}` is not installed and not a declared hafiz "
        f"dep. If hafiz needs it, file an issue; otherwise install it "
        f"in the environment that runs hafiz.",
        ctx,
    )


# pgvector message fragments are stable Postgres wording (server-side,
# not driver-side), so substring matching here is reasonably durable.
_PGVECTOR_MARKERS = (
    'extension "vector" does not exist',
    'type "vector" does not exist',
)


def _recognize_pgvector_missing(
    exc: BaseException, *, argv: list[str], traceback_text: str
) -> _Suggestion | None:
    """Postgres has no pgvector extension installed in the target DB.
    Surfaces as sqlalchemy ProgrammingError (or asyncpg's own variant)
    with a stable server-side error message."""
    if not _exc_inherits_from(
        exc,
        "sqlalchemy.exc.ProgrammingError",
        "asyncpg.exceptions.UndefinedObjectError",
        "asyncpg.exceptions.UndefinedFileError",
    ):
        return None
    msg = str(exc).lower()
    if not any(marker in msg for marker in _PGVECTOR_MARKERS):
        return None
    return (
        "Your Postgres database is missing the pgvector extension. "
        "Fix: install the `postgresql-NN-pgvector` system package "
        "(or build from https://github.com/pgvector/pgvector), then run "
        "`hafiz init` to create the extension and tables.",
        {"missing_extension": "vector"},
    )


def _recognize_db_connectivity(
    exc: BaseException, *, argv: list[str], traceback_text: str
) -> _Suggestion | None:
    """Sqlalchemy ``OperationalError`` — usually unreachable DB, refused
    connection, bad credentials, or no such database. We don't try to
    classify the sub-cause; we point the user at the diagnose path."""
    if not _exc_inherits_from(exc, "sqlalchemy.exc.OperationalError"):
        return None
    # Skip if the embedded message is a pgvector miss — that recognizer
    # ran first and would have matched. Defensive belt + braces.
    msg_lower = str(exc).lower()
    if any(marker in msg_lower for marker in _PGVECTOR_MARKERS):
        return None
    return (
        "Hafiz couldn't talk to Postgres. Run `hafiz status --diagnose` "
        "to see whether the server is reachable and the configured URL "
        "is correct (check `[database].url` in your hafiz.toml or the "
        "HAFIZ_DATABASE__URL env var).",
        {"db_error_class": type(exc).__name__},
    )


def _recognize_config_validation(
    exc: BaseException, *, argv: list[str], traceback_text: str
) -> _Suggestion | None:
    """Pydantic ``ValidationError`` raised while loading hafiz.toml /
    HAFIZ_* env vars. We require evidence that the failure came from
    our config loader, not from arbitrary downstream code, so we
    don't misfire on unrelated pydantic models."""
    if not _exc_inherits_from(
        exc,
        "pydantic.ValidationError",
        "pydantic_core._pydantic_core.ValidationError",
    ):
        return None
    if "hafiz/core/config.py" not in traceback_text:
        return None
    invalid_keys: list[str] = []
    try:
        errors_method = getattr(exc, "errors", None)
        if callable(errors_method):
            for err in errors_method():
                loc = err.get("loc") if isinstance(err, dict) else None
                if loc:
                    invalid_keys.append(".".join(str(p) for p in loc))
    except Exception:  # noqa: BLE001 — recognizer must never crash
        pass
    return (
        "Your hafiz config didn't validate. Run `hafiz config show` to "
        "see resolved values and per-key sources, then fix the offending "
        "key with `hafiz config set <key> <value>` or correct hafiz.toml. "
        "If a HAFIZ_*__* env var is overriding things, unset it to fall "
        "through to TOML/sticky/default.",
        {"invalid_keys": invalid_keys},
    )


# Walked in order; first match wins. Order matters: pgvector must come
# before db_connectivity (the same OperationalError can carry a
# pgvector message, and the more-specific suggestion should win).
_RECOGNIZERS = (
    _recognize_module_not_found,
    _recognize_pgvector_missing,
    _recognize_db_connectivity,
    _recognize_config_validation,
)


def _suggest_action(
    exc: BaseException, *, argv: list[str], traceback_text: str = ""
) -> tuple[str | None, dict[str, Any]]:
    """Walk the recognizer registry. Returns the first match, or
    ``(None, {})`` when nothing recognized the exception."""
    for recognizer in _RECOGNIZERS:
        try:
            result = recognizer(exc, argv=argv, traceback_text=traceback_text)
        except Exception:  # noqa: BLE001 — a buggy recognizer must not break logging
            logger.warning(
                "Recognizer %s raised while inspecting %s",
                recognizer.__name__,
                type(exc).__name__,
                exc_info=True,
            )
            continue
        if result is not None:
            return result[0], result[1]
    return None, {}


def _is_declared_runtime_dep(module_name: str) -> bool:
    """Check whether ``module_name`` appears in pyproject.toml's
    [project].dependencies. Top-level module name comparison, ignores
    versions and package-vs-module casing for the common Python cases."""
    try:
        pass  # for project root locator
    except Exception:
        return False

    # Walk up from this file to find pyproject.toml. Keeps the check
    # working in both installed and editable modes.
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        cand = parent / "pyproject.toml"
        if cand.is_file():
            try:
                import tomllib

                with open(cand, "rb") as f:
                    data = tomllib.load(f)
                deps = (data.get("project") or {}).get("dependencies") or []
                want = module_name.lower()
                for dep in deps:
                    name = _dep_package_name(dep).lower()
                    # numpy / scipy / networkx are the direct common case;
                    # also tolerate "-" vs "_" where the Python module
                    # name doesn't match the PyPI name exactly.
                    if name == want or name.replace("-", "_") == want:
                        return True
                return False
            except Exception:
                return False
    return False


def _dep_package_name(dep_spec: str) -> str:
    """Extract the PyPI name from a PEP 508-ish dep string like
    ``sqlalchemy[asyncio]>=2.0`` → ``sqlalchemy``."""
    s = dep_spec.strip()
    for sep in ("[", "=", "<", ">", "~", "!", ";", " "):
        idx = s.find(sep)
        if idx != -1:
            s = s[:idx]
    return s.strip()


# ---------------------------------------------------------------------------
# Build + write
# ---------------------------------------------------------------------------


def _hafiz_version() -> str | None:
    try:
        import importlib.metadata as _md

        return _md.version("hafiz")
    except Exception:
        return None


def _git_context() -> tuple[str | None, bool | None]:
    try:
        from hafiz.core.git_context import current_git_context, is_git_repo

        cwd = Path.cwd()
        if not is_git_repo(cwd):
            return None, None
        ctx = current_git_context(cwd)
        return ctx.get("branch"), ctx.get("is_dirty")
    except Exception:
        return None, None


def _host_fingerprint() -> str | None:
    try:
        from hafiz.core.host_probe import probe_host

        return probe_host().fingerprint
    except Exception:
        return None


def build_record(
    exc: BaseException,
    *,
    argv: list[str] | None = None,
) -> ErrorRecord:
    """Construct an :class:`ErrorRecord` from an exception. Does not
    touch disk — call :func:`append` separately to persist."""
    argv = list(argv) if argv is not None else list(sys.argv[1:])
    command = _command_from_argv(argv)
    tb_s = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))

    branch, dirty = _git_context()
    suggestion, ctx = _suggest_action(exc, argv=argv, traceback_text=tb_s)

    return ErrorRecord(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        command=command,
        argv=argv,
        exception_type=type(exc).__name__,
        message=str(exc)[:500] or type(exc).__name__,
        traceback=tb_s,
        cwd=str(Path.cwd()),
        hafiz_version=_hafiz_version(),
        git_branch=branch,
        git_dirty=dirty,
        host_fingerprint=_host_fingerprint(),
        suggested_action=suggestion,
        context=ctx,
    )


def _command_from_argv(argv: list[str]) -> str:
    """Best-effort: join non-flag leading args. ``hafiz config set foo 1``
    → ``config set``; ``hafiz ingest . --project x`` → ``ingest``. Good
    enough for grouping by command class."""
    parts: list[str] = []
    for a in argv:
        if a.startswith("-"):
            break
        parts.append(a)
    return " ".join(parts) or "(unknown)"


def append(record: ErrorRecord) -> bool:
    """Persist ``record``. Rotates the log if either cap is exceeded.
    Returns True on success, False on any write failure (logged to
    stderr as a warning — the caller should not be interrupted)."""
    path = log_file_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.as_jsonable(), ensure_ascii=False) + "\n"
        # Rotate first if we're about to exceed a cap.
        _rotate_if_needed(path, incoming_bytes=len(line.encode("utf-8")))
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        return True
    except OSError as exc:
        logger.warning("Could not append error record at %s: %s", path, exc)
        return False


def _rotate_if_needed(path: Path, *, incoming_bytes: int) -> None:
    """Keep the log under caps. If adding ``incoming_bytes`` would push
    past MAX_BYTES, or the current entry count is at MAX_ENTRIES,
    rewrite the file keeping the newest ``MAX_ENTRIES - 1`` entries so
    the incoming write makes it exactly ``MAX_ENTRIES``."""
    try:
        if not path.is_file():
            return
        size = path.stat().st_size
        # Fast path: well under limits.
        if size + incoming_bytes <= MAX_BYTES:
            # Still check line count for the entries cap.
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) < MAX_ENTRIES:
                return
        else:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        keep = lines[-(MAX_ENTRIES - 1) :] if MAX_ENTRIES > 1 else []
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(keep)
    except OSError as exc:
        logger.warning("Rotation check failed at %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _parse_line(line: str) -> ErrorRecord | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "id" not in data:
        return None
    try:
        return ErrorRecord(
            id=data["id"],
            timestamp=data.get("timestamp", ""),
            command=data.get("command", ""),
            argv=list(data.get("argv", [])),
            exception_type=data.get("exception_type", ""),
            message=data.get("message", ""),
            traceback=data.get("traceback", ""),
            cwd=data.get("cwd", ""),
            hafiz_version=data.get("hafiz_version"),
            git_branch=data.get("git_branch"),
            git_dirty=data.get("git_dirty"),
            host_fingerprint=data.get("host_fingerprint"),
            suggested_action=data.get("suggested_action"),
            context=dict(data.get("context", {}) or {}),
        )
    except (KeyError, TypeError):
        return None


def _load_all() -> list[ErrorRecord]:
    path = log_file_path()
    if not path.is_file():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    out: list[ErrorRecord] = []
    for line in lines:
        rec = _parse_line(line.strip())
        if rec is not None:
            out.append(rec)
    return out


def tail(
    *,
    since: str | None = None,
    limit: int | None = None,
) -> list[ErrorRecord]:
    """Return records newest-first, optionally filtered by age.

    ``since`` is a shorthand duration like ``1h``, ``30m``, ``2d`` —
    records with ``timestamp`` older than *now - since* are dropped.
    ``limit`` caps the returned count after age filtering.
    """
    records = _load_all()
    records.reverse()  # newest first
    if since is not None:
        cutoff = _resolve_cutoff(since)
        if cutoff is not None:
            records = [r for r in records if r.timestamp >= cutoff]
    if limit is not None and limit > 0:
        records = records[:limit]
    return records


def count_recent(*, since: str) -> int:
    return len(tail(since=since))


def group_by_exception_type(records: list[ErrorRecord]) -> list[dict[str, Any]]:
    """Group ``records`` by ``exception_type``.

    Records are expected newest-first (the shape ``tail`` returns). The
    first record per type therefore drives ``most_recent_*`` and
    ``sample_*`` fields. Groups come out sorted by ``count`` desc, then
    by most-recent timestamp desc — agents glancing at index 0 see the
    most-frequent, freshest class.

    Sample fields exist so an agent can act without a second
    ``errors show`` round-trip; they're truncated to keep the response
    small.
    """
    by_type: dict[str, list[ErrorRecord]] = {}
    for r in records:
        by_type.setdefault(r.exception_type, []).append(r)

    groups: list[dict[str, Any]] = []
    for exc_type, rs in by_type.items():
        head = rs[0]
        sample_msg = head.message
        if len(sample_msg) > 200:
            sample_msg = sample_msg[:200] + "…"
        groups.append(
            {
                "exception_type": exc_type,
                "count": len(rs),
                "with_suggestions": sum(1 for r in rs if r.suggested_action),
                "most_recent_id": head.id,
                "most_recent_timestamp": head.timestamp,
                "sample_command": head.command,
                "sample_message": sample_msg,
            }
        )

    groups.sort(
        key=lambda g: (g["count"], g["most_recent_timestamp"]),
        reverse=True,
    )
    return groups


def get(record_id: str) -> ErrorRecord | None:
    """Lookup by id. Supports full-id and unique-prefix matches."""
    rid = record_id.strip()
    if not rid:
        return None
    records = _load_all()
    # exact first
    for r in records:
        if r.id == rid:
            return r
    # unique prefix
    hits = [r for r in records if r.id.startswith(rid)]
    if len(hits) == 1:
        return hits[0]
    return None


def clear() -> int:
    """Remove the log file. Returns the number of records discarded."""
    path = log_file_path()
    if not path.is_file():
        return 0
    count = len(_load_all())
    try:
        path.unlink()
    except OSError as exc:
        logger.warning("Could not clear error log at %s: %s", path, exc)
        return 0
    return count


# ---------------------------------------------------------------------------
# Duration parsing (mirrors annotations --since/--expires-in shape)
# ---------------------------------------------------------------------------


def _resolve_cutoff(since: str) -> str | None:
    """Convert a relative duration like ``1d`` into an ISO timestamp
    cutoff; compared lexicographically since our timestamps are
    ISO-8601 with fixed format. Returns None on parse failure — the
    caller treats that as "no filter", which beats silently hiding
    records."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 86400 * 7}
    s = since.strip().lower()
    if not s:
        return None
    try:
        unit = s[-1]
        if unit not in units:
            return None
        value = int(s[:-1])
    except (ValueError, IndexError):
        return None
    seconds = value * units[unit]
    from datetime import timedelta

    cutoff = datetime.now(UTC) - timedelta(seconds=seconds)
    return cutoff.isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Convenience entry point used by the top-level handler
# ---------------------------------------------------------------------------


def log_exception(exc: BaseException, *, argv: list[str] | None = None) -> ErrorRecord:
    """Build and persist an error record for ``exc``. Always returns the
    record so the caller can echo the id in the user-facing message,
    even if the write itself failed."""
    record = build_record(exc, argv=argv)
    append(record)
    return record
