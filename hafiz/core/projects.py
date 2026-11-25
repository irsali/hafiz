"""One project, one name — resolving what a caller typed to the name on file.

A 2026-11 audit of a live store found ten products filed under two or three
spellings each (``Ledger API`` / ``ledger-api``, ``Shop Front_v2`` /
``Shop_Front_v2`` / ``shop-front-v2``). The code index used the repo folder
name; agents wrote notes under the ids they knew. Every project filter is an
exact match, so a query under one spelling silently missed everything filed
under the others — and a misspelt ``--project`` returned an empty answer that
looked exactly like "nothing is known".

Spelling alone cannot fix this. The same store held ``Billing Portal`` (a
retired repo) and ``billing-portal`` (its replacement): one key apart, two
different codebases. So the rules are:

* **Synonyms are declared, not guessed.** ``project_names`` maps alias → main
  name and marks projects retired. It lives in the database, next to the data
  it describes, so the CLI, the daemon and MCP all read the same list
  regardless of which ``hafiz.toml`` a working directory happens to pick up.
* **Reads correct confidently, and refuse when confused.** Exact name, then
  declared alias, then the same name ignoring case/space/``_``/``-``, then a
  close spelling. Several candidates left → prefer the one not retired; still
  several → :class:`UnknownProjectError` listing them.
* **Writes never guess from a close spelling.** A brand-new name that looks
  like an existing one is refused unless the caller says it is new — that is
  how the variants above were created.

Resolution happens in core because the daemon and MCP call core directly.
"""

from __future__ import annotations

import difflib
import logging
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

#: A close spelling is offered at this similarity (0-1, on normalized keys) …
SUGGEST_RATIO = 0.75
#: … and used without asking at this one, when nothing else is as close.
AUTO_RATIO = 0.9
#: How long the name table is trusted before a re-read. A miss re-reads anyway.
_TTL_SECONDS = 60.0

_SEPARATORS = re.compile(r"[\s_\-]+")


def project_key(name: str) -> str:
    """What two spellings of one name have in common: ``Shop_Front v2`` → ``shop-front-v2``."""
    return _SEPARATORS.sub("-", name.casefold()).strip("-")


class UnknownProjectError(ValueError):
    """A project name that matched nothing, or matched more than one project.

    A ``ValueError`` so the commands that already turn bad input into exit 2
    keep doing so; ``candidates`` lets a caller offer the fix.
    """

    def __init__(self, message: str, *, name: str, candidates: list[str]):
        super().__init__(message)
        self.name = name
        self.candidates = candidates


@dataclass
class ProjectNames:
    """Every name the store knows, and what each one means."""

    #: Name → rows filed under it, across notes, index, transcripts and sessions.
    used: dict[str, int] = field(default_factory=dict)
    #: Declared synonym → main name.
    aliases: dict[str, str] = field(default_factory=dict)
    #: Main names marked retired.
    retired: set[str] = field(default_factory=set)

    def main(self, name: str) -> str:
        return self.aliases.get(name, name)

    def spellings(self) -> set[str]:
        return set(self.used) | set(self.aliases) | set(self.aliases.values())

    def label(self, name: str) -> str:
        return f"{name} (retired)" if name in self.retired else name


#: Set by the daemon around each request, so a correction reaches the person
#: who typed the name instead of the daemon's log. Unset → log it here.
notices: ContextVar[list[str] | None] = ContextVar("hafiz_project_notices", default=None)


def _tell(message: str) -> None:
    collected = notices.get()
    if collected is not None:
        collected.append(message)
    else:
        logger.warning(message)


# ── Loading ─────────────────────────────────────────────────────────────

_cache: tuple[float, ProjectNames] | None = None


def invalidate() -> None:
    """Forget the cached name table (after a write that adds or renames one)."""
    global _cache
    _cache = None


async def load_names(*, fresh: bool = False) -> ProjectNames:
    """The name table, cached for a minute. ``fresh`` forces a re-read."""
    global _cache
    if not fresh and _cache and time.monotonic() - _cache[0] < _TTL_SECONDS:
        return _cache[1]
    names = await _read_names()
    _cache = (time.monotonic(), names)
    return names


async def _read_names() -> ProjectNames:
    from sqlalchemy import text

    from hafiz.core.database import get_session_factory

    out = ProjectNames()
    factory = get_session_factory()
    async with factory() as session:
        rows = (
            await session.execute(
                text(
                    "SELECT project, count(*) FROM annotations WHERE project IS NOT NULL "
                    "GROUP BY project "
                    "UNION ALL SELECT project, count(*) FROM files WHERE project IS NOT NULL "
                    "GROUP BY project "
                    "UNION ALL SELECT scope_value, count(*) FROM communications "
                    "WHERE scope_kind = 'project' AND scope_value IS NOT NULL GROUP BY scope_value "
                    "UNION ALL SELECT scope_value, count(*) FROM sessions "
                    "WHERE scope_kind = 'project' AND scope_value IS NOT NULL GROUP BY scope_value"
                )
            )
        ).all()
        for name, n in rows:
            out.used[name] = out.used.get(name, 0) + int(n or 0)
        try:
            declared = (
                await session.execute(text("SELECT name, project, retired_at FROM project_names"))
            ).all()
        except Exception as exc:  # noqa: BLE001 — a store not yet migrated has no synonyms
            logger.debug("project_names unavailable (%s)", exc)
            declared = []
    for name, project, retired_at in declared:
        if name != project:
            out.aliases[name] = project
        elif retired_at is not None:
            out.retired.add(project)
    return out


# ── Resolving ───────────────────────────────────────────────────────────


def _key_matches(name: str, names: ProjectNames) -> set[str]:
    key = project_key(name)
    return {names.main(s) for s in names.spellings() if project_key(s) == key}


_DIGITS = re.compile(r"\d+")


def _versions_differ(a: str, b: str) -> bool:
    """``shop-front-v3`` vs ``shop-front-v4``: same shape, different number —
    a sibling project, never a typo."""
    return a != b and _DIGITS.sub("#", a) == _DIGITS.sub("#", b)


def _same_numbers(a: str, b: str) -> bool:
    """Numbers name versions, so a typo never changes them: ``shop-front-v4``
    is not a misspelling of ``Shop Front_v2_4``, however close the letters."""
    return _DIGITS.findall(project_key(a)) == _DIGITS.findall(project_key(b))


def _close(name: str, names: ProjectNames, cutoff: float) -> list[tuple[float, str]]:
    """Main names whose spelling is close to ``name``, best first. Names that
    differ only in a number are siblings, not typos, and never count as close."""
    key = project_key(name)
    best: dict[str, float] = {}
    for spelling in names.spellings():
        other = project_key(spelling)
        if _versions_differ(key, other):
            continue
        ratio = difflib.SequenceMatcher(None, key, other).ratio()
        # "ledger" → "ledger-api": a whole-word prefix is a strong hint even
        # when the ratio is modest.
        if other.startswith(key + "-") or key.startswith(other + "-"):
            ratio = max(ratio, SUGGEST_RATIO)  # a hint to offer, never enough to act on
        if ratio >= cutoff:
            main = names.main(spelling)
            best[main] = max(best.get(main, 0.0), ratio)
    return sorted(((r, m) for m, r in best.items()), key=lambda t: (-t[0], t[1]))


def _prefer_active(candidates: set[str], names: ProjectNames) -> set[str]:
    active = {c for c in candidates if c not in names.retired}
    return active if len(active) == 1 else candidates


def _ambiguous(name: str, candidates: set[str], names: ProjectNames) -> UnknownProjectError:
    ordered = sorted(candidates, key=lambda c: (c in names.retired, -names.used.get(c, 0), c))
    listed = ", ".join(names.label(c) for c in ordered)
    return UnknownProjectError(
        f"project {name!r} matches more than one project: {listed}. Use the exact name.",
        name=name,
        candidates=ordered,
    )


async def resolve_project(name: str | None) -> str | None:
    """The project a *read* means by ``name``. ``None``/blank → no filter.

    Raises :class:`UnknownProjectError` when nothing fits or several do —
    an empty answer for a misspelt name is indistinguishable from "nothing
    known", which is the failure this module exists to remove.
    """
    cleaned = (name or "").strip()
    if not cleaned:
        return None
    names = await load_names()
    found = _match(cleaned, names)
    if found is None:
        names = await load_names(fresh=True)  # written since the cache filled?
        found = _match(cleaned, names)
    if found is not None:
        return found

    close = _close(cleaned, names, SUGGEST_RATIO)
    if close:
        top_ratio, top = close[0]
        runner_up = close[1][0] if len(close) > 1 else 0.0
        if (
            top_ratio >= AUTO_RATIO
            and top_ratio - runner_up >= 0.05
            and _same_numbers(cleaned, top)
        ):
            _tell(f"hafiz: no project {cleaned!r}; using {top!r} (closest name)")
            return top
        raise _unknown(cleaned, [m for _, m in close], names)
    raise _unknown(cleaned, [m for _, m in _close(cleaned, names, 0.5)], names)


def _match(name: str, names: ProjectNames) -> str | None:
    """Exact, declared, or same-key match — the corrections needing no guess."""
    if name in names.aliases:
        return names.aliases[name]
    if name in names.used or name in names.aliases.values():
        return name
    candidates = _key_matches(name, names)
    if not candidates:
        return None
    candidates = _prefer_active(candidates, names)
    if len(candidates) == 1:
        return next(iter(candidates))
    raise _ambiguous(name, candidates, names)


def _unknown(name: str, suggestions: list[str], names: ProjectNames) -> UnknownProjectError:
    hint = (
        f" Did you mean {', '.join(repr(names.label(s)) for s in suggestions[:5])}?"
        if suggestions
        else " Run `hafiz projects` to list them."
    )
    return UnknownProjectError(
        f"no project named {name!r}.{hint}", name=name, candidates=suggestions[:5]
    )


async def resolve_projects(names: str | list[str] | None) -> str | list[str] | None:
    """:func:`resolve_project` over the ``str | list`` shape some filters take."""
    if isinstance(names, list):
        return [p for p in [await resolve_project(n) for n in names] if p]
    return await resolve_project(names)


async def canonical_project(name: str | None, *, new_project: bool = False) -> str | None:
    """The name a *write* should file under.

    Exact, declared and same-key matches map to the main name. A name that
    matches nothing is a new project — unless it is close to an existing one,
    which is how spelling variants were born; then this refuses, and
    ``new_project=True`` is how a caller says "yes, really a different one".
    """
    cleaned = (name or "").strip()
    if not cleaned:
        return None
    names = await load_names()
    found = _match(cleaned, names)
    if found is None:
        names = await load_names(fresh=True)
        found = _match(cleaned, names)
    if found is not None or new_project:
        if found is None:
            invalidate()
        return found or cleaned
    # Only a near-certain typo blocks a write: a looser bar would refuse
    # legitimate siblings ("weather-app" next to "weather-app-trunk").
    close = [m for _, m in _close(cleaned, names, AUTO_RATIO) if _same_numbers(cleaned, m)]
    if close:
        listed = ", ".join(repr(names.label(c)) for c in close[:5])
        raise UnknownProjectError(
            f"no project named {cleaned!r}, but it looks like {listed}. Use that name, "
            "or pass --new-project if this really is a different project.",
            name=cleaned,
            candidates=close[:5],
        )
    invalidate()
    return cleaned


async def project_from_cwd(cwd: Path | str | None = None) -> str | None:
    """The project a working directory belongs to, for filling in a write.

    Whichever is closer to ``cwd`` names it: the indexed project whose tree
    holds it, or the git repo's folder (mapped through declared synonyms). A
    repo nested inside a broadly indexed folder is its own project. ``None``
    outside both — no project beats a guessed one.
    """
    from hafiz.core.store import indexed_root_per_project, project_for_path

    here = Path(cwd or Path.cwd()).expanduser().resolve()
    try:
        roots = await indexed_root_per_project()
        indexed = await project_for_path(here, roots=roots)
    except Exception:  # noqa: BLE001 — a fill-in must never fail the write
        roots, indexed = {}, None
    repo = _git_root(here)
    names = await load_names()
    if indexed and (repo is None or len(Path(roots[indexed]).resolve().parts) >= len(repo.parts)):
        return names.main(indexed)
    if repo is None:
        return None
    try:
        return _match(repo.name, names) or repo.name
    except UnknownProjectError:
        return None  # the folder name fits several projects: leave it unset


def _git_root(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


# ── Managing the name table ─────────────────────────────────────────────


async def list_projects() -> dict:
    """Every name on file with its row count, synonyms, retirement, and
    undeclared look-alikes (same key, different main name) to review."""
    names = await load_names(fresh=True)
    mains = sorted(
        {names.main(s) for s in names.spellings()},
        key=lambda m: project_key(m),
    )
    by_key: dict[str, set[str]] = {}
    for m in mains:
        by_key.setdefault(project_key(m), set()).add(m)
    alias_of: dict[str, list[str]] = {}
    for alias, main in names.aliases.items():
        alias_of.setdefault(main, []).append(alias)
    return {
        "projects": [
            {
                "name": m,
                "rows": names.used.get(m, 0),
                "retired": m in names.retired,
                "aliases": sorted(alias_of.get(m, [])),
                "unmerged_alias_rows": sum(names.used.get(a, 0) for a in alias_of.get(m, [])),
            }
            for m in mains
        ],
        "look_alikes": [sorted(group) for group in by_key.values() if len(group) > 1],
    }


async def add_alias(alias: str, project: str) -> dict:
    """Declare ``alias`` a synonym of ``project``. Rows stay where they are
    until :func:`merge_aliases` moves them."""
    alias, project = alias.strip(), project.strip()
    if not alias or not project or alias == project:
        raise ValueError("an alias needs two different, non-blank names")
    names = await load_names(fresh=True)
    if alias in names.aliases.values():
        raise ValueError(f"{alias!r} is itself the main name of other synonyms")
    project = names.main(project)
    await _upsert(alias, project)
    await _upsert(project, project)
    invalidate()
    return {"alias": alias, "project": project}


async def remove_alias(alias: str) -> dict:
    from sqlalchemy import text

    from hafiz.core.database import get_session_factory

    async with get_session_factory()() as session:
        result = await session.execute(
            text("DELETE FROM project_names WHERE name = :n AND project <> :n"), {"n": alias}
        )
        await session.commit()
    invalidate()
    return {"alias": alias, "removed": bool(result.rowcount)}


async def set_retired(project: str, retired: bool = True) -> dict:
    from sqlalchemy import text

    from hafiz.core.database import get_session_factory

    names = await load_names(fresh=True)
    project = names.main(project.strip())
    if project not in names.spellings():
        raise _unknown(project, [m for _, m in _close(project, names, 0.5)], names)
    await _upsert(project, project)
    async with get_session_factory()() as session:
        await session.execute(
            text("UPDATE project_names SET retired_at = :at WHERE name = :n"),
            {"at": datetime.now(UTC) if retired else None, "n": project},
        )
        await session.commit()
    invalidate()
    return {"project": project, "retired": retired}


async def _upsert(name: str, project: str) -> None:
    from sqlalchemy import select

    from hafiz.core.database import ProjectName, get_session_factory

    async with get_session_factory()() as session:
        row = (
            await session.execute(select(ProjectName).where(ProjectName.name == name))
        ).scalar_one_or_none()
        if row is None:
            session.add(ProjectName(name=name, project=project))
        else:
            row.project = project
        await session.commit()


#: Where a project name is stored, and the column holding it. The code index
#: (files, commits) is deliberately absent: renaming it would orphan the git
#: hooks pinned to the old name, and an alias already routes reads to it.
_RENAMEABLE = (
    ("annotations", "project", None),
    ("communications", "scope_value", "scope_kind = 'project'"),
    ("sessions", "scope_value", "scope_kind = 'project'"),
)


async def merge_aliases(*, apply: bool = False) -> dict:
    """Move rows filed under a declared synonym to its main name.

    Preview unless ``apply``. Only declared synonyms move — never a spelling
    that merely looks alike. A synonym that owns indexed files is reported and
    left alone (see :data:`_RENAMEABLE`).
    """
    from sqlalchemy import text

    from hafiz.core.database import get_session_factory

    names = await load_names(fresh=True)
    moves: list[dict] = []
    skipped: list[dict] = []
    async with get_session_factory()() as session:
        for alias, main in sorted(names.aliases.items()):
            indexed = (
                await session.execute(
                    text("SELECT count(*) FROM files WHERE project = :a"), {"a": alias}
                )
            ).scalar() or 0
            if indexed:
                skipped.append({"alias": alias, "project": main, "reason": "owns indexed files"})
                continue
            counts = {}
            for table, column, scope in _RENAMEABLE:
                where = f"{column} = :a" + (f" AND {scope}" if scope else "")
                n = (
                    await session.execute(
                        text(f"SELECT count(*) FROM {table} WHERE {where}"), {"a": alias}
                    )
                ).scalar() or 0
                if n and apply:
                    await session.execute(
                        text(f"UPDATE {table} SET {column} = :m WHERE {where}"),
                        {"a": alias, "m": main},
                    )
                if n:
                    counts[table] = n
            if counts:
                moves.append({"alias": alias, "project": main, "rows": counts})
        if apply:
            await session.commit()
    if apply:
        invalidate()
    return {"applied": apply, "moves": moves, "skipped": skipped}
