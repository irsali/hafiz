"""Vector similarity search over the structural-grounding schema.

Queries hit the ``embeddings`` table and join back to
``unit_revisions → units → files`` for context. Only current revisions
(``superseded_at IS NULL``) are searched — old bodies stay in the DB for
history but don't pollute retrieval.
"""

from __future__ import annotations

import re

from sqlalchemy import func, not_, or_, select

from hafiz.core import telemetry
from hafiz.core.database import (
    Embedding,
    File,
    Unit,
    UnitRevision,
    get_session_factory,
)
from hafiz.core.dialect import cosine_distance, similarity
from hafiz.core.embeddings import embed_query
from hafiz.core.results import SearchResult  # re-exported


def _normalize_domains(domains: list[str] | None) -> list[str]:
    """Lowercase, strip, and dedupe a list of domain names.

    Empty/None input → []. Each entry is validated as a single
    dotless token (raises ValueError otherwise) — domains are the
    prefix of ``kind`` before the dot, never a full ``kind``.
    """
    if not domains:
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in domains:
        d = (raw or "").strip().lower()
        if not d:
            continue
        if "." in d:
            raise ValueError(
                f"Domain {raw!r} must be a single token (e.g. 'code', "
                "'doc') — use --type for exact kinds like 'code.function'."
            )
        if d not in seen:
            seen.add(d)
            cleaned.append(d)
    return cleaned


def _validate_domain_filters(include: list[str], exclude: list[str]) -> None:
    """Raise if include and exclude share a domain."""
    overlap = set(include) & set(exclude)
    if overlap:
        raise ValueError(f"--include-domain and --exclude-domain overlap: {sorted(overlap)}")


class EmptyQueryError(ValueError):
    """Raised when a retrieval call gets a blank query.

    An empty query embeds to a near-zero vector, against which empty
    documents (blank ``.scss`` files, whitespace-only headings) score a
    perfect 1.0. That turns a caller's broken variable interpolation into
    confidently-ranked garbage instead of an error — a silent-failure
    amplifier. Guarded in core, not just the CLI, so the daemon and any
    library caller hit it too.

    Blank *annotation content* is the write-side of the same defect and raises
    the same error: such a row would score near-perfectly against every future
    query, becoming a permanent noise magnet in recall.
    """

    def __init__(self, what: str = "query", *, hint: str = "nothing to search for"):
        super().__init__(
            f"empty {what} — {hint}. "
            "Check that the caller's variable interpolation produced a value."
        )


def require_query(
    text: str | None, *, what: str = "query", hint: str = "nothing to search for"
) -> str:
    """Return ``text`` stripped, or raise :class:`EmptyQueryError` if blank."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise EmptyQueryError(what, hint=hint)
    return cleaned


# Agent harnesses feed text into the conversation that the user never typed, and
# a per-prompt recall hook forwards it verbatim. Two shapes, handled differently:
#
# * A *preamble* is context glued to the front of a real prompt ("The user opened
#   file X…"). The prompt behind it is genuine, so the block is stripped and the
#   rest searched.
# * A *message* is the whole turn — a sub-agent's final report arriving as a
#   task notification. It is an answer, not a question: across 468 recorded
#   recalls, 88% came back empty, and they were the single largest source of
#   "unanswered" noise in the retrieval log. Searching it is skipped outright.
#
# Only *leading* tags count, and only these names: text that merely mentions a
# tag, or starts with an unknown one, is searched as-is.
_HARNESS_PREAMBLES = frozenset({"ide_opened_file", "ide_selection", "system-reminder"})
_HARNESS_MESSAGES = frozenset({"task-notification"})
_LEADING_BLOCK = re.compile(r"<([a-z][\w-]*)>.*?</\1>\s*", re.DOTALL)
_LEADING_TAG = re.compile(r"<([a-z][\w-]*)>")


def strip_harness_wrappers(text: str) -> tuple[str, str | None]:
    """Return ``(query, skipped)`` for a stripped, non-blank ``text``.

    ``skipped`` names the harness wrapper (``"<task-notification>"``) when there
    is nothing of the user's left to search; ``query`` is then ``""``.
    """
    rest, last = text, None
    while (m := _LEADING_BLOCK.match(rest)) and m.group(1) in _HARNESS_PREAMBLES:
        rest, last = rest[m.end() :], m.group(1)
    rest = rest.strip()
    m = _LEADING_TAG.match(rest)
    if m and m.group(1) in _HARNESS_MESSAGES:
        return "", f"<{m.group(1)}>"
    if not rest:
        return "", f"<{last}>"  # the prompt was nothing but preamble
    return rest, None


async def prepare_query(text: str | None, *, telemetry_command: str | None) -> str | None:
    """:func:`require_query`, then drop harness wrappers.

    Returns the text to embed, or ``None`` when the caller should return an
    empty result set. A skip is ordinary — hooks see it on every sub-agent
    completion — so it is not an error, and it is logged as a one-line marker
    rather than the full message, which the retrieval log has no use for.
    """
    query, skipped = strip_harness_wrappers(require_query(text))
    if skipped and telemetry_command:
        await telemetry.record_skip(command=telemetry_command, wrapper=skipped)
    return None if skipped else query


# How far to over-fetch when collapsing duplicates, so the caller still gets
# ``limit`` *distinct* rows. The markdown parser attaches each paragraph to every
# ancestor heading, so one paragraph can be indexed once per heading level.
# Measured on a 34k-unit doc corpus: 7,490 paragraphs appear once, 3,420 twice,
# 4,926 three times, with a thin tail to eight. 4 covers the mass; the tail costs
# a slightly short result set, never a wrong one.
_DEDUP_OVERFETCH = 4


async def vector_search(
    query: str,
    *,
    limit: int = 10,
    project: str | list[str] | None = None,
    kind: str | None = None,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    similarity_threshold: float = 0.0,
    dedup: bool = True,
    telemetry_command: str | None = telemetry.QUERY,
) -> list[SearchResult]:
    """Search embeddings by cosine similarity and return enriched results.

    Args:
        query: Search query text (will be embedded).
        limit: Maximum results.
        project: Filter by project name (str), multiple projects (list),
            or None for no project filter.
        kind: Filter by unit kind (e.g. ``"code.function"``,
            ``"doc.heading"``). Matches on ``units.kind``.
        include_domains: Restrict results to units whose ``kind`` starts
            with one of these domain prefixes (e.g. ``["code"]``,
            ``["doc", "chat"]``). Domains are the part of ``kind`` before
            the first dot.
        exclude_domains: Drop results whose ``kind`` starts with any of
            these domain prefixes. Mutually exclusive with the same
            domain appearing in ``include_domains``.
        similarity_threshold: Drop results below this cosine similarity.
            Applied in SQL, *before* the ``limit`` is taken, so the caller
            gets the top ``limit`` results that clear the floor rather than
            the survivors of the top ``limit`` overall.
        dedup: Collapse byte-identical content within the same file, keeping
            the copy with the deepest heading path. On by default: the same
            bytes repeated in one result set is never what a caller wants,
            and it was costing ~36% of every doc/code result set.
        telemetry_command: Label for this search in the ``retrievals`` log.
            Defaults to ``"query"`` so a new caller is recorded without having
            to know telemetry exists; pass ``None`` to record nothing (internal
            fan-out that isn't a caller-visible search).

    Returns:
        List of SearchResult ordered by similarity (best first).

    Raises:
        EmptyQueryError: if ``query`` is blank.
    """
    inc = _normalize_domains(include_domains)
    exc = _normalize_domains(exclude_domains)
    _validate_domain_filters(inc, exc)
    query = await prepare_query(query, telemetry_command=telemetry_command)
    if query is not None:
        from hafiz.core.projects import resolve_projects

        project = await resolve_projects(project)
    if query is None:
        return []
    query_embedding = await embed_query(query)

    similarity_expr = similarity(Embedding.embedding, query_embedding)
    similarity_label = similarity_expr.label("similarity")

    stmt = (
        select(Embedding, UnitRevision, Unit, File, similarity_label)
        .join(UnitRevision, UnitRevision.id == Embedding.unit_revision_id)
        .join(Unit, Unit.id == UnitRevision.unit_id)
        .join(File, File.id == Unit.file_id)
        .where(Embedding.embedding.isnot(None))
        .where(UnitRevision.superseded_at.is_(None))
        .where(Unit.valid_until.is_(None))
        .where(File.valid_until.is_(None))
        .order_by(cosine_distance(Embedding.embedding, query_embedding))
        # Over-fetch so dedup happens *before* the cap, not after — trimming
        # first would spend the caller's limit on copies and then delete them.
        .limit(limit * _DEDUP_OVERFETCH if dedup else limit)
    )
    if similarity_threshold > 0:
        stmt = stmt.where(similarity_expr >= similarity_threshold)

    if isinstance(project, list):
        stmt = stmt.where(File.project.in_(project))
    elif project:
        stmt = stmt.where(File.project == project)
    if kind:
        stmt = stmt.where(Unit.kind == kind)
    if inc:
        stmt = stmt.where(or_(*(Unit.kind.like(f"{d}.%") for d in inc)))
    if exc:
        stmt = stmt.where(not_(or_(*(Unit.kind.like(f"{d}.%") for d in exc))))

    session_factory = get_session_factory()
    async with session_factory() as session:
        rows = (await session.execute(stmt)).all()

    # (file, content) → the row we're keeping for it. Identical content embeds
    # identically, so duplicates tie on score and collapsing can't reorder
    # anything; the only choice is which name to keep, and the deepest heading
    # path is the most informative ("X > Integrations > Geolocation Services"
    # beats "X"). Keyed on the file, so the same text in two files stays twice —
    # those are genuinely different sources.
    at: dict[tuple[str, str], int] = {}
    results: list[SearchResult] = []
    for emb, rev, unit, file, sim in rows:
        sim_f = float(sim)
        if sim_f < similarity_threshold:
            continue
        result = SearchResult(
            id=str(emb.id),
            unit_id=str(unit.id),
            unit_name=unit.name,
            kind=unit.kind,
            content=emb.content,
            source_file=file.path,
            line_start=rev.line_start,
            line_end=rev.line_end,
            language=file.language,
            project=file.project,
            part_index=emb.part_index,
            score=round(sim_f, 4),
        )
        if not dedup:
            results.append(result)
            continue

        key = (file.path, emb.content_hash)
        index = at.get(key)
        if index is None:
            at[key] = len(results)
            results.append(result)
        elif len(result.unit_name or "") > len(results[index].unit_name or ""):
            results[index] = result

    final = results[:limit] if dedup else results

    if telemetry_command:
        await telemetry.record_retrieval(
            command=telemetry_command,
            query=query,
            result_ids=[r.id for r in final],
            top_score=final[0].score if final else None,
            filters={
                "project": project,
                "kind": kind,
                "include_domains": inc or None,
                "exclude_domains": exc or None,
                "limit": limit,
                "min_score": similarity_threshold or None,
            },
        )

    return final


async def search_with_staleness(query: str, **kwargs) -> tuple[list[SearchResult], dict]:
    """:func:`vector_search`, plus freshness for the projects the hits came from.

    Scoped to those projects, not the whole store, so it costs a few ms rather
    than ~40. Without it a caller cannot tell a current result from a
    28-commit-stale one, and the only safe policy is to distrust the whole
    index (which a real integrator adopted). One function so the daemon and
    the in-process fallback cannot disagree.
    """
    from hafiz.core.freshness import index_staleness, stale_projects

    results = await vector_search(query, **kwargs)
    try:
        staleness = stale_projects(
            await index_staleness(sorted({r.project for r in results if r.project}))
        )
    except Exception:  # noqa: BLE001 — a git hiccup must not fail a search
        staleness = {}
    return results, staleness


async def count_embeddings(project: str | None = None) -> int:
    """Count current embedding rows, optionally filtered by project."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(func.count())
            .select_from(Embedding)
            .join(UnitRevision, UnitRevision.id == Embedding.unit_revision_id)
            .join(Unit, Unit.id == UnitRevision.unit_id)
            .join(File, File.id == Unit.file_id)
            .where(UnitRevision.superseded_at.is_(None))
            .where(Unit.valid_until.is_(None))
            .where(File.valid_until.is_(None))
        )
        if project:
            stmt = stmt.where(File.project == project)
        return (await session.execute(stmt)).scalar() or 0
