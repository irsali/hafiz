"""Annotation storage and retrieval with vector similarity search.

The "wisdom layer" — decisions, facts, learnings, patterns, warnings, notes.
Annotations may optionally link to a unit (`unit_id`) so they survive body
changes across revisions, or float free as project-level or session-level
knowledge.

Phase 5 adds **polymorphic ``derived_from``**: an annotation may cite
other annotations (knowledge layer) OR communication_messages (source
layer) OR sessions / communications. The link is recorded in the
``annotation_targets`` pivot with ``relation='derived_from'``. The
legacy ``metadata.derived_from`` list is also preserved during the
transition for back-compat with existing readers.

This module replaces the old ``observations.py``. The schema renamed
`observations` → `annotations` and `obs_type` → `kind` as part of the
structural-grounding work (see workitems/done/structural-grounding.md).
The CLI verb stays `hafiz observe` — that's a user-facing name, not a
model reference.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select

from hafiz.core.database import (
    Annotation,
    AnnotationTarget,
    Communication,
    CommunicationMessage,
    get_session_factory,
)
from hafiz.core.database import (
    Session as SessionRow,
)
from hafiz.core.embeddings import embed_query
from hafiz.core.git_context import current_git_context


async def _classify_target_kind(target_uuid: uuid.UUID) -> str | None:
    """Return ``'annotation'|'message'|'communication'|'session'`` for
    a uuid, or None if no matching row exists. Used to decide what
    ``annotation_targets.target_kind`` to write.
    """
    factory = get_session_factory()
    async with factory() as s:
        if await s.get(Annotation, target_uuid):
            return "annotation"
        if await s.get(CommunicationMessage, target_uuid):
            return "message"
        if await s.get(Communication, target_uuid):
            return "communication"
        if await s.get(SessionRow, target_uuid):
            return "session"
    return None


async def write_derived_from_links(annotation_id: uuid.UUID, derived_from: list[str]) -> list[dict]:
    """Insert annotation_targets rows for each derived_from id.

    Returns one summary dict per id describing how it was classified.
    Unknown uuids are skipped (recorded as ``target_kind=None`` with a
    note) — write-time integrity matters less here than not blocking
    the annotation write itself, since lineage is best-effort metadata.
    """
    if not derived_from:
        return []
    summary: list[dict] = []
    factory = get_session_factory()
    async with factory() as s:
        for raw in derived_from:
            entry: dict = {"id": raw}
            try:
                target_uuid = uuid.UUID(raw)
            except ValueError:
                entry["target_kind"] = None
                entry["note"] = "not-a-uuid"
                summary.append(entry)
                continue
            kind = await _classify_target_kind(target_uuid)
            entry["target_kind"] = kind
            if kind is None:
                entry["note"] = "no-matching-row"
                summary.append(entry)
                continue
            link = AnnotationTarget(
                id=uuid.uuid4(),
                annotation_id=annotation_id,
                target_kind=kind,
                target_id=target_uuid,
                relation="derived_from",
            )
            s.add(link)
            summary.append(entry)
        await s.commit()
    return summary


@dataclass
class AnnotationResult:
    """A single annotation search result with similarity score."""

    id: str
    content: str
    kind: str
    source: str | None
    project: str | None
    tags: list[str] | None
    confidence: float
    valid_from: datetime
    valid_until: datetime | None
    unit_id: str | None
    metadata: dict
    score: float


async def store_annotation(
    content: str,
    *,
    kind: str = "fact",
    source: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 1.0,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    unit_id: str | None = None,
    session_id: str | uuid.UUID | None = None,
    task: str | None = None,
    commit_hash: str | None = None,
    supersedes_id: str | None = None,
    derived_from: list[str] | None = None,
    metadata: dict | None = None,
) -> Annotation:
    """Store a new annotation with its embedding.

    Args:
        content: The annotation text.
        kind: fact, decision, learning, pattern, warning, note, …
        source: Origin (e.g. ``"agent:claude-code"``, ``"user:you"``).
        project: Project name.
        tags: Categorization tags.
        confidence: Confidence score 0.0–1.0.
        valid_from: When the annotation becomes valid (default: now).
        valid_until: When the annotation expires (None = forever).
        unit_id: Optional UUID of a unit this annotation is attached to.
            Survives body revisions via the stable `units.identity_key`.
        session_id: Thread of work — see :mod:`hafiz.core.session`.
        task: Named task within the session.
        commit_hash: Git HEAD when the annotation was made. Auto-captured
            if not provided.
        supersedes_id: UUID of an annotation this one replaces. Sets the
            previous row's ``valid_until = now`` and links via
            ``supersedes_id``. Raises ValueError if the target is missing.
        derived_from: Lineage — list of annotation UUIDs this one was
            distilled from. Stored in ``metadata.derived_from``.
        metadata: Arbitrary JSONB metadata. ``commit_hash`` key is
            promoted to the dedicated column and stripped.

    Returns:
        The stored Annotation ORM object.
    """
    embedding = await embed_query(content)

    merged_metadata = dict(metadata or {})
    git_ctx = current_git_context()

    legacy_from_meta = merged_metadata.pop("commit_hash", None)
    resolved_commit_hash = commit_hash or legacy_from_meta or git_ctx.get("commit_hash")

    for key in ("branch", "is_dirty"):
        if key not in merged_metadata and key in git_ctx:
            merged_metadata[key] = git_ctx[key]

    if derived_from:
        merged_metadata["derived_from"] = list(derived_from)

    # Phase 2 session resolution. ``session_id`` may arrive as:
    #   - a real uuid (Phase 2+ callers; importer; future code)
    #   - a slug string (legacy CLI / per-TTY cursor)
    # When it's a slug, look up the sessions table; if a row exists,
    # populate the uuid FK *and* keep the slug on legacy_session_id
    # for human-readable display in journal/distill output. If no row
    # is found, the slug lands in legacy_session_id only.
    legacy_session_value: str | None = None
    session_uuid_value: uuid.UUID | None = None
    if isinstance(session_id, uuid.UUID):
        session_uuid_value = session_id
        from hafiz.core.sessions import get_session_by_id

        found_row = await get_session_by_id(session_id)
        if found_row is not None:
            legacy_session_value = found_row.slug
    elif session_id is not None:
        raw = str(session_id).strip()
        if raw:
            try:
                session_uuid_value = uuid.UUID(raw)
                from hafiz.core.sessions import get_session_by_id

                found_row = await get_session_by_id(session_uuid_value)
                if found_row is not None:
                    legacy_session_value = found_row.slug
            except ValueError:
                # Treat as slug. Look up; if missing, keep as legacy text.
                from hafiz.core.sessions import get_session_by_slug

                found = await get_session_by_slug(raw)
                if found is not None:
                    session_uuid_value = found.id
                    legacy_session_value = found.slug
                else:
                    legacy_session_value = raw

    now = datetime.now(UTC)
    new_ann = Annotation(
        id=uuid.uuid4(),
        content=content,
        embedding=embedding,
        kind=kind,
        source=source,
        project=project,
        tags=tags,
        confidence=confidence,
        valid_from=valid_from or now,
        valid_until=valid_until,
        unit_id=uuid.UUID(unit_id) if unit_id else None,
        legacy_session_id=legacy_session_value,
        session_id=session_uuid_value,
        task=task,
        commit_hash=resolved_commit_hash,
        supersedes_id=uuid.UUID(supersedes_id) if supersedes_id else None,
        metadata_=merged_metadata,
    )

    session_factory = get_session_factory()
    async with session_factory() as session:
        if supersedes_id:
            target = await session.get(Annotation, uuid.UUID(supersedes_id))
            if target is None:
                raise ValueError(f"Cannot supersede {supersedes_id!r}: annotation not found.")
            if target.valid_until is None or target.valid_until > now:
                target.valid_until = now
        session.add(new_ann)
        await session.commit()
        await session.refresh(new_ann)

    # Phase 5 — polymorphic lineage. Write annotation_targets rows for
    # each derived_from id. Done in a separate transaction so a missing
    # target row doesn't roll back the annotation itself.
    if derived_from:
        await write_derived_from_links(new_ann.id, list(derived_from))

    return new_ann


async def search_annotations(
    query: str,
    *,
    limit: int = 10,
    project: str | list[str] | None = None,
    kind: str | None = None,
    source: str | None = None,
    active_only: bool = True,
    rerank: bool | None = None,
) -> list[AnnotationResult]:
    """Search annotations by vector similarity, optionally cross-encoder reranked.

    When ``rerank`` is True (or None and ``rerank.enabled`` config is set), the
    vector stage over-fetches ``limit × candidate_multiplier`` candidates and a
    cross-encoder reorders them by joint (query, content) relevance before
    truncating to ``limit``. Reranking is strictly a reordering: on any failure
    it falls back to the vector order. ``rerank=False`` forces pure vector.
    """
    from hafiz.core.config import load_settings
    from hafiz.core.reranker import rerank as _rerank_items

    rerank_cfg = load_settings().rerank
    do_rerank = rerank_cfg.enabled if rerank is None else rerank
    # Over-fetch candidates for the reranker to reorder; it can only improve on
    # what vector recall surfaced, so a wider net helps. Pure-vector path keeps
    # the tight limit.
    fetch_limit = max(limit * rerank_cfg.candidate_multiplier, limit) if do_rerank else limit

    query_embedding = await embed_query(query)

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(
                Annotation,
                (1 - Annotation.embedding.cosine_distance(query_embedding)).label("similarity"),
            )
            .where(Annotation.embedding.isnot(None))
            .order_by(Annotation.embedding.cosine_distance(query_embedding))
            .limit(fetch_limit)
        )

        if isinstance(project, list):
            stmt = stmt.where(Annotation.project.in_(project))
        elif project:
            stmt = stmt.where(Annotation.project == project)
        if kind:
            stmt = stmt.where(Annotation.kind == kind)
        if source:
            stmt = stmt.where(Annotation.source == source)
        if active_only:
            now = datetime.now(UTC)
            stmt = stmt.where(Annotation.valid_from <= now)
            stmt = stmt.where((Annotation.valid_until.is_(None)) | (Annotation.valid_until > now))

        result = await session.execute(stmt)
        rows = result.all()

    candidates = [
        AnnotationResult(
            id=str(ann.id),
            content=ann.content,
            kind=ann.kind,
            source=ann.source,
            project=ann.project,
            tags=ann.tags,
            confidence=ann.confidence,
            valid_from=ann.valid_from,
            valid_until=ann.valid_until,
            unit_id=str(ann.unit_id) if ann.unit_id else None,
            metadata=ann.metadata_ or {},
            score=round(float(similarity), 4),
        )
        for ann, similarity in rows
    ]

    if do_rerank and len(candidates) > 1:
        return await _rerank_items(query, candidates, text_of=lambda r: r.content, top_n=limit)
    return candidates[:limit]


async def list_annotations(
    *,
    project: str | None = None,
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Annotation]:
    """List annotations with optional filters, newest first."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Annotation).order_by(Annotation.valid_from.desc()).limit(limit).offset(offset)
        if project:
            stmt = stmt.where(Annotation.project == project)
        if kind:
            stmt = stmt.where(Annotation.kind == kind)

        result = await session.execute(stmt)
        return list(result.scalars().all())


async def invalidate_annotation(ann_id: str) -> Annotation | None:
    """Invalidate an annotation by setting ``valid_until = now``."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Annotation).where(Annotation.id == uuid.UUID(ann_id))
        result = await session.execute(stmt)
        ann = result.scalar_one_or_none()
        if ann is None:
            return None
        ann.valid_until = datetime.now(UTC)
        await session.commit()
        await session.refresh(ann)
        return ann
