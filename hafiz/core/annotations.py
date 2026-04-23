"""Annotation storage and retrieval with vector similarity search.

The "wisdom layer" — decisions, facts, learnings, patterns, warnings, notes.
Annotations may optionally link to a unit (`unit_id`) so they survive body
changes across revisions, or float free as project-level or session-level
knowledge.

This module replaces the old ``observations.py``. The schema renamed
`observations` → `annotations` and `obs_type` → `kind` as part of the
structural-grounding work (see workitems/active/structural-grounding.md).
The CLI verb stays `hafiz observe` — that's a user-facing name, not a
model reference.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select

from hafiz.core.database import Annotation, get_session_factory
from hafiz.core.embeddings import embed_query
from hafiz.core.git_context import current_git_context


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
    session_id: str | None = None,
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

    now = datetime.now(timezone.utc)
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
        session_id=session_id,
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
                raise ValueError(
                    f"Cannot supersede {supersedes_id!r}: annotation not found."
                )
            if target.valid_until is None or target.valid_until > now:
                target.valid_until = now
        session.add(new_ann)
        await session.commit()
        await session.refresh(new_ann)
        return new_ann


async def search_annotations(
    query: str,
    *,
    limit: int = 10,
    project: str | list[str] | None = None,
    kind: str | None = None,
    source: str | None = None,
    active_only: bool = True,
) -> list[AnnotationResult]:
    """Search annotations by vector similarity."""
    query_embedding = await embed_query(query)

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(
                Annotation,
                (1 - Annotation.embedding.cosine_distance(query_embedding)).label(
                    "similarity"
                ),
            )
            .where(Annotation.embedding.isnot(None))
            .order_by(Annotation.embedding.cosine_distance(query_embedding))
            .limit(limit)
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
            now = datetime.now(timezone.utc)
            stmt = stmt.where(Annotation.valid_from <= now)
            stmt = stmt.where(
                (Annotation.valid_until.is_(None))
                | (Annotation.valid_until > now)
            )

        result = await session.execute(stmt)
        rows = result.all()

        return [
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
        stmt = (
            select(Annotation)
            .order_by(Annotation.valid_from.desc())
            .limit(limit)
            .offset(offset)
        )
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
        ann.valid_until = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(ann)
        return ann
