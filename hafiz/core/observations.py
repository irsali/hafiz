"""Observation storage and retrieval with vector similarity search.

Store high-level decisions, facts, and learnings as embeddings for semantic recall.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, func, update

from hafiz.core.database import Observation, get_session_factory
from hafiz.core.embeddings import embed_query
from hafiz.core.git_context import current_git_context


@dataclass
class ObservationResult:
    """A single observation search result with similarity score."""

    id: str
    content: str
    obs_type: str
    source: str | None
    project: str | None
    tags: list[str] | None
    confidence: float
    valid_from: datetime
    valid_until: datetime | None
    metadata: dict
    score: float


async def store_observation(
    content: str,
    *,
    obs_type: str = "fact",
    source: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 1.0,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    session_id: str | None = None,
    task: str | None = None,
    commit_hash: str | None = None,
    supersedes_id: str | None = None,
    derived_from: list[str] | None = None,
    metadata: dict | None = None,
) -> Observation:
    """Store a new observation with its embedding.

    Args:
        content: The observation text.
        obs_type: fact, decision, learning, pattern, warning, note.
        source: Origin (e.g. ``"agent:claude-code"``, ``"user:you"``).
        project: Project name.
        tags: Categorization tags.
        confidence: Confidence score 0.0–1.0.
        valid_from: When the observation becomes valid (default: now).
        valid_until: When the observation expires (None = forever).
        session_id: Thread of work this belongs to — see :mod:`hafiz.core.session`.
        task: Named task within the session.
        commit_hash: Git HEAD when the observation was made. If None and the
            caller did not pre-populate ``metadata["commit_hash"]``, it's
            auto-captured from the current cwd via
            :func:`hafiz.core.git_context.current_git_context`.
        supersedes_id: UUID of an observation this one replaces. Atomically
            sets that row's ``valid_until = now`` and records the link on the
            new row's ``supersedes_id`` column. Raises ``ValueError`` if the
            target does not exist. Supersession is non-destructive — the old
            row stays queryable via ``--include-superseded``.
        derived_from: Source observation ids this row was distilled from.
            Stored in ``metadata.derived_from`` as a list of UUID strings —
            lineage, not replacement (use ``supersedes_id`` for that).
        metadata: Arbitrary JSONB metadata. Any ``commit_hash`` key is
            promoted into the dedicated column and stripped from the dict.

    Returns:
        The stored Observation ORM object.
    """
    embedding = await embed_query(content)

    # Start by accepting whatever the caller provided.
    merged_metadata = dict(metadata or {})
    git_ctx = current_git_context()

    # commit_hash is now a first-class column. Accept it from (in priority):
    #  1. explicit commit_hash kwarg,
    #  2. a commit_hash key in caller-supplied metadata (legacy callers),
    #  3. auto-captured git HEAD.
    # Strip from metadata either way so it doesn't duplicate.
    legacy_from_meta = merged_metadata.pop("commit_hash", None)
    resolved_commit_hash = commit_hash or legacy_from_meta or git_ctx.get("commit_hash")

    # branch and is_dirty still live in metadata — they have no dedicated column.
    for key in ("branch", "is_dirty"):
        if key not in merged_metadata and key in git_ctx:
            merged_metadata[key] = git_ctx[key]

    # derived_from is lineage, not replacement — store in JSONB.
    if derived_from:
        merged_metadata["derived_from"] = list(derived_from)

    now = datetime.now(timezone.utc)
    new_obs = Observation(
        id=uuid.uuid4(),
        content=content,
        embedding=embedding,
        obs_type=obs_type,
        source=source,
        project=project,
        tags=tags,
        confidence=confidence,
        valid_from=valid_from or now,
        valid_until=valid_until,
        session_id=session_id,
        task=task,
        commit_hash=resolved_commit_hash,
        supersedes_id=uuid.UUID(supersedes_id) if supersedes_id else None,
        metadata_=merged_metadata,
    )

    session_factory = get_session_factory()
    async with session_factory() as session:
        # Atomic: if this write supersedes a prior row, invalidate it in the
        # same transaction so readers never see both as active simultaneously.
        if supersedes_id:
            target = await session.get(Observation, uuid.UUID(supersedes_id))
            if target is None:
                raise ValueError(
                    f"Cannot supersede {supersedes_id!r}: observation not found."
                )
            if target.valid_until is None or target.valid_until > now:
                target.valid_until = now
        session.add(new_obs)
        await session.commit()
        await session.refresh(new_obs)
        return new_obs


async def search_observations(
    query: str,
    *,
    limit: int = 10,
    project: str | list[str] | None = None,
    obs_type: str | None = None,
    active_only: bool = True,
) -> list[ObservationResult]:
    """Search observations by vector similarity.

    Args:
        query: The search query text.
        limit: Maximum number of results.
        project: Filter by project name (str), multiple projects (list), or None for all.
        obs_type: Filter by observation type.
        active_only: Only return currently valid observations.

    Returns:
        List of ObservationResult sorted by similarity (highest first).
    """
    query_embedding = await embed_query(query)

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(
                Observation,
                (1 - Observation.embedding.cosine_distance(query_embedding)).label(
                    "similarity"
                ),
            )
            .where(Observation.embedding.isnot(None))
            .order_by(Observation.embedding.cosine_distance(query_embedding))
            .limit(limit)
        )

        if isinstance(project, list):
            stmt = stmt.where(Observation.project.in_(project))
        elif project:
            stmt = stmt.where(Observation.project == project)
        if obs_type:
            stmt = stmt.where(Observation.obs_type == obs_type)
        if active_only:
            now = datetime.now(timezone.utc)
            stmt = stmt.where(Observation.valid_from <= now)
            stmt = stmt.where(
                (Observation.valid_until.is_(None)) | (Observation.valid_until > now)
            )

        result = await session.execute(stmt)
        rows = result.all()

        return [
            ObservationResult(
                id=str(obs.id),
                content=obs.content,
                obs_type=obs.obs_type,
                source=obs.source,
                project=obs.project,
                tags=obs.tags,
                confidence=obs.confidence,
                valid_from=obs.valid_from,
                valid_until=obs.valid_until,
                metadata=obs.metadata_ or {},
                score=round(float(similarity), 4),
            )
            for obs, similarity in rows
        ]


async def list_observations(
    *,
    project: str | None = None,
    obs_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Observation]:
    """List observations with optional filters.

    Args:
        project: Filter by project name.
        obs_type: Filter by observation type.
        limit: Maximum number of results.
        offset: Skip this many results.

    Returns:
        List of Observation ORM objects.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(Observation)
            .order_by(Observation.valid_from.desc())
            .limit(limit)
            .offset(offset)
        )
        if project:
            stmt = stmt.where(Observation.project == project)
        if obs_type:
            stmt = stmt.where(Observation.obs_type == obs_type)

        result = await session.execute(stmt)
        return list(result.scalars().all())


async def update_observation(
    obs_id: str,
    *,
    content: str | None = None,
    obs_type: str | None = None,
    confidence: float | None = None,
    valid_until: datetime | None = None,
) -> Observation | None:
    """Update an observation. Re-embeds if content changes.

    Args:
        obs_id: UUID of the observation.
        content: New content (triggers re-embedding).
        obs_type: New observation type.
        confidence: New confidence score.
        valid_until: New expiration datetime.

    Returns:
        The updated Observation, or None if not found.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Observation).where(Observation.id == uuid.UUID(obs_id))
        result = await session.execute(stmt)
        obs = result.scalar_one_or_none()

        if obs is None:
            return None

        if content is not None and content != obs.content:
            obs.content = content
            obs.embedding = await embed_query(content)
        if obs_type is not None:
            obs.obs_type = obs_type
        if confidence is not None:
            obs.confidence = confidence
        if valid_until is not None:
            obs.valid_until = valid_until

        await session.commit()
        await session.refresh(obs)
        return obs


async def invalidate_observation(obs_id: str) -> Observation | None:
    """Invalidate an observation by setting valid_until = now.

    Args:
        obs_id: UUID of the observation.

    Returns:
        The updated Observation, or None if not found.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Observation).where(Observation.id == uuid.UUID(obs_id))
        result = await session.execute(stmt)
        obs = result.scalar_one_or_none()

        if obs is None:
            return None

        obs.valid_until = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(obs)
        return obs
