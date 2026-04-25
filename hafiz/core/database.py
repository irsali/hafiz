"""SQLAlchemy 2.0 async models and database connection for Hafiz.

Seven-table model built on identity/body/embedding separation:

  files          - One row per file ever seen (tombstoned via valid_until).
  units          - Stable identity of an addressable thing (function, heading,
                   section, whole-file fallback). kind is namespaced
                   `domain.subtype` (code.function, doc.heading, mail.message,
                   …) by convention.
  unit_revisions - Append-only body. At most one current revision per unit
                   (partial unique on superseded_at IS NULL).
  embeddings     - 1:N vector search index over revisions. Small units: one
                   row. Oversized units: many rows with part_index + token
                   spans, each content-hashed independently.
  edges          - Append-only relations between units. source ∈ {ast, agent,
                   user}. target_name kept for unresolved/external refs.
  annotations    - Decisions / facts / learnings. May link to a unit or float
                   free. Temporal primitives (valid_from/until, supersedes_id).
  commits        - Git axis as a first-class citizen. rewritten_at /
                   rewritten_to track rebase/amend/squash (Phase 5b).

See workitems/active/structural-grounding.md and docs/roadmap.md for design.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from hafiz.core.config import get_settings


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Commits — git axis
# ---------------------------------------------------------------------------

class Commit(Base):
    __tablename__ = "commits"

    hash: Mapped[str] = mapped_column(Text, primary_key=True)
    project: Mapped[str | None] = mapped_column(Text, nullable=True)
    author: Mapped[str | None] = mapped_column(Text, nullable=True)
    committed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    rewritten_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    rewritten_to: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    __table_args__ = (
        Index("idx_commits_project", "project"),
        Index("idx_commits_committed_at", "committed_at"),
        Index("idx_commits_rewritten_at", "rewritten_at"),
    )


# ---------------------------------------------------------------------------
# Files — one row per file ever seen
# ---------------------------------------------------------------------------

class File(Base):
    __tablename__ = "files"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    project: Mapped[str | None] = mapped_column(Text, nullable=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str | None] = mapped_column(Text, nullable=True)
    first_seen_commit: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_seen_commit: Mapped[str | None] = mapped_column(Text, nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    units: Mapped[list["Unit"]] = relationship(
        "Unit", back_populates="file", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("project", "path", name="uq_files_project_path"),
        Index("idx_files_project", "project"),
        Index("idx_files_path", "path"),
        Index("idx_files_valid_until", "valid_until"),
    )


# ---------------------------------------------------------------------------
# Units — stable identity
# ---------------------------------------------------------------------------

class Unit(Base):
    __tablename__ = "units"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    parent_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    identity_key: Mapped[str] = mapped_column(Text, nullable=False)
    first_seen_commit: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_seen_commit: Mapped[str | None] = mapped_column(Text, nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    file: Mapped[File] = relationship("File", back_populates="units")
    revisions: Mapped[list["UnitRevision"]] = relationship(
        "UnitRevision", back_populates="unit", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("identity_key", name="uq_units_identity_key"),
        Index("idx_units_file_id", "file_id"),
        Index("idx_units_kind", "kind"),
        Index("idx_units_name", "name"),
        Index("idx_units_valid_until", "valid_until"),
    )


# ---------------------------------------------------------------------------
# Unit revisions — versioned body (append-only)
# ---------------------------------------------------------------------------

class UnitRevision(Base):
    __tablename__ = "unit_revisions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    unit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("units.id", ondelete="CASCADE"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    line_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    commit_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    superseded_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("unit_revisions.id", ondelete="SET NULL"),
        nullable=True,
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    unit: Mapped[Unit] = relationship("Unit", back_populates="revisions")
    embeddings: Mapped[list["Embedding"]] = relationship(
        "Embedding", back_populates="revision", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "source IN ('ast', 'parser', 'agent', 'user')",
            name="ck_unit_revisions_source",
        ),
        Index("idx_unit_revisions_unit_id", "unit_id"),
        Index("idx_unit_revisions_content_hash", "content_hash"),
        Index("idx_unit_revisions_commit_hash", "commit_hash"),
        Index(
            "uq_unit_revisions_current",
            "unit_id",
            unique=True,
            postgresql_where=text("superseded_at IS NULL"),
        ),
    )


# ---------------------------------------------------------------------------
# Embeddings — vector search index over revisions (1:N)
# ---------------------------------------------------------------------------

class Embedding(Base):
    __tablename__ = "embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    unit_revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("unit_revisions.id", ondelete="CASCADE"),
        nullable=False,
    )
    part_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(768), nullable=True)
    token_span_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_span_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    revision: Mapped[UnitRevision] = relationship(
        "UnitRevision", back_populates="embeddings"
    )

    __table_args__ = (
        UniqueConstraint(
            "unit_revision_id", "part_index", name="uq_embeddings_revision_part"
        ),
        Index("idx_embeddings_revision", "unit_revision_id"),
        Index("idx_embeddings_content_hash", "content_hash"),
    )


# ---------------------------------------------------------------------------
# Edges — relations between units (append-only)
# ---------------------------------------------------------------------------

class Edge(Base):
    __tablename__ = "edges"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_unit_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("units.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_unit_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("units.id", ondelete="SET NULL"),
        nullable=True,
    )
    target_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    relation: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    commit_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    superseded_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    source_unit: Mapped[Unit] = relationship(
        "Unit", foreign_keys=[source_unit_id]
    )
    target_unit: Mapped[Unit | None] = relationship(
        "Unit", foreign_keys=[target_unit_id]
    )

    __table_args__ = (
        CheckConstraint(
            "source IN ('ast', 'agent', 'user')",
            name="ck_edges_source",
        ),
        Index("idx_edges_source", "source_unit_id"),
        Index("idx_edges_target", "target_unit_id"),
        Index("idx_edges_target_name", "target_name"),
        Index("idx_edges_relation", "relation"),
        Index("idx_edges_commit", "commit_hash"),
    )


# ---------------------------------------------------------------------------
# Annotations — decisions, facts, learnings, patterns, warnings
# ---------------------------------------------------------------------------

class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(768), nullable=True)
    kind: Mapped[str] = mapped_column(Text, default="fact")
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    project: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    unit_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("units.id", ondelete="SET NULL"),
        nullable=True,
    )
    session_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    task: Mapped[str | None] = mapped_column(Text, nullable=True)
    commit_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    valid_from: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    supersedes_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("annotations.id", ondelete="SET NULL"),
        nullable=True,
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    unit: Mapped[Unit | None] = relationship("Unit", foreign_keys=[unit_id])

    __table_args__ = (
        Index("idx_annotations_kind", "kind"),
        Index("idx_annotations_unit_id", "unit_id"),
        Index("idx_annotations_project", "project"),
        Index("idx_annotations_session", "session_id"),
        Index("idx_annotations_task", "task"),
        Index("idx_annotations_commit", "commit_hash"),
        Index("idx_annotations_supersedes", "supersedes_id"),
    )


# ---------------------------------------------------------------------------
# Transition stubs — Phase 3 removes these
# ---------------------------------------------------------------------------
# Several modules (hafiz.core.search / context / capture / journal /
# graph_analysis / observations / distill / store / extractor, and several
# commands) still import Chunk / Entity / Relation / Observation at module
# scope. Migration 0005 dropped those tables; the callers get rewired in
# Phase 3 of workitems/active/structural-grounding.md.
#
# Until then these stubs keep the import chain alive so the CLI loads and
# unrelated tests run. Any attempt to instantiate or ORM-query them fails
# loudly — there is no silent compatibility path.

class _RemovedInV5:
    """Placeholder for a model removed in 0005_structural_grounding."""

    _name = "<unknown>"

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"{type(self)._name} was removed in migration 0005 "
            "(structural-grounding). Callers are rewired in Phase 3; "
            "see workitems/active/structural-grounding.md."
        )


class Chunk(_RemovedInV5):
    _name = "Chunk"


class Entity(_RemovedInV5):
    _name = "Entity"


class Relation(_RemovedInV5):
    _name = "Relation"


class Observation(_RemovedInV5):
    _name = "Observation"


# ---------------------------------------------------------------------------
# Engine / Session factory
# ---------------------------------------------------------------------------

_engine = None
_session_factory = None


def get_engine(url: str | None = None):
    """Get or create the async engine."""
    global _engine
    if _engine is None:
        db_url = url or get_settings().database.url
        _engine = create_async_engine(db_url, echo=False, pool_size=5, max_overflow=10)
    return _engine


def get_session_factory(url: str | None = None) -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine(url)
        _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


def _alembic_config(url: str | None = None):
    """Build an Alembic Config pointing at the packaged alembic/ directory."""
    from pathlib import Path
    from alembic.config import Config

    import hafiz

    hafiz_root = Path(hafiz.__file__).resolve().parent.parent
    cfg = Config(str(hafiz_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(hafiz_root / "alembic"))
    cfg.set_main_option(
        "sqlalchemy.url", url or get_settings().database.url
    )
    return cfg


async def create_tables(url: str | None = None) -> None:
    """Initialize schema by running Alembic migrations to head.

    Alembic is the single source of truth for the schema. On a fresh DB this
    runs every migration in order; on an existing Alembic-tracked DB it applies
    only missing ones.

    IMPORTANT: the `0005_structural_grounding` migration is destructive. It
    drops the old `chunks` / `entities` / `relations` / `observations` tables
    and replaces them with the seven-table identity/body/embedding model.
    Callers should communicate "re-ingest required" to the user on first run.
    """
    import asyncio

    from alembic import command

    cfg = _alembic_config(url)
    await asyncio.get_running_loop().run_in_executor(
        None, command.upgrade, cfg, "head"
    )


async def close_engine() -> None:
    """Dispose the engine connection pool."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
