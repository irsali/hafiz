"""SQLAlchemy 2.0 async models and database connection for Hafiz.

Two storage layers (see docs/architecture.md "Storage layers"):

**Knowledge layer** — identity-stable, mid-volume, embedding-first:

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

**Source layer** — high-volume, time-series, immutable, retention-bounded
(see workitems/active/communications-and-sessions.md):

  sessions               - Engineer/agent threads of work. Promoted from
                           per-TTY JSON to a real DB row. Annotations FK
                           against ``sessions.id`` via the new uuid
                           ``Annotation.session_id`` column; the original
                           text slug column is preserved as
                           ``legacy_session_id``.
  communications         - Agent transcripts, chat threads, email threads.
                           One row per session-shaped exchange. Optional FK
                           to ``sessions``.
  communication_messages - Append-only turns within a communication. Raw
                           ``content`` is canonical; ``embedding`` is
                           nullable and populated only per the
                           selective-embed policy.
  annotation_targets     - Polymorphic pivot. Lets an annotation cite a
                           unit, another annotation, a message, a
                           communication, or a session. Replaces the
                           ``metadata.derived_from`` pattern as it gets
                           adopted; old metadata stays readable.

See workitems/done/structural-grounding.md and
workitems/active/communications-and-sessions.md for design.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
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
    # The historical text slug is preserved as ``legacy_session_id`` so
    # existing readers (journal, distill, recall filters) still work.
    # New writes from Phase 2+ populate ``session_id`` (uuid FK) instead.
    legacy_session_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
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
        Index("idx_annotations_legacy_session", "legacy_session_id"),
        Index("idx_annotations_task", "task"),
        Index("idx_annotations_commit", "commit_hash"),
        Index("idx_annotations_supersedes", "supersedes_id"),
    )


# ---------------------------------------------------------------------------
# Source-layer tables — high-volume, time-series, retention-bounded
# (See docs/architecture.md "Storage layers — knowledge vs source".)
# ---------------------------------------------------------------------------


class Session(Base):
    """Engineer/agent thread of work — promoted from per-TTY JSON.

    The historical slug (e.g. ``"phase-3-a3f19c"``) is kept on ``slug``;
    ``id`` is the canonical uuid that other tables FK against. Sessions
    have a ``valid_until`` for tombstoning but no ``retention_until`` —
    they are typically referenced by long-lived annotations and should
    not auto-expire.
    """

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope_kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    task: Mapped[str | None] = mapped_column(Text, nullable=True)
    tty: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    __table_args__ = (
        UniqueConstraint("slug", name="uq_sessions_slug"),
        Index("idx_sessions_agent", "agent"),
        Index("idx_sessions_scope", "scope_kind", "scope_value"),
        Index("idx_sessions_started_at", "started_at"),
        Index("idx_sessions_ended_at", "ended_at"),
        Index("idx_sessions_valid_until", "valid_until"),
    )


class Communication(Base):
    """Agent transcript, chat thread, or email thread.

    One row per session-shaped exchange. ``external_id`` is the
    agent-harness's own identifier (e.g. a Claude Code session uuid)
    and is the basis for importer idempotency — re-importing the same
    JSONL is a no-op.

    Retention is bounded: ``retention_until`` defaults in app code to
    ``started_at + 90 days``. The retention sweeper tombstones rows
    past their retention via ``valid_until``.
    """

    __tablename__ = "communications"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="SET NULL"),
        nullable=True,
    )
    external_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent: Mapped[str] = mapped_column(Text, nullable=False)
    channel: Mapped[str | None] = mapped_column(Text, nullable=True)
    participants: Mapped[list] = mapped_column(JSONB, default=list)
    scope_kind: Mapped[str | None] = mapped_column(Text, nullable=True)
    scope_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    retention_until: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    messages: Mapped[list["CommunicationMessage"]] = relationship(
        "CommunicationMessage",
        back_populates="communication",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index(
            "uq_communications_agent_external",
            "agent",
            "external_id",
            unique=True,
            postgresql_where=text("external_id IS NOT NULL"),
        ),
        Index("idx_communications_session", "session_id"),
        Index("idx_communications_agent", "agent"),
        Index("idx_communications_started_at", "started_at"),
        Index("idx_communications_retention", "retention_until"),
        Index("idx_communications_valid_until", "valid_until"),
        Index(
            "idx_communications_scope", "scope_kind", "scope_value"
        ),
    )


class CommunicationMessage(Base):
    """Append-only turn within a communication.

    Raw ``content`` is canonical (NOT NULL); ``embedding`` is nullable
    and populated only per the selective-embed policy (skip < 30
    tokens; skip pure tool-result echoes; mark-salient override).
    Storing only a vector without the source row inverts source-of-truth
    and is forbidden — the column is required.
    """

    __tablename__ = "communication_messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    communication_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("communications.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    author: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(
        Text, default="text/markdown"
    )
    tool_calls: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    parent_message_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("communication_messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    ts: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
    embedding = mapped_column(Vector(768), nullable=True)
    chunk_window: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    marked_salient: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    communication: Mapped[Communication] = relationship(
        "Communication", back_populates="messages"
    )

    __table_args__ = (
        UniqueConstraint(
            "communication_id", "seq", name="uq_messages_comm_seq"
        ),
        CheckConstraint(
            "role IN ('user', 'assistant', 'tool', 'system')",
            name="ck_messages_role",
        ),
        Index("idx_messages_comm_seq", "communication_id", "seq"),
        Index("idx_messages_ts", "ts"),
        Index("idx_messages_role", "role"),
        Index("idx_messages_parent", "parent_message_id"),
        Index(
            "idx_messages_salient",
            "marked_salient",
            postgresql_where=text("marked_salient = true"),
        ),
    )


class AnnotationTarget(Base):
    """Polymorphic pivot — bridges annotations to any source-layer row.

    An annotation may target a unit (knowledge), another annotation
    (knowledge), a message / communication / session (source). The
    ``relation`` column is curated; the CHECK constraint enforces the
    allowed vocabulary. ``metadata.derived_from`` lists on annotations
    remain readable for back-compat; new code writes to this table.
    """

    __tablename__ = "annotation_targets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    annotation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("annotations.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_kind: Mapped[str] = mapped_column(Text, nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    relation: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    observed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        CheckConstraint(
            "target_kind IN ('unit', 'annotation', 'message', "
            "'communication', 'session')",
            name="ck_annotation_targets_kind",
        ),
        CheckConstraint(
            "relation IN ('derived_from', 'about', 'supersedes', "
            "'cites', 'rebuts', 'related_to')",
            name="ck_annotation_targets_relation",
        ),
        Index("idx_ann_targets_annotation", "annotation_id"),
        Index("idx_ann_targets_target", "target_kind", "target_id"),
        Index("idx_ann_targets_relation", "relation"),
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
