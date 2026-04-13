"""SQLAlchemy 2.0 async models and database connection for Hafiz.

Tables: chunks, entities, relations, observations
Uses pgvector for embedding columns.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from hafiz.core.config import get_settings


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Chunks — raw content, chunked and embedded
# ---------------------------------------------------------------------------

class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(768), nullable=True)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    line_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_type: Mapped[str] = mapped_column(Text, default="code")
    language: Mapped[str | None] = mapped_column(Text, nullable=True)
    project: Mapped[str | None] = mapped_column(Text, nullable=True)
    checksum: Mapped[str | None] = mapped_column(Text, nullable=True)
    indexed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    __table_args__ = (
        Index("idx_chunks_project", "project"),
        Index("idx_chunks_source", "source_file"),
        Index("idx_chunks_checksum", "checksum"),
    )


# ---------------------------------------------------------------------------
# Entities — the "nouns" of the codebase
# ---------------------------------------------------------------------------

class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    entity_type: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    project: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_file: Mapped[str | None] = mapped_column(Text, nullable=True)
    properties: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    outgoing_relations: Mapped[list[Relation]] = relationship(
        "Relation", foreign_keys="Relation.source_id", back_populates="source", cascade="all, delete"
    )
    incoming_relations: Mapped[list[Relation]] = relationship(
        "Relation", foreign_keys="Relation.target_id", back_populates="target", cascade="all, delete"
    )

    __table_args__ = (
        Index("idx_entities_type", "entity_type"),
        Index("idx_entities_project", "project"),
    )


# ---------------------------------------------------------------------------
# Relations — the "verbs" between entities
# ---------------------------------------------------------------------------

class Relation(Base):
    __tablename__ = "relations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entities.id", ondelete="CASCADE"),
        nullable=False,
    )
    relation_type: Mapped[str] = mapped_column(Text, nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    evidence: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    # Relationships
    source: Mapped[Entity] = relationship(
        "Entity", foreign_keys=[source_id], back_populates="outgoing_relations"
    )
    target: Mapped[Entity] = relationship(
        "Entity", foreign_keys=[target_id], back_populates="incoming_relations"
    )

    __table_args__ = (
        Index("idx_relations_source", "source_id"),
        Index("idx_relations_target", "target_id"),
    )


# ---------------------------------------------------------------------------
# Observations — high-level decisions, facts, and learnings
# ---------------------------------------------------------------------------

class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(768), nullable=True)
    obs_type: Mapped[str] = mapped_column(Text, default="fact")
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
    project: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    valid_from: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    __table_args__ = (
        Index("idx_observations_type", "obs_type"),
    )


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


async def create_tables(url: str | None = None) -> None:
    """Create all tables (used by hafiz init)."""
    engine = get_engine(url)
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def close_engine() -> None:
    """Dispose the engine connection pool."""
    global _engine, _session_factory
    if _engine:
        await _engine.dispose()
        _engine = None
        _session_factory = None
