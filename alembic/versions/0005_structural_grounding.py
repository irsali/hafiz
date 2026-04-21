"""structural grounding — greenfield reshape to units/revisions/embeddings/edges/annotations/files/commits

Revision ID: 0005
Revises: 0004
Create Date: 2026-04-21

Drops the original four-table model (chunks, entities, relations, observations)
and replaces it with a seven-table model built on identity/body/embedding
separation. See workitems/active/structural-grounding.md for the design.

Destructive by design — pre-1.0 greenfield grant. No backfill. Users re-ingest.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ─── Drop the old world ─────────────────────────────────────
    op.drop_index("idx_observations_supersedes", table_name="observations")
    op.drop_index("idx_observations_type", table_name="observations")
    op.drop_table("observations")

    op.drop_index("idx_relations_target", table_name="relations")
    op.drop_index("idx_relations_source", table_name="relations")
    op.drop_table("relations")

    op.drop_index("idx_entities_project", table_name="entities")
    op.drop_index("idx_entities_type", table_name="entities")
    op.drop_table("entities")

    op.drop_index("idx_chunks_checksum", table_name="chunks")
    op.drop_index("idx_chunks_source", table_name="chunks")
    op.drop_index("idx_chunks_project", table_name="chunks")
    op.drop_table("chunks")

    # ─── commits ────────────────────────────────────────────────
    # Git axis as first-class. hash is the PK; rewritten_at/rewritten_to
    # track rebase/amend/squash per Phase 5b.
    op.create_table(
        "commits",
        sa.Column("hash", sa.Text(), primary_key=True),
        sa.Column("project", sa.Text(), nullable=True),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column(
            "committed_at", postgresql.TIMESTAMP(timezone=True), nullable=True
        ),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "rewritten_at", postgresql.TIMESTAMP(timezone=True), nullable=True
        ),
        sa.Column("rewritten_to", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index("idx_commits_project", "commits", ["project"])
    op.create_index("idx_commits_committed_at", "commits", ["committed_at"])
    op.create_index("idx_commits_rewritten_at", "commits", ["rewritten_at"])

    # ─── files ──────────────────────────────────────────────────
    # One row per file ever seen. (project, path) is unique.
    # valid_until null means currently present; tombstoned otherwise.
    op.create_table(
        "files",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("project", sa.Text(), nullable=True),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), nullable=True),
        sa.Column("first_seen_commit", sa.Text(), nullable=True),
        sa.Column("last_seen_commit", sa.Text(), nullable=True),
        sa.Column(
            "valid_until", postgresql.TIMESTAMP(timezone=True), nullable=True
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.UniqueConstraint("project", "path", name="uq_files_project_path"),
    )
    op.create_index("idx_files_project", "files", ["project"])
    op.create_index("idx_files_path", "files", ["path"])
    op.create_index("idx_files_valid_until", "files", ["valid_until"])

    # ─── units ──────────────────────────────────────────────────
    # Stable identity of an addressable thing. identity_key is unique
    # globally across time — re-appearing units reuse the same row.
    op.create_table(
        "units",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "file_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("files.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("parent_name", sa.Text(), nullable=True),
        sa.Column("identity_key", sa.Text(), nullable=False),
        sa.Column("first_seen_commit", sa.Text(), nullable=True),
        sa.Column("last_seen_commit", sa.Text(), nullable=True),
        sa.Column(
            "valid_until", postgresql.TIMESTAMP(timezone=True), nullable=True
        ),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("identity_key", name="uq_units_identity_key"),
    )
    op.create_index("idx_units_file_id", "units", ["file_id"])
    op.create_index("idx_units_kind", "units", ["kind"])
    op.create_index("idx_units_name", "units", ["name"])
    op.create_index("idx_units_valid_until", "units", ["valid_until"])

    # ─── unit_revisions ─────────────────────────────────────────
    # Append-only body. Partial unique index keeps exactly one
    # current revision per unit (superseded_at IS NULL).
    # source ∈ {ast, parser, agent, user}.
    op.create_table(
        "unit_revisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "unit_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("units.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("line_start", sa.Integer(), nullable=True),
        sa.Column("line_end", sa.Integer(), nullable=True),
        sa.Column("commit_hash", sa.Text(), nullable=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column(
            "observed_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "superseded_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "superseded_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("unit_revisions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.CheckConstraint(
            "source IN ('ast', 'parser', 'agent', 'user')",
            name="ck_unit_revisions_source",
        ),
    )
    op.create_index(
        "idx_unit_revisions_unit_id", "unit_revisions", ["unit_id"]
    )
    op.create_index(
        "idx_unit_revisions_content_hash", "unit_revisions", ["content_hash"]
    )
    op.create_index(
        "idx_unit_revisions_commit_hash", "unit_revisions", ["commit_hash"]
    )
    op.create_index(
        "uq_unit_revisions_current",
        "unit_revisions",
        ["unit_id"],
        unique=True,
        postgresql_where=sa.text("superseded_at IS NULL"),
    )

    # ─── embeddings ─────────────────────────────────────────────
    # 1:N from unit_revisions. One row per small unit; many rows for
    # oversized bodies (part_index + token span). Each part has its
    # own content_hash so partial edits re-embed only changed parts.
    op.create_table(
        "embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "unit_revision_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("unit_revisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "part_index", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("token_span_start", sa.Integer(), nullable=True),
        sa.Column("token_span_end", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "unit_revision_id",
            "part_index",
            name="uq_embeddings_revision_part",
        ),
    )
    op.create_index(
        "idx_embeddings_revision", "embeddings", ["unit_revision_id"]
    )
    op.create_index(
        "idx_embeddings_content_hash", "embeddings", ["content_hash"]
    )

    # ─── edges ──────────────────────────────────────────────────
    # Append-only relations between units. target_unit_id nullable
    # for unresolved / external references (target_name holds the
    # raw string until a resolver binds it, or forever for externals).
    # source ∈ {ast, agent, user}.
    op.create_table(
        "edges",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "source_unit_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("units.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_unit_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("units.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("target_name", sa.Text(), nullable=True),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("commit_hash", sa.Text(), nullable=True),
        sa.Column(
            "observed_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "superseded_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.CheckConstraint(
            "source IN ('ast', 'agent', 'user')",
            name="ck_edges_source",
        ),
    )
    op.create_index("idx_edges_source", "edges", ["source_unit_id"])
    op.create_index("idx_edges_target", "edges", ["target_unit_id"])
    op.create_index("idx_edges_target_name", "edges", ["target_name"])
    op.create_index("idx_edges_relation", "edges", ["relation"])
    op.create_index("idx_edges_commit", "edges", ["commit_hash"])

    # ─── annotations ────────────────────────────────────────────
    # Decisions, facts, learnings. May link to a unit (unit_id) or
    # float free. source is free-form like "agent:claude-code" or
    # "user:irshad" — different shape from parser-source columns.
    op.create_table(
        "annotations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("kind", sa.Text(), nullable=False, server_default="fact"),
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("project", sa.Text(), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column(
            "confidence", sa.Float(), nullable=False, server_default="1.0"
        ),
        sa.Column(
            "unit_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("units.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("session_id", sa.Text(), nullable=True),
        sa.Column("task", sa.Text(), nullable=True),
        sa.Column("commit_hash", sa.Text(), nullable=True),
        sa.Column(
            "valid_from",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "valid_until",
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "supersedes_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("annotations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index("idx_annotations_kind", "annotations", ["kind"])
    op.create_index("idx_annotations_unit_id", "annotations", ["unit_id"])
    op.create_index("idx_annotations_project", "annotations", ["project"])
    op.create_index("idx_annotations_session", "annotations", ["session_id"])
    op.create_index("idx_annotations_task", "annotations", ["task"])
    op.create_index("idx_annotations_commit", "annotations", ["commit_hash"])
    op.create_index(
        "idx_annotations_supersedes", "annotations", ["supersedes_id"]
    )


def downgrade() -> None:
    raise NotImplementedError(
        "Greenfield structural-grounding migration has no downgrade path. "
        "Restore from backup if you need the old chunks/entities/relations/observations schema."
    )
