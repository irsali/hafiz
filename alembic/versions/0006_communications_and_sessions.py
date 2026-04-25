"""communications + sessions — first source-layer tables

Revision ID: 0006
Revises: 0005
Create Date: 2026-04-25

Adds the first **source-layer** tables (per docs/architecture.md "Storage layers
— knowledge vs source"). The seven knowledge-layer tables remain unchanged in
shape; this migration adds:

    sessions               — engineer/agent threads of work, promoted from
                             per-TTY JSON to a real DB row.
    communications         — agent transcripts, chat threads, etc.
                             (one row per session-shaped exchange).
    communication_messages — append-only turns within a communication.
    annotation_targets     — polymorphic linkage so annotations can cite
                             units, messages, sessions, or other
                             annotations (replaces the metadata.derived_from
                             pattern as it gets adopted; old metadata
                             remains readable).

It also pivots ``annotations.session_id`` from the historical ``text`` slug
column to a real ``uuid`` FK to ``sessions.id``:

    1. The existing ``session_id text`` column is renamed to
       ``legacy_session_id`` and kept indexed for back-compat. Every
       reader of session info still works; nothing is destroyed.
    2. A new ``session_id uuid`` FK column is added (nullable). Phase 2
       starts populating it as ``hafiz session start`` writes to the DB
       instead of just the per-TTY JSON cache.

See workitems/active/communications-and-sessions.md for the full design
(decisions, embedding policy, retention, agent contract).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ─── sessions ───────────────────────────────────────────────
    # Promoted from per-TTY JSON. ``slug`` is the human-friendly
    # identifier (the old session_id format, e.g. "phase-3-a3f19c");
    # ``id`` is the canonical uuid that annotations FK against.
    op.create_table(
        "sessions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=True),
        sa.Column("agent", sa.Text(), nullable=True),
        sa.Column("scope_kind", sa.Text(), nullable=True),
        sa.Column("scope_value", sa.Text(), nullable=True),
        sa.Column("task", sa.Text(), nullable=True),
        sa.Column("tty", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "ended_at", postgresql.TIMESTAMP(timezone=True), nullable=True
        ),
        sa.Column(
            "valid_until", postgresql.TIMESTAMP(timezone=True), nullable=True
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.UniqueConstraint("slug", name="uq_sessions_slug"),
    )
    op.create_index("idx_sessions_agent", "sessions", ["agent"])
    op.create_index(
        "idx_sessions_scope", "sessions", ["scope_kind", "scope_value"]
    )
    op.create_index("idx_sessions_started_at", "sessions", ["started_at"])
    op.create_index("idx_sessions_ended_at", "sessions", ["ended_at"])
    op.create_index("idx_sessions_valid_until", "sessions", ["valid_until"])

    # ─── communications ─────────────────────────────────────────
    # One row per session-shaped exchange. ``external_id`` is the
    # agent-harness's own identifier (Claude Code session uuid, Cursor
    # row id, etc.) and is what makes import idempotent.
    op.create_table(
        "communications",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("external_id", sa.Text(), nullable=True),
        sa.Column("agent", sa.Text(), nullable=False),
        sa.Column("channel", sa.Text(), nullable=True),
        sa.Column(
            "participants",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("scope_kind", sa.Text(), nullable=True),
        sa.Column("scope_value", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "ended_at", postgresql.TIMESTAMP(timezone=True), nullable=True
        ),
        sa.Column(
            "retention_until",
            postgresql.TIMESTAMP(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "valid_until", postgresql.TIMESTAMP(timezone=True), nullable=True
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    # Idempotency: re-importing the same Claude Code session uuid is a
    # no-op. Partial unique because external_id is nullable for native
    # captures that don't come from a third-party harness.
    op.create_index(
        "uq_communications_agent_external",
        "communications",
        ["agent", "external_id"],
        unique=True,
        postgresql_where=sa.text("external_id IS NOT NULL"),
    )
    op.create_index("idx_communications_session", "communications", ["session_id"])
    op.create_index("idx_communications_agent", "communications", ["agent"])
    op.create_index(
        "idx_communications_started_at", "communications", ["started_at"]
    )
    op.create_index(
        "idx_communications_retention", "communications", ["retention_until"]
    )
    op.create_index(
        "idx_communications_valid_until", "communications", ["valid_until"]
    )
    op.create_index(
        "idx_communications_scope",
        "communications",
        ["scope_kind", "scope_value"],
    )

    # ─── communication_messages ─────────────────────────────────
    # Append-only turns. ``content`` is canonical (raw is canonical;
    # embedding is a derived index). ``embedding`` is nullable and
    # populated only per the selective-embed policy. ``seq`` is
    # monotonic per-communication; ``parent_message_id`` allows for
    # branching/edits but is usually null for linear harness exports.
    op.create_table(
        "communication_messages",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "communication_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("communications.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "content_type",
            sa.Text(),
            nullable=False,
            server_default="text/markdown",
        ),
        sa.Column(
            "tool_calls",
            postgresql.JSONB(),
            nullable=True,
        ),
        sa.Column(
            "parent_message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("communication_messages.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("ts", postgresql.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column(
            "chunk_window",
            postgresql.JSONB(),
            nullable=True,
        ),
        sa.Column(
            "marked_salient",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.UniqueConstraint(
            "communication_id", "seq", name="uq_messages_comm_seq"
        ),
        sa.CheckConstraint(
            "role IN ('user', 'assistant', 'tool', 'system')",
            name="ck_messages_role",
        ),
    )
    op.create_index(
        "idx_messages_comm_seq", "communication_messages", ["communication_id", "seq"]
    )
    op.create_index("idx_messages_ts", "communication_messages", ["ts"])
    op.create_index(
        "idx_messages_role", "communication_messages", ["role"]
    )
    op.create_index(
        "idx_messages_parent", "communication_messages", ["parent_message_id"]
    )
    op.create_index(
        "idx_messages_salient",
        "communication_messages",
        ["marked_salient"],
        postgresql_where=sa.text("marked_salient = true"),
    )

    # ─── annotation_targets ─────────────────────────────────────
    # Polymorphic pivot — the bridge between knowledge and source
    # layers. An annotation may target a unit, another annotation, a
    # message, a communication, or a session. ``relation`` is curated;
    # the ``CHECK`` enforces the agreed vocabulary.
    op.create_table(
        "annotation_targets",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "annotation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("annotations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("target_kind", sa.Text(), nullable=False),
        sa.Column("target_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "observed_at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "target_kind IN ('unit', 'annotation', 'message', "
            "'communication', 'session')",
            name="ck_annotation_targets_kind",
        ),
        sa.CheckConstraint(
            "relation IN ('derived_from', 'about', 'supersedes', "
            "'cites', 'rebuts', 'related_to')",
            name="ck_annotation_targets_relation",
        ),
    )
    op.create_index(
        "idx_ann_targets_annotation", "annotation_targets", ["annotation_id"]
    )
    op.create_index(
        "idx_ann_targets_target",
        "annotation_targets",
        ["target_kind", "target_id"],
    )
    op.create_index(
        "idx_ann_targets_relation", "annotation_targets", ["relation"]
    )

    # ─── annotations.session_id pivot ───────────────────────────
    # Rename the existing text column to legacy_session_id (kept for
    # back-compat — current readers continue working). Add a new uuid
    # FK column that Phase 2 starts populating as `hafiz session
    # start` becomes DB-backed.
    op.drop_index("idx_annotations_session", table_name="annotations")
    op.alter_column(
        "annotations",
        "session_id",
        new_column_name="legacy_session_id",
    )
    op.create_index(
        "idx_annotations_legacy_session",
        "annotations",
        ["legacy_session_id"],
    )
    op.add_column(
        "annotations",
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index(
        "idx_annotations_session", "annotations", ["session_id"]
    )


def downgrade() -> None:
    raise NotImplementedError(
        "Migration 0006 (communications + sessions) has no downgrade. "
        "Pre-1.0; restore from backup if you need the prior schema."
    )
