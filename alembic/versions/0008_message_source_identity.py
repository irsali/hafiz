"""communication_messages.source_message_id — identity, not position

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-04

Transcript import deduped on ``(communication_id, seq)``, where ``seq`` is a
turn's position within one source *file*. That is not an identity, and the
difference cost about a third of every imported conversation.

Claude Code spreads a single session across many JSONL files — resumed sessions
and sidechains all carry the same ``sessionId`` — and each file restarts ``seq``
at 0. Every file after the first therefore collided with the first file's
positions and had its turns discarded as "already present". Measured on a real
``~/.claude/projects``: 201 files over 77 sessions, one session spanning 25
files, and **11,214 of 38,249 parsed turns silently dropped (29.3%)**. The worst
single session parsed 5,526 turns (4,795 distinct) and stored 3,181.

The fix is to dedupe on the identity the source already assigns each turn —
Claude Code's per-record ``uuid``, which is stable across files and preserved
when a session resumes. ``seq`` reverts to what it can actually be: an
append-only ordering key within a communication.

Three parts:

* ``source_message_id`` (nullable text). Nullable on purpose — ``hafiz capture``
  and hand-built rows have no source-native id and keep deduping positionally.
* A **partial** unique index over ``(communication_id, source_message_id)``,
  excluding NULLs so identity-less rows don't collide with each other.
* A backfill from ``metadata->>'claude_uuid'``, which the importer has been
  storing all along. That is what lets an existing store re-import and *heal*:
  turns lost to the old collision are recognised as absent and finally land,
  instead of colliding a second time.

The pre-existing ``uq_messages_comm_seq`` constraint is deliberately kept. It
still holds (allocation is append-only per communication) and it remains the
uniqueness guarantee for the identity-less rows.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "communication_messages",
        sa.Column("source_message_id", sa.Text(), nullable=True),
    )

    # Backfill from the metadata the claude-code importer already wrote.
    # Rows whose metadata has no claude_uuid (captures, other agents) stay
    # NULL and continue on the positional path.
    op.execute(
        """
        UPDATE communication_messages
           SET source_message_id = metadata->>'claude_uuid'
         WHERE metadata ? 'claude_uuid'
           AND metadata->>'claude_uuid' IS NOT NULL
           AND source_message_id IS NULL
        """
    )

    # Defensive: the backfill can only produce duplicates if the same turn
    # was somehow stored twice under one communication. Keep the earliest
    # row's id and clear the rest rather than failing the migration — a
    # duplicate is a data smell, not a reason to block an upgrade.
    op.execute(
        """
        UPDATE communication_messages m
           SET source_message_id = NULL
          FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY communication_id, source_message_id
                           ORDER BY seq, id
                       ) AS rn
                  FROM communication_messages
                 WHERE source_message_id IS NOT NULL
               ) d
         WHERE m.id = d.id
           AND d.rn > 1
        """
    )

    op.create_index(
        "uq_messages_comm_source_id",
        "communication_messages",
        ["communication_id", "source_message_id"],
        unique=True,
        postgresql_where=sa.text("source_message_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("uq_messages_comm_source_id", table_name="communication_messages")
    op.drop_column("communication_messages", "source_message_id")
