"""add session_id, task, commit_hash columns + backfill commit_hash from JSONB

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-21
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # observations: session_id + task + commit_hash (promoted from metadata JSONB)
    op.add_column("observations", sa.Column("session_id", sa.Text(), nullable=True))
    op.add_column("observations", sa.Column("task", sa.Text(), nullable=True))
    op.add_column("observations", sa.Column("commit_hash", sa.Text(), nullable=True))
    op.create_index("idx_observations_session", "observations", ["session_id"])
    op.create_index("idx_observations_task", "observations", ["task"])
    op.create_index("idx_observations_commit", "observations", ["commit_hash"])

    # Backfill commit_hash from existing metadata JSONB so older observations
    # stay queryable via the new column.
    op.execute(
        "UPDATE observations "
        "SET commit_hash = metadata->>'commit_hash' "
        "WHERE metadata ? 'commit_hash' AND commit_hash IS NULL"
    )

    # chunks: session_id + task (for transcript session membership)
    op.add_column("chunks", sa.Column("session_id", sa.Text(), nullable=True))
    op.add_column("chunks", sa.Column("task", sa.Text(), nullable=True))
    op.create_index("idx_chunks_session", "chunks", ["session_id"])
    op.create_index("idx_chunks_task", "chunks", ["task"])


def downgrade() -> None:
    op.drop_index("idx_chunks_task", table_name="chunks")
    op.drop_index("idx_chunks_session", table_name="chunks")
    op.drop_column("chunks", "task")
    op.drop_column("chunks", "session_id")

    op.drop_index("idx_observations_commit", table_name="observations")
    op.drop_index("idx_observations_task", table_name="observations")
    op.drop_index("idx_observations_session", table_name="observations")
    op.drop_column("observations", "commit_hash")
    op.drop_column("observations", "task")
    op.drop_column("observations", "session_id")
