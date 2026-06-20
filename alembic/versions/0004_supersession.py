"""add observations.supersedes_id (symmetric, non-destructive supersession)

Revision ID: 0004
Revises: 0003
Create Date: 2026-04-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "observations",
        sa.Column(
            "supersedes_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("observations.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("idx_observations_supersedes", "observations", ["supersedes_id"])


def downgrade() -> None:
    op.drop_index("idx_observations_supersedes", table_name="observations")
    op.drop_column("observations", "supersedes_id")
