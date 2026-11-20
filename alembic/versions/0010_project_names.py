"""project_names — declared synonyms and retired projects

Revision ID: 0010
Revises: 0009
Create Date: 2026-11-13

A live store held ten products under two or three spellings each, because the
code index used repo folder names and agents wrote notes under the ids they
knew. Every project filter is an exact match, so each spelling hid the others.
Look-alike spellings cannot simply be folded together — the same store held a
retired repo and its replacement one key apart — so synonyms are declared here
and :mod:`hafiz.core.projects` resolves through them.

Plain types only: embedded (SQLite) stores run this migration too.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "project_names",
        sa.Column("name", sa.Text(), primary_key=True),
        sa.Column("project", sa.Text(), nullable=False),
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("project_names")
