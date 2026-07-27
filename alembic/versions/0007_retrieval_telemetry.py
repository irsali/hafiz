"""retrievals — a local record of what the store was asked for

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-27

Hafiz could not answer the most basic question about itself. Auditing a 3.5-week
deployment for "is this earning its keep?" required parsing 169 Claude Code
transcripts, because hafiz kept no record of its own reads — it could not say
which annotations had ever been recalled, which surfaced and were useful, or
which had never come up once. Every quality mechanism that might grow from here
(decay dead knowledge, promote proven knowledge, notice recall quality
regressing) depends on data it wasn't collecting.

One append-only table, one INSERT per search. It records what happened, not an
interpretation of what happened:

    retrievals — one row per search: what was asked, what came back, how well
                 it scored, and who asked.

The most valuable column is ``query_text``, and specifically the rows where
``n_results = 0``: the gap between what agents look for and what the store holds
is the only signal that says what to write down *next*.

**This is a new data category for the store.** Existing rows are conclusions;
these are what somebody was looking for. So it lands in the source layer with
the source layer's guarantees — bounded ``retention_until``, tombstoneable via
``valid_until``, and never leaving the machine — and it can be switched off
entirely with ``[telemetry] retrieval = false``.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "retrievals",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "at",
            postgresql.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        # 'query' | 'query --observations' | 'context' | 'recall' — which
        # surface was used, so the two layers can be evaluated separately.
        sa.Column("command", sa.Text(), nullable=False),
        sa.Column("query_text", sa.Text(), nullable=False),
        # project / kind / source / limit / min_score / domains, as passed.
        sa.Column(
            "filters",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        # What came back, in rank order. An array rather than a pivot table:
        # one INSERT instead of a fan-out, and `= ANY(result_ids)` /
        # `unnest(result_ids)` answer both the "was this ever recalled?" and
        # "how often?" questions directly.
        sa.Column(
            "result_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        sa.Column("n_results", sa.Integer(), nullable=False, server_default=sa.text("0")),
        # Best *ranking* score (rerank score where reranking ran, else cosine).
        # NULL when nothing came back — which is the interesting case.
        sa.Column("top_score", sa.Float(), nullable=True),
        sa.Column("reranked", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("sessions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        # agent:claude-code / user:anjum when the caller declared itself.
        sa.Column("source", sa.Text(), nullable=True),
        sa.Column("retention_until", postgresql.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("valid_until", postgresql.TIMESTAMP(timezone=True), nullable=True),
    )
    # Time-ordered reads ("what happened this week") are the common access
    # pattern for every report built on this.
    op.create_index("idx_retrievals_at", "retrievals", ["at"])
    op.create_index("idx_retrievals_command", "retrievals", ["command"])
    # The retention sweep and the "still live" reports both filter on this.
    op.create_index("idx_retrievals_retention", "retrievals", ["retention_until"])
    # "Which annotations were never recalled?" is a containment query over the
    # array; without GIN it degenerates into a scan per annotation.
    op.create_index(
        "idx_retrievals_result_ids",
        "retrievals",
        ["result_ids"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("idx_retrievals_result_ids", table_name="retrievals")
    op.drop_index("idx_retrievals_retention", table_name="retrievals")
    op.drop_index("idx_retrievals_command", table_name="retrievals")
    op.drop_index("idx_retrievals_at", table_name="retrievals")
    op.drop_table("retrievals")
