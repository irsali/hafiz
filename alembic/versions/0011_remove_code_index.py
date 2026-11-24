"""Remove the code index — code units and the edges table

Revision ID: 0011
Revises: 0010
Create Date: 2026-11-23

Hafiz used to parse Python and JS/TS into functions and classes (``code.*``
units) and record calls, imports and inheritance between them (``edges``).
Measured on a live store over 120 days, agents never ran the commands that read
that graph, no note was ever attached to a code unit, and the code units were
~8% of the store. The parsers, the graph commands and ``extract`` are gone, so
nothing would write these rows again or read them.

Code files are not dropped from search: they are indexed as whole files
(``file.raw``) like any other text. Ingest only re-reads files that changed
since the last indexed commit, so this migration clears ``last_seen_commit`` for
every project that had code. The next ingest of such a project then walks every
file once; unchanged documents are skipped by hash, and code comes back as
whole files.

Plain SQL only: embedded (SQLite) stores run this migration too, and SQLite
applies ``ON DELETE CASCADE`` only when foreign keys are switched on, so rows
are deleted child-first instead of relying on it.

Downgrade recreates an empty ``edges`` table. The deleted units cannot be
restored — the parsers that produced them no longer exist.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_CODE_UNITS = "SELECT id FROM units WHERE kind LIKE 'code.%'"
_CODE_REVISIONS = f"SELECT id FROM unit_revisions WHERE unit_id IN ({_CODE_UNITS})"


def upgrade() -> None:
    # Make the next ingest of every affected project a full walk. The second
    # statement covers code files with no project, which ``IN`` cannot match.
    code_files = "SELECT file_id FROM units WHERE kind LIKE 'code.%'"
    op.execute(
        "UPDATE files SET last_seen_commit = NULL WHERE project IN "
        f"(SELECT project FROM files WHERE id IN ({code_files}))"
    )
    op.execute(f"UPDATE files SET last_seen_commit = NULL WHERE id IN ({code_files})")

    # The table references units, so it goes before them.
    op.execute("DROP TABLE IF EXISTS edges")

    op.execute(f"DELETE FROM embeddings WHERE unit_revision_id IN ({_CODE_REVISIONS})")
    op.execute(f"UPDATE annotations SET unit_id = NULL WHERE unit_id IN ({_CODE_UNITS})")
    op.execute(
        "DELETE FROM annotation_targets WHERE target_kind = 'unit' "
        f"AND target_id IN ({_CODE_UNITS})"
    )
    op.execute(
        f"UPDATE unit_revisions SET superseded_by = NULL WHERE superseded_by IN ({_CODE_REVISIONS})"
    )
    op.execute(f"DELETE FROM unit_revisions WHERE unit_id IN ({_CODE_UNITS})")
    op.execute("DELETE FROM units WHERE kind LIKE 'code.%'")


def downgrade() -> None:
    op.create_table(
        "edges",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "source_unit_id",
            sa.Uuid(),
            sa.ForeignKey("units.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "target_unit_id",
            sa.Uuid(),
            sa.ForeignKey("units.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("target_name", sa.Text(), nullable=True),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("commit_hash", sa.Text(), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.CheckConstraint("source IN ('ast', 'agent', 'user')", name="ck_edges_source"),
    )
    op.create_index("idx_edges_source", "edges", ["source_unit_id"])
    op.create_index("idx_edges_target", "edges", ["target_unit_id"])
    op.create_index("idx_edges_target_name", "edges", ["target_name"])
    op.create_index("idx_edges_relation", "edges", ["relation"])
    op.create_index("idx_edges_commit", "edges", ["commit_hash"])
