"""Drop embeddings of superseded revisions

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-30

Bodies are canonical, embeddings are an index. Search only ever reads current
revisions (``superseded_at IS NULL``), and no code path un-supersedes one — a
unit that reappears gets a fresh revision with fresh vectors. So the vectors of
a superseded revision can never be read again, yet ingest kept every one.

Measured on a live store: 97,227 of 179,977 embeddings (54%) belonged to
superseded revisions — about 500 MB of a 902 MB table — and every unit search
still had to scan past them.

From this revision on, ingest drops a revision's vectors when it supersedes it
(``hafiz.core.store._supersede``). This migration clears the backlog. The old
bodies stay in ``unit_revisions``; history loses nothing.

Postgres returns the space to the table for reuse, not to the filesystem. To
shrink the file itself, run ``VACUUM FULL embeddings`` once (it locks the table
while it rewrites it).

Downgrade is a no-op: the rows were unreachable, so there is nothing to restore.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "DELETE FROM embeddings WHERE unit_revision_id IN "
        "(SELECT id FROM unit_revisions WHERE superseded_at IS NOT NULL)"
    )


def downgrade() -> None:
    pass
