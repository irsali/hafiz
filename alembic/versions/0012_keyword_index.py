"""Full-text index on annotation content

Revision ID: 0012
Revises: 0011
Create Date: 2026-11-26

Recall was vector-only, and an embedding smears exact strings — a port, an
error code, a file name. Recall now also runs a keyword search over annotation
text and merges its hits with the vector candidates before the cross-encoder.

Postgres: a stored generated ``tsvector`` column and a GIN index. The database
keeps the column in sync, so no write path changes. Adding a stored generated
column rewrites the table once.

SQLite: an FTS5 table kept in sync by triggers, backfilled from existing rows.

Both are created by ``hafiz.core.dialect.ensure_keyword_index``, shared with
the places that build a schema without replaying migrations (a new embedded
store, ``migrate-backend``'s target). The ORM does not model either object;
``alembic/env.py`` hides them from autogenerate.

Downgrade drops the index. Nothing else depends on it: recall falls back to
vector-only when the index is missing.
"""

from collections.abc import Sequence

from alembic import op
from hafiz.core.dialect import drop_keyword_index, ensure_keyword_index

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    ensure_keyword_index(op.get_bind())


def downgrade() -> None:
    drop_keyword_index(op.get_bind())
