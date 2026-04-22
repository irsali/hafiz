"""Write path for the structural-grounding schema.

``index_file`` is the heart of the ingest pipeline:

    enumerate units (via a Parser)
      upsert File row
      for each ParsedUnit:
          upsert Unit (keyed by identity_key)
          hash-compare body against current revision
          if changed: insert new revision + supersede old
                      split body into embedding parts
                      embed + insert embeddings
      tombstone units seen in DB but not in this parse

``tombstone_vanished_files`` handles file-level deletions across a ingest
pass: files present in DB under the project but not in the seen set get
their ``valid_until`` set and their current units cascade-tombstoned.

Nothing here knows about AST, Markdown, or Go — that's the Parser's job.
This module only knows "here's a parse result, persist it idempotently".
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hafiz.core.chunker import (
    EmbeddingPart,
    compute_hash,
    prepare_embedding_parts,
)
from hafiz.core.database import (
    Embedding,
    File,
    Unit,
    UnitRevision,
    get_session_factory,
)
from hafiz.core.embeddings import embed_texts
from hafiz.core.parsers import ParsedUnit, Parser, get_registry


EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass
class IndexedFileResult:
    """Return shape from :func:`index_file` — enough for the ingest
    command's progress output and for tests to assert idempotency."""

    file_id: uuid.UUID
    parser_name: str
    units_seen: int
    revisions_created: int
    embeddings_written: int
    units_tombstoned: int


def compute_identity_key(
    *,
    project: str | None,
    path: str,
    kind: str,
    name: str,
    parent_name: str | None,
) -> str:
    """Stable hash of the identity tuple. A unit's identity is what does
    **not** change when its body changes. Re-appearing units (renamed
    back, resurrected) hit the same key and re-use the same row."""
    parts = [project or "", path, kind, name, parent_name or ""]
    joined = "\x1f".join(parts)  # ASCII unit separator, can't appear in names
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


async def _upsert_file(
    session: AsyncSession,
    *,
    project: str | None,
    path: str,
    language: str | None,
    commit_hash: str | None,
) -> File:
    """Find the File row for (project, path); create if missing. If the
    row exists but was tombstoned (valid_until set), clear it — the file
    has re-appeared."""
    now = datetime.now(timezone.utc)
    stmt = select(File).where(File.project == project, File.path == path)
    existing = (await session.execute(stmt)).scalar_one_or_none()

    if existing is not None:
        existing.last_seen_commit = commit_hash
        existing.valid_until = None  # un-tombstone if re-appeared
        if language and not existing.language:
            existing.language = language
        return existing

    new_file = File(
        id=uuid.uuid4(),
        project=project,
        path=path,
        language=language,
        first_seen_commit=commit_hash,
        last_seen_commit=commit_hash,
        created_at=now,
    )
    session.add(new_file)
    await session.flush()
    return new_file


async def _current_revision(
    session: AsyncSession, unit_id: uuid.UUID
) -> UnitRevision | None:
    stmt = (
        select(UnitRevision)
        .where(
            UnitRevision.unit_id == unit_id,
            UnitRevision.superseded_at.is_(None),
        )
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()


async def _upsert_unit(
    session: AsyncSession,
    *,
    file: File,
    parsed: ParsedUnit,
    identity_key: str,
    commit_hash: str | None,
) -> Unit:
    """Find-or-create a Unit for this parsed unit's identity. Touches
    last_seen_commit either way; clears any tombstone if the unit
    re-appeared."""
    stmt = select(Unit).where(Unit.identity_key == identity_key)
    existing = (await session.execute(stmt)).scalar_one_or_none()

    if existing is not None:
        existing.last_seen_commit = commit_hash
        existing.valid_until = None
        return existing

    new_unit = Unit(
        id=uuid.uuid4(),
        file_id=file.id,
        kind=parsed.kind,
        name=parsed.name,
        parent_name=parsed.parent_name,
        identity_key=identity_key,
        first_seen_commit=commit_hash,
        last_seen_commit=commit_hash,
    )
    session.add(new_unit)
    await session.flush()
    return new_unit


def _source_tag(parser: Parser) -> str:
    """Map a Parser to its ``unit_revisions.source`` value. AST-flavored
    parsers get ``'ast'``; everything else is generic ``'parser'``."""
    return "ast" if "ast" in parser.name else "parser"


async def index_file(
    abs_path: Path,
    content: str,
    *,
    project: str | None = None,
    commit_hash: str | None = None,
    session: AsyncSession | None = None,
    embed_fn: EmbedFn | None = None,
) -> IndexedFileResult:
    """Parse ``abs_path`` and upsert its units / revisions / embeddings.

    Idempotent: calling twice with unchanged content produces zero new
    revisions and zero new embeddings. A changed body triggers exactly
    one new revision + re-embedding of only its parts.

    The caller is responsible for committing the session. If no session
    is passed, a fresh session is opened and committed before returning.
    """
    if embed_fn is None:
        embed_fn = embed_texts

    owns_session = session is None
    if owns_session:
        factory = get_session_factory()
        session = factory()

    try:
        parser = get_registry().for_path(abs_path)
        parse_result = parser.parse(abs_path, content)
        now = datetime.now(timezone.utc)
        source_tag = _source_tag(parser)

        file = await _upsert_file(
            session,
            project=project,
            path=str(abs_path),
            language=parse_result.language,
            commit_hash=commit_hash,
        )

        seen_identity_keys: set[str] = set()
        changed_revisions: list[tuple[UnitRevision, str]] = []

        for parsed in parse_result.units:
            identity_key = compute_identity_key(
                project=project,
                path=str(abs_path),
                kind=parsed.kind,
                name=parsed.name,
                parent_name=parsed.parent_name,
            )
            seen_identity_keys.add(identity_key)

            unit = await _upsert_unit(
                session,
                file=file,
                parsed=parsed,
                identity_key=identity_key,
                commit_hash=commit_hash,
            )

            content_hash = compute_hash(parsed.content)
            current = await _current_revision(session, unit.id)

            if current is not None and current.content_hash == content_hash:
                # Unchanged — already the current revision. Nothing to do.
                continue

            # Supersede the old revision FIRST, then flush — otherwise the
            # partial unique index (unit_id) WHERE superseded_at IS NULL sees
            # two current revisions for a moment and rejects the insert.
            if current is not None:
                current.superseded_at = now
                await session.flush()

            new_rev = UnitRevision(
                id=uuid.uuid4(),
                unit_id=unit.id,
                content=parsed.content,
                content_hash=content_hash,
                line_start=parsed.line_start,
                line_end=parsed.line_end,
                commit_hash=commit_hash,
                source=source_tag,
                observed_at=now,
            )
            session.add(new_rev)
            await session.flush()

            if current is not None:
                current.superseded_by = new_rev.id
                await session.flush()

            changed_revisions.append((new_rev, parsed.content))

        # Batch-embed all changed revisions together. Group by revision
        # so we can assign parts back.
        per_rev_parts: list[tuple[UnitRevision, list[EmbeddingPart]]] = []
        for rev, rev_content in changed_revisions:
            per_rev_parts.append((rev, prepare_embedding_parts(rev_content)))

        all_part_texts: list[str] = [
            p.content for _, parts in per_rev_parts for p in parts
        ]

        embeddings_written = 0
        if all_part_texts:
            vectors = await embed_fn(all_part_texts)
            if len(vectors) != len(all_part_texts):
                raise RuntimeError(
                    f"Embedder returned {len(vectors)} vectors for "
                    f"{len(all_part_texts)} inputs"
                )
            v_idx = 0
            for rev, parts in per_rev_parts:
                for part in parts:
                    session.add(
                        Embedding(
                            id=uuid.uuid4(),
                            unit_revision_id=rev.id,
                            part_index=part.part_index,
                            content=part.content,
                            content_hash=compute_hash(part.content),
                            embedding=vectors[v_idx],
                            token_span_start=part.token_span_start,
                            token_span_end=part.token_span_end,
                        )
                    )
                    v_idx += 1
                    embeddings_written += 1

        # Tombstone units of this file that weren't in the parse result.
        vanished_stmt = select(Unit).where(
            Unit.file_id == file.id,
            Unit.valid_until.is_(None),
            ~Unit.identity_key.in_(seen_identity_keys or [""]),
        )
        vanished = (await session.execute(vanished_stmt)).scalars().all()
        units_tombstoned = 0
        for v in vanished:
            v.valid_until = now
            # Also supersede its current revision so searches skip it.
            current = await _current_revision(session, v.id)
            if current is not None:
                current.superseded_at = now
            units_tombstoned += 1

        if owns_session:
            await session.commit()

        return IndexedFileResult(
            file_id=file.id,
            parser_name=parser.name,
            units_seen=len(parse_result.units),
            revisions_created=len(changed_revisions),
            embeddings_written=embeddings_written,
            units_tombstoned=units_tombstoned,
        )
    finally:
        if owns_session:
            await session.close()


async def tombstone_vanished_files(
    project: str | None,
    seen_paths: set[str],
    *,
    session: AsyncSession | None = None,
) -> int:
    """Mark files in DB under ``project`` whose paths weren't seen this
    pass as tombstoned. Their units cascade-tombstone via the same
    pass in the next ingest or via explicit prune. Returns the count."""
    now = datetime.now(timezone.utc)
    owns_session = session is None
    if owns_session:
        factory = get_session_factory()
        session = factory()
    try:
        stmt = select(File).where(
            File.project == project, File.valid_until.is_(None)
        )
        all_files = (await session.execute(stmt)).scalars().all()
        tombstoned = 0
        for f in all_files:
            if f.path not in seen_paths:
                f.valid_until = now
                tombstoned += 1
        if owns_session:
            await session.commit()
        return tombstoned
    finally:
        if owns_session:
            await session.close()
