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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hafiz.core.chunker import (
    EmbeddingPart,
    compute_hash,
    prepare_embedding_parts,
)
from hafiz.core.database import (
    Commit,
    Edge,
    Embedding,
    File,
    Unit,
    UnitRevision,
    get_session_factory,
)
from hafiz.core.embeddings import embed_texts
from hafiz.core.git_context import commit_metadata, is_commit_reachable
from hafiz.core.parsers import ParsedEdge, ParsedUnit, Parser, get_registry
from hafiz.core.tunables import resolve as resolve_tunable

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
    edges_written: int = 0
    edges_superseded: int = 0


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
    now = datetime.now(UTC)
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


async def _current_revision(session: AsyncSession, unit_id: uuid.UUID) -> UnitRevision | None:
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
        now = datetime.now(UTC)
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
        # so we can assign parts back. Part size comes from the Tunable
        # registry so `hafiz config` / `hafiz doctor --apply` can adjust
        # it per host without touching callsites.
        max_chars = resolve_tunable("embedding.max_part_chars")
        per_rev_parts: list[tuple[UnitRevision, list[EmbeddingPart]]] = []
        for rev, rev_content in changed_revisions:
            per_rev_parts.append((rev, prepare_embedding_parts(rev_content, max_chars=max_chars)))

        all_part_texts: list[str] = [p.content for _, parts in per_rev_parts for p in parts]

        embeddings_written = 0
        if all_part_texts:
            vectors = await embed_fn(all_part_texts)
            if len(vectors) != len(all_part_texts):
                raise RuntimeError(
                    f"Embedder returned {len(vectors)} vectors for {len(all_part_texts)} inputs"
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

        # Edges — same-file resolution, append-only with supersession.
        edges_written, edges_superseded = await _sync_edges(
            session,
            file=file,
            parsed_edges=parse_result.edges,
            source_tag=source_tag,
            commit_hash=commit_hash,
            now=now,
        )

        if owns_session:
            await session.commit()

        return IndexedFileResult(
            file_id=file.id,
            parser_name=parser.name,
            units_seen=len(parse_result.units),
            revisions_created=len(changed_revisions),
            embeddings_written=embeddings_written,
            units_tombstoned=units_tombstoned,
            edges_written=edges_written,
            edges_superseded=edges_superseded,
        )
    finally:
        if owns_session:
            await session.close()


async def _sync_edges(
    session: AsyncSession,
    *,
    file: File,
    parsed_edges: list[ParsedEdge],
    source_tag: str,
    commit_hash: str | None,
    now: datetime,
) -> tuple[int, int]:
    """Reconcile the edges table against a fresh parse of ``file``.

    Same-file resolution only: ``target_name`` binds to a local Unit's id
    if the name exactly matches; otherwise ``target_unit_id`` stays null
    and ``target_name`` carries the unresolved (probably external)
    reference. Cross-file / cross-project resolution is a later concern
    (scoped to Phase 4b).

    Edges are compared by their identity tuple ``(source_unit_id,
    target_unit_id, target_name, relation)``. Edges in the new parse but
    not in the current edge set are inserted; edges in the current set
    but not in the new parse get ``superseded_at = now``. Edges that
    appear in both are untouched — zero DB churn on re-ingest of
    unchanged content.

    Edges whose ``source_name`` can't be resolved to a unit are dropped
    silently. That only happens for parser bugs or edges synthesised
    outside the parsed file — genuine code edges always have a resolvable
    source.

    The ``edges`` CHECK constraint restricts ``source`` to
    ``{'ast','agent','user'}``. Parser callers with ``source_tag ==
    'parser'`` (prose / whole-file) produce no edges in practice — their
    :class:`~hafiz.core.parsers.ParseResult.edges` is empty. Guard here
    anyway: map ``'parser'`` → skip entirely so we don't violate the CHECK.
    """
    if source_tag not in ("ast", "agent", "user"):
        # Non-edge-producing parsers (prose, whole_file) — skip cleanly.
        if parsed_edges:
            # This would be a parser bug; fail loudly so it's caught in tests.
            raise ValueError(
                f"Parser with source_tag={source_tag!r} produced "
                f"{len(parsed_edges)} edges; only 'ast' may write to edges."
            )
        # Still supersede any stale edges from a previous run.
        parsed_edges = []
        source_tag_for_writes = "ast"  # unreachable; placate type checker
    else:
        source_tag_for_writes = source_tag

    # ── Resolve source_name / target_name against the file's current units ──
    file_unit_stmt = select(Unit).where(Unit.file_id == file.id, Unit.valid_until.is_(None))
    file_units = (await session.execute(file_unit_stmt)).scalars().all()
    by_name: dict[str, Unit] = {u.name: u for u in file_units}

    new_edges: list[dict] = []
    for pe in parsed_edges:
        src = by_name.get(pe.source_name)
        if src is None:
            # Source didn't resolve — probably a parser quirk. Drop.
            continue
        tgt = by_name.get(pe.target_name)
        new_edges.append(
            {
                "source_unit_id": src.id,
                "target_unit_id": tgt.id if tgt is not None else None,
                "target_name": pe.target_name,
                "relation": pe.relation,
                "evidence": pe.evidence,
                "line": pe.line,
            }
        )

    # Dedupe new edges by identity tuple — two syntactically identical
    # edges from the parser (e.g. the same import emitted twice) collapse.
    def _fp(e: dict) -> tuple:
        return (e["source_unit_id"], e["target_unit_id"], e["target_name"], e["relation"])

    new_by_fp: dict[tuple, dict] = {}
    for e in new_edges:
        new_by_fp.setdefault(_fp(e), e)

    # ── Fetch current (non-superseded) edges whose source is in this file ──
    if file_units:
        current_stmt = select(Edge).where(
            Edge.source_unit_id.in_([u.id for u in file_units]),
            Edge.superseded_at.is_(None),
        )
        current_edges = (await session.execute(current_stmt)).scalars().all()
    else:
        current_edges = []

    current_by_fp: dict[tuple, Edge] = {}
    for e in current_edges:
        fp = (e.source_unit_id, e.target_unit_id, e.target_name, e.relation)
        current_by_fp[fp] = e

    # ── Supersede edges that no longer appear in the parse ──
    superseded = 0
    for fp, edge in current_by_fp.items():
        if fp not in new_by_fp:
            edge.superseded_at = now
            superseded += 1

    # ── Insert edges that aren't in the current set ──
    written = 0
    for fp, e in new_by_fp.items():
        if fp in current_by_fp:
            continue
        session.add(
            Edge(
                id=uuid.uuid4(),
                source_unit_id=e["source_unit_id"],
                target_unit_id=e["target_unit_id"],
                target_name=e["target_name"],
                relation=e["relation"],
                source=source_tag_for_writes,
                evidence=e["evidence"],
                commit_hash=commit_hash,
                observed_at=now,
                metadata_={"line": e["line"]} if e["line"] else {},
            )
        )
        written += 1

    if written or superseded:
        await session.flush()
    return written, superseded


async def upsert_commit(
    sha: str,
    *,
    project: str | None,
    cwd: Path,
    session: AsyncSession | None = None,
) -> Commit | None:
    """Record a commit's metadata in the ``commits`` table. Idempotent —
    an existing row for the same hash is updated with fresh metadata
    (author / summary can drift after an amend).

    Returns the ORM object, or None if the commit isn't reachable from
    ``cwd`` (e.g. rewritten away) — callers can treat that as "we
    couldn't pin the commit, keep going".
    """
    if not sha:
        return None
    meta = commit_metadata(sha, cwd)
    if meta is None:
        return None

    owns_session = session is None
    if owns_session:
        factory = get_session_factory()
        session = factory()
    try:
        existing = await session.get(Commit, sha)
        if existing is None:
            row = Commit(
                hash=sha,
                project=project,
                author=meta["author"],
                committed_at=meta["committed_at"],
                summary=meta["summary"],
            )
            session.add(row)
        else:
            existing.project = existing.project or project
            existing.author = meta["author"]
            existing.committed_at = meta["committed_at"]
            existing.summary = meta["summary"]
            row = existing
        await session.flush()
        if owns_session:
            await session.commit()
        return row
    finally:
        if owns_session:
            await session.close()


async def latest_indexed_commit(project: str | None) -> str | None:
    """Return the most-recent ``last_seen_commit`` across the project's
    currently-present files. Used as the base SHA for diff-driven
    ingest: files unchanged since this commit can be skipped."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(File.last_seen_commit)
            .where(File.valid_until.is_(None))
            .where(File.last_seen_commit.is_not(None))
        )
        if project is not None:
            stmt = stmt.where(File.project == project)
        rows = (await session.execute(stmt)).scalars().all()
    if not rows:
        return None
    # "Latest" here means whatever SHA appears most often as the last-seen
    # value — a reasonable proxy for "the commit the project was ingested at".
    # We don't need strict max-by-committed-at because the caller validates
    # reachability via `git merge-base --is-ancestor` anyway.
    from collections import Counter

    counter = Counter(rows)
    return counter.most_common(1)[0][0]


async def reconcile_orphaned_commits(
    project: str | None,
    cwd: Path,
) -> int:
    """Mark commits orphaned by rebase / force-push as ``rewritten_at = now``.

    Belt-and-braces for the ``post-rewrite`` hook: even if the hook was
    skipped (or never installed), this pass keeps the ``commits`` table
    honest over time. Walks every commit row whose ``rewritten_at`` is
    still null and checks git reachability from any ref. Unreachable
    rows get a timestamp so downstream queries can distinguish history
    from "still live".

    ``rewritten_to`` is left null here — inferring the successor
    requires reflog / patch-id matching, which is Phase 5b-follow-up.
    The ``post-rewrite`` hook is the primary path for populating it.
    """
    if not is_commit_reachable("HEAD", cwd):
        # Either not in a git repo or HEAD itself is orphaned — both
        # cases mean we can't draw reliable conclusions. Skip.
        return 0

    session_factory = get_session_factory()
    now = datetime.now(UTC)
    reconciled = 0

    async with session_factory() as session:
        stmt = select(Commit).where(Commit.rewritten_at.is_(None))
        if project is not None:
            stmt = stmt.where(Commit.project == project)
        commits = (await session.execute(stmt)).scalars().all()

        for commit in commits:
            if not is_commit_reachable(commit.hash, cwd):
                commit.rewritten_at = now
                reconciled += 1

        if reconciled:
            await session.commit()

    return reconciled


async def tombstone_vanished_files(
    project: str | None,
    seen_paths: set[str],
    *,
    session: AsyncSession | None = None,
) -> int:
    """Mark files in DB under ``project`` whose paths weren't seen this
    pass as tombstoned. Their units cascade-tombstone via the same
    pass in the next ingest or via explicit prune. Returns the count."""
    now = datetime.now(UTC)
    owns_session = session is None
    if owns_session:
        factory = get_session_factory()
        session = factory()
    try:
        stmt = select(File).where(File.project == project, File.valid_until.is_(None))
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
