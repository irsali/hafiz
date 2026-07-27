"""Annotation storage and retrieval with vector similarity search.

The "wisdom layer" — decisions, facts, learnings, patterns, warnings, notes.
Annotations may optionally link to a unit (`unit_id`) so they survive body
changes across revisions, or float free as project-level or session-level
knowledge.

Phase 5 adds **polymorphic ``derived_from``**: an annotation may cite
other annotations (knowledge layer) OR communication_messages (source
layer) OR sessions / communications. The link is recorded in the
``annotation_targets`` pivot with ``relation='derived_from'``. The
legacy ``metadata.derived_from`` list is also preserved during the
transition for back-compat with existing readers.

This module replaces the old ``observations.py``. The schema renamed
`observations` → `annotations` and `obs_type` → `kind` as part of the
structural-grounding work (see workitems/done/structural-grounding.md).
The CLI verb stays `hafiz observe` — that's a user-facing name, not a
model reference.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select

from hafiz.core import telemetry
from hafiz.core.database import (
    Annotation,
    AnnotationTarget,
    Communication,
    CommunicationMessage,
    get_session_factory,
)
from hafiz.core.database import (
    Session as SessionRow,
)
from hafiz.core.embeddings import embed_query
from hafiz.core.git_context import current_git_context


async def _classify_target_kind(target_uuid: uuid.UUID) -> str | None:
    """Return ``'annotation'|'message'|'communication'|'session'`` for
    a uuid, or None if no matching row exists. Used to decide what
    ``annotation_targets.target_kind`` to write.
    """
    factory = get_session_factory()
    async with factory() as s:
        if await s.get(Annotation, target_uuid):
            return "annotation"
        if await s.get(CommunicationMessage, target_uuid):
            return "message"
        if await s.get(Communication, target_uuid):
            return "communication"
        if await s.get(SessionRow, target_uuid):
            return "session"
    return None


async def write_derived_from_links(annotation_id: uuid.UUID, derived_from: list[str]) -> list[dict]:
    """Insert annotation_targets rows for each derived_from id.

    Returns one summary dict per id describing how it was classified.
    Unknown uuids are skipped (recorded as ``target_kind=None`` with a
    note) — write-time integrity matters less here than not blocking
    the annotation write itself, since lineage is best-effort metadata.
    """
    if not derived_from:
        return []
    summary: list[dict] = []
    factory = get_session_factory()
    async with factory() as s:
        for raw in derived_from:
            entry: dict = {"id": raw}
            try:
                target_uuid = uuid.UUID(raw)
            except ValueError:
                entry["target_kind"] = None
                entry["note"] = "not-a-uuid"
                summary.append(entry)
                continue
            kind = await _classify_target_kind(target_uuid)
            entry["target_kind"] = kind
            if kind is None:
                entry["note"] = "no-matching-row"
                summary.append(entry)
                continue
            link = AnnotationTarget(
                id=uuid.uuid4(),
                annotation_id=annotation_id,
                target_kind=kind,
                target_id=target_uuid,
                relation="derived_from",
            )
            s.add(link)
            summary.append(entry)
        await s.commit()
    return summary


@dataclass
class AnnotationResult:
    """A single annotation search result with its relevance scores.

    Two scores, deliberately both surfaced:

    * ``score`` — cosine similarity from the vector stage, 0–1. Kept as-is
      for back-compat; live agent hooks read this field.
    * ``rerank_score`` — the cross-encoder's 0–1 relevance, or ``None`` when
      reranking did not run. This is the score the results are *ordered* by
      when it is present, and the one a relevance floor must filter on.

    ``score`` alone cannot tell you which stage ranked the row: under
    reranking it is non-monotonic down the result list, so filtering on it
    fights the ordering.
    """

    id: str
    content: str
    kind: str
    source: str | None
    project: str | None
    tags: list[str] | None
    confidence: float
    valid_from: datetime
    valid_until: datetime | None
    unit_id: str | None
    metadata: dict
    score: float
    rerank_score: float | None = None

    @property
    def ranking_score(self) -> float:
        """The score this row was actually ordered by — rerank if it ran."""
        return self.score if self.rerank_score is None else self.rerank_score


@dataclass
class NearDuplicate:
    """An existing live annotation that closely resembles a pending write."""

    id: str
    content: str
    kind: str
    score: float


class DuplicateAnnotationError(Exception):
    """Raised in strict mode when a near-duplicate exists and the caller
    neither superseded it nor opted out via ``allow_duplicate``.

    Carries the offending ``duplicates`` so the caller can show the agent
    exactly which ids to supersede.
    """

    def __init__(self, duplicates: list[NearDuplicate]):
        self.duplicates = duplicates
        ids = ", ".join(d.id for d in duplicates)
        super().__init__(f"near-duplicate live annotation(s) exist: {ids}")


class ExactDuplicateAnnotationError(Exception):
    """Raised when the write is byte-for-byte an existing live annotation.

    Distinct from :class:`DuplicateAnnotationError`, and deliberately *not*
    subject to the ``strict`` setting. Surface-don't-block is the right design
    for *near*-duplicates, where only the author can judge whether the new row
    refines, contradicts, or merely resembles the old one. An **exact** match on
    the same kind + source + project admits no such judgement: it is never a
    legitimate new annotation. Left unenforced it accumulated silently — a real
    store reached 34 near-duplicate clusters over 80 live annotations, several
    of them character-for-character identical.

    Carries ``existing_id`` so the caller can cite the row it already has.
    """

    def __init__(self, existing_id: str, kind: str):
        self.existing_id = existing_id
        self.kind = kind
        super().__init__(
            f"identical live {kind} already exists: {existing_id} "
            "(supersede it, edit the text, or pass --allow-duplicate)"
        )


async def find_near_duplicates(
    embedding: list[float],
    *,
    kind: str,
    project: str | None,
    threshold: float,
    limit: int = 5,
    exclude_id: uuid.UUID | None = None,
) -> list[NearDuplicate]:
    """Return live annotations of the same ``kind``/``project`` whose cosine
    similarity to ``embedding`` is at or above ``threshold``.

    Scoped to same kind + same project deliberately: a ``decision`` rarely
    duplicates a ``warning``, and cross-project collisions are noise. Only
    *live* rows count — a superseded/expired row is already retired, so
    re-stating its content is not a duplicate. ``exclude_id`` skips a row by
    id (e.g. the freshly-inserted annotation itself).
    """
    now = datetime.now(UTC)
    session_factory = get_session_factory()
    async with session_factory() as session:
        similarity = (1 - Annotation.embedding.cosine_distance(embedding)).label("similarity")
        stmt = (
            select(Annotation, similarity)
            .where(Annotation.embedding.isnot(None))
            .where(Annotation.kind == kind)
            .where(Annotation.valid_from <= now)
            .where((Annotation.valid_until.is_(None)) | (Annotation.valid_until > now))
            .order_by(Annotation.embedding.cosine_distance(embedding))
            .limit(limit)
        )
        if project:
            stmt = stmt.where(Annotation.project == project)
        else:
            stmt = stmt.where(Annotation.project.is_(None))
        if exclude_id is not None:
            stmt = stmt.where(Annotation.id != exclude_id)

        rows = (await session.execute(stmt)).all()

    return [
        NearDuplicate(id=str(ann.id), content=ann.content, kind=ann.kind, score=round(float(s), 4))
        for ann, s in rows
        if float(s) >= threshold
    ]


async def find_exact_duplicate(
    content: str,
    *,
    kind: str,
    source: str | None,
    project: str | None,
) -> Annotation | None:
    """Return the live annotation identical to this write, if one exists.

    Exact string equality on ``content``, scoped to the same kind + source +
    project. Deliberately cheaper than :func:`find_near_duplicates` — no
    embedding, no vector scan — so it can run on every write including the
    ``note`` firehose.

    Only *live* rows count: re-stating something that was superseded or expired
    is a legitimate new assertion, not a duplicate.
    """
    now = datetime.now(UTC)
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(Annotation)
            .where(Annotation.content == content)
            .where(Annotation.kind == kind)
            .where(Annotation.valid_from <= now)
            .where((Annotation.valid_until.is_(None)) | (Annotation.valid_until > now))
            .order_by(Annotation.valid_from)
            .limit(1)
        )
        # NULL-safe comparison: `is_distinct_from` matches NULL == NULL, which a
        # plain `==` would not, so an untagged rewrite of an untagged row still
        # counts as a duplicate.
        stmt = stmt.where(Annotation.source.is_not_distinct_from(source))
        stmt = stmt.where(Annotation.project.is_not_distinct_from(project))
        return (await session.execute(stmt)).scalar_one_or_none()


async def store_annotation(
    content: str,
    *,
    kind: str = "fact",
    source: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 1.0,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    unit_id: str | None = None,
    session_id: str | uuid.UUID | None = None,
    task: str | None = None,
    commit_hash: str | None = None,
    supersedes_id: str | None = None,
    derived_from: list[str] | None = None,
    metadata: dict | None = None,
) -> Annotation:
    """Store a new annotation with its embedding.

    Args:
        content: The annotation text.
        kind: fact, decision, learning, pattern, warning, note, …
        source: Origin (e.g. ``"agent:claude-code"``, ``"user:you"``).
        project: Project name.
        tags: Categorization tags.
        confidence: Confidence score 0.0–1.0.
        valid_from: When the annotation becomes valid (default: now).
        valid_until: When the annotation expires (None = forever).
        unit_id: Optional UUID of a unit this annotation is attached to.
            Survives body revisions via the stable `units.identity_key`.
        session_id: Thread of work — see :mod:`hafiz.core.session`.
        task: Named task within the session.
        commit_hash: Git HEAD when the annotation was made. Auto-captured
            if not provided.
        supersedes_id: UUID of an annotation this one replaces. Sets the
            previous row's ``valid_until = now`` and links via
            ``supersedes_id``. Raises ValueError if the target is missing.
        derived_from: Lineage — list of annotation UUIDs this one was
            distilled from. Stored in ``metadata.derived_from``.
        metadata: Arbitrary JSONB metadata. ``commit_hash`` key is
            promoted to the dedicated column and stripped.

    Returns:
        The stored Annotation ORM object.

    Raises:
        EmptyQueryError: if ``content`` is blank. Blank content is the write-side
            of the empty-query defect: it embeds to a near-zero vector, so the
            row then scores near-perfectly against *every* future query and acts
            as a permanent noise magnet in recall. Refusing the write is the
            only fix that doesn't require sweeping the table later.

    Note:
        Near-duplicate detection is *not* run here — bulk writers (importer,
        extractor, daemon) must stay fast and unconditional. The ``observe``
        command runs :func:`find_near_duplicates` itself before calling this.
    """
    from hafiz.core.search import require_query

    content = require_query(content, what="annotation content", hint="nothing to store")
    embedding = await embed_query(content)

    merged_metadata = dict(metadata or {})
    git_ctx = current_git_context()

    legacy_from_meta = merged_metadata.pop("commit_hash", None)
    resolved_commit_hash = commit_hash or legacy_from_meta or git_ctx.get("commit_hash")

    for key in ("branch", "is_dirty"):
        if key not in merged_metadata and key in git_ctx:
            merged_metadata[key] = git_ctx[key]

    if derived_from:
        merged_metadata["derived_from"] = list(derived_from)

    # Phase 2 session resolution. ``session_id`` may arrive as:
    #   - a real uuid (Phase 2+ callers; importer; future code)
    #   - a slug string (legacy CLI / per-TTY cursor)
    # When it's a slug, look up the sessions table; if a row exists,
    # populate the uuid FK *and* keep the slug on legacy_session_id
    # for human-readable display in journal/distill output. If no row
    # is found, the slug lands in legacy_session_id only.
    legacy_session_value: str | None = None
    session_uuid_value: uuid.UUID | None = None
    if isinstance(session_id, uuid.UUID):
        session_uuid_value = session_id
        from hafiz.core.sessions import get_session_by_id

        found_row = await get_session_by_id(session_id)
        if found_row is not None:
            legacy_session_value = found_row.slug
    elif session_id is not None:
        raw = str(session_id).strip()
        if raw:
            try:
                session_uuid_value = uuid.UUID(raw)
                from hafiz.core.sessions import get_session_by_id

                found_row = await get_session_by_id(session_uuid_value)
                if found_row is not None:
                    legacy_session_value = found_row.slug
            except ValueError:
                # Treat as slug. Look up; if missing, keep as legacy text.
                from hafiz.core.sessions import get_session_by_slug

                found = await get_session_by_slug(raw)
                if found is not None:
                    session_uuid_value = found.id
                    legacy_session_value = found.slug
                else:
                    legacy_session_value = raw

    now = datetime.now(UTC)
    new_ann = Annotation(
        id=uuid.uuid4(),
        content=content,
        embedding=embedding,
        kind=kind,
        source=source,
        project=project,
        tags=tags,
        confidence=confidence,
        valid_from=valid_from or now,
        valid_until=valid_until,
        unit_id=uuid.UUID(unit_id) if unit_id else None,
        legacy_session_id=legacy_session_value,
        session_id=session_uuid_value,
        task=task,
        commit_hash=resolved_commit_hash,
        supersedes_id=uuid.UUID(supersedes_id) if supersedes_id else None,
        metadata_=merged_metadata,
    )

    session_factory = get_session_factory()
    async with session_factory() as session:
        if supersedes_id:
            target = await session.get(Annotation, uuid.UUID(supersedes_id))
            if target is None:
                raise ValueError(f"Cannot supersede {supersedes_id!r}: annotation not found.")
            if target.valid_until is None or target.valid_until > now:
                target.valid_until = now
        session.add(new_ann)
        await session.commit()
        await session.refresh(new_ann)

    # Phase 5 — polymorphic lineage. Write annotation_targets rows for
    # each derived_from id. Done in a separate transaction so a missing
    # target row doesn't roll back the annotation itself.
    if derived_from:
        await write_derived_from_links(new_ann.id, list(derived_from))

    return new_ann


@dataclass
class StoreResult:
    """Outcome of a checked annotation write.

    ``deduped`` is True when nothing was written because an identical live row
    already existed and the caller's lane treats that as success (the ``note``
    firehose). ``annotation`` is then the pre-existing row.
    """

    annotation: Annotation
    near_duplicates: list[NearDuplicate]
    deduped: bool = False


async def store_annotation_checked(
    content: str,
    *,
    kind: str = "fact",
    source: str | None = None,
    project: str | None = None,
    supersedes_id: str | None = None,
    allow_duplicate: bool = False,
    detect_near: bool = True,
    dedupe_silently: bool = False,
    **kwargs,
) -> StoreResult:
    """Check for duplicates, then store — the ``observe`` / ``note`` entry point.

    Two independent checks, because exact and approximate duplication are
    different problems:

    **Exact** (``content`` + ``kind`` + ``source`` + ``project`` all identical to
    a live row) always runs — it is a cheap string comparison, needs no
    embedding, and is never a legitimate write. It ignores ``dedup.strict``,
    which governs only the approximate case. What happens next depends on the
    lane:

    * ``dedupe_silently=False`` (``observe``): raise
      :class:`ExactDuplicateAnnotationError`. The caller *may* have meant
      ``--supersedes``, and a non-zero exit is what makes them look.
    * ``dedupe_silently=True`` (``note``): return the existing row with
      ``deduped=True`` and no error. "Raw capture is never gated" protects the
      caller from having to stop and deliberate; it does not entitle the store
      to keep identical rows. The write is idempotent rather than refused, so
      no caller has to change.

    **Near** (cosine similarity at/above ``dedup.threshold``) runs only when
    ``detect_near`` and ``dedup.enabled`` are set and no ``supersedes_id`` was
    given. Behaviour is unchanged: surface-only by default, blocking under
    ``dedup.strict``.

    ``allow_duplicate`` forces the write past both checks.

    Bulk writers (importer, extractor, daemon) should call
    :func:`store_annotation` directly — they must stay fast and unconditional.
    """
    from hafiz.core.config import load_settings

    dedup_cfg = load_settings().dedup
    near_duplicates: list[NearDuplicate] = []

    if dedup_cfg.enabled and not supersedes_id and not allow_duplicate:
        exact = await find_exact_duplicate(content, kind=kind, source=source, project=project)
        if exact is not None:
            if dedupe_silently:
                return StoreResult(annotation=exact, near_duplicates=[], deduped=True)
            raise ExactDuplicateAnnotationError(str(exact.id), kind)

    if detect_near and dedup_cfg.enabled and not supersedes_id:
        embedding = await embed_query(content)
        near_duplicates = await find_near_duplicates(
            embedding,
            kind=kind,
            project=project,
            threshold=dedup_cfg.threshold,
            limit=dedup_cfg.max_candidates,
        )
        if near_duplicates and dedup_cfg.strict and not allow_duplicate:
            raise DuplicateAnnotationError(near_duplicates)

    ann = await store_annotation(
        content,
        kind=kind,
        source=source,
        project=project,
        supersedes_id=supersedes_id,
        **kwargs,
    )
    return StoreResult(annotation=ann, near_duplicates=near_duplicates)


#: A newest row materially shorter than a sibling it would replace is not a safe
#: plain retire — the longer row probably carries a qualifier (a scope limit, an
#: excluded case) that the restatement dropped. Below this ratio the proposal
#: downgrades to a merge the operator has to write.
_MERGE_IF_SHORTER_THAN = 0.8


@dataclass
class ClusterMember:
    """One annotation inside a near-duplicate cluster.

    Carries what an operator needs to overrule the suggestion — how long it is
    and when it was written — not just the text.
    """

    id: str
    content: str
    kind: str
    score: float
    source: str | None
    valid_from: datetime
    primary: bool


@dataclass
class DuplicateCluster:
    """A group of mutually near-duplicate live annotations, with a proposal.

    Every proposal has the same shape — one row is ``primary``, the rest are
    retired — and ``action`` says what happens to the primary:

    * ``"retire"`` — the primary is the newest row and no shorter than the ones
      it replaces, so it survives untouched and the others are simply dropped.
    * ``"merge"`` — the newest row is materially shorter than a sibling, so the
      primary is instead the *longest* row and the operator has to write merged
      text that supersedes it. Retiring the longer row against a shorter
      restatement would silently drop whatever only it says.
    """

    kind: str
    project: str | None
    members: list[ClusterMember]
    action: str

    @property
    def primary(self) -> ClusterMember:
        """The row the proposal is built around — survivor, or supersede target."""
        return next(m for m in self.members if m.primary)

    @property
    def others(self) -> list[ClusterMember]:
        """The rows the proposal retires."""
        return [m for m in self.members if not m.primary]


@dataclass
class ReconcileReport:
    """Clusters plus the coverage they were derived from.

    Coverage is part of the result, not a footnote: the previous version
    scanned the newest ``limit=500`` rows and reported the clusters it found as
    if they were the whole store. On a 1,099-row deployment that reported 34
    clusters where a full sweep finds 65 — an undercount with nothing in the
    output to suggest one.
    """

    clusters: list[DuplicateCluster]
    scanned: int
    total_live: int
    truncated: bool
    threshold: float


async def reconcile_duplicates(
    *,
    project: str | None = None,
    kind: str | None = None,
    threshold: float | None = None,
    limit: int | None = None,
) -> ReconcileReport:
    """Find clusters of near-duplicate *live* annotations — a read-only sweep.

    The after-the-fact backstop to write-time detection: surfaces drift that
    slipped through (writes made before detection existed, or via bulk paths).
    Resolution stays explicit and manual — this function never mutates; it
    proposes a keeper, and the operator runs ``forget --annotation`` or
    ``observe --supersedes``.

    Clusters are built by single-linkage: each annotation is compared to the
    others of its kind/project via cosine similarity; rows at/above
    ``threshold`` are linked transitively into one cluster. ``threshold``
    defaults to the configured ``dedup.threshold``.

    ``limit`` caps how many live annotations are scanned, newest first;
    ``None`` (the default) scans every one of them and is what makes the
    reported cluster count trustworthy. When a cap does bite, the report says
    so via ``truncated`` — a partial sweep is never silent.
    """
    from hafiz.core.config import load_settings

    cfg = load_settings().dedup
    thr = cfg.threshold if threshold is None else threshold

    now = datetime.now(UTC)
    live = (
        (Annotation.embedding.isnot(None)),
        (Annotation.valid_from <= now),
        ((Annotation.valid_until.is_(None)) | (Annotation.valid_until > now)),
    )
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Annotation).where(*live).order_by(Annotation.valid_from.desc())
        count_stmt = select(func.count()).select_from(Annotation).where(*live)
        if project:
            stmt = stmt.where(Annotation.project == project)
            count_stmt = count_stmt.where(Annotation.project == project)
        if kind:
            stmt = stmt.where(Annotation.kind == kind)
            count_stmt = count_stmt.where(Annotation.kind == kind)
        if limit:
            stmt = stmt.limit(limit)
        rows = list((await session.execute(stmt)).scalars().all())
        total_live = (await session.execute(count_stmt)).scalar() or 0

    # Group by (kind, project), then single-linkage cluster within each group
    # using cosine similarity over the stored embeddings (no re-embedding).
    from collections import defaultdict

    groups: dict[tuple[str, str | None], list[Annotation]] = defaultdict(list)
    for ann in rows:
        groups[(ann.kind, ann.project)].append(ann)

    clusters: list[DuplicateCluster] = []
    for (grp_kind, grp_project), anns in groups.items():
        if len(anns) < 2:
            continue
        # One normalized matrix per group, so the O(n²) comparison happens in
        # BLAS rather than a Python double loop. At 1,099 live rows the pure
        # Python version took ~10s, which is what made scanning everything look
        # unaffordable and the 500-row cap look reasonable.
        sim = _similarity_matrix([a.embedding for a in anns])
        for idxs in _single_linkage(sim, thr):
            if len(idxs) < 2:
                continue
            clusters.append(_build_cluster(grp_kind, grp_project, anns, idxs, sim))

    clusters.sort(key=lambda c: max(m.score for m in c.members), reverse=True)
    return ReconcileReport(
        clusters=clusters,
        scanned=len(rows),
        total_live=total_live,
        truncated=len(rows) < total_live,
        threshold=thr,
    )


def _similarity_matrix(embeddings: list):
    """Pairwise cosine similarity for a group of embeddings."""
    import numpy as np

    mat = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero vector is similar to nothing, not everything
    unit = mat / norms
    return unit @ unit.T


def _single_linkage(sim, threshold: float) -> list[list[int]]:
    """Group indices transitively linked by ``sim >= threshold``."""
    n = sim.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    from collections import defaultdict

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        groups[find(idx)].append(idx)
    return list(groups.values())


def _build_cluster(
    kind: str, project: str | None, anns: list, idxs: list[int], sim
) -> DuplicateCluster:
    """Assemble one cluster and propose which row the resolution is built around.

    Default to the newest row: a near-duplicate written later is usually the
    same insight restated, and the later statement is the one currently
    believed. But when that newest row is materially *shorter* than a sibling,
    keeping it would drop text, so the primary becomes the longest row and the
    action becomes ``merge`` — the operator writes text that supersedes it.

    Either way this is a *suggestion*. Every member ships with its length and
    date so the operator can overrule it.
    """
    newest = max(idxs, key=lambda i: anns[i].valid_from)
    longest = max(idxs, key=lambda i: len(anns[i].content))
    shrinks = len(anns[newest].content) < _MERGE_IF_SHORTER_THAN * len(anns[longest].content)
    action = "merge" if shrinks else "retire"
    primary = longest if shrinks else newest

    members = [
        ClusterMember(
            id=str(anns[i].id),
            content=anns[i].content,
            kind=anns[i].kind,
            # Best similarity to any sibling, so the display leads with the
            # tightest match rather than an arbitrary member.
            score=round(float(max(sim[i, k] for k in idxs if k != i)), 4),
            source=anns[i].source,
            valid_from=anns[i].valid_from,
            primary=(i == primary),
        )
        for i in idxs
    ]
    members.sort(key=lambda m: (not m.primary, -m.score))
    return DuplicateCluster(kind=kind, project=project, members=members, action=action)


async def count_clustered_annotations(threshold: float | None = None) -> int:
    """How many live annotations have at least one near-duplicate sibling.

    The cheap counterpart to :func:`reconcile_duplicates`, for health output
    that only needs a number to act on. Runs entirely in Postgres — no
    embeddings cross the wire — because the full sweep ships every vector to
    Python and is far too slow to sit in a command anyone runs casually.

    Still O(n²) distance computations: ``annotations.embedding`` carries no
    vector index, so this is a sequential scan and it grows quadratically.
    Measured at ~310 ms for 1,099 rows, which is why it belongs in ``doctor``
    and not in ``status``.
    """
    from sqlalchemy import text

    from hafiz.core.config import load_settings

    thr = load_settings().dedup.threshold if threshold is None else threshold
    session_factory = get_session_factory()
    async with session_factory() as session:
        return (
            await session.execute(
                text(
                    "SELECT count(*) FROM annotations a "
                    "WHERE a.valid_until IS NULL AND a.embedding IS NOT NULL "
                    "AND EXISTS (SELECT 1 FROM annotations b "
                    "  WHERE b.valid_until IS NULL AND b.embedding IS NOT NULL "
                    "    AND b.id <> a.id AND b.kind = a.kind "
                    "    AND b.project IS NOT DISTINCT FROM a.project "
                    "    AND (1 - (a.embedding <=> b.embedding)) >= :thr)"
                ),
                {"thr": thr},
            )
        ).scalar() or 0


async def search_annotations(
    query: str,
    *,
    limit: int = 10,
    project: str | list[str] | None = None,
    kind: str | None = None,
    source: str | None = None,
    active_only: bool = True,
    rerank: bool | None = None,
    min_score: float | None = None,
    telemetry_command: str | None = telemetry.OBSERVATIONS,
) -> list[AnnotationResult]:
    """Search annotations by vector similarity, optionally cross-encoder reranked.

    When ``rerank`` is True (or None and ``rerank.enabled`` config is set), the
    vector stage over-fetches ``limit × candidate_multiplier`` candidates and a
    cross-encoder reorders them by joint (query, content) relevance before
    truncating to ``limit``. Reranking is strictly a reordering: on any failure
    it falls back to the vector order. ``rerank=False`` forces pure vector.

    ``telemetry_command`` labels the search in the ``retrievals`` log (default
    ``"query --observations"``); pass ``None`` explicitly to record nothing.

    ``min_score`` is a 0–1 relevance floor applied to
    :attr:`AnnotationResult.ranking_score` — the cross-encoder score when
    reranking ran, the cosine similarity otherwise. It is applied **after**
    reranking and **before** the ``limit`` truncation, so the caller gets up
    to ``limit`` rows that all clear the floor.

    Raises:
        EmptyQueryError: if ``query`` is blank.
    """
    from hafiz.core.config import load_settings
    from hafiz.core.reranker import rerank_scored
    from hafiz.core.search import require_query

    query = require_query(query)
    rerank_cfg = load_settings().rerank
    do_rerank = rerank_cfg.enabled if rerank is None else rerank
    # Over-fetch candidates for the reranker to reorder; it can only improve on
    # what vector recall surfaced, so a wider net helps. Pure-vector path keeps
    # the tight limit.
    fetch_limit = max(limit * rerank_cfg.candidate_multiplier, limit) if do_rerank else limit

    query_embedding = await embed_query(query)

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(
                Annotation,
                (1 - Annotation.embedding.cosine_distance(query_embedding)).label("similarity"),
            )
            .where(Annotation.embedding.isnot(None))
            .order_by(Annotation.embedding.cosine_distance(query_embedding))
            .limit(fetch_limit)
        )

        if isinstance(project, list):
            stmt = stmt.where(Annotation.project.in_(project))
        elif project:
            stmt = stmt.where(Annotation.project == project)
        if kind:
            stmt = stmt.where(Annotation.kind == kind)
        if source:
            stmt = stmt.where(Annotation.source == source)
        if active_only:
            now = datetime.now(UTC)
            stmt = stmt.where(Annotation.valid_from <= now)
            stmt = stmt.where((Annotation.valid_until.is_(None)) | (Annotation.valid_until > now))

        result = await session.execute(stmt)
        rows = result.all()

    candidates = [
        AnnotationResult(
            id=str(ann.id),
            content=ann.content,
            kind=ann.kind,
            source=ann.source,
            project=ann.project,
            tags=ann.tags,
            confidence=ann.confidence,
            valid_from=ann.valid_from,
            valid_until=ann.valid_until,
            unit_id=str(ann.unit_id) if ann.unit_id else None,
            metadata=ann.metadata_ or {},
            score=round(float(similarity), 4),
        )
        for ann, similarity in rows
    ]

    if do_rerank and len(candidates) > 1:
        # Rerank the whole candidate pool without truncating: the floor has to
        # be applied to the reranked order, and truncating first would discard
        # rows that clear it in favour of rows that don't.
        for result, rerank_score in await rerank_scored(
            query, candidates, text_of=lambda r: r.content
        ):
            result.rerank_score = rerank_score
        candidates.sort(key=lambda r: r.ranking_score, reverse=True)

    if min_score is not None:
        candidates = [r for r in candidates if r.ranking_score >= min_score]
    final = candidates[:limit]

    # Recorded here rather than in the CLI: hafiz/core/daemon.py calls this
    # function directly, so command-layer telemetry would silently miss every
    # warm request.
    if telemetry_command:
        await telemetry.record_retrieval(
            command=telemetry_command,
            query=query,
            result_ids=[r.id for r in final],
            top_score=final[0].ranking_score if final else None,
            reranked=any(r.rerank_score is not None for r in final),
            filters={
                "project": project,
                "kind": kind,
                "source": source,
                "limit": limit,
                "min_score": min_score,
                "include_superseded": None if active_only else True,
            },
        )
    return final


async def list_annotations(
    *,
    project: str | None = None,
    kind: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Annotation]:
    """List annotations with optional filters, newest first."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Annotation).order_by(Annotation.valid_from.desc()).limit(limit).offset(offset)
        if project:
            stmt = stmt.where(Annotation.project == project)
        if kind:
            stmt = stmt.where(Annotation.kind == kind)

        result = await session.execute(stmt)
        return list(result.scalars().all())


async def invalidate_annotation(ann_id: str) -> Annotation | None:
    """Invalidate an annotation by setting ``valid_until = now``."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Annotation).where(Annotation.id == uuid.UUID(ann_id))
        result = await session.execute(stmt)
        ann = result.scalar_one_or_none()
        if ann is None:
            return None
        ann.valid_until = datetime.now(UTC)
        await session.commit()
        await session.refresh(ann)
        return ann
