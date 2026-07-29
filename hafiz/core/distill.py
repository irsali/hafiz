"""Distill — surface recent raw captures (notes + transcripts + messages)
as promotable candidates.

Propose, don't auto-apply. This module is a **scanner**: it returns the
ids + content of recent ``kind="note"`` annotations, legacy transcripts,
and (Phase 5) source-layer ``communication_messages`` so the agent /
user can read them and decide what (if anything) to promote into a
``decision`` / ``learning`` / ``pattern`` via a follow-up ``hafiz
observe`` call with ``--derived-from``.

Explicitly NOT an LLM call. Hafiz stays sovereign; the distillation
judgement is delegated to whoever reads the candidates.

Phase 5 enrichment: when a session filter is provided (or there's an
active session), source-layer message ids are surfaced as well, so a
distilled decision can cite turns directly via the polymorphic
``annotation_targets`` pivot.

Capture-table transcripts: currently empty (Phase 3b-2 rewires
:mod:`hafiz.core.capture` onto the new schema). Until then, only
note-kind annotations and source-layer messages are surfaced.

**The backlog drains itself.** A candidate stops being a candidate two
ways, both of which fall out of state that already exists — there is no
"distilled" flag and no proposals table:

- **Promoted** — some live annotation cites it via ``annotation_targets``
  with ``relation='derived_from'``. That is exactly what a successful
  promotion writes, so citing a note removes it from the queue.
- **Declined** — ``hafiz forget <id> --annotation`` retires the note. The
  candidate query only ever returned live rows, so a retired note is
  already gone.

Without this the same notes resurface every run forever, which is worse
than having no backlog at all.

**Themes.** Candidates are grouped by embedding similarity so the reader
sees "these four captures are one topic, here is the one ``observe`` that
covers them" instead of a flat list. Same single-linkage clustering
``reconcile`` uses, at a much lower threshold — see
:class:`~hafiz.core.config.DistillSettings`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select

from hafiz.core.database import (
    Annotation,
    AnnotationTarget,
    Communication,
    CommunicationMessage,
    get_session_factory,
)
from hafiz.core.journal import JournalCapture, fetch_captures


@dataclass
class NoteCandidate:
    id: str
    content: str
    valid_from: datetime
    source: str | None
    project: str | None
    tags: list[str] | None
    session_id: str | None
    task: str | None
    # A live annotation already cites this note via --derived-from, so it
    # has been distilled. Excluded from the backlog unless asked for.
    promoted: bool = False


@dataclass
class MessageCandidate:
    """Source-layer turn surfaced as a distillation source."""

    id: str
    communication_id: str
    seq: int
    role: str
    author: str | None
    content: str
    ts: datetime
    marked_salient: bool


@dataclass
class ThemeMember:
    """One capture inside a theme, flattened to what a reader judges on."""

    id: str
    kind: str  # "note" | "message"
    content: str
    ts: datetime
    label: str | None  # note source, or message role


@dataclass
class Theme:
    """A group of captures that look like they're about the same thing.

    Singletons are themes too — a capture nobody else resembles is still one
    unit of distillation work, and dropping it would silently shrink the
    backlog. ``size == 1`` distinguishes them.
    """

    members: list[ThemeMember]
    score: float  # best pairwise similarity inside the theme; 1.0 alone

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def newest(self) -> datetime:
        return max(m.ts for m in self.members)

    @property
    def oldest(self) -> datetime:
        return min(m.ts for m in self.members)


@dataclass
class Backlog:
    """The queue's shape, so a caller can gate on it without reading rows."""

    pending: int
    promoted: int
    oldest_pending_age_days: float | None
    themes: int
    clustered: int  # members sitting in a theme of size >= 2
    skipped_unembedded: int  # turns dropped from themes for having no vector

    def to_dict(self) -> dict:
        return {
            "pending": self.pending,
            "promoted": self.promoted,
            "oldest_pending_age_days": self.oldest_pending_age_days,
            "themes": self.themes,
            "clustered": self.clustered,
            "skipped_unembedded": self.skipped_unembedded,
        }


@dataclass
class DistillBundle:
    window_start: datetime
    window_end: datetime
    notes: list[NoteCandidate] = field(default_factory=list)
    transcripts: list[JournalCapture] = field(default_factory=list)
    messages: list[MessageCandidate] = field(default_factory=list)
    themes: list[Theme] = field(default_factory=list)
    backlog: Backlog | None = None


async def find_distill_candidates(
    *,
    since: timedelta | None = None,
    project: str | list[str] | None = None,
    session_id: str | None = None,
    task: str | None = None,
    include_transcripts: bool = True,
    include_promoted: bool = False,
    limit: int = 200,
    message_limit: int | None = None,
    cluster_threshold: float | None = None,
) -> DistillBundle:
    """Return the promotable backlog for ``[now-since, now]``.

    Expired / superseded notes are excluded — already handled elsewhere,
    don't drag them into fresh distillation. Notes a live annotation already
    cites via ``--derived-from`` are excluded too: they've been distilled,
    and re-offering them is how a backlog stops being believable.
    ``include_promoted`` keeps them, flagged, for auditing the drain.

    Candidates are grouped into :class:`Theme` s by embedding similarity, and
    the queue's shape is summarised in :class:`Backlog` so a caller can gate
    on it — see ``hafiz distill --brief``.
    """
    from hafiz.core.config import load_settings

    cfg = load_settings().distill
    msg_limit = cfg.message_limit if message_limit is None else message_limit
    thr = cfg.cluster_threshold if cluster_threshold is None else cluster_threshold

    now = datetime.now(UTC)
    start = now - (since or timedelta(days=7))
    end = now

    def _scoped(stmt):
        """Window + scope filters, shared by the candidate and count queries."""
        stmt = (
            stmt.where(Annotation.kind == "note")
            .where(Annotation.valid_from >= start)
            .where(Annotation.valid_from <= end)
            .where((Annotation.valid_until.is_(None)) | (Annotation.valid_until > now))
        )
        if isinstance(project, list):
            stmt = stmt.where(Annotation.project.in_(project))
        elif project:
            stmt = stmt.where(Annotation.project == project)
        if session_id:
            # Filter on legacy_session_id (the historical text slug). Phase 2
            # will resolve user-supplied slugs to the new uuid FK and union.
            stmt = stmt.where(Annotation.legacy_session_id == session_id)
        if task:
            stmt = stmt.where(Annotation.task == task)
        return stmt

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            _scoped(select(Annotation, _promoted_flag()))
            .order_by(Annotation.valid_from.desc())
            .limit(limit)
        )
        # Drop promoted rows in SQL, not after, so ``limit`` is spent on
        # candidates. Filtering in Python meant a window whose notes were
        # mostly distilled returned a near-empty backlog and reported the cap
        # as if it had bitten on real work.
        if not include_promoted:
            stmt = stmt.where(~_promoted_exists())
        rows = list((await session.execute(stmt)).all())

        # Counted separately, and deliberately un-capped: ``promoted`` is the
        # metric that says whether the loop is working at all, so it must not
        # depend on how many candidates happened to fit under ``limit``.
        promoted_count = (
            await session.execute(
                _scoped(select(func.count()).select_from(Annotation)).where(_promoted_exists())
            )
        ).scalar() or 0

    notes: list[NoteCandidate] = []
    vectors: dict[str, list[float]] = {}
    for ann, is_promoted in rows:
        notes.append(
            NoteCandidate(
                id=str(ann.id),
                content=ann.content,
                valid_from=ann.valid_from,
                source=ann.source,
                project=ann.project,
                tags=ann.tags,
                session_id=(
                    ann.legacy_session_id or (str(ann.session_id) if ann.session_id else None)
                ),
                task=ann.task,
                promoted=bool(is_promoted),
            )
        )
        if ann.embedding is not None:
            vectors[str(ann.id)] = list(ann.embedding)

    transcripts: list[JournalCapture] = []
    if include_transcripts:
        transcripts = await fetch_captures(
            start=start,
            end=end,
            project=project,
            session_id=session_id,
            task=task,
        )

    messages, message_vectors = await _fetch_message_candidates(
        start=start,
        end=end,
        project=project,
        session_slug=session_id,
        limit=msg_limit,
    )
    vectors.update(message_vectors)

    themes = cluster_candidates(notes, messages, vectors, threshold=thr)
    # Pending is derived from the themes, not from the raw candidate lists, so
    # "56 pending in 40 themes" can't disagree with itself and the ``--brief``
    # gate counts only work a reader can actually act on.
    members = [m for t in themes for m in t.members]
    oldest = min((m.ts for m in members), default=None)

    return DistillBundle(
        window_start=start,
        window_end=end,
        notes=notes,
        transcripts=transcripts,
        messages=messages,
        themes=themes,
        backlog=Backlog(
            pending=len(members),
            promoted=promoted_count,
            oldest_pending_age_days=(
                round((now - oldest).total_seconds() / 86400, 1) if oldest else None
            ),
            themes=len(themes),
            clustered=sum(t.size for t in themes if t.size > 1),
            skipped_unembedded=sum(1 for m in messages if m.id not in vectors),
        ),
    )


def _promoted_flag():
    """:func:`_promoted_exists` as a selectable ``promoted`` column."""
    return _promoted_exists().label("promoted")


def _promoted_exists():
    """``EXISTS``: a live annotation cites this note via ``--derived-from``.

    That citation is written by ``observe --derived-from`` into
    ``annotation_targets``, so it is the promotion receipt — no extra column
    needed. A citation from an annotation that was itself retired or expired
    doesn't count: the distilled belief is gone, so the note is promotable
    again.
    """
    citing = Annotation.__table__.alias("citing")
    now = datetime.now(UTC)
    return (
        select(1)
        .select_from(AnnotationTarget)
        .join(citing, citing.c.id == AnnotationTarget.annotation_id)
        .where(AnnotationTarget.target_kind == "annotation")
        .where(AnnotationTarget.target_id == Annotation.id)
        .where(AnnotationTarget.relation == "derived_from")
        .where(citing.c.valid_from <= now)
        .where(or_(citing.c.valid_until.is_(None), citing.c.valid_until > now))
        .exists()
    )


def cluster_candidates(
    notes: list[NoteCandidate],
    messages: list[MessageCandidate],
    vectors: dict[str, list[float]],
    *,
    threshold: float,
) -> list[Theme]:
    """Group candidates into themes by embedding similarity.

    Reuses ``reconcile``'s single-linkage clustering — same maths, looser
    threshold.

    **An unembedded turn is not a candidate.** Selective embedding already
    declined it at import: short turns and pure tool-result echoes never get a
    vector. Surfacing them here would contradict that judgement, and it does
    so at the worst possible ratio — on a real window it buried two useful
    themes under ~26 singletons reading "Let me connect to your browser" and
    raw ``<toolCall>`` echoes. Notes are different: a note is a deliberate
    capture, so an unembedded one stays as a singleton rather than vanishing.

    Ordered biggest theme first, then most recent, so the head of the list is
    where distillation pays best.
    """
    from hafiz.core.annotations import _similarity_matrix, _single_linkage

    members = [
        ThemeMember(id=n.id, kind="note", content=n.content, ts=n.valid_from, label=n.source)
        for n in notes
        if not n.promoted
    ] + [
        ThemeMember(id=m.id, kind="message", content=m.content, ts=m.ts, label=m.role)
        for m in messages
        if m.id in vectors
    ]
    if not members:
        return []

    embedded = [m for m in members if m.id in vectors]
    themes = [Theme(members=[m], score=1.0) for m in members if m.id not in vectors]

    if embedded:
        sim = _similarity_matrix([vectors[m.id] for m in embedded])
        for idxs in _single_linkage(sim, threshold):
            group = [embedded[i] for i in idxs]
            best = (
                max(float(sim[i, j]) for i in idxs for j in idxs if i != j)
                if len(idxs) > 1
                else 1.0
            )
            themes.append(Theme(members=sorted(group, key=lambda m: m.ts), score=best))

    themes.sort(key=lambda t: (t.size, t.newest), reverse=True)
    return themes


async def _fetch_message_candidates(
    *,
    start: datetime,
    end: datetime,
    project: str | list[str] | None,
    session_slug: str | None,
    limit: int = 50,
) -> tuple[list[MessageCandidate], dict[str, list[float]]]:
    """Surface source-layer turns in the distillation window.

    Filters:
    - ``ts`` within [start, end]
    - communication is not tombstoned **and not past its retention_until**
    - if session_slug is set, restrict to communications belonging to
      that session (so distillation lineage can cite the actual turns)
    - if project filter is set, restrict to communications scoped to it

    Retention is a filter here, not a nicety. Distilling an expired turn
    would mint a fresh annotation carrying its content, and that annotation
    has no ``retention_until`` — the source data would outlive the window it
    was promised to expire in. The sweep (``forget --all-expired``) can lag;
    this must not.

    Ordering is salient-first, then newest — the cap has to fall on the least
    useful end of a busy window. It used to be ``ts ASC`` under a hardcoded
    50, so a wide window surfaced the fifty *oldest* turns and ignored
    ``marked_salient`` entirely.

    Returns the candidates plus whatever embeddings they carry, so themes can
    be built without a second pass. Selective embedding means many turns have
    none; those are returned without a vector rather than skipped.
    """
    session_factory = get_session_factory()
    now = datetime.now(UTC)
    async with session_factory() as session:
        from hafiz.core.sessions import get_session_by_slug

        stmt = (
            select(CommunicationMessage)
            .join(
                Communication,
                Communication.id == CommunicationMessage.communication_id,
            )
            .where(CommunicationMessage.ts >= start)
            .where(CommunicationMessage.ts <= end)
            .where(Communication.valid_until.is_(None))
            .where(
                or_(
                    Communication.retention_until.is_(None),
                    Communication.retention_until > now,
                )
            )
            .order_by(
                CommunicationMessage.marked_salient.desc(),
                CommunicationMessage.ts.desc(),
            )
            .limit(limit)
        )

        if isinstance(project, list):
            stmt = stmt.where(Communication.scope_value.in_(project))
        elif project:
            stmt = stmt.where(Communication.scope_value == project)

        if session_slug:
            sess = await get_session_by_slug(session_slug)
            if sess is not None:
                stmt = stmt.where(Communication.session_id == sess.id)
            else:
                # Session slug doesn't resolve — surface nothing rather
                # than every message in the window.
                return [], {}

        rows = (await session.execute(stmt)).scalars().all()

    candidates = [
        MessageCandidate(
            id=str(m.id),
            communication_id=str(m.communication_id),
            seq=m.seq,
            role=m.role,
            author=m.author,
            content=m.content,
            ts=m.ts,
            marked_salient=m.marked_salient,
        )
        for m in rows
    ]
    vectors = {str(m.id): list(m.embedding) for m in rows if m.embedding is not None}
    return candidates, vectors


def brief_gate_open(backlog: Backlog | None, *, min_pending: int, min_age_days: float) -> bool:
    """Is the backlog worth interrupting a session for?

    Either condition fires: enough pending captures, or one that's been
    waiting long enough. Below both, ``--brief`` prints nothing — a session
    hook needs "quiet" to be the ordinary answer, or it gets removed.
    """
    if backlog is None or backlog.pending <= 0:
        return False
    if backlog.pending >= min_pending:
        return True
    age = backlog.oldest_pending_age_days
    return age is not None and age >= min_age_days
