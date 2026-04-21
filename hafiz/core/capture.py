"""Transcript capture — ingest multi-page text as ``chunk_type="transcript"``.

Reuses the existing ``chunks`` pipeline (embed → store → search) with no
migration. Each transcript becomes a group of chunks sharing a
``metadata.transcript_id`` + ``metadata.turn_index`` so retrieval can
reassemble surrounding context.

Synthetic ``source_file`` paths under ``<cwd>/captures/`` tag transcript
rows for prune + display; no actual file is written to disk. ``prune``
is aware and leaves ``chunk_type="transcript"`` rows alone.
"""

from __future__ import annotations

import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import and_, select

from hafiz.core.chunker import ChunkResult, compute_checksum
from hafiz.core.database import Chunk, get_session_factory
from hafiz.core.embeddings import embed_texts
from hafiz.core.search import SearchResult
from hafiz.core.store import store_chunks

TRANSCRIPT_CHUNK_TYPE = "transcript"
EMBED_BATCH_SIZE = 64

_TURN_SPLITTER = re.compile(r"\n\s*\n+")
_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


@dataclass
class TranscriptStored:
    """Summary returned by :func:`store_transcript`."""

    transcript_id: str
    title: str | None
    source_file: str
    turn_count: int
    chunks_stored: int


def split_transcript(text: str) -> list[str]:
    """Split raw transcript text into turn-sized chunks.

    Splits on blank lines (one or more newlines with only whitespace),
    which handles paragraph-based notes, speaker-delimited dialogues
    ("Q: ...\\n\\nA: ..."), and chat logs equally well. Empty turns
    are skipped.
    """
    turns = [t.strip() for t in _TURN_SPLITTER.split(text or "")]
    return [t for t in turns if t]


def _slugify(title: str | None) -> str:
    """Produce a URL-safe short slug from a title, with random suffix."""
    suffix = secrets.token_hex(3)
    if not title:
        return suffix
    base = _SLUG_STRIP.sub("-", title.strip().lower()).strip("-")
    base = base[:40] if base else "capture"
    return f"{base}-{suffix}"


def _synthetic_source_file(title: str | None) -> str:
    """Build an absolute synthetic path for the transcript (no file is written)."""
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return str(Path.cwd().resolve() / "captures" / f"{date}-{_slugify(title)}.md")


async def store_transcript(
    text: str,
    *,
    title: str | None = None,
    project: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    session_id: str | None = None,
    task: str | None = None,
) -> TranscriptStored:
    """Chunk, embed, and store a transcript.

    Each resulting chunk carries ``metadata.transcript_id``,
    ``metadata.turn_index``, ``metadata.title``, and — when given — the
    ``session_id`` / ``task`` columns so the transcript is recoverable
    as a unit via session-scoped filters.
    """
    turns = split_transcript(text)
    if not turns:
        raise ValueError("Transcript is empty after splitting — nothing to store.")

    transcript_id = str(uuid.uuid4())
    source_file = _synthetic_source_file(title)
    total = len(turns)

    chunks: list[ChunkResult] = [
        ChunkResult(
            content=turn,
            source_file=source_file,
            chunk_type=TRANSCRIPT_CHUNK_TYPE,
            language="markdown",
            checksum=compute_checksum(turn),
            metadata={
                "transcript_id": transcript_id,
                "turn_index": idx,
                "total_turns": total,
                "title": title,
                "source": source,
                "tags": tags,
            },
        )
        for idx, turn in enumerate(turns)
    ]

    # Embed in batches — mirrors the ingest pipeline so large transcripts
    # don't blow the embedding model's context in a single call.
    all_embeddings: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        all_embeddings.extend(await embed_texts([c.content for c in batch]))

    stored = await store_chunks(
        chunks,
        all_embeddings,
        project=project,
        session_id=session_id,
        task=task,
    )

    return TranscriptStored(
        transcript_id=transcript_id,
        title=title,
        source_file=source_file,
        turn_count=total,
        chunks_stored=stored,
    )


async def fetch_transcript_neighbors(
    seeds: list[tuple[str, int]], *, radius: int = 1
) -> list[dict]:
    """Return chunks within ``radius`` turns of each ``(transcript_id, turn_index)`` seed.

    Output is a list of plain dicts shaped like :class:`hafiz.core.search.SearchResult`
    fields, so callers can convert to ``SearchResult`` without needing the ORM.
    Seeds themselves are **not** filtered out — the caller knows which IDs are
    seeds vs. neighbors.
    """
    if not seeds:
        return []

    # Build (transcript_id, turn_index) index range filter per transcript.
    by_tid: dict[str, set[int]] = {}
    for tid, turn in seeds:
        wanted = by_tid.setdefault(tid, set())
        for offset in range(-radius, radius + 1):
            if turn + offset >= 0:
                wanted.add(turn + offset)

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = select(Chunk).where(Chunk.chunk_type == TRANSCRIPT_CHUNK_TYPE)
        # One OR clause per transcript — small, linear in # of seeded transcripts.
        from sqlalchemy import or_

        clauses = []
        for tid, turns in by_tid.items():
            clauses.append(
                and_(
                    Chunk.metadata_["transcript_id"].astext == tid,
                    Chunk.metadata_["turn_index"].astext.in_(
                        [str(t) for t in turns]
                    ),
                )
            )
        stmt = stmt.where(or_(*clauses))

        rows = (await session.execute(stmt)).scalars().all()

    return [
        {
            "id": str(c.id),
            "content": c.content,
            "source_file": c.source_file,
            "line_start": c.line_start,
            "line_end": c.line_end,
            "chunk_type": c.chunk_type,
            "language": c.language,
            "project": c.project,
            "metadata": c.metadata_ or {},
        }
        for c in rows
    ]


async def expand_transcript_neighbors(
    chunks: list[SearchResult], *, radius: int = 1
) -> list[SearchResult]:
    """Interleave ±``radius`` turn-neighbors after each transcript chunk.

    Non-transcript chunks pass through unchanged. Neighbors are appended
    right after their parent with ``is_neighbor=True`` and inherit the
    parent's score so ranking stays stable. If a neighbor is already in
    the seed set (or was added via another seed), it's skipped — no
    duplicates.
    """
    seeds: list[tuple[str, int]] = []
    for c in chunks:
        if c.chunk_type != TRANSCRIPT_CHUNK_TYPE:
            continue
        tid = c.metadata.get("transcript_id")
        if not tid:
            continue
        try:
            turn = int(c.metadata.get("turn_index", 0))
        except (TypeError, ValueError):
            continue
        seeds.append((tid, turn))

    if not seeds:
        return list(chunks)

    raw = await fetch_transcript_neighbors(seeds, radius=radius)
    neighbor_map: dict[tuple[str, int], dict] = {}
    for row in raw:
        meta = row["metadata"]
        tid = meta.get("transcript_id")
        try:
            turn = int(meta.get("turn_index"))
        except (TypeError, ValueError):
            continue
        if tid:
            neighbor_map[(tid, turn)] = row

    seen_ids = {c.id for c in chunks}
    result: list[SearchResult] = []
    for c in chunks:
        result.append(c)
        if c.chunk_type != TRANSCRIPT_CHUNK_TYPE:
            continue
        tid = c.metadata.get("transcript_id")
        if not tid:
            continue
        try:
            turn = int(c.metadata.get("turn_index", 0))
        except (TypeError, ValueError):
            continue
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            neighbor = neighbor_map.get((tid, turn + offset))
            if not neighbor or neighbor["id"] in seen_ids:
                continue
            result.append(
                SearchResult(
                    id=neighbor["id"],
                    content=neighbor["content"],
                    source_file=neighbor["source_file"],
                    line_start=neighbor["line_start"],
                    line_end=neighbor["line_end"],
                    chunk_type=neighbor["chunk_type"],
                    language=neighbor["language"],
                    project=neighbor["project"],
                    score=c.score,
                    metadata=neighbor["metadata"],
                    is_neighbor=True,
                )
            )
            seen_ids.add(neighbor["id"])

    return result
