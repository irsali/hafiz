"""hafiz export — one-way sovereignty dump of the brain's wisdom to plain files.

This is the data-portability half of the sovereignty story; ``forget`` is
the deletion half. The two are kept consistent by a single load-bearing
invariant:

    **Export never resurrects what ``forget`` removed.**

Every query here filters tombstoned rows (``valid_until IS NULL``) and,
for the source layer, retention-expired rows (``retention_until > now``).
A row the user forgot — or that aged past its retention window — does not
appear in any export.

**Scope: wisdom, not code.** Code and its AST structure (units, revisions,
files, commits, edges, embeddings) live in git — the repository is already
the sovereign copy. Dumping them here would just duplicate the source tree.
Export carries only the *irreplaceable* layer:

* annotations — decisions / facts / learnings / patterns / warnings / notes,
  the agent- and user-authored meaning that exists nowhere else.
* transcripts — agent conversations (opt-in via ``include_transcripts``),
  the source-layer record of how that meaning came to be.

Two formats, selected by the caller:

* ``md``   — a human-readable directory tree (observations grouped by
             kind, optionally transcripts). The default. "Eject and read
             it anywhere."
* ``json`` — per-table JSONL, lossless except embeddings. Designed to
             *enable* a future ``hafiz import``; import itself is a
             separate work item (see workitems/active/sovereignty-export.md).

Writes are atomic: everything lands in a sibling temp directory which is
renamed onto the target only after the manifest is written, so a partial
dump never masquerades as complete.
"""

from __future__ import annotations

import json
import shutil
import uuid
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hafiz import __version__
from hafiz.core.database import (
    Annotation,
    Communication,
    CommunicationMessage,
    get_session_factory,
)

# The Alembic head this build expects. Stamped into the manifest so a
# future ``import`` can refuse a dump from an incompatible schema.
SCHEMA_VERSION = "0006"

VALID_FORMATS = ("md", "json")

# Secrets-in-content is a known sharp edge until the Tier-3a redactor
# lands (see workitems/active/second-brain-coverage.md). Annotations and
# transcript turns can contain secrets a user pasted; export writes them
# to plaintext files faithfully. Surfaced to the user on every run.
SECRETS_WARNING = (
    "Export writes raw content to plaintext files. If any secret (token, "
    "key, credential) was ever recorded in an observation or transcript, "
    "it will be written to disk unredacted. Treat the export directory as "
    "sensitive."
)


def _jsonable(value: Any) -> Any:
    """Coerce a column value into something ``json.dumps`` accepts."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def _row_to_dict(row: Any, *, drop: Iterable[str] = ()) -> dict[str, Any]:
    """Serialize an ORM row to a plain dict via its mapped columns.

    ``metadata_`` is emitted under its DB name ``metadata``. Columns in
    ``drop`` are skipped (used to omit the huge ``embedding`` vector).
    """
    drop_set = set(drop)
    out: dict[str, Any] = {}
    for col in row.__table__.columns:
        if col.name in drop_set:
            continue
        attr = "metadata_" if col.name == "metadata" else col.name
        out[col.name] = _jsonable(getattr(row, attr))
    return out


# ---------------------------------------------------------------------------
# Row streaming — every query here applies the sovereignty filters
# ---------------------------------------------------------------------------


async def _live_annotations(session: AsyncSession, project: str | None) -> list[Annotation]:
    stmt = (
        select(Annotation)
        .where(Annotation.valid_until.is_(None))
        .order_by(Annotation.kind, Annotation.valid_from)
    )
    if project is not None:
        stmt = stmt.where(Annotation.project == project)
    return list((await session.execute(stmt)).scalars().all())


async def _live_communications(
    session: AsyncSession, project: str | None, now: datetime
) -> list[Communication]:
    """Source-layer transcripts that are neither tombstoned nor expired."""
    stmt = (
        select(Communication)
        .where(Communication.valid_until.is_(None))
        .where((Communication.retention_until.is_(None)) | (Communication.retention_until > now))
        .order_by(Communication.started_at)
    )
    if project is not None:
        stmt = stmt.where(
            (Communication.scope_kind == "project") & (Communication.scope_value == project)
        )
    return list((await session.execute(stmt)).scalars().all())


async def _messages_for(
    session: AsyncSession, comm_ids: list[uuid.UUID]
) -> dict[uuid.UUID, list[CommunicationMessage]]:
    if not comm_ids:
        return {}
    stmt = (
        select(CommunicationMessage)
        .where(CommunicationMessage.communication_id.in_(comm_ids))
        .order_by(CommunicationMessage.communication_id, CommunicationMessage.seq)
    )
    grouped: dict[uuid.UUID, list[CommunicationMessage]] = defaultdict(list)
    for msg in (await session.execute(stmt)).scalars().all():
        grouped[msg.communication_id].append(msg)
    return grouped


# ---------------------------------------------------------------------------
# Bundle — gather every in-scope row once, reused by both serializers
# ---------------------------------------------------------------------------


class _Bundle:
    """All in-scope rows, gathered once and shared between formats."""

    def __init__(self) -> None:
        self.annotations: list[Annotation] = []
        self.communications: list[Communication] = []
        self.messages: dict[uuid.UUID, list[CommunicationMessage]] = {}

    def counts(self) -> dict[str, int]:
        return {
            "annotations": len(self.annotations),
            "communications": len(self.communications),
            "communication_messages": sum(len(m) for m in self.messages.values()),
        }


async def _gather(*, project: str | None, include_transcripts: bool) -> _Bundle:
    now = datetime.now(UTC)
    factory = get_session_factory()
    bundle = _Bundle()
    async with factory() as session:
        bundle.annotations = await _live_annotations(session, project)
        if include_transcripts:
            bundle.communications = await _live_communications(session, project, now)
            bundle.messages = await _messages_for(session, [c.id for c in bundle.communications])
    return bundle


# ---------------------------------------------------------------------------
# JSON serializer — per-table JSONL, lossless except embeddings
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def _render_json(bundle: _Bundle, root: Path) -> None:
    knowledge = root / "knowledge"
    knowledge.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        knowledge / "annotations.jsonl",
        (_row_to_dict(a, drop=("embedding",)) for a in bundle.annotations),
    )

    if bundle.communications:
        source = root / "source"
        source.mkdir(parents=True, exist_ok=True)
        _write_jsonl(
            source / "communications.jsonl",
            (_row_to_dict(c) for c in bundle.communications),
        )
        all_messages = (
            _row_to_dict(m, drop=("embedding",)) for msgs in bundle.messages.values() for m in msgs
        )
        _write_jsonl(source / "communication_messages.jsonl", all_messages)


# ---------------------------------------------------------------------------
# Markdown serializer — human-readable tree
# ---------------------------------------------------------------------------


def _safe_filename(text: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in text)
    return safe.strip("-") or "untitled"


def _render_markdown(bundle: _Bundle, root: Path, *, include_transcripts: bool) -> None:
    _render_md_observations(bundle, root)
    if include_transcripts and bundle.communications:
        _render_md_transcripts(bundle, root)
    _render_md_readme(bundle, root, include_transcripts=include_transcripts)


def _render_md_observations(bundle: _Bundle, root: Path) -> None:
    """One file per annotation kind, chronological within."""
    if not bundle.annotations:
        return
    obs_dir = root / "observations"
    obs_dir.mkdir(parents=True, exist_ok=True)

    by_kind: dict[str, list[Annotation]] = defaultdict(list)
    for ann in bundle.annotations:
        by_kind[ann.kind or "note"].append(ann)

    for kind, anns in sorted(by_kind.items()):
        lines = [f"# {kind.capitalize()} ({len(anns)})", ""]
        for ann in anns:
            when = ann.valid_from.date().isoformat() if ann.valid_from else "?"
            src = ann.source or "unknown"
            lines.append(f"## {when} — {src}")
            if ann.project:
                lines.append(f"*project: {ann.project}*")
            if ann.tags:
                lines.append(f"*tags: {', '.join(ann.tags)}*")
            lines.append("")
            lines.append(ann.content.strip())
            lines.append("")
            lines.append(f"<sub>id: `{ann.id}` · confidence: {ann.confidence}</sub>")
            lines.append("")
        (obs_dir / f"{_safe_filename(kind)}.md").write_text("\n".join(lines), encoding="utf-8")


def _render_md_transcripts(bundle: _Bundle, root: Path) -> None:
    """One file per communication — ordered turns, role-labelled."""
    tx_dir = root / "transcripts"
    tx_dir.mkdir(parents=True, exist_ok=True)
    for comm in bundle.communications:
        msgs = bundle.messages.get(comm.id, [])
        started = comm.started_at.date().isoformat() if comm.started_at else "?"
        title = f"{started}-{comm.agent}-{str(comm.id)[:8]}"
        lines = [f"# Transcript — {comm.agent}", ""]
        lines.append(f"*started: {comm.started_at.isoformat() if comm.started_at else '?'}*")
        if comm.scope_value:
            lines.append(f"*scope: {comm.scope_kind}={comm.scope_value}*")
        lines.append("")
        for msg in msgs:
            lines.append(f"### {msg.role}" + (f" ({msg.author})" if msg.author else ""))
            lines.append("")
            lines.append((msg.content or "").strip())
            lines.append("")
        (tx_dir / f"{_safe_filename(title)}.md").write_text("\n".join(lines), encoding="utf-8")


def _render_md_readme(bundle: _Bundle, root: Path, *, include_transcripts: bool) -> None:
    counts = bundle.counts()
    lines = [
        "# Hafiz export",
        "",
        "A sovereign dump of your second brain's wisdom — the decisions,",
        "facts, learnings, patterns, and warnings that live nowhere else.",
        "Plain Markdown, readable without Hafiz. Code is intentionally",
        "excluded: your repository is already its sovereign copy.",
        "",
        "See `manifest.json` for exact counts and provenance.",
        "",
        "## Contents",
        "",
        f"- `observations/` — {counts['annotations']} decisions / facts / "
        "learnings / patterns / warnings, grouped by kind",
    ]
    if include_transcripts:
        lines.append(
            f"- `transcripts/` — {counts['communications']} agent conversations "
            f"({counts['communication_messages']} turns)"
        )
    lines += [
        "",
        "## Note",
        "",
        f"> {SECRETS_WARNING}",
        "",
        "Forgotten and retention-expired rows are intentionally excluded —",
        "this export reflects only live data, consistent with `hafiz forget`.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Manifest + atomic orchestration
# ---------------------------------------------------------------------------


def _write_manifest(
    root: Path,
    bundle: _Bundle,
    *,
    fmt: str,
    project: str | None,
    include_transcripts: bool,
    generated_at: datetime,
) -> dict[str, Any]:
    manifest = {
        "tool": "hafiz",
        "tool_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "format": fmt,
        "generated_at": generated_at.isoformat(),
        "scope": {"project": project},
        "flags": {"include_transcripts": include_transcripts},
        "counts": bundle.counts(),
        "excludes": "code/structure; tombstoned (valid_until) and retention-expired rows",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


async def export_brain(
    *,
    out_dir: str | Path,
    fmt: str = "md",
    project: str | None = None,
    include_transcripts: bool = False,
) -> dict[str, Any]:
    """Dump the live wisdom layer to ``out_dir`` and return a summary dict.

    Atomic: rows are written into a sibling ``.tmp`` directory which is
    renamed onto ``out_dir`` only after the manifest lands. An existing
    ``out_dir`` is replaced.

    Returns ``{"ok": True, "path": ..., "format": ..., "counts": {...},
    "warning": ...}`` on success, or ``{"ok": False, "error": ...}`` on a
    bad format.
    """
    if fmt not in VALID_FORMATS:
        return {
            "ok": False,
            "error": f"unknown format {fmt!r}; expected one of {', '.join(VALID_FORMATS)}",
        }

    generated_at = datetime.now(UTC)
    out_path = Path(out_dir).expanduser().resolve()
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    bundle = await _gather(project=project, include_transcripts=include_transcripts)

    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    if fmt == "json":
        _render_json(bundle, tmp_path)
    else:
        _render_markdown(bundle, tmp_path, include_transcripts=include_transcripts)

    manifest = _write_manifest(
        tmp_path,
        bundle,
        fmt=fmt,
        project=project,
        include_transcripts=include_transcripts,
        generated_at=generated_at,
    )

    # Atomic swap: replace any existing export only once the new one is whole.
    if out_path.exists():
        shutil.rmtree(out_path)
    tmp_path.rename(out_path)

    return {
        "ok": True,
        "path": str(out_path),
        "format": fmt,
        "counts": manifest["counts"],
        "warning": SECRETS_WARNING,
    }
