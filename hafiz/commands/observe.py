"""hafiz observe / note / recall — store and search annotations.

The CLI verb stays ``observe`` (and ``note`` / ``recall``); internally these
write/read the `annotations` table via :mod:`hafiz.core.annotations`. See
workitems/active/structural-grounding.md for the rename rationale.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hafiz.core.database import close_engine
from hafiz.core.durations import parse_duration
from hafiz.core.session import resolve_session_tag

console = Console()


def _compute_valid_until(expires_in: str | None, expires: str | None) -> datetime | None:
    """Resolve --expires-in / --expires into an absolute UTC datetime, or None.

    Mutually exclusive — providing both is a user error.
    """
    if expires_in and expires:
        console.print("[red]Error:[/red] --expires-in and --expires are mutually exclusive.")
        raise SystemExit(1)
    if expires_in:
        try:
            return datetime.now(UTC) + parse_duration(expires_in)
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1)
    if expires:
        try:
            parsed = datetime.fromisoformat(expires)
        except ValueError:
            console.print(
                f"[red]Error:[/red] --expires must be an ISO date/datetime "
                f"(e.g. 2026-06-01), got {expires!r}"
            )
            raise SystemExit(1)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    return None


def _parse_uuid_list(raw: str | None) -> list[str] | None:
    """Parse a comma-separated UUID list; error cleanly on bad input."""
    if not raw:
        return None
    import uuid as _uuid

    ids: list[str] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        try:
            ids.append(str(_uuid.UUID(s)))
        except ValueError:
            console.print(f"[red]Error:[/red] not a valid UUID: {s!r}")
            raise SystemExit(1)
    return ids or None


def run_observe(
    text: str,
    *,
    kind: str = "fact",
    source: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 1.0,
    expires_in: str | None = None,
    expires: str | None = None,
    session: str | None = None,
    task: str | None = None,
    supersedes: str | None = None,
    derived_from: str | None = None,
    output_json: bool = False,
) -> None:
    """Store an annotation and print confirmation."""
    valid_until = _compute_valid_until(expires_in, expires)
    resolved_session_id, resolved_task = resolve_session_tag(
        session_override=session, task_override=task
    )
    derived_ids = _parse_uuid_list(derived_from)

    if supersedes:
        import uuid as _uuid

        try:
            _uuid.UUID(supersedes)
        except ValueError:
            console.print(f"[red]Error:[/red] --supersedes not a valid UUID: {supersedes!r}")
            raise SystemExit(1)

    async def _store():
        try:
            from hafiz.core.annotations import store_annotation

            ann = await store_annotation(
                text,
                kind=kind,
                source=source,
                project=project,
                tags=tags,
                confidence=confidence,
                valid_until=valid_until,
                session_id=resolved_session_id,
                task=resolved_task,
                supersedes_id=supersedes,
                derived_from=derived_ids,
            )
            return ann
        finally:
            await close_engine()

    try:
        ann = asyncio.run(_store())
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    if output_json:
        data = {
            "action": "observe",
            "annotation": {
                "id": str(ann.id),
                "content": ann.content,
                "kind": ann.kind,
                "source": ann.source,
                "project": ann.project,
                "tags": ann.tags,
                "confidence": ann.confidence,
                "valid_from": ann.valid_from.isoformat(),
                "valid_until": ann.valid_until.isoformat() if ann.valid_until else None,
                "unit_id": str(ann.unit_id) if ann.unit_id else None,
                "session_id": ann.legacy_session_id
                or (str(ann.session_id) if ann.session_id else None),
                "task": ann.task,
                "commit_hash": ann.commit_hash,
                "supersedes_id": str(ann.supersedes_id) if ann.supersedes_id else None,
                "derived_from": (ann.metadata_ or {}).get("derived_from"),
            },
        }
        console.print_json(json.dumps(data))
        return

    tags_str = ", ".join(ann.tags) if ann.tags else "none"
    session_display = ann.legacy_session_id or (str(ann.session_id) if ann.session_id else None)
    session_line = ""
    if session_display or ann.task:
        session_line = (
            f"  [bold]Session:[/bold]    {session_display or '—'}\n"
            f"  [bold]Task:[/bold]       {ann.task or '—'}\n"
        )
    info = (
        f"[bold green]Annotation stored[/bold green]\n\n"
        f"  [bold]ID:[/bold]         {ann.id}\n"
        f"  [bold]Kind:[/bold]       {ann.kind}\n"
        f"  [bold]Source:[/bold]     {ann.source or '—'}\n"
        f"  [bold]Project:[/bold]    {ann.project or '—'}\n"
        f"  [bold]Tags:[/bold]       {tags_str}\n"
        f"  [bold]Confidence:[/bold] {ann.confidence:.0%}\n"
        f"{session_line}"
        f"  [bold]Content:[/bold]    {ann.content[:200]}"
    )
    console.print(Panel(info, border_style="cyan"))


def run_note(
    text: str,
    *,
    source: str | None = None,
    project: str | None = None,
    tags: list[str] | None = None,
    confidence: float = 1.0,
    expires_in: str | None = None,
    expires: str | None = None,
    session: str | None = None,
    task: str | None = None,
    supersedes: str | None = None,
    derived_from: str | None = None,
    output_json: bool = False,
) -> None:
    """Low-bar capture — stores as ``kind="note"``."""
    run_observe(
        text,
        kind="note",
        source=source,
        project=project,
        tags=tags,
        confidence=confidence,
        expires_in=expires_in,
        expires=expires,
        session=session,
        task=task,
        supersedes=supersedes,
        derived_from=derived_from,
        output_json=output_json,
    )


STALE_DAYS = 90


def _age(valid_from: datetime) -> tuple[str, int, bool]:
    """(human label, age in days, stale flag) for a ``valid_from``."""
    now = datetime.now(UTC)
    days = (now - valid_from.astimezone(UTC)).days
    if days < 0:
        return "future", days, False
    if days == 0:
        return "today", 0, False
    if days == 1:
        return "1d ago", 1, False
    if days < 30:
        return f"{days}d ago", days, days > STALE_DAYS
    if days < 365:
        return f"{round(days / 30)}mo ago", days, days > STALE_DAYS
    return f"{round(days / 365)}y ago", days, True


def run_recall(
    query: str,
    *,
    limit: int = 10,
    project: str | None = None,
    workspace: bool = False,
    kind: str | None = None,
    source: str | None = None,
    include_superseded: bool = False,
    rerank: bool = True,
    output_json: bool = False,
) -> None:
    """Search annotations by semantic similarity and display results."""

    async def _search():
        try:
            from hafiz.core.annotations import search_annotations

            search_project: str | list[str] | None = project
            if workspace:
                # resolve_workspace_projects lives in context.py which still
                # depends on the old schema; keep workspace-fanout stubbed
                # until Phase 3b rewires context.
                console.print(
                    "[yellow]--workspace fanout is disabled until "
                    "hafiz.core.context is rewired (Phase 3b). "
                    "Falling back to --project filter.[/yellow]"
                )
            results = await search_annotations(
                query,
                limit=limit,
                project=search_project,
                kind=kind,
                source=source,
                active_only=not include_superseded,
                rerank=rerank,
            )
            return results
        finally:
            await close_engine()

    results = asyncio.run(_search())

    def _is_inactive(r) -> bool:
        if r.valid_until is None:
            return False
        return r.valid_until < datetime.now(UTC)

    if output_json:
        data = {
            "query": query,
            "results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "kind": r.kind,
                    "source": r.source,
                    "project": r.project,
                    "tags": r.tags,
                    "confidence": r.confidence,
                    "valid_from": r.valid_from.isoformat(),
                    "valid_until": r.valid_until.isoformat() if r.valid_until else None,
                    "unit_id": r.unit_id,
                    "age_days": _age(r.valid_from)[1],
                    "stale": _age(r.valid_from)[2],
                    "inactive": _is_inactive(r),
                    "score": r.score,
                }
                for r in results
            ],
            "total": len(results),
        }
        console.print_json(json.dumps(data))
        return

    if not results:
        console.print("[yellow]No annotations found.[/yellow]")
        return

    console.print()
    table = Table(
        title=f'Recall: "{query}" ({len(results)} results)',
        border_style="cyan",
    )
    table.add_column("Kind", style="yellow", width=10)
    table.add_column("Content", ratio=3)
    table.add_column("Source", style="dim", width=16)
    table.add_column("Age", style="dim", width=8)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Score", justify="right", width=8)

    for r in results:
        score_color = "green" if r.score > 0.7 else "yellow" if r.score > 0.5 else "red"
        content_preview = r.content[:120]
        if len(r.content) > 120:
            content_preview += "..."
        age_label, _, stale = _age(r.valid_from)
        inactive = _is_inactive(r)
        row_style = "dim" if stale or inactive else None
        kind_label = f"{r.kind} (superseded)" if inactive else r.kind
        table.add_row(
            kind_label,
            content_preview,
            r.source or "—",
            age_label,
            f"{r.confidence:.0%}",
            f"[{score_color}]{r.score:.2%}[/{score_color}]",
            style=row_style,
        )

    console.print(table)
    console.print()
