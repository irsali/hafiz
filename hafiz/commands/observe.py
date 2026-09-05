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
from hafiz.core.durations import age_label, parse_duration
from hafiz.core.formats import OutputFormat, annotation_compact, annotation_md
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
    session_key: str | None = None,
    supersedes: str | None = None,
    derived_from: str | None = None,
    allow_duplicate: bool = False,
    detect_duplicates: bool = True,
    output_json: bool = False,
) -> None:
    """Store an annotation and print confirmation.

    Runs near-duplicate detection (unless ``detect_duplicates`` is False, as
    for the ``note`` firehose). In surface-only mode any matches are reported
    alongside the stored row; in strict mode a match aborts the write with a
    non-zero exit unless ``--supersedes`` or ``--allow-duplicate`` was given.
    """
    valid_until = _compute_valid_until(expires_in, expires)
    resolved_session_id, resolved_task = resolve_session_tag(
        session_override=session, task_override=task, session_key=session_key
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
            from hafiz.core.annotations import store_annotation_checked

            return await store_annotation_checked(
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
                allow_duplicate=allow_duplicate,
                # The note firehose skips *near*-duplicate detection by design
                # (raw capture is never gated) but still collapses byte-identical
                # writes, silently, into the row it already has.
                detect_near=detect_duplicates,
                dedupe_silently=not detect_duplicates,
            )
        finally:
            await close_engine()

    try:
        from hafiz.core.annotations import (
            DuplicateAnnotationError,
            ExactDuplicateAnnotationError,
        )

        result = asyncio.run(_store())
        ann, near_dupes, deduped = result.annotation, result.near_duplicates, result.deduped
    except ExactDuplicateAnnotationError as e:
        _report_exact_duplicate(e, output_json)
        raise SystemExit(2)
    except DuplicateAnnotationError as e:
        _report_strict_block(e.duplicates, output_json)
        raise SystemExit(2)
    except ValueError as e:
        # Blank content, a missing --supersedes target, etc. Agents parse
        # stdout, so honor the project's machine-readable failure shape rather
        # than emitting a Rich panel they can't read.
        if output_json:
            console.print_json(json.dumps({"ok": False, "error": str(e)}))
        else:
            console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    from hafiz.core.annotations import oversized_warning

    # Advisory only, and computed from what was actually stored so a deduped
    # write reports on the row the caller ends up with.
    oversized = oversized_warning(ann.content, kind=ann.kind)

    if output_json:
        data = {
            "action": "observe",
            # True when an identical live row already existed and nothing new
            # was written — the annotation below is that pre-existing row.
            "deduped": deduped,
            # None unless the record is long enough to be several records.
            # Never a failure: the write above already succeeded.
            "oversized": oversized,
            "near_duplicates": [
                {"id": d.id, "content": d.content, "kind": d.kind, "score": d.score}
                for d in near_dupes
            ],
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
    headline = (
        "[bold yellow]Already recorded[/bold yellow] [dim](identical row — nothing written)[/dim]"
        if deduped
        else "[bold green]Annotation stored[/bold green]"
    )
    info = (
        f"{headline}\n\n"
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

    if oversized:
        console.print(
            f"[yellow]⚠ {oversized['chars']} chars — long enough to be several "
            f"annotations[/yellow] [dim](soft limit {oversized['limit']})[/dim]\n"
            "[dim]One claim per record recalls better. Split it and link the parts "
            "with --derived-from.[/dim]"
        )

    if near_dupes:
        _print_dupe_hint(near_dupes)


def _print_dupe_hint(duplicates: list) -> None:
    """Surface near-duplicate live annotations after a successful write."""
    lines = [
        "[bold yellow]⚠ Similar live annotation(s) already exist[/bold yellow]",
        "[dim]If this replaces one of them, supersede it:[/dim]",
        "",
    ]
    for d in duplicates:
        preview = d.content[:80] + ("…" if len(d.content) > 80 else "")
        lines.append(f"  [cyan]{d.id}[/cyan]  [dim]({d.score:.0%})[/dim]  {preview}")
    lines += [
        "",
        f'[dim]→ hafiz observe "<text>" --supersedes {duplicates[0].id}[/dim]',
    ]
    console.print(Panel("\n".join(lines), border_style="yellow"))


def _report_exact_duplicate(err, output_json: bool) -> None:
    """Report a write refused for being byte-identical to a live row."""
    if output_json:
        data = {
            "ok": False,
            "error": str(err),
            "existing_id": err.existing_id,
            "hint": (
                "The text is identical to a live annotation. Supersede it if the "
                "belief changed, edit the text if it's a refinement, or pass "
                "--allow-duplicate to force."
            ),
        }
        console.print_json(json.dumps(data))
        return
    console.print(
        Panel(
            "[bold red]✗ Write refused — identical annotation already live[/bold red]\n\n"
            f"  [cyan]{err.existing_id}[/cyan]\n\n"
            f'[dim]→ belief changed:  hafiz observe "<new text>" '
            f"--supersedes {err.existing_id}[/dim]\n"
            '[dim]→ force the write: hafiz observe "<text>" --allow-duplicate[/dim]',
            border_style="red",
        )
    )


def _report_strict_block(duplicates: list, output_json: bool) -> None:
    """Report a strict-mode write that was refused for near-duplication."""
    if output_json:
        data = {
            "ok": False,
            "error": "near-duplicate annotation exists (strict mode)",
            "near_duplicates": [
                {"id": d.id, "content": d.content, "kind": d.kind, "score": d.score}
                for d in duplicates
            ],
            "hint": "pass --supersedes <id> to replace one, or --allow-duplicate to force.",
        }
        console.print_json(json.dumps(data))
        return
    lines = [
        "[bold red]✗ Write refused — near-duplicate exists (strict mode)[/bold red]",
        "",
    ]
    for d in duplicates:
        preview = d.content[:80] + ("…" if len(d.content) > 80 else "")
        lines.append(f"  [cyan]{d.id}[/cyan]  [dim]({d.score:.0%})[/dim]  {preview}")
    lines += [
        "",
        f'[dim]→ supersede it:   hafiz observe "<text>" --supersedes {duplicates[0].id}[/dim]',
        '[dim]→ or force write: hafiz observe "<text>" --allow-duplicate[/dim]',
    ]
    console.print(Panel("\n".join(lines), border_style="red"))


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
    session_key: str | None = None,
    supersedes: str | None = None,
    derived_from: str | None = None,
    output_json: bool = False,
) -> None:
    """Low-bar capture — stores as ``kind="note"``.

    The note firehose skips near-duplicate detection by design: raw capture
    should never be gated.
    """
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
        session_key=session_key,
        supersedes=supersedes,
        derived_from=derived_from,
        detect_duplicates=False,
        output_json=output_json,
    )


def _age(valid_from: datetime) -> tuple[str, int, bool]:
    """(human label, age in days, stale flag) for a ``valid_from``.

    Thin alias over :func:`hafiz.core.durations.age_label`, kept because the
    compact/markdown formatters in core need the same rendering.
    """
    return age_label(valid_from)


def run_recall(
    query: str,
    *,
    limit: int = 10,
    project: str | None = None,
    workspace: bool = False,
    kind: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    include_superseded: bool = False,
    rerank: bool = True,
    min_score: float | None = None,
    output_format: OutputFormat = OutputFormat.RICH,
    with_ids: bool = False,
) -> None:
    """Search annotations by semantic similarity and display results."""

    async def _search():
        try:
            # Daemon-first. `query_recall` returns AnnotationResult either
            # way and falls back to direct in-process execution on any
            # daemon problem, so this can only be faster, never different.
            from hafiz.core.daemon_client import query_recall

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
            results = await query_recall(
                query,
                limit=limit,
                project=search_project,
                kind=kind,
                source=source,
                tags=tags,
                active_only=not include_superseded,
                rerank=rerank,
                min_score=min_score,
            )
            # Only the two formats that exist to be injected into a prompt get
            # trimmed; `json` and `rich` keep the full record, so nothing is
            # ever unreachable. Best-effort — a failure here leaves `snippet`
            # unset and the caller renders full content.
            if output_format in (OutputFormat.COMPACT, OutputFormat.MD):
                from hafiz.core.config import load_settings
                from hafiz.core.snippets import attach_snippets

                await attach_snippets(
                    query, results, budget=load_settings().annotations.snippet_chars
                )
            return results
        finally:
            await close_engine()

    results = asyncio.run(_search())

    def _is_inactive(r) -> bool:
        if r.valid_until is None:
            return False
        return r.valid_until < datetime.now(UTC)

    if output_format is OutputFormat.JSON:
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
                    # Additive: which stage ranked this row, and how strongly.
                    # None means reranking did not run (--no-rerank, blank
                    # candidate pool, or model failure), so `score` is the
                    # ranking score. Without this, reranked and vector output
                    # are indistinguishable.
                    "rerank_score": r.rerank_score,
                }
                for r in results
            ],
            "total": len(results),
            "reranked": any(r.rerank_score is not None for r in results),
        }
        console.print_json(json.dumps(data))
        return

    if output_format is OutputFormat.COMPACT:
        data = {
            "query": query,
            "results": [annotation_compact(r, with_ids=with_ids) for r in results],
            "total": len(results),
        }
        console.print_json(json.dumps(data))
        return

    if output_format is OutputFormat.MD:
        if not results:
            # Silence, not a placeholder. `md` exists to be injected into a
            # prompt, and a hook shouldn't have to filter out hafiz saying
            # nothing — under a floor, "no rows" is the not-relevant signal and
            # it fires on every off-topic prompt. Callers wanting the count have
            # `json` / `compact`.
            return
        print(f"## Recall: {query}\n")
        for r in results:
            print(annotation_md(r, with_ids=with_ids))
        return

    if not results:
        console.print("[yellow]No annotations found.[/yellow]")
        return

    console.print()
    reranked = any(r.rerank_score is not None for r in results)
    table = Table(
        title=f'Recall: "{query}" ({len(results)} results)',
        border_style="cyan",
    )
    table.add_column("Kind", style="yellow", width=10)
    table.add_column("Content", ratio=3)
    table.add_column("Source", style="dim", width=16)
    table.add_column("Age", style="dim", width=8)
    table.add_column("Confidence", justify="right", width=10)
    # Name the score in the header — a "Score" column that silently switches
    # between cosine similarity and cross-encoder relevance is how the two got
    # conflated in the first place.
    table.add_column("Relevance" if reranked else "Similarity", justify="right", width=10)

    for r in results:
        shown = r.ranking_score
        score_color = "green" if shown > 0.7 else "yellow" if shown > 0.5 else "red"
        content_preview = r.content[:120]
        if len(r.content) > 120:
            content_preview += "..."
        age, _, stale = _age(r.valid_from)
        inactive = _is_inactive(r)
        row_style = "dim" if stale or inactive else None
        kind_label = f"{r.kind} (superseded)" if inactive else r.kind
        table.add_row(
            kind_label,
            content_preview,
            r.source or "—",
            age,
            f"{r.confidence:.0%}",
            f"[{score_color}]{shown:.2%}[/{score_color}]",
            style=row_style,
        )

    console.print(table)
    console.print()
