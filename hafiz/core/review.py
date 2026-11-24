"""Self-review engine — analyzes hafiz knowledge quality and suggests improvements.

This is Layer 2 (evolving) — separate from skills.md (Layer 1, stable contract).
It helps users and agents understand the health of their knowledge base and
surfaces actionable improvements without being prescriptive.

Reads the knowledge layer: ``units`` (addressable things), ``annotations``
(the wisdom layer), and ``embeddings`` (the vector index). Only *live* rows are
counted — tombstoned units (``valid_until`` set) and expired annotations
(``valid_until`` in the past) are excluded so findings reflect the current
state, not history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select

from hafiz.core.database import (
    Annotation,
    Embedding,
    File,
    Unit,
    UnitRevision,
    get_session_factory,
)


@dataclass
class ReviewFinding:
    """A single review finding with actionable suggestion."""

    category: str  # annotations, staleness
    severity: str  # info, suggestion, warning
    title: str
    detail: str
    action: str | None = None


@dataclass
class ReviewReport:
    """Complete review of the hafiz knowledge base."""

    findings: list[ReviewFinding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stats": self.stats,
            "findings": [
                {
                    "category": f.category,
                    "severity": f.severity,
                    "title": f.title,
                    "detail": f.detail,
                    "action": f.action,
                }
                for f in self.findings
            ],
            "summary": {
                "total": len(self.findings),
                "warnings": sum(1 for f in self.findings if f.severity == "warning"),
                "suggestions": sum(1 for f in self.findings if f.severity == "suggestion"),
                "info": sum(1 for f in self.findings if f.severity == "info"),
            },
        }

    def to_markdown(self) -> str:
        sections = ["# Hafiz Knowledge Review"]

        # Stats summary
        sections.append("\n## Overview")
        for k, v in self.stats.items():
            sections.append(f"- **{k}**: {v}")

        # Group findings by category
        by_category: dict[str, list[ReviewFinding]] = {}
        for f in self.findings:
            by_category.setdefault(f.category, []).append(f)

        severity_icon = {"warning": "!", "suggestion": "~", "info": "-"}

        for category, findings in by_category.items():
            sections.append(f"\n## {category.title()}")
            for f in findings:
                icon = severity_icon.get(f.severity, "-")
                sections.append(f"\n[{icon}] **{f.title}**")
                sections.append(f"  {f.detail}")
                if f.action:
                    sections.append(f"  -> {f.action}")

        if not self.findings:
            sections.append("\n_No issues found. Knowledge base looks healthy._")

        return "\n".join(sections)


async def run_review(project: str | None = None) -> ReviewReport:
    """Analyze the hafiz knowledge base and produce a review report.

    Checks:
    - Annotation quality: type distribution, low confidence, staleness
    """
    from hafiz.core.projects import resolve_project

    project = await resolve_project(project)
    report = ReviewReport()
    session_factory = get_session_factory()

    async with session_factory() as session:
        # ── Gather stats (live rows only) ───────────────────────────────
        embedding_count = (await session.execute(_embedding_count(project))).scalar() or 0
        unit_count = (await session.execute(_unit_count(project))).scalar() or 0
        ann_count = (await session.execute(_annotation_count(project))).scalar() or 0

        report.stats = {
            "units": unit_count,
            "embeddings": embedding_count,
            "annotations": ann_count,
        }

        # ── Annotation checks ───────────────────────────────────────────

        # Kind distribution
        kind_rows = (
            await session.execute(
                _ann_filter(
                    select(Annotation.kind, func.count()).group_by(Annotation.kind),
                    project,
                )
            )
        ).all()

        kind_dist = {k: c for k, c in kind_rows}
        report.stats["annotation_kinds"] = kind_dist

        if ann_count > 0 and not kind_dist.get("decision"):
            report.findings.append(
                ReviewFinding(
                    category="annotations",
                    severity="suggestion",
                    title="No decisions recorded",
                    detail="Decisions are the most durable annotation kind — "
                    "they capture why, not just what.",
                    action='hafiz observe "<decision>" --type decision --source agent:<name>',
                )
            )

        if ann_count > 0 and not kind_dist.get("warning"):
            report.findings.append(
                ReviewFinding(
                    category="annotations",
                    severity="info",
                    title="No warnings recorded",
                    detail="Warnings capture gotchas and non-obvious behaviors "
                    "that prevent repeated mistakes.",
                )
            )

        # Low-confidence annotations
        low_conf = (
            await session.execute(
                _ann_filter(
                    select(func.count())
                    .select_from(Annotation)
                    .where(_ann_live())
                    .where(Annotation.confidence < 0.5),
                    project,
                )
            )
        ).scalar() or 0

        if low_conf > 0:
            report.findings.append(
                ReviewFinding(
                    category="annotations",
                    severity="suggestion",
                    title=f"{low_conf} low-confidence annotations",
                    detail="Annotations with confidence < 50% may add noise. "
                    "Review and either boost or remove.",
                    action="hafiz query '' --observations --limit 50 --json",
                )
            )

        # Stale annotations (older than 90 days)
        cutoff = datetime.now(UTC) - timedelta(days=90)
        stale = (
            await session.execute(
                _ann_filter(
                    select(func.count())
                    .select_from(Annotation)
                    .where(_ann_live())
                    .where(Annotation.valid_from < cutoff),
                    project,
                )
            )
        ).scalar() or 0

        if stale > 0:
            report.findings.append(
                ReviewFinding(
                    category="staleness",
                    severity="info",
                    title=f"{stale} annotations older than 90 days",
                    detail="Older annotations may still be valid, but periodic "
                    "review keeps knowledge fresh.",
                    action="hafiz journal --since 90d --json  # review and supersede if outdated",
                )
            )

        # Unit-to-embedding ratio (rough index coverage signal)
        if embedding_count > 0 and unit_count > 0:
            report.stats["unit_embedding_ratio"] = round(unit_count / embedding_count, 3)

    return report


# ---------------------------------------------------------------------------
# Query builders. ``project`` lives on ``files`` (units join through file_id),
# on ``annotations`` directly, and is reached for embeddings via their unit.
# We scope by joining to ``files`` where needed.
# ---------------------------------------------------------------------------


def _ann_live():
    """Predicate: annotation is not expired."""
    now = datetime.now(UTC)
    return or_(Annotation.valid_until.is_(None), Annotation.valid_until > now)


def _ann_filter(stmt, project: str | None):
    if project:
        stmt = stmt.where(Annotation.project == project)
    return stmt


def _annotation_count(project: str | None):
    stmt = select(func.count()).select_from(Annotation).where(_ann_live())
    return _ann_filter(stmt, project)


def _unit_filter(stmt, project: str | None):
    """Scope a Unit-based count to a project via the files join."""
    if project:
        stmt = stmt.where(Unit.file_id.in_(select(File.id).where(File.project == project)))
    return stmt


def _unit_count(project: str | None):
    stmt = select(func.count()).select_from(Unit).where(Unit.valid_until.is_(None))
    return _unit_filter(stmt, project)


def _embedding_count(project: str | None):
    stmt = (
        select(func.count())
        .select_from(Embedding)
        .join(UnitRevision, Embedding.unit_revision_id == UnitRevision.id)
        .where(UnitRevision.superseded_at.is_(None))
    )
    if project:
        stmt = stmt.where(
            UnitRevision.unit_id.in_(
                select(Unit.id).where(
                    Unit.file_id.in_(select(File.id).where(File.project == project))
                )
            )
        )
    return stmt
