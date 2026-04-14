"""Self-review engine — analyzes hafiz knowledge quality and suggests improvements.

This is Layer 2 (evolving) — separate from skills.md (Layer 1, stable contract).
It helps users and agents understand the health of their knowledge base and
surfaces actionable improvements without being prescriptive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from sqlalchemy import select, func, text

from hafiz.core.database import (
    Chunk,
    Entity,
    Relation,
    Observation,
    get_session_factory,
)


@dataclass
class ReviewFinding:
    """A single review finding with actionable suggestion."""

    category: str  # observations, graph, coverage, staleness
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
    - Observation quality: duplicates, low confidence, type distribution
    - Graph coverage: orphan entities, entities without descriptions
    - Index coverage: projects without entities, stale files
    - Staleness: old observations that may need re-evaluation
    """
    report = ReviewReport()
    session_factory = get_session_factory()

    async with session_factory() as session:
        # ── Gather stats ────────────────────────────────────────────────
        chunk_count = (
            await session.execute(
                _count_query(Chunk, project)
            )
        ).scalar() or 0
        entity_count = (
            await session.execute(
                _count_query(Entity, project)
            )
        ).scalar() or 0
        relation_count = (
            await session.execute(
                _count_query(Relation, project, field_name="source_id")
            )
        ).scalar() or 0
        obs_count = (
            await session.execute(
                _count_query(Observation, project)
            )
        ).scalar() or 0

        report.stats = {
            "chunks": chunk_count,
            "entities": entity_count,
            "relations": relation_count,
            "observations": obs_count,
        }

        # ── Observation checks ──────────────────────────────────────────

        # Type distribution
        obs_types = (
            await session.execute(
                _filtered(
                    select(Observation.obs_type, func.count())
                    .group_by(Observation.obs_type),
                    Observation,
                    project,
                )
            )
        ).all()

        type_dist = {t: c for t, c in obs_types}
        report.stats["observation_types"] = type_dist

        if obs_count > 0 and not type_dist.get("decision"):
            report.findings.append(ReviewFinding(
                category="observations",
                severity="suggestion",
                title="No decisions recorded",
                detail="Decisions are the most durable observation type — they capture why, not just what.",
                action='hafiz observe "<decision>" --type decision --source agent:<name>',
            ))

        if obs_count > 0 and not type_dist.get("warning"):
            report.findings.append(ReviewFinding(
                category="observations",
                severity="info",
                title="No warnings recorded",
                detail="Warnings capture gotchas and non-obvious behaviors that prevent repeated mistakes.",
            ))

        # Low confidence observations
        low_conf = (
            await session.execute(
                _filtered(
                    select(func.count()).select_from(Observation)
                    .where(Observation.confidence < 0.5),
                    Observation,
                    project,
                )
            )
        ).scalar() or 0

        if low_conf > 0:
            report.findings.append(ReviewFinding(
                category="observations",
                severity="suggestion",
                title=f"{low_conf} low-confidence observations",
                detail="Observations with confidence < 50% may add noise. Review and either boost or remove.",
                action="hafiz recall '' --limit 50 --json  # then filter by confidence",
            ))

        # Stale observations (older than 90 days)
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        stale_obs = (
            await session.execute(
                _filtered(
                    select(func.count()).select_from(Observation)
                    .where(Observation.valid_from < cutoff),
                    Observation,
                    project,
                )
            )
        ).scalar() or 0

        if stale_obs > 0:
            report.findings.append(ReviewFinding(
                category="staleness",
                severity="info",
                title=f"{stale_obs} observations older than 90 days",
                detail="Older observations may still be valid, but periodic review keeps knowledge fresh.",
                action="hafiz recall '' --limit 20 --json  # review and invalidate if outdated",
            ))

        # ── Graph checks ───────────────────────────────────────────────

        # Orphan entities (no relations)
        if entity_count > 0:
            connected_ids = (
                select(Relation.source_id).union(select(Relation.target_id))
            ).subquery()

            orphan_count = (
                await session.execute(
                    _filtered(
                        select(func.count()).select_from(Entity)
                        .where(Entity.id.notin_(select(connected_ids.c[0]))),
                        Entity,
                        project,
                    )
                )
            ).scalar() or 0

            if orphan_count > 0:
                pct = round(orphan_count / entity_count * 100)
                report.findings.append(ReviewFinding(
                    category="graph",
                    severity="suggestion" if pct > 30 else "info",
                    title=f"{orphan_count} orphan entities ({pct}%)",
                    detail="Entities without relations are isolated — they don't contribute to dependency analysis.",
                    action="hafiz graph show <entity> --json  # check if relations are missing",
                ))

        # Entities without descriptions
        if entity_count > 0:
            no_desc = (
                await session.execute(
                    _filtered(
                        select(func.count()).select_from(Entity)
                        .where(
                            (Entity.description.is_(None))
                            | (Entity.description == "")
                        ),
                        Entity,
                        project,
                    )
                )
            ).scalar() or 0

            if no_desc > 0:
                pct = round(no_desc / entity_count * 100)
                report.findings.append(ReviewFinding(
                    category="graph",
                    severity="suggestion" if pct > 50 else "info",
                    title=f"{no_desc} entities without descriptions ({pct}%)",
                    detail="Descriptions help semantic search find entities by concept, not just name.",
                ))

        # ── Coverage checks ────────────────────────────────────────────

        # Projects with chunks but no entities
        project_chunks = (
            await session.execute(
                select(Chunk.project, func.count())
                .where(Chunk.project.isnot(None))
                .group_by(Chunk.project)
            )
        ).all()

        project_entities = (
            await session.execute(
                select(Entity.project, func.count())
                .where(Entity.project.isnot(None))
                .group_by(Entity.project)
            )
        ).all()

        entity_projects = {p for p, _ in project_entities}
        for proj, count in project_chunks:
            if proj and proj not in entity_projects:
                report.findings.append(ReviewFinding(
                    category="coverage",
                    severity="suggestion",
                    title=f"Project '{proj}' has {count} chunks but no entities",
                    detail="Entity extraction hasn't been run for this project. Graph queries won't return results.",
                    action=f"hafiz chunks export --project {proj} --limit 200  # then extract entities",
                ))

        # Entity-to-chunk ratio
        if chunk_count > 0 and entity_count > 0:
            ratio = entity_count / chunk_count
            report.stats["entity_chunk_ratio"] = round(ratio, 3)

    return report


def _count_query(model, project: str | None, field_name: str = "project"):
    """Build a count query with optional project filter."""
    stmt = select(func.count()).select_from(model)
    if project:
        stmt = stmt.where(getattr(model, field_name if field_name == "project" else "id").isnot(None))
        if hasattr(model, "project"):
            stmt = select(func.count()).select_from(model).where(model.project == project)
    return stmt


def _filtered(stmt, model, project: str | None):
    """Add project filter to an existing statement if project is specified."""
    if project and hasattr(model, "project"):
        stmt = stmt.where(model.project == project)
    return stmt
