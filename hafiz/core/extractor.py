"""Agent extraction v2 — annotations + semantic edges.

Post-structural-grounding, the agent's job narrows. Parsers own
**structure**: entities (functions, classes, modules, docs) and their
syntactic relations (calls, imports, inherits). Agents own **meaning**:
concepts, patterns, decisions, warnings, and semantic relations between
units (``implements_pattern``, ``is_workaround_for``,
``depends_on_concept``, ``supersedes_approach``).

The v2 import contract rejects AST-territory facts at the door — the
import layer is the enforcer of the non-duplication rule. Agents who
still send v1 payloads (``entity_type`` / ``relation_type`` vocabulary)
get a clear migration error.

JSON shape::

    {
      "version": 2,
      "annotations": [
        {
          "content": "Auth canonicalized through this entry point",
          "kind": "pattern",
          "source": "agent:claude-code",
          "unit_identity_key": "<sha256 hex>",   # preferred
          "unit_name": "UserService.authenticate", # fallback
          "source_file": "/abs/path/auth.py",      # fallback helper
          "tags": ["auth"],
          "confidence": 0.9
        }
      ],
      "edges": [
        {
          "source_name": "UserService",
          "source_file": "/abs/path/auth.py",
          "target_name": "SecurityPolicy",
          "target_file": "/abs/path/policy.py",
          "relation": "implements_pattern",
          "evidence": "canonicalized via policy_engine.enforce(...)"
        }
      ]
    }
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from hafiz.core.annotations import store_annotation
from hafiz.core.database import Edge, File, Unit, get_session_factory

EXTRACT_CONTRACT_VERSION = 2


# ── Vocabulary enforcement ────────────────────────────────────────────────

# Kinds an AST parser emits. Agents that try to write any of these get
# rejected with a clear "that's the parser's job" error.
AST_UNIT_KIND_PREFIXES = ("code.",)

# Relations the Python AST (and future tree-sitter) parsers own. Same
# rejection rule.
AST_RELATION_NAMES = frozenset({"calls", "imports", "inherits", "references"})

# Annotation kinds the agent is free to use. Open set (anything not in
# AST_UNIT_KIND_PREFIXES), but the canonical values are enumerated so
# tooling can validate autocomplete and so the contract is legible.
AGENT_ANNOTATION_KINDS = frozenset(
    {
        "fact",
        "decision",
        "learning",
        "pattern",
        "warning",
        "note",
        "concept",
        "service",
    }
)

# Semantic relations the agent may add. Open set — tooling should warn
# rather than reject if a new name shows up. Structural relations
# (AST_RELATION_NAMES) are hard-rejected regardless.
AGENT_RELATION_NAMES = frozenset(
    {
        "implements_pattern",
        "is_workaround_for",
        "supersedes_approach",
        "depends_on_concept",
        "related_to",
        "documents",
        "configures",
    }
)


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class ExtractedAnnotation:
    content: str
    kind: str = "fact"
    source: str | None = None
    unit_identity_key: str | None = None
    unit_name: str | None = None
    source_file: str | None = None
    project: str | None = None
    tags: list[str] | None = None
    confidence: float = 1.0


@dataclass
class ExtractedEdge:
    source_name: str
    target_name: str
    relation: str
    source_file: str | None = None
    target_file: str | None = None
    evidence: str | None = None


@dataclass
class ExtractionResult:
    annotations: list[ExtractedAnnotation] = field(default_factory=list)
    edges: list[ExtractedEdge] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ── Parsing + validation ──────────────────────────────────────────────────


class ExtractContractError(ValueError):
    """Raised when an import payload doesn't match v2."""


def parse_extraction_payload(data: dict[str, Any]) -> ExtractionResult:
    """Parse and validate a v2 extraction payload.

    Rejects v1 payloads (``entities`` / ``relations`` with
    ``entity_type`` / ``relation_type``) with a migration message.
    Rejects AST-territory kinds and relations. Collects warnings for
    annotations that reference units without resolution info.
    """
    if "entities" in data or "relations" in data:
        raise ExtractContractError(
            "This payload looks like agent extraction v1 "
            "('entities'/'relations' with entity_type/relation_type). "
            "The contract bumped to v2 with the structural-grounding work "
            "(see workitems/active/structural-grounding.md). Agents now "
            "write annotations + semantic edges only; structural facts "
            "(calls/imports/inherits/classes/functions) are owned by the "
            "parser. Update your extractor or re-run `hafiz agent install` "
            "to pick up the current skills.md."
        )

    version = data.get("version")
    if version != EXTRACT_CONTRACT_VERSION:
        raise ExtractContractError(
            f"extraction payload missing or wrong version "
            f"(got {version!r}, want {EXTRACT_CONTRACT_VERSION}). "
            "Set top-level {'version': 2} and see `hafiz agent install` "
            "for the current skills.md."
        )

    warnings: list[str] = []

    raw_anns = data.get("annotations", [])
    if not isinstance(raw_anns, list):
        raise ExtractContractError("'annotations' must be a list.")
    annotations: list[ExtractedAnnotation] = []
    for i, row in enumerate(raw_anns):
        if not isinstance(row, dict):
            raise ExtractContractError(f"annotations[{i}] must be an object.")
        content = row.get("content")
        if not content or not isinstance(content, str):
            raise ExtractContractError(f"annotations[{i}].content is required.")
        kind = row.get("kind", "fact")
        if any(kind.startswith(p) for p in AST_UNIT_KIND_PREFIXES):
            raise ExtractContractError(
                f"annotations[{i}].kind={kind!r} is AST-territory "
                f"({', '.join(AST_UNIT_KIND_PREFIXES)}*). Parsers own "
                "structural units; agents add annotations with kinds "
                f"like {sorted(AGENT_ANNOTATION_KINDS)}."
            )
        if kind not in AGENT_ANNOTATION_KINDS:
            warnings.append(
                f"annotations[{i}].kind={kind!r} is not in the canonical "
                f"agent vocabulary ({sorted(AGENT_ANNOTATION_KINDS)}); "
                "accepted but may confuse downstream tooling."
            )
        annotations.append(
            ExtractedAnnotation(
                content=content,
                kind=kind,
                source=row.get("source"),
                unit_identity_key=row.get("unit_identity_key"),
                unit_name=row.get("unit_name"),
                source_file=row.get("source_file"),
                project=row.get("project"),
                tags=row.get("tags"),
                confidence=float(row.get("confidence", 1.0)),
            )
        )

    raw_edges = data.get("edges", [])
    if not isinstance(raw_edges, list):
        raise ExtractContractError("'edges' must be a list.")
    edges: list[ExtractedEdge] = []
    for i, row in enumerate(raw_edges):
        if not isinstance(row, dict):
            raise ExtractContractError(f"edges[{i}] must be an object.")
        for required in ("source_name", "target_name", "relation"):
            if not row.get(required):
                raise ExtractContractError(f"edges[{i}].{required} is required.")
        relation = row["relation"]
        if relation in AST_RELATION_NAMES:
            raise ExtractContractError(
                f"edges[{i}].relation={relation!r} is AST-territory "
                f"({sorted(AST_RELATION_NAMES)}). Parsers own structural "
                "relations; agents add semantic ones like "
                f"{sorted(AGENT_RELATION_NAMES)}."
            )
        if relation not in AGENT_RELATION_NAMES:
            warnings.append(
                f"edges[{i}].relation={relation!r} is not in the canonical "
                f"agent vocabulary ({sorted(AGENT_RELATION_NAMES)}); "
                "accepted but may confuse downstream tooling."
            )
        edges.append(
            ExtractedEdge(
                source_name=row["source_name"],
                target_name=row["target_name"],
                relation=relation,
                source_file=row.get("source_file"),
                target_file=row.get("target_file"),
                evidence=row.get("evidence"),
            )
        )

    return ExtractionResult(
        annotations=annotations,
        edges=edges,
        warnings=warnings,
    )


# ── Unit resolution ───────────────────────────────────────────────────────


async def _resolve_unit(
    session: AsyncSession,
    *,
    identity_key: str | None,
    name: str | None,
    source_file: str | None,
    project: str | None,
) -> Unit | None:
    """Find a Unit the agent referenced. Preference order:

    1. ``identity_key`` exact match — the stable, correct handle.
    2. ``(name, source_file, project)`` fallback — ``name`` is the
       unit's qualified name; ``source_file`` narrows the lookup.
    3. ``(name, project)`` if no source_file — may pick wrong unit
       when a name is shared across files; caller takes the risk.
    """
    if identity_key:
        row = (
            await session.execute(
                select(Unit).where(
                    Unit.identity_key == identity_key,
                    Unit.valid_until.is_(None),
                )
            )
        ).scalar_one_or_none()
        if row is not None:
            return row

    if not name:
        return None

    stmt = (
        select(Unit)
        .join(File, File.id == Unit.file_id)
        .where(Unit.name == name, Unit.valid_until.is_(None))
    )
    if source_file:
        stmt = stmt.where(File.path == source_file)
    if project is not None:
        stmt = stmt.where(File.project == project)
    return (await session.execute(stmt.limit(1))).scalar_one_or_none()


# ── Storage ───────────────────────────────────────────────────────────────


async def store_extraction(
    result: ExtractionResult,
    *,
    project: str | None = None,
) -> tuple[int, int, int]:
    """Persist a parsed ExtractionResult.

    Returns ``(annotations_written, edges_written, unresolved_count)``
    where ``unresolved_count`` is the number of rows whose unit /
    target couldn't be bound — they're still stored, just with
    ``unit_id=None`` on annotations or ``target_unit_id=None`` on
    edges.
    """
    ann_written = 0
    edge_written = 0
    unresolved = 0

    session_factory = get_session_factory()

    # Annotations go through the standard annotation store so git
    # context, metadata merging, and embedding all happen uniformly.
    for ann in result.annotations:
        unit_id: str | None = None
        async with session_factory() as session:
            unit = await _resolve_unit(
                session,
                identity_key=ann.unit_identity_key,
                name=ann.unit_name,
                source_file=ann.source_file,
                project=ann.project or project,
            )
            if unit is not None:
                unit_id = str(unit.id)
            elif ann.unit_identity_key or ann.unit_name:
                unresolved += 1

        await store_annotation(
            ann.content,
            kind=ann.kind,
            source=ann.source,
            project=ann.project or project,
            tags=ann.tags,
            confidence=ann.confidence,
            unit_id=unit_id,
        )
        ann_written += 1

    # Semantic edges are written directly to the edges table with
    # source='agent'. Source must resolve to a real unit (or the edge is
    # dropped); target may stay unresolved.
    async with session_factory() as session:
        async with session.begin():
            for edge in result.edges:
                src_unit = await _resolve_unit(
                    session,
                    identity_key=None,
                    name=edge.source_name,
                    source_file=edge.source_file,
                    project=project,
                )
                if src_unit is None:
                    unresolved += 1
                    continue

                tgt_unit = await _resolve_unit(
                    session,
                    identity_key=None,
                    name=edge.target_name,
                    source_file=edge.target_file,
                    project=project,
                )
                if tgt_unit is None:
                    unresolved += 1

                session.add(
                    Edge(
                        id=uuid.uuid4(),
                        source_unit_id=src_unit.id,
                        target_unit_id=tgt_unit.id if tgt_unit else None,
                        target_name=edge.target_name,
                        relation=edge.relation,
                        source="agent",
                        evidence=edge.evidence,
                    )
                )
                edge_written += 1

    return ann_written, edge_written, unresolved
