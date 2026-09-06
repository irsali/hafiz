"""The single source of truth for the MCP tool surface.

Hafiz's contract with agents is ``hafiz`` + ``--json``. MCP adds a *second*
contract, for clients that cannot shell out — and a second contract is a
second thing to keep in sync. The whole design here exists to make that sync
structural rather than a review habit.

How drift is prevented
----------------------

A :class:`ToolSpec` names a **core function**, not a copied parameter list.
Names, types and requiredness are read from ``inspect.signature`` at schema
build time, so a parameter renamed or removed in ``hafiz/core`` changes the
tool schema in the same commit, with no MCP-side edit.

Prose descriptions are the one thing that cannot be derived — only 2 of the
9 core functions behind these tools carry Google-style ``Args:`` blocks, and
inventing descriptions from parameter names would be worse than useless to a
model choosing between tools. So descriptions live in ``ToolSpec.params``,
and :func:`check_drift` asserts that set is *exactly* the set of exposed
parameters. Add a parameter to a core function and the build fails until
someone writes a sentence about it. That is the intended cost.

The precedent is ``daemon.py``'s ``_supported_kwargs``: it forwards caller
arguments wholesale against the real signature rather than through a
hand-maintained list, after a hand-maintained list silently dropped ``tags``
and ``active_only`` on the warm path. Same lesson, applied earlier.

What is deliberately absent
---------------------------

Install and administrative commands (``init``, ``hooks``, ``agent``,
``config set``, ``embedding``, ``serve``, ``watch``), destructive ones
(``forget``, ``prune``) and long-running ingest paths (``ingest``,
``import``, ``extract``, ``review``) are **not** exposed. An MCP client can
call any listed tool without confirmation, so anything irreversible or
environment-mutating stays on the CLI, where a human is present and the act
is deliberate.

This module must not import ``mcp``. The SDK is an optional extra
(``pip install hafiz[mcp]``), and the drift tests have to run without it.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import enum
import importlib
import inspect
import uuid
from collections.abc import Mapping
from typing import Any

from hafiz.core import telemetry

#: Parameters no tool ever exposes, whatever function it points at. These are
#: internal plumbing rather than capability: the caller has no business
#: setting them and a model has no way to choose a sensible value.
_NEVER_EXPOSE = frozenset({"telemetry_command"})


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    """One MCP tool, defined by the core function it calls."""

    name: str
    summary: str
    #: ``"module:function"`` — resolved lazily so importing this module stays
    #: cheap and free of database side effects.
    target: str
    #: Exposed parameter name -> description shown to the model. Must exactly
    #: match the exposed parameters of ``target``; see :func:`check_drift`.
    params: Mapping[str, str]
    #: Parameters pinned to a fixed value. Pinned implies hidden — a value the
    #: caller cannot influence has no place in the schema.
    force: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    #: Parameters hidden without being pinned, each for a stated reason.
    exclude: Mapping[str, str] = dataclasses.field(default_factory=dict)
    #: True for tools that write. Surfaced to the client as a hint so it can
    #: apply its own confirmation policy.
    writes: bool = False

    def resolve(self):
        module_name, _, fn_name = self.target.partition(":")
        return getattr(importlib.import_module(module_name), fn_name)

    def hidden(self) -> frozenset[str]:
        return frozenset(self.force) | frozenset(self.exclude) | _NEVER_EXPOSE

    def exposed(self) -> list[inspect.Parameter]:
        """Signature parameters this tool actually offers, in order."""
        hidden = self.hidden()
        return [
            p
            for name, p in inspect.signature(self.resolve()).parameters.items()
            if name not in hidden and p.kind is not inspect.Parameter.VAR_KEYWORD
        ]

    def input_schema(self) -> dict:
        properties: dict[str, dict] = {}
        required: list[str] = []
        for p in self.exposed():
            schema = _json_type(p.annotation)
            schema["description"] = self.params[p.name]
            properties[p.name] = schema
            if p.default is inspect.Parameter.empty:
                required.append(p.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }


def _json_type(annotation: Any) -> dict:
    """Map a Python annotation to a JSON Schema fragment.

    Annotations arrive as *strings* — every core module uses
    ``from __future__ import annotations`` — so this reads the source text
    rather than resolving types. That is deliberate: ``get_type_hints`` would
    need every name in every core module's namespace to be importable here,
    and would raise on the first forward reference, turning a description
    problem into an import problem.
    """
    text = annotation if isinstance(annotation, str) else _render(annotation)
    text = text.replace(" ", "")
    optional = "|None" in text or "None|" in text
    base = text.replace("|None", "").replace("None|", "")

    if base.startswith(("list[", "Sequence[", "tuple[")):
        inner = base[base.index("[") + 1 : -1]
        schema: dict = {"type": "array", "items": _json_type(inner)}
    elif base in {"dict", "Mapping"} or base.startswith(("dict[", "Mapping[")):
        schema = {"type": "object"}
    elif base == "bool":
        schema = {"type": "boolean"}
    elif base == "int":
        schema = {"type": "integer"}
    elif base == "float":
        schema = {"type": "number"}
    elif base in {"datetime", "_datetime.datetime", "datetime.datetime"}:
        schema = {"type": "string", "format": "date-time"}
    else:
        # str, uuid.UUID, unions of them, and anything unrecognised. A string
        # is the honest default: every remaining type in these signatures is
        # something the caller writes as text.
        schema = {"type": "string"}

    if optional:
        # Advertised as nullable rather than as a union, because a model
        # reading `["string","null"]` reliably understands "may be omitted",
        # whereas anyOf branches get filled in with literal "null" strings.
        schema = {**schema, "type": [schema["type"], "null"]}
    return schema


def _render(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "str"
    return getattr(annotation, "__name__", None) or str(annotation)


# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------

#: Shared descriptions. Several core functions take the same scoping
#: parameters, and a model comparing two tools should not see two different
#: explanations of ``project``.
_PROJECT = "Restrict to one indexed project. Omit to search everything."
_LIMIT = "Maximum rows to return."
_MIN_SCORE = (
    "Relevance floor, 0-1, applied to the score results are ranked by. "
    "Reranked scores separate sharply, so 0.05 already excludes off-topic "
    "matches; an empty result is the 'not relevant' signal."
)
_DOMAINS = (
    "Data domains to include, e.g. ['code','doc']. The domain is the part of "
    "a unit kind before the dot (code, doc, chat, mail, file). Note the index "
    "is ~89% documentation and only ~4% code."
)
_EX_DOMAINS = "Data domains to exclude. Mutually exclusive with include_domains per domain."

_OBSERVE_PARAMS = {
    "content": "The text to record. One claim per annotation reads back better than a prose blob.",
    "kind": (
        "fact | decision | learning | pattern | warning | note | concept | service. "
        "Preference or rule -> learning; claim -> fact; architectural choice -> "
        "decision; gotcha -> warning."
    ),
    "source": (
        "Who is writing: 'agent:<name>' or 'user:<name>'. Defaults to the "
        "connected MCP client's own name."
    ),
    "project": _PROJECT,
    "tags": "Categorisation tags.",
    "confidence": "Confidence 0.0-1.0.",
    "valid_until": "Expiry timestamp. Expired rows drop out of recall by default.",
    "session_id": "Session uuid or slug to tag this write with.",
    "task": "Named task within the session.",
    "supersedes_id": (
        "UUID of an annotation this one replaces. Use only on genuine "
        "contradiction or replacement, never on a reword."
    ),
    "derived_from": "UUIDs this was distilled from — records lineage without replacing.",
}

_OBSERVE_HIDDEN = {
    "unit_id": "Attaching to a code unit needs an identity_key the caller does not have.",
    "commit_hash": "Auto-captured from the repo; a caller-supplied value would be a lie.",
    "valid_from": "Defaults to now. Backdating a belief is not something to expose.",
    "metadata": "Arbitrary JSONB. No stable meaning for a model to target.",
}

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="hafiz_context",
        summary=(
            "Start here for any task. Returns one bundle of relevant code/doc units, "
            "the graph neighbourhood around them, and prior decisions — the same thing "
            "`hafiz context` builds."
        ),
        target="hafiz.core.context:build_context",
        params={
            "query": "What you are about to work on, in a sentence.",
            "project": _PROJECT,
            "limit_chunks": "Maximum content units in the bundle.",
            "limit_annotations": "Maximum prior observations in the bundle.",
            "include_domains": _DOMAINS,
            "exclude_domains": _EX_DOMAINS,
            "min_score": _MIN_SCORE,
        },
    ),
    ToolSpec(
        name="hafiz_query",
        summary=(
            "Semantic search over indexed content — code, docs, notes. Reaches prose "
            "that grep cannot, because it matches meaning rather than tokens."
        ),
        target="hafiz.core.search:vector_search",
        params={
            "query": "Natural-language description of what you are looking for.",
            "limit": _LIMIT,
            "project": _PROJECT,
            "kind": "Exact unit kind, e.g. 'code.function' or 'doc.heading'.",
            "include_domains": _DOMAINS,
            "exclude_domains": _EX_DOMAINS,
            "similarity_threshold": "Cosine floor, 0-1. Useful band is roughly 0.5-0.65.",
            "dedup": "Collapse byte-identical results.",
        },
        force={"telemetry_command": telemetry.QUERY},
    ),
    ToolSpec(
        name="hafiz_recall_observations",
        summary=(
            "Search the wisdom layer: decisions, learnings, patterns, warnings and facts "
            "recorded by agents and the user. Ask this before re-deciding something."
        ),
        target="hafiz.core.annotations:search_annotations",
        params={
            "query": "Topic to recall.",
            "limit": _LIMIT,
            "project": _PROJECT,
            "kind": "fact | decision | learning | pattern | warning | note | concept | service.",
            "source": "Filter by writer, e.g. 'user:anjum' or 'agent:claude-code'.",
            "tags": "Only annotations carrying all of these tags.",
            "active_only": (
                "True (default) hides superseded and expired rows. False reads prior beliefs too."
            ),
            "rerank": "Cross-encoder reranking. On by default; sharpens precision.",
            "min_score": _MIN_SCORE,
        },
        force={"telemetry_command": telemetry.OBSERVATIONS},
    ),
    ToolSpec(
        name="hafiz_recall_session",
        summary=(
            "Search inside captured agent transcripts (the source layer). Separate from "
            "the wisdom layer on purpose — raw conversation, not curated belief."
        ),
        target="hafiz.core.communications:search_messages",
        params={
            "query": "Text to find across transcript turns.",
            "limit": _LIMIT,
            "agent": "Restrict to one harness, e.g. 'claude-code' or 'cursor'.",
            "session_id": "Restrict to one session uuid.",
            "communication_id": "Restrict to one communication uuid.",
        },
    ),
    ToolSpec(
        name="hafiz_graph",
        summary=(
            "Walk the code graph: what a unit depends on, what depends on it (blast "
            "radius before a refactor), the path between two units, or overall shape."
        ),
        target="hafiz.core.graph_ops:graph_op",
        params={
            "operation": (
                "show (unit + direct links) | deps (outgoing) | impact (blast radius, "
                "ask before refactoring) | path (shortest route between two units) | "
                "rank (most central units) | stats (overall health)."
            ),
            "name": "Unit name. Required for show, deps, impact and path.",
            "target": "Destination unit name. Required for 'path'.",
            "project": _PROJECT,
            "depth": "Hops to walk for deps/impact.",
            "metric": "Centrality metric for 'rank': pagerank | degree | betweenness.",
            "limit": _LIMIT,
        },
    ),
    ToolSpec(
        name="hafiz_journal",
        summary=(
            "What was recorded recently, grouped by day. The 'what did I learn last week' view."
        ),
        target="hafiz.core.journal:build_journal",
        params={
            "since": "Window, e.g. '7d', '2w', '3mo'.",
            "day": "A single ISO date instead of a window.",
            "project": _PROJECT,
            "source": "Filter by writer.",
            "kind": "Filter by annotation kind.",
            "session_id": "Restrict to one session.",
            "task": "Restrict to one named task.",
            "limit": _LIMIT,
        },
    ),
    ToolSpec(
        name="hafiz_distill",
        summary=(
            "The promotable backlog: raw captures and notes clustered by theme, ready to "
            "be turned into durable decisions. Hafiz never calls an LLM here — you are "
            "the distiller."
        ),
        target="hafiz.core.distill:find_distill_candidates",
        params={
            "since": "Window, e.g. '7d'.",
            "project": _PROJECT,
            "session_id": "Restrict to one session.",
            "task": "Restrict to one named task.",
            "include_transcripts": "Also pull salient turns from captured transcripts.",
            "include_promoted": "Include captures already distilled into annotations.",
            "limit": _LIMIT,
            "message_limit": "Cap on transcript turns pulled into the window.",
            "cluster_threshold": (
                "Cosine similarity at/above which two captures share a theme. Much "
                "looser than duplicate detection — 'same topic', not 'same claim'."
            ),
        },
    ),
    ToolSpec(
        name="hafiz_observe",
        summary=(
            "Record a durable decision, learning, pattern, warning or fact. Use after "
            "deciding an approach, or on discovering a gotcha — this is what makes the "
            "knowledge survive the session."
        ),
        target="hafiz.core.annotations:store_annotation",
        params=_OBSERVE_PARAMS,
        exclude=_OBSERVE_HIDDEN,
        writes=True,
    ),
    ToolSpec(
        name="hafiz_note",
        summary=(
            "Capture a half-formed thought with no ceremony. Below decision grade — "
            "`hafiz_distill` surfaces these later for promotion."
        ),
        target="hafiz.core.annotations:store_annotation",
        params={k: v for k, v in _OBSERVE_PARAMS.items() if k not in {"kind", "supersedes_id"}},
        force={"kind": "note"},
        exclude={
            **_OBSERVE_HIDDEN,
            "supersedes_id": "Raw capture never supersedes a considered belief.",
        },
        writes=True,
    ),
    ToolSpec(
        name="hafiz_capture",
        summary=(
            "Store a conversation transcript in the source layer. Split into turns, "
            "selectively embedded, retention-bounded."
        ),
        target="hafiz.core.capture:store_transcript",
        params={
            "text": "The transcript. Split into turns automatically.",
            "title": "Human-readable label.",
            "project": _PROJECT,
            "source": "Origin, e.g. 'agent:cursor'. Defaults to the connected client.",
            "tags": "Categorisation tags.",
            "session_id": "Session uuid or slug to attach to.",
            "task": "Named task within the session.",
        },
        writes=True,
    ),
    ToolSpec(
        name="hafiz_session",
        summary=(
            "Group a thread of work. Writes tagged with a session can be pulled back "
            "together later as one narrative."
        ),
        target="hafiz.core.session_ops:session_op",
        params={
            "operation": "list | show | start | end.",
            "slug": "Human-facing session identifier. Required for show, start and end.",
            "name": "Descriptive title, for 'start'.",
            "agent": "Which agent owns this session.",
            "task": "Named task within the session, for 'start'.",
            "project": _PROJECT,
            "limit": _LIMIT,
            "include_ended": "For 'list': also return closed sessions.",
        },
        writes=True,
    ),
    ToolSpec(
        name="hafiz_reconcile",
        summary=(
            "Read-only sweep for near-duplicate live annotations, clustered so you can "
            "supersede or retire them. Finds drift that predates a write."
        ),
        target="hafiz.core.annotations:reconcile_duplicates",
        params={
            "project": _PROJECT,
            "kind": "Restrict to one annotation kind.",
            "threshold": "Cosine similarity at/above which two annotations cluster.",
            "limit": _LIMIT,
        },
    ),
    ToolSpec(
        name="hafiz_status",
        summary=(
            "Health of the store: how much is indexed, which projects have fallen behind "
            "their repo, whether retention is being enforced, and whether transcripts are "
            "still arriving. Check index freshness before trusting a code search."
        ),
        target="hafiz.core.health:collect_status",
        params={
            "verbose": (
                "False (default) names only the projects actually behind their repo and "
                "adds a 'staleness_summary' count. True also lists every up-to-date "
                "project. Prefer False: the answer you want is which projects are stale, "
                "not a roll-call of the healthy ones."
            ),
        },
    ),
    ToolSpec(
        name="hafiz_retrievals",
        summary=(
            "What the store was asked for and could NOT answer — a worklist of things "
            "worth recording. The most useful signal in the telemetry."
        ),
        target="hafiz.core.telemetry:retrieval_report",
        params={
            "since_days": "Window in days.",
            "limit": _LIMIT,
        },
    ),
)


def by_name() -> dict[str, ToolSpec]:
    return {t.name: t for t in TOOLS}


def check_drift() -> list[str]:
    """Return one message per registry/signature disagreement. Empty is healthy.

    Both directions matter. A described parameter that no longer exists means
    the tool advertises something the core function will reject; an exposed
    parameter with no description means a capability was added and silently
    handed to models with no explanation of what it does.
    """
    problems: list[str] = []
    for spec in TOOLS:
        try:
            signature_params = {p.name for p in spec.exposed()}
        except Exception as e:  # noqa: BLE001 — report, don't abort the sweep
            problems.append(f"{spec.name}: cannot resolve {spec.target!r} ({e})")
            continue

        described = set(spec.params)
        for missing in sorted(signature_params - described):
            problems.append(
                f"{spec.name}: parameter {missing!r} exists on {spec.target} but has no "
                f"description. Add one to ToolSpec.params, or hide it via exclude= with "
                f"a reason."
            )
        for stale in sorted(described - signature_params):
            problems.append(
                f"{spec.name}: describes parameter {stale!r}, which {spec.target} no "
                f"longer accepts. Remove it."
            )

        real = set(inspect.signature(spec.resolve()).parameters)
        for pinned in sorted(set(spec.force) - real):
            problems.append(f"{spec.name}: force={pinned!r} is not a parameter of {spec.target}.")
        for hidden in sorted(set(spec.exclude) - real):
            problems.append(f"{spec.name}: exclude={hidden!r} is not a parameter of {spec.target}.")
    return problems


def to_jsonable(value: Any) -> Any:
    """Best-effort JSON view of whatever a core function returned.

    Core functions return dataclasses, ORM rows, dicts and scalars. This
    walks all of them rather than requiring each tool to declare a shape,
    which would be another hand-maintained list to rot.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, _datetime.datetime | _datetime.date):
        return value.isoformat()
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, "__table__"):  # SQLAlchemy ORM row
        return {
            column.name: to_jsonable(getattr(value, column.name))
            for column in value.__table__.columns
        }
    return str(value)
