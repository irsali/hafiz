"""Was recalled knowledge actually used? — the delivery side of telemetry.

``retrievals`` records what the store *returned*. That says nothing about whether
an agent did anything with it, and a 2026-11 audit found that question
unanswerable: the text a recall hook injected into a session was in the harness's
own log but dropped on import, so "is hafiz helping?" had no data behind it.

The importer now keeps each hafiz hook injection as a ``role='system'`` message
tagged :data:`INJECTION_KIND`. This module reads them back and estimates uptake.

The estimate is deliberately simple and says so in its output:

* An injection is split into its memories (one markdown bullet each).
* A memory's **novel terms** are the specific tokens it carries — identifiers,
  paths, versions, hyphenated names, long words — that had *not* already
  appeared in the session before it was injected. Terms the user or a tool had
  already put in front of the agent prove nothing.
* The memory is **likely used** when the agent's own words and tool inputs that
  follow reuse enough of those novel terms.
* A memory with too few novel terms is **unscorable** and kept out of the rate,
  rather than counted as unused.

It over-credits when the agent finds the same facts independently (say, by
reading the file a memory describes) and under-credits when a memory changes
behaviour without changing vocabulary ("run tests locally"). Treat it as a
trend line, not a verdict on any single memory.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

#: ``metadata.kind`` on a source-layer message holding a hafiz hook injection.
INJECTION_KIND = "hafiz_injection"

#: Hook events whose memories stay relevant for the whole session (standing
#: rules), versus a prompt's memories, which are judged on that prompt's turn.
SESSION_WIDE_EVENTS = frozenset({"SessionStart"})

#: A memory needs at least this many novel terms to be judged at all.
MIN_NOVEL_TERMS = 3
#: ...and is "likely used" when the agent reuses this many of them...
MIN_REUSED_TERMS = 3
#: ...making up at least this share of its novel terms.
MIN_REUSED_SHARE = 0.2

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./#:@-]*[A-Za-z0-9]")
_SPECIFIC_CHARS = re.compile(r"[0-9_./#:@-]")
# "- **[decision | user:anjum]** text" — the label is presentation, not content.
_BULLET_LABEL = re.compile(r"^\s*-\s+(\*\*\[[^\]]*\]\*\*\s*)?")

#: Long, common words that are not evidence of anything. Small on purpose: the
#: novelty rule already discards words the session had used before.
_COMMON = frozenset(
    """
    something everything anything otherwise therefore whatever whenever because
    important different available currently including following previously
    actually basically generally information implementation application
    configuration documentation recommendation additionally specifically
    """.split()
)


def parse_bullets(text: str) -> list[str]:
    """Split a hook injection into its memories, one per top-level bullet.

    Continuation lines (``hafiz``'s metadata lines under a bullet) stay with
    their bullet; ``##`` headings belong to no memory and are dropped.
    """
    bullets: list[list[str]] = []
    for line in (text or "").splitlines():
        if line.startswith("## "):
            continue
        if line.startswith("- "):
            bullets.append([_BULLET_LABEL.sub("", line, count=1)])
        elif bullets and line.strip():
            bullets[-1].append(line.strip())
    return [" ".join(b).strip() for b in bullets if " ".join(b).strip()]


def specific_terms(text: str) -> set[str]:
    """The tokens in ``text`` specific enough to count as evidence of reuse.

    A token qualifies when it carries a digit or a structural character
    (``web-channel``, ``v2.4``, ``hafiz/core``, ``#47471``) or is long (9+
    characters) and not a stock word. Short plain words never qualify — "use",
    "test", "merge" turn up in every session.
    """
    out: set[str] = set()
    for raw in _TOKEN.findall(text or ""):
        tok = raw.lower().strip("./:-")
        if len(tok) < 4 or tok in _COMMON:
            continue
        if _SPECIFIC_CHARS.search(tok) or len(tok) >= 9:
            out.add(tok)
    return out


@dataclass
class MemoryVerdict:
    """How one injected memory fared."""

    preview: str
    novel_terms: int
    reused_terms: int

    @property
    def scorable(self) -> bool:
        return self.novel_terms >= MIN_NOVEL_TERMS

    @property
    def likely_used(self) -> bool:
        return (
            self.scorable
            and self.reused_terms >= MIN_REUSED_TERMS
            and self.reused_terms / self.novel_terms >= MIN_REUSED_SHARE
        )


def judge_memory(memory: str, *, seen_before: set[str], agent_after: set[str]) -> MemoryVerdict:
    """Judge one memory against what the session had already seen and what the
    agent went on to say/do. Both sets are :func:`specific_terms` outputs."""
    novel = specific_terms(memory) - seen_before
    return MemoryVerdict(
        preview=preview_of(memory),
        novel_terms=len(novel),
        reused_terms=len(novel & agent_after),
    )


def preview_of(memory: str, width: int = 100) -> str:
    """A stable key for grouping the same memory across injections.

    Recall excerpts long memories and may open them with an ellipsis, so the
    leading ellipsis is dropped before truncating.
    """
    flat = " ".join(memory.split()).lstrip("…. ")
    return flat[:width]


# ---------------------------------------------------------------------------
# Session walk
# ---------------------------------------------------------------------------


@dataclass
class _Msg:
    ts: datetime
    role: str
    content: str
    tool_calls: list | None
    metadata: dict = field(default_factory=dict)

    @property
    def is_injection(self) -> bool:
        return self.role == "system" and self.metadata.get("kind") == INJECTION_KIND

    def agent_text(self) -> str:
        """What the agent itself said or did: its text plus its tool inputs."""
        parts = [self.content or ""]
        for call in self.tool_calls or []:
            if isinstance(call, dict) and call.get("kind") == "tool_use":
                parts.append(str(call.get("input") or ""))
        return "\n".join(parts)

    def seen_text(self) -> str:
        """Everything this message put in front of the agent."""
        parts = [self.content or ""]
        for call in self.tool_calls or []:
            if isinstance(call, dict):
                parts.append(str(call.get("input") or call.get("content_preview") or ""))
        return "\n".join(parts)


def judge_session(messages: list[_Msg]) -> list[tuple[_Msg, list[MemoryVerdict]]]:
    """Judge every injection in one session. ``messages`` in any order.

    Ordered by ``ts``, not ``seq``: backfilled injections are appended with a
    later ``seq`` than the turns around them, so ``seq`` is not a time axis.
    """
    ordered = sorted(messages, key=lambda m: m.ts)
    # Tokenised once per message: a long session holds hundreds of injections,
    # and re-tokenising its history for each one made the report quadratic.
    agent_terms = [
        specific_terms(m.agent_text()) if m.role == "assistant" else None for m in ordered
    ]
    seen: set[str] = set()  # grows in time order; read before each injection
    out: list[tuple[_Msg, list[MemoryVerdict]]] = []
    for i, msg in enumerate(ordered):
        if not msg.is_injection:
            seen |= specific_terms(msg.seen_text())
            continue
        # Injections themselves never count as "seen", so a repeat is judged
        # on its own merits rather than against its earlier copy.
        session_wide = msg.metadata.get("hook_event") in SESSION_WIDE_EVENTS
        after: set[str] = set()
        started = False
        for m, terms in zip(ordered[i + 1 :], agent_terms[i + 1 :], strict=True):
            if m.role == "user":
                # A prompt-time injection lands just after its prompt; the
                # agent's reply runs until the *next* prompt.
                if started and not session_wide:
                    break
                continue
            if terms is not None:
                started = True
                after |= terms
        verdicts = [
            judge_memory(b, seen_before=seen, agent_after=after) for b in parse_bullets(msg.content)
        ]
        out.append((msg, verdicts))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


METHOD = (
    "rough: a memory counts as likely used when the agent's following turns reuse "
    f"at least {MIN_REUSED_TERMS} of its specific terms (and {MIN_REUSED_SHARE:.0%} of them) "
    "that the session had not seen before it was shown; memories with fewer than "
    f"{MIN_NOVEL_TERMS} such terms are unscorable and left out of the rate"
)


def _pct(sorted_values: list[int], q: float) -> int | None:
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


async def delivery_report(*, since_days: int = 30, limit: int = 10) -> dict:
    """Aggregate uptake over every captured injection in the window."""
    from sqlalchemy import select

    from hafiz.core.database import CommunicationMessage, get_session_factory

    since = datetime.now(UTC) - timedelta(days=since_days)
    factory = get_session_factory()
    async with factory() as session:
        # Filter on role in SQL and on metadata in Python: JSON-path predicates
        # differ between the two supported backends, and system rows are rare.
        comm_ids = {
            cid
            for cid, meta in (
                await session.execute(
                    select(CommunicationMessage.communication_id, CommunicationMessage.metadata_)
                    .where(CommunicationMessage.role == "system")
                    .where(CommunicationMessage.ts >= since)
                )
            ).all()
            if (meta or {}).get("kind") == INJECTION_KIND
        }
        rows = []
        if comm_ids:
            rows = (
                await session.execute(
                    select(
                        CommunicationMessage.communication_id,
                        CommunicationMessage.ts,
                        CommunicationMessage.role,
                        CommunicationMessage.content,
                        CommunicationMessage.tool_calls,
                        CommunicationMessage.metadata_,
                    ).where(CommunicationMessage.communication_id.in_(comm_ids))
                )
            ).all()

    by_comm: dict = defaultdict(list)
    for cid, ts, role, content, tool_calls, meta in rows:
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        by_comm[cid].append(_Msg(ts, role, content or "", tool_calls, meta or {}))

    events: dict[str, dict] = defaultdict(
        lambda: {
            "injections": 0,
            "memories_shown": 0,
            "scorable": 0,
            "likely_used": 0,
            "chars": [],
            "ms": [],
        }
    )
    per_memory: dict[str, dict] = defaultdict(lambda: {"shown": 0, "scorable": 0, "used": 0})
    sessions = 0
    for msgs in by_comm.values():
        judged = [(inj, v) for inj, v in judge_session(msgs) if inj.ts >= since]
        if judged:
            sessions += 1
        for inj, verdicts in judged:
            ev = events[inj.metadata.get("hook_event") or "unknown"]
            ev["injections"] += 1
            ev["memories_shown"] += len(verdicts)
            ev["chars"].append(len(inj.content))
            if isinstance(inj.metadata.get("duration_ms"), int):
                ev["ms"].append(inj.metadata["duration_ms"])
            for v in verdicts:
                mem = per_memory[v.preview]
                mem["shown"] += 1
                if v.scorable:
                    ev["scorable"] += 1
                    mem["scorable"] += 1
                if v.likely_used:
                    ev["likely_used"] += 1
                    mem["used"] += 1

    by_event = []
    for name, ev in sorted(events.items(), key=lambda kv: -kv[1]["injections"]):
        ms = sorted(ev["ms"])
        by_event.append(
            {
                "event": name,
                "injections": ev["injections"],
                "memories_shown": ev["memories_shown"],
                "scorable": ev["scorable"],
                "likely_used": ev["likely_used"],
                "used_rate": round(ev["likely_used"] / ev["scorable"], 3)
                if ev["scorable"]
                else None,
                "avg_chars": round(statistics.mean(ev["chars"])) if ev["chars"] else None,
                "p50_ms": _pct(ms, 0.5),
                "p95_ms": _pct(ms, 0.95),
            }
        )

    most_used = sorted(
        ({"preview": p, **m} for p, m in per_memory.items() if m["used"]),
        key=lambda r: (-r["used"], -r["shown"]),
    )[:limit]
    # Shown repeatedly, judgeable every time, never once picked up: the rows to
    # reword, retarget, or unpin.
    ignored = sorted(
        (
            {"preview": p, **m}
            for p, m in per_memory.items()
            if m["scorable"] >= 3 and not m["used"]
        ),
        key=lambda r: -r["shown"],
    )[:limit]

    return {
        "injections": sum(e["injections"] for e in by_event),
        "sessions": sessions,
        "by_event": by_event,
        "most_used": most_used,
        "shown_never_used": ignored,
        "method": METHOD,
    }
