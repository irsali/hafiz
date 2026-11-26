"""Every-turn memory recall for agent harnesses — the logic behind ``hafiz hook``.

Two entry points, both driven by harness hooks (see ``hafiz agent install
claude-code --hooks``):

``prompt``   UserPromptSubmit. Reads the harness JSON on stdin, takes ``.prompt``
             and recalls the observations relevant to *that* task.
``session``  SessionStart. Injects the PINNED always-on rules, then a recall
             scoped to the repo and branch the session opened in.

Why session mode has two parts
------------------------------
A fixed session-start query surfaced the same handful of rows in every session
(a 30-day audit of one store: 411 firings, ONE distinct query, 11 distinct
annotations across 2,466 slots, while 114 of 125 standing rules never appeared).
It was a hardcoded digest dressed up as retrieval. Two jobs were being conflated:

  PIN      Rules that apply regardless of topic. "Always applies" is not a
           semantic property, so no query text retrieves it — they are addressed
           by **tag** (``always``), with reranking off and no relevance floor:
           both exist to make ranking judgements, and a pin is the absence of
           one. Curate by tagging, not by tuning a query.

  TOPICAL  Everything else, scoped to where the session opened. Repo name and
           branch are the only context a SessionStart hook gets (there is no
           prompt yet), and they make a good query: a branch called
           ``feat/iab-tcf-2.3`` recalls the never-merge-this rule on its own.

Pins render **whole**. ``md`` / ``compact`` excerpt long records to the span
best matching the query; under a tag pin the query text carries no meaning, so
the span is arbitrary and can open a rule at its rationale with the rule itself
elided. A prohibition with its scope clause cut off is worse than an absent one.

The output cap
--------------
Claude Code cuts hook output over 10,000 characters: it saves the text to a file
and the model sees only a ~2 KB preview. A pin set that grows past that silently
stops delivering its *tail* — so output is capped (``[hooks] max_chars``), pins
fill the budget first, then topical rows, and whatever did not fit is counted in
a single closing line that names the command to list it.

Contract: never fail a turn. :func:`run_hook` catches everything and returns
``""`` on any failure; the caller always exits 0.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
from datetime import datetime
from pathlib import Path

PIN_TAG = "always"

#: "continue", "yes", "go on" retrieve nothing useful.
MIN_PROMPT_CHARS = 12
#: Only the head of a long prompt carries the topic, and the models truncate anyway.
MAX_QUERY_CHARS = 2000

#: 0–1 floor on the cross-encoder score recall results are ranked by. Reranked
#: scores separate by orders of magnitude (on-topic 0.96+, off-topic ~0.0002), so
#: one low floor does the whole job and "nothing came back" IS the not-relevant
#: signal. Do not raise it toward the cosine band (~0.5–0.65): different scale.
MIN_SCORE = 0.05

PROMPT_LIMIT = 6
TOPICAL_LIMIT = 5
#: Sanity bound on the pinned set, not a selector. If the tagged set outgrows it
#: the footer says so (more rows than this come back as exactly this many).
PIN_LIMIT = 60

PIN_HEADING = "## Always applies (hafiz — pinned)"
PROMPT_HEADING = "## Relevant prior decisions (hafiz)"

_PIN_LIST_CMD = (
    f'hafiz query "{PIN_TAG}" --observations --tags {PIN_TAG} --no-rerank '
    f"--limit {PIN_LIMIT} --json"
)
_PROMPT_LIST_CMD = (
    'hafiz query "<your task>" --observations --min-score 0.05 --limit 20 --format md'
)

# Blocks an editor/harness prepends to a prompt; they are not the user's topic.
_HARNESS_BLOCK = re.compile(
    r"<(ide_opened_file|ide_selection|ide_diagnostics|system-reminder)>.*?</\1>", re.DOTALL
)


# ── Prompt filtering ──────────────────────────────────────────────────────


def clean_prompt(text: str | None) -> str:
    """The user's own words: editor preambles stripped, whitespace trimmed."""
    return _HARNESS_BLOCK.sub("", text or "").strip()


def should_skip_prompt(text: str) -> bool:
    """True for prompts that have no topic worth recalling for.

    Short acknowledgements, harness messages (a sub-agent's ``<task-notification>``
    report is not a question) and autonomous ``/loop`` ticks — the last fire
    repeatedly with the same text, so recalling for them is pure noise.
    """
    if len(text) < MIN_PROMPT_CHARS:
        return True
    if text.startswith("<task-notification"):
        return True
    return text.startswith("/loop") or "<<autonomous-loop" in text


# ── Rendering ─────────────────────────────────────────────────────────────


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def render_pin(row: dict) -> str | None:
    """A pinned rule, whole, as one bullet. ``None`` for an empty row."""
    content = _one_line(row.get("content") or "")
    if not content:
        return None
    return f"- **[{row.get('kind') or 'note'} | {row.get('source') or '?'}]** {content}"


def _age(value) -> str:
    from hafiz.core.durations import age_label

    try:
        ts = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        return age_label(ts)[0]
    except (TypeError, ValueError):
        return "?"


def render_topical(row: dict) -> str | None:
    """A recalled row as a bullet, excerpted if the daemon/core trimmed it.

    An excerpt is labelled in the metadata line as well as elided with ``…`` —
    a decision read without its scope clause must never pass for the whole.
    """
    content = row.get("content") or ""
    if not content.strip():
        return None
    snippet = row.get("snippet")
    meta = [row.get("source") or "unknown source", _age(row.get("valid_from"))]
    if snippet:
        meta.append(f"excerpt of {len(content)} chars")
    return f"- **[{row.get('kind') or 'note'}]** {snippet or content}\n  _({' · '.join(meta)})_"


def assemble(
    sections: list[tuple[str, list[str]]],
    *,
    max_chars: int,
    footer,
) -> str:
    """Join ``(heading, bullets)`` sections under a hard character cap.

    Sections fill in order, so earlier ones (pins) have first claim on the
    budget. Within a section a bullet that does not fit is skipped and later,
    shorter ones may still fit — more rules delivered beats strict order, and
    the order inside a pin set carries no meaning. ``footer(dropped)`` is called
    only when something was dropped; its line is reserved out of the budget so
    the total never exceeds ``max_chars``.
    """

    def build(budget: int) -> tuple[str, list[int]]:
        out, used, dropped = [], 0, []
        for heading, bullets in sections:
            head = f"{heading}\n\n"
            # A blank line separates sections, so the heading costs one more.
            head_cost = len(head) + (1 if out else 0)
            shown, cost = [], 0
            for b in bullets:
                if used + head_cost + cost + len(b) + 1 <= budget:
                    shown.append(b)
                    cost += len(b) + 1
            dropped.append(len(bullets) - len(shown))
            if shown:
                out.append(("\n" if out else "") + head + "\n".join(shown) + "\n")
                used += head_cost + cost
        return "".join(out), dropped

    text, dropped = build(max_chars)
    if not any(dropped):
        return text
    reserve = len(footer(dropped)) + 2
    text, dropped = build(max_chars - reserve)
    line = footer(dropped)
    # The reserve was computed from a worst-case count, so the line only shrinks.
    return (text + ("\n" if text else "") + line + "\n")[:max_chars]


# ── Retrieval ─────────────────────────────────────────────────────────────


def _settings():
    """The ``[hooks]`` budgets. A config problem degrades to defaults, never a failure."""
    from hafiz.core.config import HooksSettings

    try:
        from hafiz.core.config import load_settings

        return load_settings().hooks
    except Exception:  # noqa: BLE001 — a hook may not fail on a bad config
        return HooksSettings()


async def _recall(
    query: str,
    *,
    limit: int,
    tags: list[str] | None = None,
    rerank: bool | None = None,
    min_score: float | None = None,
    snippet: bool = False,
) -> list[dict]:
    """One annotation recall, daemon-first so the embedding model stays warm.

    ``snippet`` asks for the ``--format md`` style excerpt on long records;
    pins leave it off and are rendered whole.
    """
    from hafiz.core.config import load_settings
    from hafiz.core.daemon_client import query_recall

    kwargs: dict = {"limit": limit}
    if tags:
        kwargs["tags"] = tags
    if rerank is not None:
        kwargs["rerank"] = rerank
    if min_score is not None:
        kwargs["min_score"] = min_score
    budget = load_settings().annotations.snippet_chars if snippet else 0
    rows = await query_recall(query, snippet_budget=budget, **kwargs)
    return [dataclasses.asdict(r) for r in rows]


async def _session_projects(cwd: str | None) -> frozenset[str]:
    """Projects whose pins apply to a session opened in ``cwd``.

    The scope is the **workspace**, not the single repo: the session's own
    project (the indexed root containing ``cwd``, as the importers resolve it)
    plus its sibling repos (the repo root's siblings matched to indexed
    projects, as ``--workspace`` does). A rule pinned to one repo of a
    multi-repo workspace — "run tests locally before deploy" — must reach a
    session opened in any of its siblings, and must not leak into an unrelated
    workspace.

    Each half is best-effort and an empty set is the honest "don't know":
    only unscoped pins then show, rather than guessing a project.
    """
    if not cwd:
        return frozenset()
    projects: set[str] = set()
    try:
        from hafiz.core.store import project_for_path

        own = await project_for_path(cwd)
        if own:
            projects.add(own)
    except Exception:  # noqa: BLE001
        pass
    try:
        from hafiz.core.context import resolve_workspace_projects
        from hafiz.core.git_context import _git

        path = Path(cwd)
        # Siblings are siblings of the *repo*, not of whichever subdirectory
        # the session happened to open in.
        root = await asyncio.to_thread(_git, ["rev-parse", "--show-toplevel"], path)
        projects.update(await resolve_workspace_projects(Path(root) if root else path))
    except Exception:  # noqa: BLE001
        pass
    return frozenset(projects)


def pin_visible(row: dict, projects: frozenset[str]) -> bool:
    """Unscoped pins always apply; a scoped pin only inside its own workspace."""
    scope = row.get("project")
    return not scope or scope in projects


def repo_context(cwd: str | None) -> str:
    """Query text for where a session opened: repo name + branch subject.

    Falls back to the directory name outside a git repo and to ``""`` when there
    is no usable cwd — a blank query recalls nothing rather than embedding noise.
    """
    from hafiz.core.git_context import _git

    if not cwd:
        return ""
    path = Path(cwd)
    if not path.is_dir():
        return ""
    root = _git(["rev-parse", "--show-toplevel"], path)
    name = Path(root).name if root else path.name
    branch = _git(["branch", "--show-current"], path)
    # A branch name is topical on purpose: feat/iab-tcf-2.3 should pull the rule
    # about never merging it. Drop the type prefix, keep the subject.
    subject = branch.split("/", 1)[-1].replace("-", " ") if branch else ""
    return " ".join(p for p in (name, subject) if p)


# ── Modes ─────────────────────────────────────────────────────────────────


def _payload(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


async def run_prompt(payload: dict, *, max_chars: int) -> str:
    text = clean_prompt(payload.get("prompt"))
    if should_skip_prompt(text):
        return ""
    rows = await _recall(
        text[:MAX_QUERY_CHARS], limit=PROMPT_LIMIT, min_score=MIN_SCORE, snippet=True
    )
    bullets = [b for b in (render_topical(r) for r in rows) if b]

    def footer(dropped: list[int]) -> str:
        return (
            f"_hafiz: {sum(dropped)} more recalled row(s) not shown (hook output capped at "
            f"{max_chars} chars). List them: {_PROMPT_LIST_CMD}_"
        )

    return assemble([(PROMPT_HEADING, bullets)], max_chars=max_chars, footer=footer)


async def run_session(payload: dict, *, max_chars: int) -> str:
    cwd = payload.get("cwd") if isinstance(payload.get("cwd"), str) else None
    if not cwd:
        try:
            cwd = os.getcwd()
        except OSError:  # the directory was deleted under us
            cwd = None

    projects = await _session_projects(cwd)
    # Exact set: no floor, no reranking, rendered whole, narrowed to this
    # session's workspace in Python (a handful of rows, so no SQL needed).
    pin_rows = await _recall(PIN_TAG, limit=PIN_LIMIT, tags=[PIN_TAG], rerank=False)
    pin_rows = [r for r in pin_rows if pin_visible(r, projects)]
    pins = [(r.get("id"), render_pin(r)) for r in pin_rows]
    pins = [(i, b) for i, b in pins if b]

    topic = await asyncio.to_thread(repo_context, cwd)
    topical: list[tuple[str | None, str]] = []
    if topic:
        rows = await _recall(topic, limit=TOPICAL_LIMIT, min_score=MIN_SCORE, snippet=True)
        shown_ids = {i for i, _ in pins}
        # A pinned rule can also be topical; show it once, under the pin.
        topical = [
            (r.get("id"), b)
            for r in rows
            if r.get("id") not in shown_ids and (b := render_topical(r))
        ]

    label = f"## Relevant to {topic} (hafiz)" if topic else "## Relevant here (hafiz)"
    sections = [(PIN_HEADING, [b for _, b in pins]), (label, [b for _, b in topical])]
    topic_cmd = (
        f'hafiz query "{topic}" --observations --min-score 0.05 --limit {TOPICAL_LIMIT} --format md'
    )

    def footer(dropped: list[int]) -> str:
        pins_dropped, topical_dropped = dropped
        cmd = _PIN_LIST_CMD if pins_dropped else topic_cmd
        what = "pinned rule(s)" if pins_dropped and not topical_dropped else "rule(s)"
        return (
            f"_hafiz: {pins_dropped + topical_dropped} {what} not shown (hook output capped at "
            f"{max_chars} chars). List them: {cmd}_"
        )

    return assemble(sections, max_chars=max_chars, footer=footer)


async def _run(mode: str, raw_stdin: str, max_chars: int) -> str:
    from hafiz.core.database import close_engine

    payload = _payload(raw_stdin)
    try:
        if mode == "session":
            return await run_session(payload, max_chars=max_chars)
        return await run_prompt(payload, max_chars=max_chars)
    finally:
        try:
            await close_engine()
        except Exception:  # noqa: BLE001
            pass


def run_hook(
    mode: str,
    raw_stdin: str,
    *,
    max_chars: int | None = None,
    timeout: float | None = None,
) -> str:
    """Run one hook call and return the text to print — ``""`` on ANY failure.

    ``max_chars`` / ``timeout`` default to ``[hooks]`` in the config. Everything,
    including a timeout, an unreachable database and a malformed payload, is
    swallowed: a memory layer that can break the conversation gets removed.
    """
    try:
        cfg = _settings()
        cap = max_chars if max_chars is not None else cfg.max_chars
        limit = timeout if timeout is not None else cfg.timeout_seconds
        return asyncio.run(asyncio.wait_for(_run(mode, raw_stdin, cap), timeout=limit)) or ""
    except Exception:  # noqa: BLE001 — never fail a turn
        return ""
