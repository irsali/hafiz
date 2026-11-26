"""Pure scoring logic for the recall eval — no database, no subprocesses.

Everything here is a function of plain data, so it is unit-tested without a
store (``tests/test_eval_scoring.py``). ``eval/run.py`` does the I/O and calls
into this module.

Vocabulary
----------
* **dataset id** — the stable id in ``eval/data/*.jsonl`` (``f01``, ``d02c``).
* **real id** — the uuid hafiz assigned when the memory was loaded. The runner
  keeps the dataset-id → real-id map; scoring converts both ways.
* **returned** — the ordered list of annotation ids a recall returned.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

#: The recall hook injects at most this many memories (``--limit 6``).
K = 6

MEASURES = ("exact", "decision", "offtopic")

#: The shipped per-prompt hook ignores prompts shorter than this.
MIN_PROMPT_CHARS = 12


# ── Loading and validation ──────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
    return rows


def validate_dataset(
    memories: Sequence[dict[str, Any]], questions: Sequence[dict[str, Any]]
) -> list[str]:
    """Return a list of problems with the dataset (empty means it is sound)."""
    problems: list[str] = []
    content_by_id: dict[str, str] = {}
    for m in memories:
        mid = m.get("id")
        if not mid or mid in content_by_id:
            problems.append(f"memory id missing or duplicated: {mid!r}")
            continue
        if not m.get("content", "").strip():
            problems.append(f"memory {mid}: empty content")
        sup = m.get("supersedes")
        if sup is not None and sup not in content_by_id:
            problems.append(f"memory {mid}: supersedes {sup!r}, which is not defined before it")
        content_by_id[mid] = m.get("content", "")
    contents = list(content_by_id.values())
    if len(set(contents)) != len(contents):
        problems.append("two memories have byte-identical content (hafiz refuses duplicates)")

    seen: set[str] = set()
    for q in questions:
        qid = q.get("id")
        if not qid or qid in seen:
            problems.append(f"question id missing or duplicated: {qid!r}")
            continue
        seen.add(qid)
        if q.get("measure") not in MEASURES:
            problems.append(f"question {qid}: unknown measure {q.get('measure')!r}")
        if len(q.get("query", "")) < MIN_PROMPT_CHARS:
            problems.append(f"question {qid}: query shorter than the hook's {MIN_PROMPT_CHARS}")
        expect = q.get("expect", [])
        if q.get("measure") == "offtopic":
            if expect:
                problems.append(f"question {qid}: an off-topic prompt must expect nothing")
            continue
        if not expect:
            problems.append(f"question {qid}: no expected memory")
        for ref in list(expect) + list(q.get("forbid", [])):
            if ref not in content_by_id:
                problems.append(f"question {qid}: unknown memory id {ref!r}")
        needle = q.get("needle")
        if needle and not any(needle in content_by_id.get(e, "") for e in expect):
            problems.append(f"question {qid}: needle {needle!r} is not in any expected memory")
    return problems


# ── Per-question scoring ────────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Rough token count: characters / 4, rounded up. Good enough to compare
    runs; not a tokenizer. An empty injection is exactly 0."""
    return math.ceil(len(text) / 4) if text else 0


def returned_ids(payload: dict[str, Any]) -> list[str]:
    """Ordered annotation ids from ``hafiz query --observations --format json``."""
    return [r["id"] for r in payload.get("results", [])]


def score_question(
    question: dict[str, Any],
    returned: Sequence[str],
    injected: str,
    id_map: dict[str, str],
) -> dict[str, Any]:
    """Score one recall.

    ``returned`` are real ids, in rank order; ``injected`` is the markdown the
    hook would have put in the prompt; ``id_map`` maps dataset id → real id.
    """
    top = list(returned)[:K]
    expect = {id_map[e] for e in question.get("expect", []) if e in id_map}
    forbid = {id_map[f] for f in question.get("forbid", []) if f in id_map}

    rank = next((i for i, rid in enumerate(top, 1) if rid in expect), None)
    needle = question.get("needle")
    return {
        "id": question["id"],
        "measure": question["measure"],
        "rank": rank,
        "returned": top,
        "had_forbid": bool(forbid),
        "hit1": rank == 1,
        "hitk": rank is not None,
        "leaked": sorted(rid for rid in top if rid in forbid),
        "needle_surfaced": (needle in injected) if needle else None,
        "n_returned": len(top),
        "tokens": estimate_tokens(injected),
        "silent": not top and not injected.strip(),
    }


# ── Aggregation ─────────────────────────────────────────────────────────


def _rate(flags: Iterable[bool]) -> float | None:
    flags = list(flags)
    return sum(flags) / len(flags) if flags else None


def _mrr(rows: Sequence[dict[str, Any]]) -> float | None:
    return sum(1 / r["rank"] if r["rank"] else 0.0 for r in rows) / len(rows) if rows else None


def percentile(values: Sequence[int], pct: float) -> int:
    """Nearest-rank percentile; deterministic, no interpolation."""
    if not values:
        return 0
    ordered = sorted(values)
    idx = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[idx]


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collapse per-question rows into the three measures."""
    by = {m: [r for r in rows if r["measure"] == m] for m in MEASURES}

    exact = by["exact"]
    decision = by["decision"]
    chained = [r for r in decision if r.get("had_forbid")]
    off = by["offtopic"]
    on_topic = exact + decision
    on_tokens = [r["tokens"] for r in on_topic]

    return {
        "exact": {
            "n": len(exact),
            "hit_at_1": _rate(r["hit1"] for r in exact),
            "hit_at_6": _rate(r["hitk"] for r in exact),
            "mrr": _mrr(exact),
            "needle_surfaced": _rate(
                r["needle_surfaced"] for r in exact if r["needle_surfaced"] is not None
            ),
        },
        "decision": {
            "n": len(decision),
            "hit_at_1": _rate(r["hit1"] for r in decision),
            "hit_at_6": _rate(r["hitk"] for r in decision),
            "mrr": _mrr(decision),
            "n_superseded_chain": len(chained),
            "stale_leak_rate": _rate(bool(r["leaked"]) for r in chained),
        },
        "tokens": {
            "on_topic_n": len(on_topic),
            "on_topic_mean": statistics.fmean(on_tokens) if on_tokens else None,
            "on_topic_median": statistics.median(on_tokens) if on_tokens else None,
            "on_topic_p95": percentile(on_tokens, 95),
            "on_topic_max": max(on_tokens, default=0),
            "off_topic_n": len(off),
            "off_topic_silent_rate": _rate(r["silent"] for r in off),
            "off_topic_mean": statistics.fmean(r["tokens"] for r in off) if off else None,
        },
    }


# ── Rendering ───────────────────────────────────────────────────────────


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _num(v: float | None, digits: int = 0) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def metric_rows(agg: dict[str, Any]) -> list[tuple[str, str]]:
    """(label, formatted value) pairs, in the order the docs table shows them."""
    e, d, t = agg["exact"], agg["decision"], agg["tokens"]
    return [
        (f"Exact-string recall hit@1 (n={e['n']})", _pct(e["hit_at_1"])),
        ("Exact-string recall hit@6", _pct(e["hit_at_6"])),
        ("Exact string visible in injected text", _pct(e["needle_surfaced"])),
        (f"Decision recall hit@1 (n={d['n']})", _pct(d["hit_at_1"])),
        ("Decision recall hit@6", _pct(d["hit_at_6"])),
        (f"Stale-belief leak rate (n={d['n_superseded_chain']})", _pct(d["stale_leak_rate"])),
        (
            f"Tokens injected, on-topic prompts, mean (n={t['on_topic_n']})",
            _num(t["on_topic_mean"], 0),
        ),
        ("Tokens injected, on-topic, p95", _num(t["on_topic_p95"])),
        (
            f"Off-topic prompts that inject nothing (n={t['off_topic_n']})",
            _pct(t["off_topic_silent_rate"]),
        ),
        ("Tokens injected, off-topic prompts, mean", _num(t["off_topic_mean"], 1)),
    ]


def render_summary(result: dict[str, Any]) -> str:
    meta = result["meta"]
    lines = [
        f"## Recall eval: {meta['label']} (commit {meta['commit']}"
        f"{', dirty' if meta.get('dirty') else ''})",
        "",
        "| Measure | Result |",
        "|---|---|",
    ]
    lines += [f"| {label} | {value} |" for label, value in metric_rows(result["aggregate"])]
    return "\n".join(lines) + "\n"


def render_comparison(results: Sequence[dict[str, Any]]) -> str:
    """One table, one column per run, in the order given."""
    if not results:
        return "_No runs recorded yet._\n"
    heads = [f"{r['meta']['label']} (`{r['meta']['commit']}`)" for r in results]
    labels = [label for label, _ in metric_rows(results[0]["aggregate"])]
    cols = [[v for _, v in metric_rows(r["aggregate"])] for r in results]
    lines = ["| Measure | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]
    for i, label in enumerate(labels):
        lines.append(f"| {label} | " + " | ".join(c[i] for c in cols) + " |")
    return "\n".join(lines) + "\n"


def failures(
    rows: Sequence[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    reverse_map: dict[str, str],
) -> list[str]:
    """Human-readable lines for every question the eval got wrong."""
    out: list[str] = []
    for r in rows:
        q = questions[r["id"]]
        got = [reverse_map.get(i, i) for i in r.get("returned", [])]
        got_txt = ", ".join(got) if got else "(nothing)"
        if r["measure"] == "offtopic":
            if not r["silent"]:
                out.append(
                    f"[{r['id']}] off-topic prompt injected {r['tokens']} tokens: "
                    f"{q['query']!r} -> {got_txt}"
                )
            continue
        problems = []
        if not r["hit1"]:
            where = f"rank {r['rank']}" if r["rank"] else f"not in top {K}"
            problems.append(f"expected {','.join(q['expect'])} {where}")
        if r["leaked"]:
            leaked = [reverse_map.get(i, i) for i in r["leaked"]]
            problems.append(f"STALE LEAK {','.join(leaked)}")
        if r["needle_surfaced"] is False:
            problems.append(f"needle {q['needle']!r} not in injected text")
        if problems:
            out.append(f"[{r['id']}] {q['query']!r}: " + "; ".join(problems) + f" | got: {got_txt}")
    return out
