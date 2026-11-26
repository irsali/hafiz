"""Replay observation recall with the keyword channel off and on.

A one-off measurement harness, not part of the product. It answers one
question about ``recall.keyword_channel``: does it find the rows an embedding
misses (exact strings) without disturbing ordinary prompts, and what does it
cost in latency?

It reads two prompt files **you** build from your own store and keep out of the
repository — the prompts are your memory, not test data:

* ``--exact``: JSON list of ``{"q": "<question>", "targets": ["<annotation id>", ...],
  "type": "<optional label>"}``. A question is a *hit* when any target is in the
  top ``--limit`` results. Build it by picking annotations that contain a
  distinctive token (path, port, flag, error name, ticket id) and writing a short
  question that uses the token plus ordinary words.
* ``--ordinary``: JSON list of prompt strings (the ``retrievals`` table has real
  past ones). Measured for *stability*: top-``limit`` overlap between off and on.

**Run it against a scratch copy of the store, never the live one.** It only
reads, but the live store is not the place to experiment, and the migration that
adds the index must have been applied to the copy::

    createdb hafiz_replay && pg_dump "$LIVE_URL" | psql hafiz_replay
    HAFIZ_DATABASE__URL=<scratch async url> HAFIZ_NO_DAEMON=1 \\
    HAFIZ_TELEMETRY__RETRIEVAL=false hafiz init   # applies migrations
    HAFIZ_DATABASE__URL=<scratch async url> HAFIZ_NO_DAEMON=1 \\
    HAFIZ_TELEMETRY__RETRIEVAL=false python scripts/replay_keyword_recall.py \\
        --exact exact.json --ordinary ordinary.json

The script refuses a database whose name does not contain ``replay``,
``scratch`` or ``test``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from urllib.parse import urlparse

FLOOR = 0.05


def _pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


async def _search(query: str, keyword: bool, limit: int):
    import hafiz.core.config as config
    from hafiz.core.annotations import search_annotations

    real = config.load_settings

    def patched():
        settings = real()
        settings.recall.keyword_channel = keyword
        return settings

    config.load_settings = patched
    try:
        started = time.perf_counter()
        results = await search_annotations(query, limit=limit, telemetry_command=None)
        return results, time.perf_counter() - started
    finally:
        config.load_settings = real


async def _replay(prompts: list[str], limit: int) -> dict[str, list]:
    """Search every prompt in both modes, alternating which goes first."""
    await _search("warm up the models", False, limit)
    await _search("warm up the models", True, limit)
    out: dict[str, list] = {"off": [], "on": []}
    for n, prompt in enumerate(prompts):
        order = (False, True) if n % 2 == 0 else (True, False)
        for keyword in order:
            results, seconds = await _search(prompt, keyword, limit)
            out["on" if keyword else "off"].append(
                {
                    "ids": [r.id for r in results],
                    "scores": [r.rerank_score for r in results],
                    "seconds": seconds,
                }
            )
    return out


def _hit(run: dict, targets: set[str], floor: float | None) -> int | None:
    """1-based rank of the first target in the result, or None."""
    for rank, (rid, score) in enumerate(zip(run["ids"], run["scores"], strict=True), start=1):
        if floor is not None and (score is None or score < floor):
            continue
        if rid in targets:
            return rank
    return None


def _latency(runs: list[dict]) -> str:
    s = [r["seconds"] for r in runs]
    return f"p50 {statistics.median(s) * 1000:.0f} ms   p95 {_pct(s, 95) * 1000:.0f} ms"


async def _report_exact(items: list[dict], limit: int) -> None:
    out = await _replay([i["q"] for i in items], limit)
    print(f"\n== (a) exact-string prompts: {len(items)}  (hit@{limit})")
    for floor in (None, FLOOR):
        label = "no floor " if floor is None else f"floor {floor}"
        for mode in ("off", "on"):
            hits = [
                _hit(run, set(item["targets"]), floor)
                for run, item in zip(out[mode], items, strict=True)
            ]
            n = sum(h is not None for h in hits)
            print(f"   {label}  keyword {mode:3}: {n:3}/{len(items)} = {n / len(items):.0%}")
    types = sorted({i.get("type", "all") for i in items})
    if len(types) > 1:
        print("   by type (no floor, off -> on):")
        for t in types:
            idx = [n for n, i in enumerate(items) if i.get("type", "all") == t]
            row = []
            for mode in ("off", "on"):
                row.append(
                    sum(_hit(out[mode][n], set(items[n]["targets"]), None) is not None for n in idx)
                )
            print(f"     {t:8} {row[0]:2} -> {row[1]:2}  of {len(idx)}")
    print(f"   latency off: {_latency(out['off'])}")
    print(f"   latency on : {_latency(out['on'])}")


async def _report_ordinary(prompts: list[str], limit: int) -> None:
    out = await _replay(prompts, limit)
    overlaps, same_set, same_top, entered, entered_above = [], 0, 0, 0, 0
    kept, lost_any, gained, shown_off, shown_on, top_changed = [], 0, 0, 0, 0, 0
    for off, on in zip(out["off"], out["on"], strict=True):
        a, b = set(off["ids"]), set(on["ids"])
        overlaps.append(len(a & b) / max(len(a | b), 1) if (a or b) else 1.0)
        same_set += a == b
        same_top += off["ids"][:1] == on["ids"][:1]
        new = [s for rid, s in zip(on["ids"], on["scores"], strict=True) if rid not in a]
        entered += len(new)
        entered_above += sum(s is not None and s >= FLOOR for s in new)
        # What a recall hook actually shows: rows at or above the floor.
        fa = [r for r, s in zip(off["ids"], off["scores"], strict=True) if s and s >= FLOOR]
        fb = [r for r, s in zip(on["ids"], on["scores"], strict=True) if s and s >= FLOOR]
        shown_off += len(fa)
        shown_on += len(fb)
        gained += len(set(fb) - set(fa))
        lost_any += bool(set(fa) - set(fb))
        top_changed += fa[:1] != fb[:1]
        if fa:
            kept.append(len(set(fa) & set(fb)) / len(fa))
    print(f"\n== (b) ordinary prompts: {len(prompts)}  (top-{limit}, off vs on)")
    print(f"   mean Jaccard overlap   : {statistics.mean(overlaps):.3f}")
    print(f"   identical result sets  : {same_set}/{len(prompts)}")
    print(f"   same top-1 row         : {same_top}/{len(prompts)}")
    print(
        f"   rows newly in top-{limit}   : {entered}  (of which rerank >= {FLOOR}: {entered_above})"
    )
    print(f"   --- as a recall hook sees it (rows with rerank >= {FLOOR}) ---")
    print(f"   rows shown off -> on   : {shown_off} -> {shown_on}  (gained {gained})")
    print(f"   off-rows still shown   : {statistics.mean(kept):.1%} on average")
    print(f"   prompts that lost a row: {lost_any}/{len(prompts)}")
    print(f"   prompts whose top row changed: {top_changed}/{len(prompts)}")
    print(f"   latency off: {_latency(out['off'])}")
    print(f"   latency on : {_latency(out['on'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--exact", help="JSON file of exact-string prompts (see module doc)")
    parser.add_argument("--ordinary", help="JSON file of ordinary prompt strings")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    from hafiz.core.config import load_settings

    name = urlparse(load_settings().database.url).path.rsplit("/", 1)[-1]
    if not any(word in name for word in ("replay", "scratch", "test")):
        raise SystemExit(f"refusing to run against {name!r}: point HAFIZ_DATABASE__URL at a copy")

    async def run() -> None:
        # One event loop for both sets: the async engine is bound to the loop
        # that first used it.
        if args.exact:
            await _report_exact(json.load(open(args.exact)), args.limit)
        if args.ordinary:
            await _report_ordinary(json.load(open(args.ordinary)), args.limit)

    asyncio.run(run())


if __name__ == "__main__":
    main()
