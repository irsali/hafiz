"""Read-time snippet extraction for annotation recall.

Annotations come back whole. On a measured store the median live annotation is
951 characters and the p90 is 1,589, so one ``--observations --limit 20
--format md`` recall spends ~4,200 tokens of an agent's context to deliver, in
most cases, two relevant sentences. Nothing shortens them: ``kind`` is the only
schema, so a decision, its rationale, its scope and its rejected alternatives
sit in one prose blob and are injected as one prose blob.

This trims what gets *injected*, and only that:

* Only ``compact`` and ``md`` — the two formats that exist to be piped into a
  prompt. ``json`` and ``rich`` keep the full record, so the whole text is
  always one flag away.
* Only records over the budget. The median record is already under it and
  passes through untouched, which also keeps the cost near zero.
* Excerpts are **marked**, both with ellipses and with an explicit flag, so a
  reader can never mistake a fragment for the whole record. A decision
  truncated past its scope qualifier ("…only for EU traffic, post-consent") is
  worse than one that was never retrieved, so the fact of truncation is not
  allowed to be silent.

Span selection is **two-stage, mirroring the search path it sits on**: a free
lexical pass narrows each record to a few finalists, then the cross-encoder —
the same model that decided the record was relevant — picks which of them is.
Cross-encoding every span was the obvious first cut and cost ~1.0 s on a
limit-20 recall (97 spans, 17.5 KB of text, versus the 5.2 KB the main rerank
sees), which is not a price a per-prompt hook can pay. Narrowing first puts it
back in the noise. If the model is unavailable the fallback is the lexical
winner, and failing that the head of the record — never an error, because this
runs on the read path and the read path may not fail.
"""

from __future__ import annotations

import re

#: Paragraph break — one or more blank lines.
_PARA = re.compile(r"\n\s*\n")
#: Sentence end followed by whitespace. Deliberately crude: the cost of a bad
#: split is a slightly ragged excerpt, not a wrong answer, and a real sentence
#: tokenizer is a dependency this does not justify.
_SENT = re.compile(r"(?<=[.!?])\s+")
#: Word characters, for the lexical pass.
_WORD = re.compile(r"[a-z0-9]+")
#: Joiner between non-adjacent spans, and the marker for a clipped edge.
_GAP = " … "

#: Spans per record handed to the cross-encoder. The lexical pass only has to
#: get the right span into a shortlist of this size, which it does far more
#: reliably than it picks a winner outright.
_FINALISTS = 3

#: Query words this common carry no signal and make every span look alike.
_STOP = frozenset(
    "a an and are as at be but by do does for from had has have how i if in into is it its "
    "not of on or should so that the their then there these they this to was we were what "
    "when where which who why will with you your".split()
)


def _lexical_scores(query: str, spans: list[str]) -> list[float]:
    """Fraction of the query's content words each span contains.

    Length-normalized only weakly (by a soft floor) — a long span genuinely is
    more likely to answer the question, and the cross-encoder settles the
    comparison anyway. This exists to shortlist, not to rank.
    """
    terms = {w for w in _WORD.findall(query.lower()) if w not in _STOP and len(w) > 2}
    if not terms:
        return [1.0] * len(spans)
    scores = []
    for span in spans:
        words = set(_WORD.findall(span.lower()))
        scores.append(len(terms & words) / len(terms))
    return scores


def split_spans(content: str, *, budget: int) -> list[str]:
    """Break a record into candidate spans, largest coherent unit first.

    Paragraphs are the natural unit — agents write "DECISION: … REJECTED: …"
    as separate blocks. A paragraph still over budget is split again into
    sentences, because a single-paragraph wall is exactly the shape this
    exists to handle.
    """
    spans: list[str] = []
    for para in _PARA.split(content):
        para = para.strip()
        if not para:
            continue
        if len(para) <= budget:
            spans.append(para)
            continue
        sentences = [s.strip() for s in _SENT.split(para) if s.strip()]
        # A "sentence" longer than the budget (no punctuation at all) is left
        # whole; the window builder clips it rather than cutting mid-word here.
        spans.extend(sentences or [para])
    return spans


def build_window(
    spans: list[str], scores: list[float], *, budget: int, best: int | None = None
) -> str:
    """Grow a contiguous window around the best span, up to ``budget``.

    Contiguous, not top-k-scattered: prose reads as a sequence, and the
    sentence after the matching one is usually the one carrying its
    qualification. Growth prefers the higher-scoring neighbour so the window
    drifts toward the relevant end of the record rather than always forward.

    ``best`` names the anchor when it was chosen by a better judge than
    ``scores`` — the cross-encoder picks the anchor, ``scores`` (lexical) only
    breaks ties about which way to grow, where being wrong costs a slightly
    off-center excerpt rather than the wrong excerpt.
    """
    if not spans:
        return ""
    if best is None:
        best = max(range(len(spans)), key=lambda i: scores[i])
    lo = hi = best
    size = len(spans[best])

    while True:
        prev_i, next_i = lo - 1, hi + 1
        prev_ok = prev_i >= 0 and size + len(spans[prev_i]) + 1 <= budget
        next_ok = next_i < len(spans) and size + len(spans[next_i]) + 1 <= budget
        if not prev_ok and not next_ok:
            break
        take_next = next_ok and (not prev_ok or scores[next_i] >= scores[prev_i])
        if take_next:
            hi = next_i
            size += len(spans[next_i]) + 1
        else:
            lo = prev_i
            size += len(spans[prev_i]) + 1

    window = " ".join(spans[lo : hi + 1])
    if len(window) > budget:  # a single span wider than the budget
        window = window[: max(0, budget - 1)].rstrip() + "…"
    if lo > 0:
        window = _GAP.lstrip() + window
    if hi < len(spans) - 1:
        window = window + _GAP.rstrip()
    return window


def _shortlist(spans: list[str], lexical: list[float]) -> list[int]:
    """Indices worth paying the cross-encoder for.

    Always includes span 0. The head of an annotation is where agents put the
    claim — it is a strong prior, it costs one slot, and it stops a record
    whose query words all appear in a trailing aside from being excerpted to
    that aside.
    """
    ranked = sorted(range(len(spans)), key=lambda i: lexical[i], reverse=True)
    return sorted({0, *ranked[:_FINALISTS]})


async def attach_snippets(query: str, results: list, *, budget: int) -> None:
    """Set ``.snippet`` on every result that needs trimming. Never raises.

    Finalists from every oversized record are scored in **one** cross-encoder
    call. Per-record calls would multiply a fixed model overhead by the result
    count, and the scores are only ever compared within a record anyway.
    """
    if budget <= 0 or not results:
        return

    oversized = [r for r in results if len(r.content or "") > budget]
    if not oversized:
        return

    per_record: list[tuple[list[str], list[float], list[int]]] = []
    finalists: list[str] = []
    for record in oversized:
        spans = split_spans(record.content, budget=budget)
        lexical = _lexical_scores(query, spans)
        picks = _shortlist(spans, lexical) if spans else []
        per_record.append((spans, lexical, picks))
        finalists.extend(spans[i] for i in picks)

    scores: list[float] | None = None
    if finalists:
        try:
            from hafiz.core.reranker import score_passages

            scores = await score_passages(query, finalists)
        except Exception:  # noqa: BLE001 — the read path may not fail
            scores = None

    cursor = 0
    for record, (spans, lexical, picks) in zip(oversized, per_record, strict=True):
        if not spans:
            continue
        if scores is None:
            # No model: the lexical winner, which for a query sharing no words
            # with the record is span 0 — the head, which is what a reader
            # would have seen first anyway. Still marked as an excerpt.
            best = max(picks, key=lambda i: lexical[i])
        else:
            window = scores[cursor : cursor + len(picks)]
            best = picks[max(range(len(picks)), key=lambda k: window[k])]
        cursor += len(picks)
        record.snippet = build_window(spans, lexical, budget=budget, best=best)
