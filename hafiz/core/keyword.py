"""Pick the search terms out of a prompt for the keyword channel.

The keyword channel (see ``search_annotations``) finds rows an embedding
misses: a port, an error code, a file name, a flag. What it needs from a
prompt is the handful of words worth looking up literally — not the question
wrapped around them. This module is that step, kept free of any database so it
can be tested on its own.

A **term** is a tuple of alphanumeric words that must appear adjacent and in
order: ``agents.py`` becomes ``("agents", "py")``, ``--min-score`` becomes
``("min", "score")``, ``5432`` becomes ``("5432",)``. Splitting on every
non-alphanumeric is what both full-text engines do to the *stored* text (see
``hafiz.core.dialect``), so a term built the same way matches however the
original was punctuated. Terms are OR'd by the caller.
"""

from __future__ import annotations

import re

#: A run of letters or digits — ``\w`` minus the underscore.
_WORD = re.compile(r"[^\W_]+")

#: Words that carry no lookup value. Without this an OR over the whole prompt
#: matches nearly every row on "the" and "is", and ranking among them is noise.
#: Deliberately plain English and short: anything distinctive must survive.
_STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been
    before being below between both but by can could did do does doing don down
    during each few for from further had has have having he her here hers him
    his how i if in into is it its just like me more most my no nor not now of
    off on once only or other our out over own same she should so some such t
    s than that the their them then there these they this those through to too
    under until up us very was we were what when where which while who whom why
    will with would you your yours ve ll re d m
    """.split()
)

#: Longest word kept. Full-text engines refuse oversized tokens, and a 200-char
#: run of alphanumerics is a pasted hash or blob, not something to look up.
_MAX_WORD = 64

#: Most terms searched. A pasted page would otherwise build a query with
#: hundreds of branches; the earliest terms are the ones the user led with.
MAX_TERMS = 24


def _keep(words: tuple[str, ...]) -> bool:
    if not words or any(len(w) > _MAX_WORD for w in words):
        return False
    if all(w in _STOPWORDS for w in words):
        return False
    # A lone single letter ("a", "x") matches too much to be a lookup; inside a
    # multi-word term it is fine ("err_x").
    return not (len(words) == 1 and len(words[0]) < 2)


def match_terms(query: str) -> list[tuple[str, ...]]:
    """Terms worth a literal lookup in ``query``, in order of appearance, deduped.

    Whitespace-separated chunks are the unit, so punctuation *inside* a chunk
    (``agents.py``, ``--min-score``, ``ERR_X``) is kept together as one phrase
    while surrounding sentence punctuation falls away. A path also yields its
    last segment, since a note says "agents.py" far more often than the full
    path a question might quote.
    """
    terms: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    def add(words: tuple[str, ...]) -> None:
        if words not in seen and _keep(words):
            seen.add(words)
            terms.append(words)

    for chunk in query.lower().split():
        add(tuple(_WORD.findall(chunk)))
        if "/" in chunk or "\\" in chunk:
            tail = re.split(r"[/\\]", chunk.rstrip("/\\"))[-1]
            add(tuple(_WORD.findall(tail)))
        if len(terms) >= MAX_TERMS:
            break
    return terms[:MAX_TERMS]


def _term_pattern(term: tuple[str, ...]) -> re.Pattern[str]:
    """Where ``term`` sits in running text, however it was punctuated.

    Words joined by any run of non-alphanumerics, bounded by non-alphanumerics —
    the same notion of a word the full-text index uses.
    """
    body = r"[\W_]+".join(re.escape(w) for w in term)
    return re.compile(rf"(?<![^\W_]){body}(?![^\W_])", re.IGNORECASE)


def focus_text(content: str, terms: list[tuple[str, ...]], budget: int) -> str:
    """What to show a reranker that only reads the first ``budget`` characters.

    Both recall channels hand the cross-encoder a row *because* of something in
    it. When that something sits past the cut, the model scores the unrelated
    opening and the row sinks — measured on exact-string questions, the rows it
    buried were almost all ones whose token started after character 400. So when
    a lookup term (``terms``: the distinctive ones the keyword channel used)
    first appears beyond the cut, the text is the row's opening plus a window
    around that term, still within ``budget``. Otherwise ``content`` unchanged,
    so a row whose evidence is already visible is scored exactly as before.
    """
    if not terms or len(content) <= budget:
        return content
    first = min(
        (m.start() for t in terms if (m := _term_pattern(t).search(content))),
        default=None,
    )
    if first is None or first < budget - 40:
        return content
    lead = budget * 2 // 5
    window = budget - lead - 3
    start = max(lead, first - window // 3)
    return f"{content[:lead]} … {content[start : start + window]}"
