"""Turn a raw prompt into something a cross-encoder can match.

Recall hooks search with the user's prompt verbatim. A 2026-11 audit of the
retrieval log found that works for short, typed prompts and fails for the rest:
prompts over 600 characters came back empty 94% of the time, and dictated ones
("uh, I do not recall exactly … Google Tag Manager or something") missed answers
the store held — the same request, trimmed to its topic, found them.

Two causes, two fixes, both deterministic (no LLM — see "No API-key-dependent
paths" in CLAUDE.md):

* **Noise** — dictation fillers, pasted element-picker dumps, code fences,
  stack traces and raw URLs dilute the text the reranker scores. :func:`clean`
  removes them.
* **Length** — a cross-encoder scores one (query, memory) pair, so a prompt
  that covers four topics matches none of them well. :func:`segments` splits a
  long prompt into its most specific sentences, each searched on its own; the
  caller merges by best score.

A short, clean prompt passes through unchanged — the common case must not move.
"""

from __future__ import annotations

import re

#: Prompts up to this length are searched whole (after cleaning).
WHOLE_QUERY_CHARS = 300
#: A long prompt is searched as at most this many sentences.
MAX_SEGMENTS = 4
#: Sentences shorter than this carry no topic of their own ("Go.", "Right?").
MIN_SEGMENT_CHARS = 25
#: A single over-long sentence is cut here.
MAX_SEGMENT_CHARS = 300

_FENCE = re.compile(r"```.*?(```|$)", re.S)
# Cursor's element picker pastes "DOM Path: …", "Position: top=…px, …" and
# "HTML Element: <tag …>text</tag>", then the user's own words on the same line.
# Each pattern stops where the dump ends so those words survive, and the
# element's visible text ("Manage in Billing Portal") is kept — it names the topic.
_PICKER = re.compile(
    r"DOM Path:[^\n]*?(?=Position:|\n|$)|Position:\s*\w+=[\w.-]+(?:,\s*\w+=[\w.-]+)*"
)
_ELEMENT = re.compile(r"HTML Element:\s*<([\w-]+)[^>]*>([^<\n]{0,200})(?:</\1>)?")
_STACK = re.compile(r"^\s*(at\s+\S+\s*\(.*\)|File \".*\", line \d+.*|Traceback .*)$", re.M)
_URL = re.compile(r"https?://([^\s/]+)(/[^\s]*)?")
# Spoken fillers only. Phrases that also carry meaning stay — "like",
# "actually", and "you know" / "I mean" ("you know that it is not yet
# implemented" lost its point when the replay stripped it).
_FILLER = re.compile(
    r"\b(u+h+|u+m+|uhm+|e+r+m*|hmm+|or something( like that)?|"
    r"kind of|sort of|i do not recall( is)? exactly|i don'?t remember exactly)\b,?",
    re.I,
)
_SPACE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,?!])")
# Words that ask rather than name: they say nothing about which memory is wanted.
_STOP = frozenset(
    """
    the and for are but not you your with this that have from was were will would
    could should can what when where which who why how all any some there their
    them they then than its it's into about also just like lets let's want need
    look see check make sure please now again here more most very much many
    does did done doing one two get got going think know because something
    """.split()
)
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+|\n+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{2,}")


def _url_words(m: re.Match) -> str:
    """Keep what a URL says (host and path words), drop the syntax."""
    host = m.group(1)
    path = " ".join(p for p in re.split(r"[/?#=&_-]+", m.group(2) or "") if p.isalpha())
    return f"{host} {path}".strip()


def clean(text: str) -> str:
    """Strip noise that dilutes relevance scoring; keep the words that ask."""
    t = _FENCE.sub(" ", text or "")
    t = _PICKER.sub(" ", t)
    t = _ELEMENT.sub(r" \2 ", t)
    t = _STACK.sub(" ", t)
    t = _URL.sub(_url_words, t)
    t = _FILLER.sub(" ", t)
    t = "\n".join(
        _SPACE_BEFORE_PUNCT.sub(r"\1", _SPACE.sub(" ", line)).strip(" ,") for line in t.splitlines()
    )
    return re.sub(r"\n{2,}", "\n", t).strip()


def _specificity(sentence: str) -> int:
    """Rough topic density: distinct content words, identifiers count double."""
    words = {w.lower() for w in _WORD.findall(sentence)} - _STOP
    return len(words) + sum(1 for w in words if re.search(r"[0-9_.-]", w))


def segments(text: str) -> list[str]:
    """What to search for ``text``: ``[cleaned]``, or that plus its key sentences.

    A long prompt yields the cleaned whole first, then up to
    :data:`MAX_SEGMENTS` of its most specific sentences in original order. The
    whole stays in because some prompts only make sense together — a replay
    lost "send only to the 13 customers with traffic" when split — so merging
    by best score can add hits but never lose what the whole prompt found.

    Returns ``[]`` only when nothing searchable is left after cleaning.
    """
    cleaned = clean(text)
    if not cleaned:
        return []
    if len(cleaned) <= WHOLE_QUERY_CHARS:
        return [cleaned]
    # dict.fromkeys: pasted text repeats itself, and a repeated sentence would
    # spend several of the few search slots on one topic.
    sentences = list(
        dict.fromkeys(
            s.strip()[:MAX_SEGMENT_CHARS]
            for s in _SENTENCE_END.split(cleaned)
            if len(s.strip()) >= MIN_SEGMENT_CHARS
        )
    )
    ranked = sorted(range(len(sentences)), key=lambda i: -_specificity(sentences[i]))
    keep = sorted(ranked[:MAX_SEGMENTS])
    return [cleaned, *(sentences[i] for i in keep if sentences[i] != cleaned)]
