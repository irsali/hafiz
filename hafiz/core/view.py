"""hafiz view — projections of the wisdom layer for visualization.

Phase 0 of the ``hafiz view`` work (see
``workitems/active/hafiz-view-visualization.md``): pure, dependency-free
functions that turn journal entries into Mermaid diagram text. No Typer, no
Rich, no DB — these take already-fetched :class:`~hafiz.core.journal.JournalEntry`
objects and return a string, so the command layer (Phase 0) and a future
local viewer (Phase 2) share one source of truth.

Two diagrams:

* :func:`supersession_to_mermaid` — a ``graph LR`` of supersession chains
  (decision A → superseded by B → C). The "how our thinking evolved" view.
* :func:`timeline_to_mermaid` — a Mermaid ``timeline`` grouped by month.
  The "what happened when" overview.

Mermaid is whitespace- and punctuation-sensitive. The traps these functions
guard against (all covered by tests):

* **Node IDs** must be syntactically safe — we mint ``n0``, ``n1``… from a
  counter, never from content or UUIDs.
* **Node text** is wrapped in double quotes and every Mermaid-significant
  character is escaped or stripped, so quotes / brackets / newlines in an
  annotation body can't break the diagram.
* **Length** — bodies are truncated for legibility; the full text always
  lives in ``hafiz journal --json`` / the CLI.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hafiz.core.journal import JournalEntry

# Max characters of annotation body shown inside a node before truncation.
_NODE_TEXT_MAX = 60


def _escape_mermaid_text(text: str) -> str:
    """Make arbitrary annotation content safe inside a quoted Mermaid node.

    Mermaid node labels are written as ``id["..."]``. Inside the quotes, a
    literal ``"`` ends the label and ``[](){}`` and newlines can confuse the
    parser. We collapse whitespace to single spaces, swap double quotes for
    typographic ones (Mermaid renders them fine and they can't terminate the
    label), and replace bracket characters with lookalikes that render but
    don't nest. ``#`` is HTML-entity-encoded since Mermaid treats ``#NN;`` as
    an entity escape.
    """
    # Collapse all whitespace (incl. newlines/tabs) to single spaces.
    collapsed = " ".join(text.split())
    out = []
    for ch in collapsed:
        if ch == '"':
            out.append("”")  # ”
        elif ch == "#":
            out.append("＃")  # fullwidth number sign — avoids entity parse
        elif ch in "[]":
            out.append("〔" if ch == "[" else "〕")  # 〔 〕
        elif ch in "{}":
            out.append("｛" if ch == "{" else "｝")  # ｛ ｝
        elif ch in "()":
            out.append("（" if ch == "(" else "）")  # （ ）
        else:
            out.append(ch)
    return "".join(out)


def _node_label(entry: JournalEntry) -> str:
    """Build the truncated, escaped label shown inside a node."""
    body = entry.content
    if len(body) > _NODE_TEXT_MAX:
        body = body[: _NODE_TEXT_MAX - 1] + "…"
    return f"{entry.kind}: {_escape_mermaid_text(body)}"


def supersession_to_mermaid(entries: Sequence[JournalEntry]) -> str:
    """Render supersession chains among ``entries`` as a Mermaid ``graph LR``.

    An edge ``old -->|superseded by| new`` is drawn for every entry whose
    ``supersedes_id`` points at another entry **in the same set**. Entries
    that neither supersede nor are superseded are emitted as standalone nodes
    so the diagram still shows the full window, not just the chains.

    Dangling links (``supersedes_id`` referencing a row outside the window)
    are surfaced as a node labelled ``(outside window)`` so the chain reads
    correctly without inventing content we don't have.
    """
    by_id = {e.id: e for e in entries}

    # Mint stable, syntactically-safe node ids from a counter.
    node_id: dict[str, str] = {}

    def _id_for(annotation_id: str) -> str:
        if annotation_id not in node_id:
            node_id[annotation_id] = f"n{len(node_id)}"
        return node_id[annotation_id]

    lines: list[str] = ["graph LR"]
    declared: set[str] = set()

    def _declare(annotation_id: str, label: str) -> str:
        nid = _id_for(annotation_id)
        if nid not in declared:
            lines.append(f'    {nid}["{label}"]')
            declared.add(nid)
        return nid

    edges: list[tuple[str, str]] = []
    superseded_ids: set[str] = set()
    has_chain: set[str] = set()

    for e in entries:
        old_id = e.supersedes_id
        if not old_id:
            continue
        new_nid = _declare(e.id, _node_label(e))
        if old_id in by_id:
            old_nid = _declare(old_id, _node_label(by_id[old_id]))
        else:
            # Chain reaches outside the journal window — keep the edge honest.
            old_nid = _declare(f"__ghost__{old_id}", "(outside window)")
        edges.append((old_nid, new_nid))
        superseded_ids.add(old_id)
        has_chain.add(e.id)
        has_chain.add(old_id)

    # Standalone nodes: in-window entries not part of any chain.
    for e in entries:
        if e.id not in has_chain:
            _declare(e.id, _node_label(e))

    for old_nid, new_nid in edges:
        lines.append(f"    {old_nid} -->|superseded by| {new_nid}")

    # Dim the superseded (now-inactive) nodes so the live decision stands out.
    for ann_id in superseded_ids:
        if ann_id in node_id:
            lines.append(f"    class {node_id[ann_id]} superseded;")
    if superseded_ids:
        lines.append("    classDef superseded stroke-dasharray: 4 3,opacity:0.6;")

    return "\n".join(lines)


def _month_key(entry: JournalEntry) -> str:
    return entry.valid_from.strftime("%Y-%m")


def timeline_to_mermaid(entries: Sequence[JournalEntry]) -> str:
    """Render ``entries`` as a Mermaid ``timeline`` grouped by month.

    Mermaid's ``timeline`` syntax is ``<period> : <event> : <event>``. We
    group by ``YYYY-MM`` (oldest first, since a timeline reads forward) and
    list each entry's escaped, truncated label as an event under its month.
    """
    buckets: dict[str, list[JournalEntry]] = defaultdict(list)
    for e in entries:
        buckets[_month_key(e)].append(e)

    lines: list[str] = ["timeline", "    title Decision & learning journal"]
    for month in sorted(buckets):
        month_entries = sorted(buckets[month], key=lambda e: e.valid_from)
        # First event sits on the same line as the period; the rest indent.
        first = True
        for e in month_entries:
            label = _node_label(e)
            if first:
                lines.append(f"    {month} : {label}")
                first = False
            else:
                lines.append(f"          : {label}")
    return "\n".join(lines)


def to_mermaid(entries: Sequence[JournalEntry], *, kind: str = "supersession") -> str:
    """Dispatch to the requested Mermaid diagram.

    ``kind`` is ``"supersession"`` (default) or ``"timeline"``. Raises
    :class:`ValueError` on an unknown kind so the CLI can report it cleanly.
    """
    if kind == "supersession":
        return supersession_to_mermaid(entries)
    if kind == "timeline":
        return timeline_to_mermaid(entries)
    raise ValueError(f"unknown mermaid kind {kind!r}; expected 'supersession' or 'timeline'")
