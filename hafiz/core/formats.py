"""Output shapes for retrieval results.

Hafiz's ``--json`` is what gets injected into an agent's context window on
every session, so its size is a running cost. The default shape is ~1/3
metadata — uuids, timestamps, null fields and float scores that a consuming
model never reads — which is fine for a debugging human and wasteful for the
actual consumer.

Four formats, one vocabulary shared by ``query`` / ``query --observations`` /
``context``:

``rich``
    Human terminal output. The default when no format is requested.
``json``
    Today's full shape, field-for-field. What ``--json`` selects, so existing
    hooks and the agent skills block keep working untouched.
``compact``
    Machine-readable, stripped to content plus enough provenance to judge and
    cite a row. ~1/5 the bytes of ``json`` on a real payload.
``md``
    Raw markdown for direct injection into a prompt — no Rich rendering, no
    JSON envelope to strip.

Shaping lives here (core: no Typer/Rich) so all three commands agree on it;
the printing lives in ``hafiz/commands/``.
"""

from __future__ import annotations

from enum import StrEnum

from hafiz.core.durations import age_label


class OutputFormat(StrEnum):
    """How a retrieval command should render its results."""

    RICH = "rich"
    JSON = "json"
    COMPACT = "compact"
    MD = "md"

    @property
    def is_machine(self) -> bool:
        """True for formats a program parses (i.e. everything but ``rich``)."""
        return self is not OutputFormat.RICH


def resolve_format(fmt: OutputFormat | None, *, json_flag: bool) -> OutputFormat:
    """Reconcile ``--format`` with the older ``--json`` boolean.

    ``--json`` predates ``--format`` and is load-bearing for installed agent
    configs, so it stays as an alias for ``--format json``. An explicit
    ``--format`` wins when both are given — that is the more specific request,
    and it lets a caller pass ``--json --format compact`` while migrating.
    """
    if fmt is not None:
        return fmt
    return OutputFormat.JSON if json_flag else OutputFormat.RICH


def error_payload(message: str) -> dict:
    """The project's standard machine-readable failure shape."""
    return {"ok": False, "error": message}


# ── Annotations (wisdom layer) ──────────────────────────────────────────


def annotation_compact(a, *, with_ids: bool = False) -> dict:
    """One annotation, stripped to what a consuming model reads.

    ``id`` is omitted unless asked for — but ask for it whenever the consumer
    might write back, because an agent that can read a decision and not cite it
    cannot ``--supersedes`` it, and the corpus silently accumulates
    contradictions instead.

    ``content`` is the best-matching excerpt when one was extracted, and
    ``excerpt: true`` says so. That flag is not decoration: a reader who
    mistakes a fragment for the whole record can act on a decision without its
    scope qualifier. The full text is always in ``--format json``.
    """
    snippet = getattr(a, "snippet", None)
    row = {
        "content": snippet or a.content,
        "kind": a.kind,
        "source": a.source,
        "age": age_label(a.valid_from)[0],
    }
    if snippet:
        row["excerpt"] = True
        row["full_chars"] = len(a.content)
    if with_ids:
        row["id"] = a.id
    return row


def annotation_md(a, *, with_ids: bool = False) -> str:
    """One annotation as a markdown bullet.

    An excerpt is labelled in the metadata line as well as being elided with
    ellipses, so the marker survives a reader skimming only the prose.
    """
    snippet = getattr(a, "snippet", None)
    meta = [a.source or "unknown source", age_label(a.valid_from)[0]]
    if snippet:
        meta.append(f"excerpt of {len(a.content)} chars")
    if with_ids:
        meta.append(a.id)
    return f"- **[{a.kind}]** {snippet or a.content}\n  _({' · '.join(meta)})_"


# ── Chunks (code / doc units) ───────────────────────────────────────────


def chunk_compact(c, *, with_ids: bool = False) -> dict:
    """One search hit, stripped to content plus its location."""
    row = {
        "content": c.content,
        "kind": c.kind,
        "unit_name": c.unit_name,
        "source_file": c.source_file,
    }
    if with_ids:
        row["id"] = c.id
        row["unit_id"] = c.unit_id
    return row


def chunk_md(c, *, with_ids: bool = False) -> str:
    """One search hit as a markdown section."""
    location = f"{c.source_file}::{c.unit_name}"
    if c.line_start and c.line_end:
        location += f":{c.line_start}-{c.line_end}"
    header = f"### {location}"
    if with_ids:
        header += f"  <!-- {c.id} -->"
    return f"{header}\n```{c.language or ''}\n{c.content}\n```"
