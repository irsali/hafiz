"""Source-layer importers — post-hoc parsers for agent transcripts.

Importers read agent-harness storage formats (JSONL, SQLite, etc.) and
produce :class:`hafiz.core.communications.MessageInput` lists, which
are then stored idempotently via ``upsert_communication`` +
``append_messages``.

Each importer is a small parser-shaped module under this package:

* :mod:`hafiz.core.importers.claude_code` — Claude Code session JSONL
  (under ``~/.claude/projects/.../*.jsonl``).
* (future) cursor — Cursor history SQLite.
* (future) slack, mail, etc.

The agent harness does not change — these are post-hoc, idempotent
readers.
"""

from __future__ import annotations

# ── Capture-freshness probes ──────────────────────────────────────────
#
# "Is anything still arriving?" is asked per agent, and only an importer
# knows where its harness keeps sessions or how to tell one apart. Each
# probe answers ``(pending, active)`` given the session ids already
# stored and the cut-off below which a session counts as settled:
# uncaptured work versus work still in progress.
#
# ChatGPT has no entry on purpose — an export is a file the user
# downloads on demand, so there is no store to fall behind.
PENDING_PROBES: dict[str, str] = {
    "claude-code": "hafiz.core.importers.claude_code",
    "cursor": "hafiz.core.importers.cursor",
    "codex": "hafiz.core.importers.codex",
}


def pending_probe(agent: str):
    """Resolve an agent's ``pending_on_disk`` probe, or None.

    Imported lazily so ``status`` does not pull in sqlite3/zipfile and
    every importer module just to print a table.
    """
    module_path = PENDING_PROBES.get(agent)
    if module_path is None:
        return None
    import importlib

    try:
        return getattr(importlib.import_module(module_path), "pending_on_disk", None)
    except ImportError:
        return None
