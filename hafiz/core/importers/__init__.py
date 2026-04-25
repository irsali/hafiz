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
