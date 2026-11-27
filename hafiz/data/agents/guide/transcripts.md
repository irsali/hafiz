<!-- summary: Source layer: transcript import, recall, retention and forget -->
# hafiz guide transcripts — Source layer: transcript import, recall, retention and forget

## Source layer — agent transcripts (opt-in)

Hafiz can ingest agent-harness transcripts into a dedicated source
layer, so prior conversations are queryable without polluting default
search. **Defaults are conservative**: source rows hide from
``hafiz query`` and ``hafiz context`` until you opt in.

```bash
# Idempotent post-hoc import. Re-running is a no-op.
hafiz import claude-code --project hafiz

# Explicit recall of one session's turns, in order.
hafiz recall <session-slug-or-uuid> --json

# Vector search inside one session.
hafiz recall <session> --query "auth flow" --json

# Add transcript matches to a normal query (clearly tagged in output).
hafiz query "auth flow" --include-transcripts --json

# Cite specific turns when distilling — polymorphic --derived-from.
hafiz observe "<distilled decision>" --type decision \
  --derived-from <message-id>,<message-id>
```

Retention is bounded (default 90 days from started_at). `hafiz import`
sweeps expired rows automatically and reports what it tombstoned, but
that trigger stops firing once imports stop — so the backlog is surfaced
as `retention.overdue` in `hafiz status --json` and as a `doctor` check.
**If you see a non-zero overdue count, tell the user** — it's a stated
retention guarantee that isn't currently being met, not a cosmetic nit.

**Automatic capture — and why `overdue: 0` is not proof of health.**
`retention.overdue` answers "is the sweep keeping up", which a store
receiving *nothing* passes trivially. The separate `capture` block on
`hafiz status --json` answers "is anything still arriving?":

```json
"capture": {"claude-code": {"days_since": 66, "pending_on_disk": 201, "live": 0}}
```

**If `pending_on_disk` is non-zero, tell the user immediately.** Unlike
every other staleness signal in hafiz, this one is not recoverable by
re-running a command later — agent harnesses rotate their own
transcripts, so uncaptured sessions are lost permanently. The remedy:

```bash
hafiz import claude-code                      # rescue what's still on disk
hafiz agent install claude-code --hooks       # stop the drift for good
```

**Five importers, one contract.** Same flags (`--project`, `--limit`,
`--since`, `--dry-run`, `--no-embed`, `--json`) and same summary shape:

| Command | Source | Notes |
|---|---|---|
| `hafiz import claude-code [path]` | `~/.claude/projects` | a dir or one `.jsonl` |
| `hafiz import cursor [path]` | Cursor's SQLite store | opened read-only |
| `hafiz import chatgpt <path>` | a ChatGPT data export | `.zip`, dir, or `conversations.json`; path required |
| `hafiz import codex [path]` | `$CODEX_HOME/sessions` | + `archived_sessions` |
| `hafiz import hermes [path]` | `$HERMES_HOME/state.db` | opened read-only; skips sessions the Hermes plugin already captured and sessions past retention (`skipped_expired`) |

`--dry-run` is a true preview on all five — it resolves the same
idempotency keys a real run would, so use it before importing a store you
have not imported before.

`--hooks` wires capture into the harness on `PreCompact` + `SessionEnd`
(and recall on `SessionStart` + `UserPromptSubmit`, see **Retrieval
hygiene**), preserving the user's own hooks. Under the hood each capture hook fires
`hafiz import claude-code --from-hook`, which reads the harness payload
on stdin, imports only that session, and always exits 0 — never use it
by hand; it's the hook's entry point.

The user's explicit redaction commands:

```bash
hafiz forget <comm-id-or-session-slug>          # soft tombstone
hafiz forget <comm-id-or-session-slug> --hard   # delete content + messages
hafiz forget --all-expired                      # sweep retention_until
```

**Selective embedding** is enforced at import time: short turns and
pure tool-result echoes don't get embedded. Salient turns are still
written (``content`` is canonical) — just not vector-indexed.
