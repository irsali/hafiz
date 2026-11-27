<!-- summary: Ingest, error reporting and self-tuning -->
# hafiz guide ops — Ingest, error reporting and self-tuning

## Ingest (for reference — users usually drive this)

```bash
hafiz ingest <path> --project <name>
```

Hafiz walks the tree (gitignore-aware), picks a parser per file via
the registry, and upserts units / revisions / embeddings. On a git
repo, re-ingests are diff-driven: only files changed since the last
indexed commit are re-parsed. Rebases and amends fire the
`post-rewrite` hook automatically if installed via `hafiz hooks install`.

## Error reporting

Hafiz captures every unhandled exception into a sovereign,
user-scope error log at `~/.cache/hafiz/errors.log` (NDJSON, capped
at 1000 entries). Use it when the user says "hafiz feels broken"
or you want to understand why a recent command misbehaved.

```bash
hafiz errors list --since 1d --json                          # newest-first, agent-consumable
hafiz errors list --group-by exception_type --since 1d --json # pattern view: counts per class + most_recent
hafiz errors show <id> --json                                 # full traceback + suggested fix
hafiz errors clear                                            # reset the log
```

Each record carries: `timestamp`, `command`, `argv`, `exception_type`,
`message`, `traceback`, `cwd`, `hafiz_version`, `git_branch`,
`host_fingerprint`, plus — for recognized error classes — a
`suggested_action` string and structured `context` (e.g.,
`{"missing_module": "asyncpg", "is_declared_dep": true}`).

Recognizer set as of v10: `ModuleNotFoundError` (declared-dep aware),
sqlalchemy `OperationalError` (DB connectivity → points at
`hafiz status --diagnose`), pgvector missing (`'extension "vector"
does not exist'` on `ProgrammingError`), pydantic `ValidationError`
raised inside the hafiz config loader (points at `hafiz config show`).

The `--group-by exception_type` shape is distinct from the flat one
— use it for the "what's been failing" lookup:
```json
{
  "since": "1d",
  "grouped_by": "exception_type",
  "total": 7,
  "with_suggestions": 4,
  "most_recent": {"id": "...", "exception_type": "...", "command": "...", "timestamp": "..."},
  "groups": [
    {"exception_type": "ModuleNotFoundError", "count": 3, "with_suggestions": 3,
     "most_recent_id": "...", "most_recent_timestamp": "...",
     "sample_command": "ingest", "sample_message": "No module named 'asyncpg'"}
  ]
}
```

The suggestion is informational. Offer it to the user; don't
auto-run remedial commands without explicit opt-in — even safe
ones like `pipx inject` mutate the user's environment.

`hafiz doctor` surfaces the last-24h error count inline so you can
see at a glance whether something's been failing.

## Self-tuning the install

Hafiz has a tunable registry (RAM-sensitive knobs like embedding
part size, policy caps like ingest file-size guard). You can probe
the host and recommend values, and you can persist those
recommendations to a sticky cache so subsequent runs pick them up.

Use this when the user asks "what are the best settings for this
machine?" or reports slowness / OOM / crashes during ingest. Do not
run it unprompted on every session.

```bash
# Read-only: check current config + recommended values for this host.
hafiz doctor --probe --json
```

The `--json` shape is documented in docs/commands.md under the Setup
section; key fields:

- `host.{ram_total_mb, ram_available_mb, cpu_count, onnx_providers, gpu_name, gpu_vram_free_mb, fingerprint}`
- `tuning[i].{key, current, recommended, rationale, confidence, probe_error}`

If the user agrees with the recommendations, persist them to the
sticky cache (user-scope, does not modify `hafiz.toml`):

```bash
hafiz config apply        # or:  hafiz doctor --apply
```

For one-off edits to the checked-in config, use `hafiz config set`
(writes TOML). Unset removes the key and falls through to sticky /
default:

```bash
hafiz config set embedding.max_part_chars 4096
hafiz config unset embedding.max_part_chars
hafiz config get embedding.max_part_chars --json   # shows source layer
hafiz config clear-sticky                          # wipe probed cache
```

**Resolution order** (every tunable reads through this chain):

    env (HAFIZ_*__*)  →  hafiz.toml  →  sticky cache  →  built-in default

Sticky is keyed to a host fingerprint (RAM class + GPU presence +
OS/arch + ORT provider set); moving a laptop-tuned cache to a
workstation is a no-op, not a hazard.

---
