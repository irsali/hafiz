# Hafiz Command Map

> Source of truth for all hafiz commands. Update this file when commands change.
> Post-structural-grounding: parsers own structure, agents own meaning — see skills.md v4 (adds self-tuning and error reporting to the agent contract).

## Brain Types

| Type | What it is | Cost | Config |
|------|-----------|------|--------|
| **—** | No model needed, pure DB/filesystem operations | Free | — |
| **Embed** | fastembed (nomic-embed-text-v1.5), runs locally via ONNX | Free | `[embedding]` in hafiz.toml |
| **Agent** | The LLM in conversation (Claude Code, Cursor, Copilot) or piped via CLI | Already paying for the session | N/A — agent reads CLI output and acts |
| **Parser** | Deterministic AST / prose parser loaded at ingest time | Free | `[parsers]` via entry points |

## Command Reference

### Setup

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `init` | Create the seven tables + pgvector extension | — | same | same |
| `status` | Count files / units / unit_revisions / embeddings / edges / annotations / commits, broken down by project and kind, plus last-indexed commit per project | — | `--json` | rich output |
| `status --diagnose` | Config / DB / pgvector / embeddings / parser-registry health | — | `--json` | rich output |
| `doctor` | Install health + host capabilities (RAM, CPU, GPU, onnxruntime) + tunable registry. Stable `--json` shape with `checks` / `host` / `tuning` keys. | — | `--json` | rich tables |
| `doctor --probe` | Same as `doctor`, plus runs per-tunable probers to recommend values for this host. Slow (loads the embedding model, runs several forward passes). | Embed | `--json` | rich tables |
| `doctor --apply [--yes]` | Implies `--probe`; persists recommendations to the sticky tuning cache (`~/.cache/hafiz/tuning_state.json`). **Interactive by default** — prompts `[Y]es / [n]o / [c]ustom` per recommendation. `--yes` skips prompts (CI); `--json` is also non-interactive. JSON response gains `applied` and `interactive` fields. | Embed | `--json --yes` | rich tables |
| `config show` | Display current hafiz.toml settings **and** per-tunable resolution sources (env / toml / sticky / default) | — | `--json` (payload gains a `tunables` array) | rich output |
| `config get <key>` | Print one tunable's effective value + source layer | — | `--json` | rich output |
| `config set <key> <value> [--local]` | Persist a tunable to hafiz.toml. User-scope by default (`~/.config/hafiz/hafiz.toml`); `--local` targets the project's `./hafiz.toml`. Validates and type-coerces the input. | — | `--json` | rich output |
| `config unset <key> [--local]` | Remove a tunable from hafiz.toml so it falls through to sticky / default. Prunes emptied tables. | — | `--json` | rich output |
| `config apply [--yes]` | Runs every prober and prompts to persist each recommendation. **Interactive by default** — `[Y]es / [n]o / [c]ustom` per row, with custom values validated through the tunable's coercer. `--yes` accepts everything; `--json` is non-interactive. JSON gains an `interactive` boolean alongside `applied`. Equivalent to `doctor --apply` with a narrower summary. | Embed | `--json --yes` | rich output |
| `config clear-sticky` | Delete the sticky tuning cache. Re-probe is required to repopulate. | — | `--json` | rich output |
| `errors list [--since] [--limit]` | Recent errors newest-first from `~/.cache/hafiz/errors.log` (NDJSON, 1000-entry cap, FIFO rotation). Each record includes `suggested_action` + `context` for recognized classes. Recognizers (v9): `ModuleNotFoundError` (declared-dep aware), sqlalchemy `OperationalError` (DB connectivity), pgvector missing (`ProgrammingError` with `'extension "vector" does not exist'`), pydantic `ValidationError` from the hafiz config loader. | — | `--json` | rich table |
| `errors list --group-by exception_type` | Pattern view: returns a distinct shape — `{since, grouped_by, total, with_suggestions, most_recent, groups}` — where each group carries `{exception_type, count, with_suggestions, most_recent_id, most_recent_timestamp, sample_command, sample_message}`. `--limit` is ignored in this mode (counts must reflect the full matching window). | — | `--json` | rich table |
| `errors show <id>` | Full structured record: traceback, cwd, git branch, host fingerprint. Accepts unique-prefix ids. | — | `--json` | rich panel |
| `errors clear` | Wipe the log; returns count discarded. | — | `--json` | rich output |
| `hooks install` | Write post-commit + post-merge + post-rewrite git hooks into a repo | — | same | same |
| `agent install` | Splice `skills.md` into an agent's config file; warns on version drift | — | same | same |
| `agent uninstall` | Remove the spliced `skills.md` block | — | same | same |
| `agent list` | Show which agents have skills installed | — | same | rich output |
| `parsers list` | List registered parsers (in-tree + entry-point-loaded) and their language coverage | — | `--json` | rich table |
| `embedding status` | Show current embedding device + provenance (config / sticky cache / probe) | — | `--json` | rich table |
| `embedding retry` | Clear sticky device cache and re-probe | Embed | `--json` | rich output |

**`hafiz doctor --json` shape** (stable — agents parse this):
```json
{
  "checks": [ {"name": "...", "passed": true, "detail": "...", "fix": "..."} ],
  "host": {
    "ram_total_mb": 64000, "ram_available_mb": 45000,
    "cpu_count": 16, "platform": "linux-x86_64",
    "onnx_providers": ["CPUExecutionProvider"],
    "gpu_name": null, "gpu_vram_total_mb": null, "gpu_vram_free_mb": null,
    "onnxruntime_version": "1.24.4",
    "fingerprint": "9c3fa..."
  },
  "tuning": [
    {
      "key": "embedding.max_part_chars",
      "current": 2000, "default": 2000,
      "description": "...", "is_policy": false,
      "recommended": 4000,          // only populated with --probe
      "rationale": "...", "confidence": "high",
      "measured": {"path": "cpu_measured", "budget_mb": 13500, "candidates": [...]},
      "probe_error": null
    }
  ]
}
```
Adding fields is safe; renaming requires a note here.

**Tunable resolution order:** `env (HAFIZ_*__*)` → `hafiz.toml` → sticky tuning cache → built-in default. The sticky layer is written by `doctor --apply` / `config apply` and keyed to a host fingerprint that invalidates itself when the RAM class, GPU presence, OS/arch, or onnxruntime provider set materially changes. Clear it with `config clear-sticky`.

**Probe safety brake.** The `embedding.max_part_chars` prober runs candidate sizes ascending in a subprocess. To stop the *measurement itself* from OOM-ing the host, the subprocess is given a `safety_ceiling_mb` (= the recommendation budget) and (a) extrapolates the next candidate's likely peak from the previous one × 3.0 and skips it if predicted to exceed the ceiling, (b) stops the moment a measured peak crosses the ceiling. Surfaced in `measured.candidates` with `_skipped` / `_stopped` sentinel rows. Without this brake, batched probing can spike RSS far above what the recommendation logic would ever pick.

### Indexing

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `ingest <path>` | Walk the tree, pick a parser per file, upsert units / revisions / embeddings. On a git repo, diff-driven: re-ingests only files changed since last indexed commit | Embed + Parser | `--json` emits NDJSON progress | rich progress |
| `ingest --git-hook` | Same pipeline, designed to run inside the installed post-commit / post-merge / post-rewrite hooks | Embed + Parser | `--json` | rich output |
| `watch <path>` | Long-running: detect file changes, re-ingest automatically. *(Phase 3b-2: still on the old API — falls through with a "not yet rewired" message.)* | Embed + Parser | `--json` events | rich output |
| `prune` | Obsolete under new pipeline — on-ingest tombstoning handles this. Kept as a CLI no-op. | — | `--json` | rich output |

**Race safety:** ingest refuses to run during a rebase / merge / cherry-pick in progress (detects `.git/<marker>` files) and exits with code 2.

**Rewrite resilience:** reconcile pass on every ingest marks commits that are no longer reachable in git as `commits.rewritten_at = now`. The installed `post-rewrite` hook triggers a fresh ingest automatically after an amend / rebase.

### Extraction (agent contract v2)

Parsers own structural facts (entities, calls, imports, inherits) — agents no longer write them. The extract pipeline is narrower and semantic-only.

| Step | Command | Brain | What happens |
|------|---------|:-----:|-------------|
| 1. Export | `extract export --project X` | — | Emits the AST-known units (with stable `identity_key`) + structural edges so the agent knows what already exists. |
| 2. Analyze | _(agent reads the output)_ | Agent | Decides which units deserve annotations (decisions / patterns / warnings) and which semantic edges to draw. |
| 3. Import | `extract import --project X` | — | Validates the v2 payload (rejects v1 / AST-territory) and writes annotations + semantic edges. |
| 4. Verify | `status --json` | — | Confirms counts across tables. |

**v2 JSON shape:**
```json
{
  "version": 2,
  "annotations": [
    {
      "content": "Canonical auth entry",
      "kind": "pattern",
      "source": "agent:claude-code",
      "unit_identity_key": "<from step 1>",
      "confidence": 0.9,
      "tags": ["auth"]
    }
  ],
  "edges": [
    {
      "source_name": "UserService", "source_file": "/abs/auth.py",
      "target_name": "SecurityPolicy", "target_file": "/abs/policy.py",
      "relation": "implements_pattern",
      "evidence": "..."
    }
  ]
}
```

Rejected at import time: `kind` starting with `code.*`; relations in `{calls, imports, inherits, references}`. Off-vocabulary kinds / relations produce warnings but are accepted.

| Command | Purpose | Brain | Key Flags |
|---------|---------|:-----:|-----------|
| `extract export` | Emit the AST-known units + edges the agent can attach to | — | `--project`, `--limit`, `--pretty` |
| `extract import` | Import a v2 payload (annotations + semantic edges) from JSON | — | `--file`, `--project` |

### Search

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `query "<text>"` | Vector similarity search over the `embeddings` table (joined back to units + files for context) | Embed | `--json` | rich output |
| `query "<text>" --recall` | Vector similarity search over annotations | Embed | `--json` | rich output |
| `context "<task>"` | Synthesize units + graph + annotations for a task | Embed | `--json` | rich panel |

**Scoping flags** (on `context`, `query`):

| Flag | Scope | How it works |
|------|-------|-------------|
| `--project X` | Single named project | Filters queries to `files.project = X` |
| `--workspace` | Sibling projects | Resolves directories in parent of cwd, matches to DB project tags (normalized) |
| _(neither)_ | Everything | No filter |

`--project` and `--workspace` are mutually exclusive.

**Domain filters** (on `context`, `query`): toggle whole data domains on/off without losing the rest of your index.

| Flag | Effect |
|------|--------|
| `--include-domain code,doc` | Restrict results to units whose `kind` starts with one of the listed domains. Comma-separated. |
| `--exclude-domain code` | Drop results whose `kind` starts with one of the listed domains. Comma-separated. |

A "domain" is the prefix of `kind` before the first dot — `code`, `doc`, `mail`, `chat`, `file`, etc. (For exact-kind filters like `code.function`, keep using `--type`.) The two flags are mutually exclusive *per-domain*: `--include-domain code,doc --exclude-domain doc` errors out before hitting the DB. A dotted value (`--include-domain code.function`) also errors — domains are dotless tokens.

When both flags are omitted on the CLI, the active session's defaults (set via `session start --include-domain` / `--exclude-domain`) are inherited. Passing *either* flag explicitly skips inheritance entirely, so a query can flip the filter without first ending the session.

### Knowledge Graph

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `graph show <name>` | Unit and its direct connections | — | `--json` | rich tree |
| `graph deps <name>` | What this unit depends on (outgoing edges) | — | `--json` | rich table |
| `graph impact <name>` | Blast radius — what depends on this unit (incoming edges) | — | `--json` | rich table |
| `graph path <src> <tgt>` | Shortest directed path | — | `--json` | rich tree |
| `graph rank` | Top units by centrality (pagerank / betweenness / degree) | — | `--json` | rich table |
| `graph stats` | Overall graph health (density, components, top-central, kind/relation breakdowns) | — | `--json` | rich tables |

Graph nodes are current units (`valid_until IS NULL`), edges are current edges (`superseded_at IS NULL`) with both endpoints resolved in scope. External references (`target_unit_id IS NULL`, `target_name` set) are excluded from traversal — visible via raw DB if needed.

### Annotations (the "wisdom layer")

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `observe "<text>"` | Embed and store a fact / decision / learning / pattern / warning / note | Embed | `--json` | rich panel |
| `note "<text>"` | Shortcut for `observe --type note` — low-bar raw capture lane | Embed | `--json` | rich panel |
| `journal` | Time-bounded digest of annotations, grouped by day | — | `--json` | rich tables |
| `distill` | Surface recent notes as promotable candidates (scanner, not promoter) | — | `--json` | rich tables |

- **Annotation kinds**: `fact` · `decision` · `learning` · `pattern` · `warning` · `note` · `concept` · `service`.
- **Auto-captured git context**: `commit_hash` column; `branch` / `is_dirty` in metadata JSONB. Captured when writing inside a git repo.
- **Expiration** (on `observe` / `note`, mutually exclusive): `--expires-in <30d|2w|6m|1y>` or `--expires <ISO-date>`. Sets `valid_until`; `--recall` hides expired rows by default.
- **Staleness hint**: `--recall` surfaces age (e.g. `3mo ago`) and dims rows older than 90 days.
- **Supersession** (on `observe` / `note`): `--supersedes <uuid>` atomically marks the target row inactive and records the link. Nothing is deleted.
- **Lineage** (on `observe` / `note`): `--derived-from <uuid>[,<uuid>...]` records distillation source without replacing.
- **Unit binding**: annotations created via `extract import` can link to a unit via `unit_identity_key`. Annotations created via `observe` are unit-free by default (can be linked later via API).

### Captures (transcripts / multi-page dumps)

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `capture [TEXT]` | *(Phase 3b-2: not yet rewired for the new schema. Falls through with a clear error. Transcript storage will land as `chat.turn` units.)* | — | — | — |

### Sessions

Per-TTY named threads of work that auto-tag subsequent `observe` / `note` writes with a `session_id` and optional `task`.

The on-disk JSON at `~/.cache/hafiz/session-<tty>.json` is now a **cursor** — the canonical record lives in the `sessions` table. The cursor carries both `session_uuid` (FK target) and `session_id` (slug) so display continuity is preserved.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `session start "<name>"` | Start a named session: creates a `sessions` row and writes the per-TTY cursor | — | `--json` | rich panel |
| `session show` | Show the active session | — | `--json` | rich panel |
| `session end` | Clear the cursor and stamp `ended_at` on the DB row | — | `--json` | rich line |

`session start` flags: `--task <name>`, `--project <name>`, `--include-domain <a,b>`, `--exclude-domain <a,b>`. The two domain flags persist into the per-TTY cursor JSON (not the DB row) and are inherited by `query` / `context` calls in this terminal.

**Auto-tagging** on `observe` / `note`: session inherited when no flag is given; explicit `--session` / `--task` always win. ``--session <slug>`` resolves the slug to a uuid via the `sessions` table; both `annotations.session_id` (uuid FK) and `annotations.legacy_session_id` (text slug) are populated, so journal/distill display stays human-readable.

### Source layer (transcripts)

Hafiz can ingest agent-harness transcripts into a dedicated source layer (see [architecture.md "Storage layers"](./architecture.md#storage-layers--knowledge-vs-source)). Source rows are **hidden from default `query` / `context`** — surfacing them is opt-in.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `import claude-code [PATH]` | Idempotent post-hoc import of Claude Code session JSONL into `communications` + messages | — | `--json` | rich table |
| `recall <target>` | Ordered messages for a session (slug or uuid) or communication; or vector search via `--query` | — | `--json` | rich table |
| `query --include-transcripts` | Add matching source-layer turns to results, tagged `layer="source"` | — | `--json` | rich panel (separate transcript section) |
| `context --include-transcripts` | Append matching transcript turns under a `transcripts` field | — | `--json` | rich panel |
| `forget <target>` | Targeted redaction: soft tombstone by default, `--hard` deletes content | — | `--json` | rich line |
| `forget --all-expired` | Sweep mode — tombstone every communication past its `retention_until` | — | `--json` | rich table |

Defaults:
- `import claude-code` source path: `~/.claude/projects/`. Idempotent by `(agent='claude-code', external_id=<jsonl session uuid>)`.
- `import claude-code` flags: `--project`, `--limit`, `--since 7d`, `--dry-run`, `--no-embed`.
- `recall` flags: `--query`, `--role`, `--from`, `--to`, `--has-tool-call` / `--no-tool-call`, `--limit`.
- `forget` flags: `--hard`, `--all-expired`, `--dry-run`.
- Retention: 90 days from `started_at` unless explicitly overridden at insert time.
- Selective embedding: skip messages under ~30 tokens; skip pure tool-result echoes; embed when `marked_salient=true` regardless of length.

### Review

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `review` | *(Phase 3b-4: still on the old schema. Fails cleanly until rewired.)* | — | — | — |

## Common Flags

| Flag | Available on | Purpose |
|------|-------------|---------|
| `--json` / `-j` | Most commands | Machine-readable output for agents |
| `--project` / `-p` | Most commands | Filter or tag by project name |
| `--workspace` / `-w` | `context`, `query`, `journal` | Scope to sibling projects in parent directory |
| `--type` / `-t` | `query`, `observe`, `journal` | Unit kind or annotation kind depending on context |
| `--limit` / `-l` | `query`, `extract export`, `journal` | Maximum results |
| `--recall` | `query` | Search annotations instead of content |
| `--since` | `journal` / `distill` | Duration window ending now (default `7d`) |
| `--day` | `journal` | Specific UTC day (ISO date). Exclusive with `--since` |
| `--expires-in` | `observe`, `note` | Expire after duration. Exclusive with `--expires` |
| `--expires` | `observe`, `note` | Expire at ISO date. Exclusive with `--expires-in` |
| `--source` | `observe`, `note`, `journal` | Origin tag (`agent:<name>`, `user:<name>`) |
| `--session` | `observe`, `note`, `journal`, `distill` | Explicit session id |
| `--task` | `observe`, `note`, `journal`, `distill` | Explicit task label |
| `--supersedes` | `observe`, `note` | UUID of annotation being replaced |
| `--derived-from` | `observe`, `note` | UUIDs this row was distilled from |
| `--include-superseded` | `query --recall` | Return superseded / expired rows |
| `--include-transcripts` | `query`, `context` | Add source-layer transcript matches to results (off by default) |
| `--include-domain` | `query`, `context`, `session start` | Comma-separated data-domain allowlist (`code`, `doc`, `chat`, …). Domain = `kind` prefix before the first dot. |
| `--exclude-domain` | `query`, `context`, `session start` | Comma-separated data-domain denylist. Mutually exclusive *per-domain* with `--include-domain`. |
| `--diagnose` | `status` | Full diagnostic checks including parser registry |

## Architecture Note

Two stability layers:

- **Layer 1 (stable contract):** `skills.md` v2 installed via `hafiz agent install`. Ownership rule (parsers own structure, agents own meaning) is load-bearing. `hafiz agent install` detects version drift and warns when refreshing an older splice.
- **Layer 2 (evolving):** `hafiz review`, `hafiz parsers list`, Phase 7 observability surfaces. Free to iterate; not referenced from `skills.md`.
