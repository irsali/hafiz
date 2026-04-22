# Hafiz Command Map

> Source of truth for all hafiz commands. Update this file when commands change.
> Post-structural-grounding: parsers own structure, agents own meaning — see skills.md v2.

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
| `config show` | Display current hafiz.toml settings | — | `--json` | rich output |
| `hooks install` | Write post-commit + post-merge + post-rewrite git hooks into a repo | — | same | same |
| `agent install` | Splice `skills.md` into an agent's config file; warns on version drift | — | same | same |
| `agent uninstall` | Remove the spliced `skills.md` block | — | same | same |
| `agent list` | Show which agents have skills installed | — | same | rich output |
| `parsers list` | List registered parsers (in-tree + entry-point-loaded) and their language coverage | — | `--json` | rich table |
| `embedding status` | Show current embedding device + provenance (config / sticky cache / probe) | — | `--json` | rich table |
| `embedding retry` | Clear sticky device cache and re-probe | Embed | `--json` | rich output |

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

Per-TTY named threads of work that auto-tag subsequent `observe` / `note` writes with a `session_id` and optional `task`. State lives in `~/.cache/hafiz/session-<tty>.json`, scoped to the controlling terminal so two shells don't pollute each other.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `session start "<name>"` | Start a named session for this terminal | — | `--json` | rich panel |
| `session show` | Show the active session | — | `--json` | rich panel |
| `session end` | Clear the session | — | `--json` | rich line |

`session start` flags: `--task <name>`, `--project <name>`.

**Auto-tagging** on `observe` / `note`: session state inherited when no flag is given; explicit `--session` / `--task` always win. Columns: `annotations.session_id` / `annotations.task` / `annotations.commit_hash`.

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
| `--diagnose` | `status` | Full diagnostic checks including parser registry |

## Architecture Note

Two stability layers:

- **Layer 1 (stable contract):** `skills.md` v2 installed via `hafiz agent install`. Ownership rule (parsers own structure, agents own meaning) is load-bearing. `hafiz agent install` detects version drift and warns when refreshing an older splice.
- **Layer 2 (evolving):** `hafiz review`, `hafiz parsers list`, Phase 7 observability surfaces. Free to iterate; not referenced from `skills.md`.
