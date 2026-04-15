# Hafiz Command Map

> Source of truth for all hafiz commands. Update this file when commands change.

## Brain Types

| Type | What it is | Cost | Config |
|------|-----------|------|--------|
| **—** | No model needed, pure DB/filesystem operations | Free | — |
| **Embed** | fastembed (nomic-embed-text-v1.5), runs locally via ONNX | Free | `[embedding]` in hafiz.toml |
| **Agent** | The LLM in conversation (Claude Code, Cursor, Copilot) or piped via CLI (`claude -p`) | Already paying for the session | N/A — agent reads CLI output and acts |

## Command Reference

### Setup

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `init` | Create DB tables + pgvector extension | — | same | same |
| `status` | Count chunks, entities, relations, observations by project | — | `--json` | rich output |
| `status --diagnose` | Check DB, pgvector, embeddings, config health | — | `--json` | rich output |
| `config show` | Display current hafiz.toml settings | — | `--json` | rich output |
| `hooks install` | Write post-commit + post-merge git hooks into a repo | — | same | same |
| `agent install` | Write skills.md to agent config directory | — | same | same |
| `agent uninstall` | Remove skills.md from agent config directory | — | same | same |
| `agent list` | Show which agents have skills installed | — | same | rich output |

### Indexing

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `ingest <path>` | Chunk files, embed, store in DB | Embed | `--json` emits NDJSON progress | rich progress bars |
| `ingest --git-hook` | Re-index only files changed in latest commit | Embed | `--json` | rich output |
| `ingest --prune` | Remove stale chunks first, then ingest | Embed | `--json` | rich output |
| `watch <path>` | Long-running: detect file changes, re-ingest automatically | Embed | `--json` events | rich output |
| `prune` | Delete chunks for files that no longer exist on disk, mark entities stale | — | `--json` | rich output |

### Extraction (agent-driven, two-phase)

Entity extraction is always agent-driven. The agent reads chunks, identifies
entities and relationships, and imports the results. No external API key needed.

| Step | Command | Brain | What happens |
|------|---------|:-----:|-------------|
| 1. Export | `extract export --unextracted` | — | Exports chunks grouped by file, filtered to files without entities |
| 2. Analyze | _(agent reads the output)_ | Agent | **Phase 1:** identify entities per file (file-scoped). **Phase 2:** identify relations across files (project-scoped). |
| 3. Import | `extract import` | — | Stores the agent-produced entities/relations JSON into the graph |
| 4. Verify | `status --json` | — | Confirm entity and relation counts |

**Terminal (no agent in session):** pipe through an LLM CLI:
```bash
hafiz extract export --unextracted --project X | claude -p "extract entities per hafiz schema" | hafiz extract import --project X
```

**Supporting commands:**

| Command | Purpose | Brain | Key Flags |
|---------|---------|:-----:|-----------|
| `extract export` | Export chunks grouped by file as JSON | — | `--project`, `--unextracted`, `--path`, `--limit`, `--offset` |
| `extract import` | Store entities/relations from JSON (file or stdin) | — | `--file`, `--project` |

### Search

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `query "<text>"` | Vector similarity search over code chunks | Embed | `--json` | rich output |
| `query "<text>" --recall` | Vector similarity search over observations | Embed | `--json` | rich output |
| `context "<task>"` | Synthesize chunks + graph + observations for a task | Embed | `--json` | rich panel |

**Scoping flags** (available on `context`, `query`):

| Flag | Scope | How it works |
|------|-------|-------------|
| `--project X` | Single named project | Filters DB queries to `project = X` |
| `--workspace` | Sibling projects | Resolves directories in parent of cwd, matches to DB project tags (normalized: case-insensitive, ignores spaces/hyphens) |
| _(neither)_ | Everything | No filter — searches all indexed content |

`--project` and `--workspace` are mutually exclusive.

### Knowledge Graph

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `graph show <name>` | Show entity and its direct connections (in + out) | — | `--json` | rich output |
| `graph deps <name>` | Show what an entity depends on (outgoing relations) | — | `--json` | rich output |
| `graph dependents <name>` | Show what depends on an entity (incoming relations) | — | `--json` | rich output |

### Observations

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `observe "<text>"` | Embed and store a fact, decision, learning, pattern, or warning | Embed | `--json` | rich panel |

### Review

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `review` | Analyze observation quality, graph coverage, extraction gaps, staleness | — | `--json` | rich panel |

## Common Flags

| Flag | Available on | Purpose |
|------|-------------|---------|
| `--json` / `-j` | Most commands | Machine-readable output for agents |
| `--project` / `-p` | Most commands | Filter or tag by project name |
| `--workspace` / `-w` | `context`, `query` | Scope to sibling projects in parent directory |
| `--type` / `-t` | `query`, `observe` | Filter by type (chunk type or observation type with --recall) |
| `--limit` / `-l` | `query`, `extract export` | Maximum number of results |
| `--recall` | `query` | Search observations instead of code chunks |
| `--diagnose` | `status` | Run full diagnostic checks (config, DB, pgvector, embeddings) |

## Architecture Note

Hafiz has two stability layers:

- **Layer 1 (stable):** `skills.md` installed via `hafiz agent install` — the contract between hafiz and AI agents. Changes here affect all agent integrations.
- **Layer 2 (evolving):** `hafiz review` — self-improvement mechanism. Evolves independently without breaking the agent contract.
