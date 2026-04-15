# Hafiz Command Map

> Source of truth for all hafiz commands. Update this file when commands change.

## Brain Types

| Type | What it is | Cost | Config |
|------|-----------|------|--------|
| **—** | No model needed, pure DB/filesystem operations | Free | — |
| **Embed** | fastembed (nomic-embed-text-v1.5), runs locally via ONNX | Free | `[embedding]` in hafiz.toml |
| **LLM** | Anthropic Claude API (extractor.py hardcoded to Anthropic SDK) | API cost per call | `[llm]` in hafiz.toml, requires `ANTHROPIC_API_KEY` |
| **Agent** | The LLM already in conversation (Claude Code, Cursor, Copilot) | Already paying for the session | N/A — agent reads CLI output and acts |

## Command Reference

### Setup

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `init` | Create DB tables + pgvector extension | — | same | same |
| `doctor` | Check DB, pgvector, embeddings, config health | — | `--json` | rich output |
| `config show` | Display current hafiz.toml settings | — | `--json` | rich output |
| `status` | Count chunks, entities, relations, observations by project | — | `--json` | rich output |
| `hooks install` | Write post-commit + post-merge git hooks into a repo | — | same | same |
| `agent install` | Write skills.md to agent config directory | — | same | same |
| `agent uninstall` | Remove skills.md from agent config directory | — | same | same |
| `agent list` | Show which agents have skills installed | — | same | rich output |

### Indexing

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `ingest <path>` | Chunk files, embed, store, optionally extract entities | Embed + LLM (optional) | `--json` emits NDJSON progress | rich progress bars |
| `ingest --no-extract` | Chunk files, embed, store (skip entity extraction) | Embed | `--json` | rich progress bars |
| `ingest --git-hook` | Re-index only files changed in latest commit | Embed | `--json` | rich output |
| `ingest --prune` | Remove stale chunks first, then ingest | Embed | `--json` | rich output |
| `watch <path>` | Long-running: detect file changes, re-ingest automatically | Embed | `--json` events | rich output |
| `prune` | Delete chunks for files that no longer exist on disk, mark entities stale | — | `--json` | rich output |

### Extraction (two paths to entities)

**Path 1 — Terminal (headless, no agent present):**

| Command | Purpose | Brain | Notes |
|---------|---------|:-----:|-------|
| `extract run` | Find chunks with no entities, call LLM API, store results | LLM | Requires `ANTHROPIC_API_KEY`. Incremental — only processes what's missing. |

**Path 2 — Agent (LLM already in conversation):**

| Step | Command | Brain | Notes |
|------|---------|:-----:|-------|
| 1. Export | `chunks export --unextracted` | — | Outputs chunks whose source files have no entities |
| 2. Analyze | Agent reads chunks, produces entities/relations JSON | Agent | The agent IS the brain — no API key needed |
| 3. Import | `extract import` | — | Stores the agent-produced JSON into the graph |

**Supporting commands:**

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `chunks export` | Export all indexed chunks as JSON | — | agent reads output | not useful alone |
| `chunks export --unextracted` | Export only chunks whose files have no entities | — | agent reads output | not useful alone |
| `extract import` | Store entities/relations from JSON (file or stdin) | — | agent pipes JSON | `--file` for JSON file |
| `extract run` | Extract entities from unextracted chunks via LLM API | LLM | not needed (agent IS the LLM) | primary use case |

### Search

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `query "<text>"` | Vector similarity search over code chunks | Embed | `--json` | rich output |
| `recall "<query>"` | Vector similarity search over observations | Embed | `--json` | rich output |
| `context "<task>"` | Synthesize chunks + graph + observations for a task | Embed | `--json` | rich panel |

**Scoping flags** (available on `context`, `query`, `recall`):

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
| `--workspace` / `-w` | `context`, `query`, `recall` | Scope to sibling projects in parent directory |
| `--type` / `-t` | `query`, `recall`, `observe` | Filter by type (chunk type or observation type) |
| `--limit` / `-l` | `query`, `recall`, `chunks export` | Maximum number of results |

## Architecture Note

Hafiz has two stability layers:

- **Layer 1 (stable):** `skills.md` installed via `hafiz agent install` — the contract between hafiz and AI agents. Changes here affect all agent integrations.
- **Layer 2 (evolving):** `hafiz review` — self-improvement mechanism. Evolves independently without breaking the agent contract.
