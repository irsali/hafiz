# 🧠 Hafiz (حافظ) — The Sovereign Intelligence Layer

> *Named after the tradition of the Hafiz — one who preserves, understands, and recalls with precision.*
> A self-sovereign, CLI-first memory system that any AI agent can plug into.
> Your codebase. Your decisions. Your knowledge. Your control.

---

## Vision

Every AI agent you use — Claude Code, Cursor, Copilot, Aider, or anything tomorrow — connects to **one shared brain**. No more scattered `.md` files, no more "I forgot what we decided last week." Hafiz is always on, always fresh, and always yours.

> **Status (2026-04-21):** Phases 1–5 shipped, plus the temporal/capture/distill layer (journal, note, capture, session, distill, expiry, supersession — see [workitems/done/temporal-session-awareness.md](workitems/done/temporal-session-awareness.md)). Core CLI surface is stable; the product is dogfooded daily. One-time migration scripts are still open. See [Shipped](#shipped) and [Open work](#open-work) below.
>
> **Dropped from scope (as of 2026-04-21):** REST API layer and MCP server. Hafiz is intentionally CLI-only; agents integrate via `hafiz` + `--json`. This decision replaces the earlier "future" positioning of those surfaces.
>
> This file is the **product vision + future backlog**. It is *not* the development guide — see [CLAUDE.md](CLAUDE.md) for conventions, layout, and how to add a command. [COMMANDS.md](COMMANDS.md) is the source of truth for command shapes and flags.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      YOUR WORKSPACE                      │
│   Code repos • Notes • Decisions • Config • Docs         │
└──────────────────────────┬──────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  LIBRARIAN  │  (Ingestion Engine)
                    │  Watches &  │  - File watcher / git hooks
                    │  Indexes    │  - Agent-driven extraction
                    └──────┬──────┘  - Entity & relationship parsing
                           │
              ┌────────────▼────────────────┐
              │    POSTGRESQL + pgvector     │
              │                             │
              │  ┌─────────┐ ┌───────────┐  │
              │  │ Chunks  │ │ Entities  │  │
              │  │ (text + │ │ (nodes)   │  │
              │  │ vectors)│ │           │  │
              │  └─────────┘ └───────────┘  │
              │  ┌─────────┐ ┌───────────┐  │
              │  │Relations│ │Observations│ │
              │  │ (edges) │ │ (facts &  │  │
              │  │         │ │ decisions)│  │
              │  └─────────┘ └───────────┘  │
              └────────────┬────────────────┘
                           │
                    ┌──────▼──────┐
                    │  BRAIN CORE │  (Python Library)
                    │  pgvector   │  - Vector search
                    │  + NetworkX │  - Graph traversal & centrality
                    │  + Custom   │  - Context synthesis
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  CLI Tool   │
                    │   `hafiz`   │
                    └──────┬──────┘
                           │
     ┌────────┬────────┬───┴──────┬─────────┐
     │        │        │          │         │
  Claude   Cursor   Copilot    Aider     Future
   Code                                  Agents
```

---

## Data Model

Four tables, all in one PostgreSQL database with `pgvector`. Alembic owns the schema — see [alembic/versions/](alembic/versions/).

```sql
-- Raw content, chunked and embedded
chunks        (id, content, embedding vector(768), source_file, line_start,
               line_end, chunk_type, language, project, checksum, indexed_at,
               metadata jsonb)

-- Extracted entities — the "nouns"
entities      (id, name, entity_type, description, project, source_file,
               properties jsonb, created_at, updated_at)

-- Relationships between entities — the "verbs"
relations     (id, source_id → entities, target_id → entities, relation_type,
               weight, evidence, metadata jsonb, created_at, updated_at)

-- High-level decisions, facts, and learnings — the "wisdom"
observations  (id, content, embedding vector(768), obs_type, source, project,
               tags text[], confidence, valid_from, valid_until, metadata jsonb)
```

**Indexes in place:** `project` / `source_file` / `checksum` on chunks; `entity_type` / `project` on entities; `source_id` / `target_id` on relations; `obs_type` on observations.

**Indexes deferred until scale demands them:** `ivfflat` on `chunks.embedding` and `observations.embedding`; GIN on `observations.tags`. Exact cosine search is fast enough at current corpus size.

---

## CLI Surface

`hafiz --help` is the authoritative list; [COMMANDS.md](COMMANDS.md) documents flags, JSON shapes, and when each command is agent-driven vs. human-driven. Current shape:

```bash
# ─── SETUP ───
hafiz init                          # create schema + pgvector extension
hafiz status [--diagnose]           # counts, health, config/DB/embedding checks
hafiz config show                   # print merged config
hafiz hooks install <repo>          # write post-commit + post-merge hooks
hafiz agent install|uninstall|list  # splice skills.md into agent config files

# ─── INGEST / MAINTAIN ───
hafiz ingest <path> [--project] [--prune] [--git-hook] [--json]
hafiz watch  <path> [--project] [--json]    # debounced re-index on change
hafiz prune         [--project] [--dry-run] [--json]

# ─── SEARCH ───
hafiz query   "<text>" [--project|--workspace] [--type] [--limit] [--json]
hafiz query   "<text>" --recall [--type] …   # search observations instead of chunks
hafiz context "<task>" [--project|--workspace] [--json]   # chunks + graph + observations

# ─── GRAPH (NetworkX-backed) ───
hafiz graph show   <entity> [--depth] [--project] [--json]
hafiz graph deps   <entity> [--depth] [--project] [--json]  # outgoing walk
hafiz graph impact <entity> [--depth] [--project] [--json]  # incoming walk (blast radius)
hafiz graph path   <src> <tgt> [--project] [--json]         # shortest directed path
hafiz graph rank   [--metric pagerank|betweenness|degree|in_degree|out_degree] [--top]
hafiz graph stats  [--project] [--top-central] [--json]

# ─── OBSERVE ───
hafiz observe "<text>" [--type] [--source] [--project] [--tags] [--confidence] [--json]

# ─── EXTRACT (agent-driven; no API key) ───
hafiz extract export [--project] [--path] [--unextracted] [--limit] [--offset]
hafiz extract import [--file] [--project]         # or stdin

# ─── REVIEW (Layer 2 — evolving) ───
hafiz review [--project] [--json]
```

**Output modes.** Human (Rich panels, tables, trees) by default. `--json` on every user-facing command; shapes are stable and documented in [COMMANDS.md](COMMANDS.md).

---

## Shipped

Phase numbering preserved for history. Deviations from the original plan are called out — they are features of the current design, not debt.

### Phase 1 — Foundation
- [x] Project structure, `pyproject.toml`, `pipx install -e .`
- [x] Config system (`hafiz.toml` + env; pydantic-settings; cwd → `~/.config/hafiz/` → `/etc/hafiz/`)
- [x] Alembic-managed schema (`chunks`, `entities`, `relations`, `observations`)
- [x] Core modules: `database.py`, `embeddings.py` (fastembed ONNX), `chunker.py` (LlamaIndex `SentenceSplitter` / `CodeSplitter`), `store.py`, `search.py`
- [x] CLI: `init`, `ingest`, `query`, `query --json`, `status`, `config show`
- [x] Tests: `test_chunker`, `test_cli`, `test_config`, `test_search`
- **Deviation:** `store.py` / `search.py` talk to pgvector directly via SQLAlchemy rather than wrapping `llama-index-vector-stores-postgres`. One fewer abstraction; the LlamaIndex vector-store dependency is not in `pyproject.toml`.

### Phase 2 — The Graph
- [x] `entities` / `relations` tables, chunk ↔ entity link via `chunk_id`
- [x] `graph show` / `deps` / `path` with `--depth`, `--project`, `--json`
- [x] **Bonus — not in original roadmap:** `graph rank` (PageRank, betweenness, degree), `graph stats` (density, components, top-central) — full NetworkX analysis layer
- **Deviation (major):** Extraction is **agent-driven**, not LlamaIndex `PropertyGraphIndex` / `SchemaLLMPathExtractor`. `extract export` dumps chunks grouped by file → the in-session agent produces the JSON → `extract import` stores it. No API key is ever required; the direct-LLM path was removed in `9440b3f`.
- **Rename:** `graph dependents` (in the original plan) shipped as `graph impact`. Same behaviour — incoming walk — clearer framing as "blast radius."
- **Deviation:** The original "hybrid search with `--depth`" on `query` is delivered by the dedicated `hafiz context` command (Phase 3), which combines chunks + graph + observations. `query` stayed focused on vector similarity.

### Phase 3 — The Wisdom Layer
- [x] `observe` with `--type`, `--source`, `--project`, `--tags`, `--confidence`
- [x] `query --recall` (semantic search over observations only) with `--type` / `--project` filters
- [x] `hafiz context "<task>"` — the killer feature. Markdown + JSON bundle: relevant chunks, graph neighbours (PageRank-scored), matching observations, project distribution
- **Deviation:** No Mem0 dependency — own `observations` table + pgvector, as planned in principle.

### Phase 4 — The Librarian
- [x] `watch` — watchdog-backed debounced re-index; checksum-gated (no-op on unchanged files)
- [x] `ingest --git-hook` — diff-based, re-indexes only files touched in the latest commit; stores commit metadata as an observation
- [x] `hooks install` — writes post-commit + post-merge hooks into a target repo
- [x] `prune` — removes chunks for deleted files, marks orphaned entities stale

### Phase 5 — Agent Integration
- [x] `hafiz agent install|uninstall|list` for **claude-code**, **cursor**, **github-copilot** — a generic registry rather than a Bilal/OpenClaw-specific adapter
- [x] **Paired-marker splicing** — `agent install` preserves user-owned content in `CLAUDE.md` / `.cursor/rules/hafiz.mdc` / `.github/copilot-instructions.md` via sentinel markers (commit `76888f6`)
- [x] [`BRAIN_AGENT_GUIDE.md`](BRAIN_AGENT_GUIDE.md) — universal agent integration playbook
- [x] [`hafiz/data/agents/skills.md`](hafiz/data/agents/skills.md) — **Layer 1 stable contract** (see [Two-Layer Stability Model](CLAUDE.md))

---

## Open work

Items that were scoped in the original roadmap but are **not yet shipped**:

- [ ] **One-time migration scripts** — `import_memory.py` (MEMORY.md → observations), `import_knowledgehub.py`, `import_chromadb.py`. No top-level `scripts/` directory exists today.
- [ ] **ChromaDB hard cutover** — not executed; no ChromaDB dependency to decommission in this repo
- [ ] **Retention policies** for stale data — `prune` is on-demand only; no age-based policy
- [ ] **Test coverage for a second agent** (Aider, Codex) beyond the three currently registered

---

## Future (Phase 6+)

Ideas, not committed — explore when the open-work list is clear.

- [ ] **Community Detection** — auto-group related entities into "modules" or "domains"
- [ ] **Cross-Project Learning** — "How did we solve rate-limiting in Project A? Apply that to Project B."
- [ ] **Dashboard** — simple web UI showing the knowledge graph visually
- [ ] **Embedding Model Migration** — install-time model choice + `hafiz embeddings migrate` to re-embed corpus when switching models (enforce one model per DB)
- [ ] **Markdown Dump/Export** — export captures, transcripts, observations, and journal entries as `.md` files for portability, backup, and human review
- [ ] **DB export/import** — `hafiz export --format json` / `hafiz import` for backup and portability (mentioned in original roadmap; not shipped)

*Temporal Queries* ("what did the auth module look like N months ago?") was in this list; the capture → distill → supersession loop (shipped in the temporal work item) covers the decision/observation side. Code-history queries remain an open direction if ever needed.

---

## Success Criteria

1. **Any agent can query Hafiz in under 2 seconds.**
2. **Hafiz stays fresh automatically** — no manual "re-index" needed (watcher + git hooks).
3. **Zero vendor lock-in** — runs entirely on your machine, your database, your models.
4. **One command to install:** `pipx install hafiz` and it's available everywhere.
5. **Observations persist across sessions** — what Cursor learns today, Claude Code knows tomorrow.

---

## Design Principles (Non-Negotiable)

- **Standalone project.** Hafiz is an independent tool — not coupled to any specific agent. Any agent that can run a CLI command can use it.
- **Two-layer stability.** Layer 1 ([skills.md](hafiz/data/agents/skills.md)) is the stable contract with every agent; Layer 2 (`hafiz review` and friends) is free to iterate. See [CLAUDE.md](CLAUDE.md).
- **Everything is configurable.** DB connection, embedding model, workspace path — all via `hafiz.toml` or environment variables. No hardcoded values.
- **Library-first, not library-captured.** We lean on LlamaIndex for chunking primitives and fastembed for embeddings, but we own the storage, search, graph, and CLI layers in plain SQL + NetworkX. Wrappers earn their place.
- **No API-key-dependent paths.** Entity/relation extraction is agent-driven by design (direct-LLM path removed in `9440b3f`). The agent *is* the brain.
- **Workspace = Unit of Knowledge.** A "workspace" is a root directory (like a VSCode workspace) that may contain multiple projects. Hafiz indexes the workspace and tags chunks/entities by project. `--workspace` scopes to sibling projects in the parent directory.
- **CLI surface is the product.** Human output must be scannable; `--json` shapes must be stable and documented in [COMMANDS.md](COMMANDS.md).

---

## Decisions (Locked In)

| # | Question | Decision |
|---|---|---|
| 1 | Database | **Dedicated DB** inside the existing `postgres` container (port 5432). Separate database name (`hafiz`). |
| 2 | Embedding model | **nomic-embed-text-v1.5** (768 dims, fastembed ONNX, local). GPU via `fastembed-gpu` extras. |
| 3 | Entity extraction | **Agent-driven.** `extract export` → agent → `extract import`. No LLM API key required. *(Revised 2026-04 from the original "LlamaIndex + Claude" plan.)* |
| 4 | Scope | **Workspace-scoped.** One Hafiz instance per workspace; a workspace contains one or more projects. `--workspace` fans out across siblings. |
| 5 | ChromaDB | **Hard cutover** when migration lands. No dual-read. |
| 6 | Stability model | **Two layers.** `skills.md` is the stable agent contract; `review` and friends evolve freely in Layer 2. |

---

*Created: 2026-04-14 · Last restructure: 2026-04-21*
*Authors: Irshad Ali
*Status: Phases 1–5 shipped · Phase 6 and open-work items active*
