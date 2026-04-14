# 🧠 Hafiz (حافظ) — The Sovereign Intelligence Layer


> *Named after the tradition of the Hafiz — one who preserves, understands, and recalls with precision.*
> A self-sovereign, CLI-first memory system that any AI agent can plug into.
> Your codebase. Your decisions. Your knowledge. Your control.

---

## Vision

Every AI agent you use — Bilal, Claude Code, Aider, Cursor, or anything tomorrow — connects to **one shared brain**. No more scattered `.md` files, no more "I forgot what we decided last week." Hafiz is always on, always fresh, and always yours.

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
                    │  Indexes    │  - LLM-powered extraction
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
              │  │Relations│ │Observations│  │
              │  │ (edges) │ │ (facts &  │  │
              │  │         │ │ decisions)│  │
              │  └─────────┘ └───────────┘  │
              └────────────┬────────────────┘
                           │
                    ┌──────▼──────┐
                    │  BRAIN CORE │  (Python Library)
                    │  LlamaIndex │  - Hybrid search
                    │  + Custom   │  - Graph traversal
                    │    Logic    │  - Context synthesis
                    └──────┬──────┘
                           │
              ┌────────────┼────────────────┐
              │            │                │
        ┌─────▼─────┐ ┌───▼────┐    ┌──────▼──────┐
        │  CLI Tool  │ │  API   │    │  MCP Server │
        │  `hafiz`   │ │ (REST) │    │  (Future)   │
        └─────┬──────┘ └───┬────┘    └──────┬──────┘
              │            │                │
     ┌────────┼──────┬─────┼────────┬───────┼──────┐
     │        │      │     │        │       │      │
   Bilal  Claude  Aider  Cursor  WebUI  Future   MCP
          Code                          Agents  Clients
```

---

## Data Model

### Core Tables

```sql
-- The raw content, chunked and embedded
CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT NOT NULL,
    embedding   vector(768),          -- nomic-embed or similar
    source_file TEXT NOT NULL,         -- relative path from workspace root
    line_start  INT,
    line_end    INT,
    chunk_type  TEXT DEFAULT 'code',   -- code | doc | note | decision
    language    TEXT,                   -- python, typescript, markdown, etc.
    project     TEXT,                   -- hu-manity, noble-wave, etc.
    checksum    TEXT,                   -- for change detection
    indexed_at  TIMESTAMPTZ DEFAULT NOW(),
    metadata    JSONB DEFAULT '{}'     -- extensible metadata bag
);

-- Extracted entities (the "nouns" of your codebase)
CREATE TABLE entities (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL,          -- class, function, module, api, table, concept
    description TEXT,                   -- LLM-generated one-liner
    project     TEXT,
    source_file TEXT,
    properties  JSONB DEFAULT '{}',    -- complexity_score, visibility, etc.
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Relationships between entities (the "verbs")
CREATE TABLE relations (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id       UUID REFERENCES entities(id) ON DELETE CASCADE,
    target_id       UUID REFERENCES entities(id) ON DELETE CASCADE,
    relation_type   TEXT NOT NULL,      -- calls, imports, inherits, depends_on, defines
    weight          FLOAT DEFAULT 1.0,  -- strength/confidence
    evidence        TEXT,               -- the code/text that proves this relationship
    metadata        JSONB DEFAULT '{}'
);

-- High-level observations, decisions, and learnings (the "wisdom")
CREATE TABLE observations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content     TEXT NOT NULL,          -- "We decided to use JWT over sessions"
    embedding   vector(768),
    obs_type    TEXT DEFAULT 'fact',    -- fact, decision, preference, lesson, warning
    source      TEXT,                   -- agent:bilal, agent:claude-code, user:irshad
    project     TEXT,
    tags        TEXT[],                 -- searchable tags
    confidence  FLOAT DEFAULT 1.0,     -- how sure are we?
    valid_from  TIMESTAMPTZ DEFAULT NOW(),
    valid_until TIMESTAMPTZ,           -- NULL = still valid
    metadata    JSONB DEFAULT '{}'
);

-- Indexes
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_project ON chunks (project);
CREATE INDEX idx_chunks_source ON chunks (source_file);
CREATE INDEX idx_entities_type ON entities (entity_type);
CREATE INDEX idx_entities_project ON entities (project);
CREATE INDEX idx_relations_source ON relations (source_id);
CREATE INDEX idx_relations_target ON relations (target_id);
CREATE INDEX idx_observations_embedding ON observations USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_observations_type ON observations (obs_type);
CREATE INDEX idx_observations_tags ON observations USING gin (tags);
```

---

## CLI Interface Design

### Command Structure

```bash
hafiz <command> [options]

# ─── SEARCH & QUERY ───
hafiz query "how does the auth module work?"          # Hybrid search (vector + graph)
hafiz query "auth module" --type code --project hu-manity  # Filtered search
hafiz query "auth module" --json                      # Machine-readable output for agents
hafiz query "auth module" --depth 2                   # Include 2-hop relationships

# ─── GRAPH NAVIGATION ───
hafiz graph show AuthController                       # Show entity and its connections
hafiz graph deps AuthController                       # Dependency tree (what it needs)
hafiz graph dependents AuthController                 # Reverse deps (what needs it)
hafiz graph path AuthController UserTable             # Shortest path between two entities

# ─── OBSERVATIONS (Agent Memory) ───
hafiz observe "JWT is preferred over sessions for auth" --type decision --project hu-manity
hafiz observe "VendorAmount is TEXT, must cast to NUMERIC" --type warning --tags sql,gotcha
hafiz recall "what do we know about Stripe integration?"  # Search observations only

# ─── INGESTION ───
hafiz ingest ./src/                                   # Index a directory
hafiz ingest ./src/auth/ --project hu-manity          # Index with project tag
hafiz ingest --watch ./src/                           # Watch mode (daemon)
hafiz ingest --git-hook                               # Run as post-commit hook

# ─── MAINTENANCE ───
hafiz status                                          # DB stats, index health
hafiz refresh ./src/auth/controller.py                # Re-index a specific file
hafiz prune --stale 90                                # Remove chunks older than 90 days with no source
hafiz export --format json > backup.json              # Export the brain
hafiz import backup.json                              # Import from backup

# ─── CONFIG ───
hafiz config show                                     # Current configuration
hafiz config set embedding_model nomic-embed-text-v1.5
hafiz config set llm_provider anthropic               # For entity extraction
hafiz config set db_url postgresql://...
```

### Output Modes

```bash
# Human-readable (default) — for interactive use
$ hafiz query "auth system"
╭─ Results (3 chunks, 2 entities, 1 observation) ──────────╮
│                                                           │
│ 📄 src/auth/controller.py (lines 12-45)                  │
│    AuthController handles JWT validation and user lookup  │
│                                                           │
│ 🔗 Related: JWTService → AuthController → UserTable      │
│                                                           │
│ 💡 Decision: "JWT preferred over sessions" (2026-04-10)  │
╰───────────────────────────────────────────────────────────╯

# JSON (--json flag) — for agents
$ hafiz query "auth system" --json
{
  "chunks": [...],
  "entities": [...],
  "relations": [...],
  "observations": [...],
  "summary": "The auth system is built around AuthController..."
}
```

---

## Implementation Phases

### Phase 1: The Foundation (Week 1-2)
> **Goal:** A working CLI that can store and search text chunks with vectors.
> **Philosophy:** Use LlamaIndex for chunking & embeddings, pgvector for storage. We write the CLI and config.

**Tasks:**
- [ ] Initialize project structure
- [ ] Set up Python project with `pyproject.toml`
  - Dependencies: `typer`, `rich`, `sqlalchemy[asyncio]`, `asyncpg`, `pgvector`, `llama-index-core`, `llama-index-vector-stores-postgres`, `llama-index-embeddings-fastembed`, `pydantic`, `pydantic-settings`, `alembic`
- [ ] Create dedicated `hafiz` database inside existing postgres container
- [ ] Configuration system (`hafiz.toml` or env vars):
  ```toml
  [database]
  url = "postgresql+asyncpg://postgres:***@localhost:5432/brain"

  [embedding]
  model = "nomic-ai/nomic-embed-text-v1.5"
  provider = "fastembed"   # fastembed | openai | ollama
  dimensions = 768

  [llm]
  provider = "anthropic"   # anthropic | openai | ollama | lmstudio
  model = "claude-sonnet-4-20250514"

  [workspace]
  root = "/path/to/workspace"
  projects = ["hu-manity", "noble-wave", "irshad"]
  ignore = [".git", "node_modules", "__pycache__", ".venv", "dist", "build"]
  ```
- [ ] Database schema + Alembic migrations (so schema evolves cleanly)
- [ ] Core modules (thin wrappers around LlamaIndex):
  - [ ] `database.py` — SQLAlchemy models + connection pool
  - [ ] `embeddings.py` — wraps `llama-index-embeddings-fastembed`
  - [ ] `chunker.py` — wraps LlamaIndex `SentenceSplitter` / `CodeSplitter` with language detection
  - [ ] `store.py` — wraps `llama-index-vector-stores-postgres` for chunk storage
  - [ ] `search.py` — wraps LlamaIndex `VectorStoreQuery` for similarity search
- [ ] CLI skeleton (`Typer` + `Rich` for pretty output):
  - [ ] `hafiz init` — create database, run migrations, write default config
  - [ ] `hafiz ingest <path>` — index files into chunks table
  - [ ] `hafiz query "<text>"` — vector similarity search
  - [ ] `hafiz query "<text>" --json` — machine-readable output for agents
  - [ ] `hafiz status` — DB stats, index health
  - [ ] `hafiz config show` — print current config
- [ ] Write tests for core functions
- [ ] Make installable via `pipx install -e .`

**Deliverable:** You can run `hafiz ingest ./src/` and `hafiz query "auth"` and get meaningful results.

---

### Phase 2: The Graph (Week 3-4)
> **Goal:** Extract entities and relationships to enable structural understanding.
> **Philosophy:** Use LlamaIndex's `PropertyGraphIndex` + `SchemaLLMPathExtractor`. We write the CLI commands and the display logic.

**Tasks:**
- [ ] Integrate LlamaIndex Property Graph:
  - [ ] Use `llama-index-graph-stores-postgresql` (or custom SQL if not available)
  - [ ] Configure `SchemaLLMPathExtractor` with Claude for entity/relation extraction
  - [ ] Define entity schema: Class, Function, Module, API, Table, Concept, Config
  - [ ] Define relation schema: calls, imports, inherits, depends_on, defines, reads, writes
- [ ] Ingestion pipeline upgrade:
  - [ ] `hafiz ingest` now also extracts entities + relations (not just chunks)
  - [ ] Store in `entities` and `relations` tables
  - [ ] Link chunks ↔ entities (which chunk mentions which entity)
- [ ] CLI graph commands:
  - [ ] `hafiz graph show <entity>` — display entity + direct connections
  - [ ] `hafiz graph deps <entity>` — recursive dependency tree
  - [ ] `hafiz graph dependents <entity>` — reverse dependency tree
  - [ ] `hafiz graph path <A> <B>` — shortest path between entities
  - [ ] All commands support `--json` for agent consumption
- [ ] Hybrid Search upgrade:
  - [ ] `hafiz query` now combines: vector similarity + graph context
  - [ ] Add `--depth` flag for multi-hop expansion
  - [ ] Results include: matching chunks + related entities + relationship paths

**Deliverable:** You can run `hafiz graph deps AuthController` and see a full dependency tree.

---

### Phase 3: The Wisdom Layer (Week 5-6)
> **Goal:** Store and retrieve high-level decisions, facts, and lessons — the "Mem0" equivalent.
> **Philosophy:** Use Mem0's conceptual model (facts, preferences, decisions with validity). Implementation via our own `observations` table + pgvector. No Mem0 dependency unless their library adds clear value.

**Tasks:**
- [ ] Implement the Observation system:
  - [ ] `hafiz observe "<text>" --type <type>` — store a fact/decision/lesson
  - [ ] `hafiz observe "<text>" --source agent:claude-code` — tag which agent stored it
  - [ ] `hafiz recall "<query>"` — semantic search over observations only
  - [ ] `hafiz recall --type decision --project hu-manity` — filtered recall
  - [ ] Validity tracking (`valid_from` / `valid_until` for time-bound facts)
  - [ ] Confidence scoring (agent can say "I'm 80% sure about this")
- [ ] Build the Context Synthesizer:
  - [ ] `hafiz context "task description"` — the killer feature
  - [ ] Combines: relevant chunks + graph neighbors + matching observations
  - [ ] Returns a structured "Context Bundle" (Markdown or JSON)
  - [ ] Designed to be pasted directly into an agent's system prompt or task description
- [ ] Migration scripts (one-time, in `scripts/`):
  - [ ] `import_memory.py` — parse `MEMORY.md` → observations table
  - [ ] `import_knowledgehub.py` — parse `KnowledgeHub/*.md` → tagged observations
  - [ ] `import_chromadb.py` — export ChromaDB → chunks table (hard cutover)

**Deliverable:** Running `hafiz context "implement a new Stripe webhook handler"` returns a rich bundle with code structure, past decisions about Stripe, and known gotchas.

---

### Phase 4: The Librarian (Week 7-8)
> **Goal:** Automate ingestion so Hafiz stays fresh without manual work.

**Tasks:**
- [ ] File Watcher Daemon:
  - [ ] `hafiz ingest --watch <path>` — uses `watchdog` to monitor file changes
  - [ ] Debounce rapid changes (e.g., during saves)
  - [ ] Re-index only changed files (checksum comparison)
- [ ] Git Integration:
  - [ ] `hafiz ingest --git-hook` — designed to run as `post-commit` hook
  - [ ] Diff-based indexing: only re-process files in the commit
  - [ ] Store commit metadata (hash, author, message) as observation
- [ ] Stale Data Management:
  - [ ] `hafiz prune` — detect chunks whose source files no longer exist
  - [ ] Mark entities as "stale" if their source was deleted/moved
  - [ ] Configurable retention policies

**Deliverable:** After a `git commit`, Hafiz automatically updates itself within seconds.

---

### Phase 5: Agent Integration (Week 9-10)
> **Goal:** Every agent in the ecosystem can use Hafiz seamlessly.
> **Philosophy:** Brain is standalone. Agents connect via CLI. We write thin adapters, not tight couplings.

**Tasks:**
- [ ] Bilal (OpenClaw) Integration:
  - [ ] Create an OpenClaw skill (`hafiz-memory`) that wraps the CLI
  - [ ] Replace ChromaDB calls in `AGENTS.md` with `hafiz query` / `hafiz observe`
  - [ ] Decommission `chroma-memory` skill after verification
- [ ] Claude Code Integration:
  - [ ] Create a `.claude/commands/brain-query.md` custom slash command
  - [ ] Add `hafiz` to Claude Code's allowed tools / CLAUDE.md instructions
  - [ ] Test: spawn Claude Code → verify it calls `hafiz query` autonomously
- [ ] Generic Agent Adapter:
  - [ ] Write `BRAIN_AGENT_GUIDE.md` — universal instructions any agent can follow
  - [ ] Provide a copy-paste "system prompt snippet":
    ```
    You have access to a workspace intelligence tool called `hafiz`.
    Use `hafiz query "<question>" --json` to search codebase knowledge.
    Use `hafiz context "<task>" --json` to get a full context bundle before starting work.
    Use `hafiz observe "<fact>" --type <type>` to record decisions or learnings.
    ```
  - [ ] Test with at least one other agent (Aider, Codex, etc.)
- [ ] Optional: REST API layer (FastAPI) for web dashboards or MCP server

**Deliverable:** You can spawn any coding agent and it automatically has access to the full workspace intelligence.

---

### Phase 6: Advanced Intelligence (Future)
> **Goal:** Move from "search" to "reasoning."

**Ideas (not committed, explore when Phase 5 is solid):**
- [ ] Community Detection: Auto-group related entities into "modules" or "domains"
- [ ] Impact Analysis: `hafiz impact <entity>` — "if I change this, what could break?"
- [ ] Temporal Queries: "What did the auth module look like 3 months ago?"
- [ ] Cross-Project Learning: "How did we solve rate-limiting in Project A? Apply that to Project B."
- [ ] MCP Server: Expose Brain as an MCP tool server for native LLM integration
- [ ] Dashboard: A simple web UI showing the knowledge graph visually

---

## Technology Stack

| Component | Technology | Role | We Build? |
|---|---|---|---|
| Language | Python 3.12+ | Everything | — |
| Database | PostgreSQL 18 + pgvector | Vector + relational storage | **No** — existing container |
| Embeddings | `llama-index-embeddings-fastembed` (nomic-embed-text-v1.5) | Text → vectors | **No** — LlamaIndex wrapper |
| Chunking | `llama-index-core` (SentenceSplitter, CodeSplitter) | File → chunks | **No** — LlamaIndex built-in |
| Vector Store | `llama-index-vector-stores-postgres` | Chunk storage + similarity search | **No** — LlamaIndex connector |
| Graph Extraction | `llama-index-core` (PropertyGraphIndex, SchemaLLMPathExtractor) | Entity + relation extraction | **No** — LlamaIndex built-in |
| LLM (extraction) | `llama-index-llms-anthropic` (configurable) | Powers entity extraction | **No** — LlamaIndex connector |
| CLI | Typer + Rich | User & agent interface | **Yes** — our code |
| Config | pydantic-settings + TOML | Configuration management | **Yes** — our code |
| ORM | SQLAlchemy 2.0 (async) | DB models, migrations | **Yes** — our code (thin) |
| Migrations | Alembic | Schema versioning | **Yes** — our code (thin) |
| File Watching | watchdog | Auto-reindex on change | **No** — existing library |
| Testing | pytest + pytest-asyncio | Tests | **Yes** — our code |
| Packaging | pipx installable | System-wide CLI | Standard Python packaging |

**What we actually write:** ~20% of the codebase. The CLI, config, glue code, and agent adapters. The rest is LlamaIndex + pgvector doing the real work.

---

## Project Structure

```
hafiz/
├── ROADMAP.md              # This file
├── README.md               # User-facing docs
├── pyproject.toml          # Package config + dependencies
├── hafiz/
│   ├── __init__.py
│   ├── cli.py              # Typer CLI entry point
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py       # Configuration management
│   │   ├── database.py     # SQLAlchemy models + connection
│   │   ├── embeddings.py   # Embedding service (fastembed)
│   │   ├── chunker.py      # File → chunks logic
│   │   ├── extractor.py    # LLM entity/relation extraction
│   │   └── search.py       # Hybrid search engine
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── query.py        # hafiz query
│   │   ├── graph.py        # hafiz graph
│   │   ├── ingest.py       # hafiz ingest
│   │   ├── observe.py      # hafiz observe / recall
│   │   └── maintenance.py  # hafiz status / prune / export
│   └── integrations/
│       ├── __init__.py
│       ├── openclaw.py     # Bilal skill adapter
│       └── claude_code.py  # Claude Code tool definition
├── tests/
│   ├── test_chunker.py
│   ├── test_search.py
│   ├── test_graph.py
│   └── test_cli.py
├── scripts/
│   ├── migrate.py          # DB migration helper
│   ├── import_memory.py    # One-time MEMORY.md import
│   └── import_chromadb.py  # One-time ChromaDB migration
└── docker/
    └── docker-compose.yml  # If we want a dedicated DB instance
```

---

## Success Criteria

1. **Any agent can query Hafiz in under 2 seconds.**
2. **Hafiz stays fresh automatically** — no manual "re-index" needed.
3. **Zero vendor lock-in** — runs entirely on your machine, your database, your models.
4. **One command to install:** `pipx install ./brain` and it's available everywhere.
5. **Observations persist across sessions** — what Bilal learns today, Claude Code knows tomorrow.

---

## Decisions (Locked In — 2026-04-14)

| # | Question | Decision |
|---|---|---|
| 1 | Database | **Dedicated DB** inside the existing `postgres` container (port 5432). Separate database name (e.g., `hafiz`). |
| 2 | Embedding model | **nomic-embed-text-v1.5** (768 dims, fastembed ONNX, local). |
| 3 | LLM for extraction | **Claude** (Anthropic API). Configurable — can swap to local Qwen or any provider. |
| 4 | Scope | **Workspace-scoped.** One Brain instance per workspace (VSCode-style). A workspace contains one or more projects. |
| 5 | ChromaDB | **Hard cutover.** Migrate once, then decommission ChromaDB. |

### Design Principles (Non-Negotiable)

- **Standalone project.** Brain is an independent tool — not coupled to OpenClaw, Bilal, or any specific agent. Any agent that can run a CLI command can use it.
- **Everything is configurable.** DB connection, embedding model, LLM provider, workspace path — all via config file or environment variables. No hardcoded values.
- **Library-first.** We do NOT reinvent: chunking (LlamaIndex), embeddings (fastembed), vector search (pgvector), graph extraction (LlamaIndex PropertyGraph). We write: the CLI, the config layer, the agent integration glue.
- **Workspace = Unit of Knowledge.** A "workspace" is a root directory (like a VSCode workspace) that may contain multiple projects. Brain indexes the workspace and tags chunks/entities by project.

---

*Created: 2026-04-14*
*Author: Bilal + Irshad*
*Status: Planning — Decisions Locked*
