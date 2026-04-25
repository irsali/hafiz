# 🧠 Hafiz (حافظ) — The Sovereign Second Brain

> *Named after the tradition of the Hafiz — one who preserves, understands, and recalls with precision.*
> A self-sovereign, CLI-first second brain that any AI agent — or any command — can read from and write to.
> **Your codebase, your research, your decisions, your notes, your conversations — anything you do on your machine, remembered.** Always on, always local, always yours.

---

## Vision

Every tool and agent you use — Claude Code, Cursor, Copilot, Aider, a research assistant, a writing assistant, or anything tomorrow — connects to **one shared second brain**. Not just your code: your notes, decisions, research, conversations, clips, and anything else you want to remember. No more scattered `.md` files, no more "I forgot what we decided last week." Hafiz is always on, always fresh, and always yours.

> **Status (2026-04-21):** Phases 1–5 shipped, plus the temporal/capture/distill layer (journal, note, capture, session, distill, expiry, supersession — see [workitems/done/temporal-session-awareness.md](../workitems/done/temporal-session-awareness.md)). Core CLI surface is stable; the product is dogfooded daily. One-time migration scripts are still open. See [Shipped](#shipped) and [Open work](#open-work) below.
>
> **Dropped from scope (as of 2026-04-21):** REST API layer and MCP server. Hafiz is intentionally CLI-only; agents integrate via `hafiz` + `--json`. This decision replaces the earlier "future" positioning of those surfaces.
>
> This file is the **product vision + future backlog**. It is *not* the development guide — see [CLAUDE.md](../CLAUDE.md) for conventions, layout, and how to add a command. [commands.md](commands.md) is the source of truth for command shapes and flags.

---

## What Hafiz Is

Hafiz is a second brain for everything you do on your system. Code is the first and best-proven workload, but the design is deliberately broader:

- **Any content** — source code, Markdown notes, exported chats, research clips, meeting logs, terminal transcripts, bookmarks, screenshots. If it can be parsed into units, it can live here.
- **Any agent** — Claude Code, Cursor, Copilot, Aider, a research assistant, a writing assistant, a personal agent. One CLI contract, many domains.
- **Any command** — humans and agents use the same CLI. Ingestion and retrieval are first-class surfaces; the web is not required.

New content types are parsers, not schema changes. New agents are CLI callers, not integrations. The architecture resists vertical lock-in by construction.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     YOUR DIGITAL LIFE                    │
│   Code • Docs • Notes • Chats • Mail • Research          │
│   Clips • Terminal • Bookmarks • Anything                │
└──────────────────────────┬──────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  LIBRARIAN  │  (Ingestion Engine)
                    │  Parsers +  │  - Parser registry (AST + prose + fallback)
                    │  Agents     │  - Agent enrichment (meaning, decisions)
                    └──────┬──────┘  - Git-aware delta ingest
                           │
              ┌────────────▼────────────────┐
              │    POSTGRESQL + pgvector     │
              │                             │
              │  ┌─────────┐ ┌───────────┐  │
              │  │  Units  │ │ Revisions │  │
              │  │(identity│ │ (versioned│  │
              │  │ +kind)  │ │  bodies)  │  │
              │  └─────────┘ └───────────┘  │
              │  ┌─────────┐ ┌───────────┐  │
              │  │  Edges  │ │Annotations│  │
              │  │(struct+ │ │ (meaning, │  │
              │  │semantic)│ │ decisions)│  │
              │  └─────────┘ └───────────┘  │
              │  ┌─────────┐ ┌───────────┐  │
              │  │  Files  │ │  Commits  │  │
              │  │ (roots) │ │ (git axis)│  │
              │  └─────────┘ └───────────┘  │
              │  ┌───────────────────────┐  │
              │  │      Embeddings       │  │
              │  │  (vector search idx)  │  │
              │  └───────────────────────┘  │
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

**Seven tables**, all in one PostgreSQL database with `pgvector`. Alembic owns the schema — see [alembic/versions/](alembic/versions/).

Built on two invariants. *Identity is stable, body evolves* — every fact that can change over time is an immutable **unit** with append-only **revisions**; branch switches, commit hops, and edits become diffs, never destructive overwrites. *Bodies are canonical, embeddings are an index* — the vector search layer sits in a separate `embeddings` table (1:N from revisions), so oversized units (long docs, huge functions, whole-file fallbacks) get multiple embedded parts while small units get one. Agent annotations reference units by stable identity, surviving body changes without orphaning; editing one paragraph of a ten-paragraph note re-embeds one part, not ten.

```sql
-- ─── Files — one row per file ever seen in the project ─────
files          (id, project, path, language, first_seen_commit,
                last_seen_commit, valid_until, metadata jsonb)

-- ─── Units — stable identity of an addressable thing ──────
-- Functions, classes, Markdown headings, config blocks, mail
-- messages, chat turns — anything parseable. `kind` is
-- namespaced by convention: code.function, code.class,
-- doc.heading, mail.message, chat.turn, …
units          (id, file_id, kind, name, parent_name, identity_key UNIQUE,
                first_seen_commit, last_seen_commit, valid_until)

-- ─── Unit revisions — versioned body (append-only) ─────────
-- Partial unique: ≤1 current revision per unit.
-- source ∈ {ast, parser, agent, user}.
unit_revisions (id, unit_id, content, content_hash,
                line_start, line_end, commit_hash, source,
                observed_at, superseded_at, superseded_by)

-- ─── Embeddings — vector search index over revisions ──────
-- 1:N from unit_revisions (UNIQUE on (unit_revision_id, part_index),
-- FK cascade on revision delete). Small units: one row per revision.
-- Oversized units (long docs, huge functions, whole-file fallback):
-- many rows with token spans. Each part is content-hashed independently,
-- so partial edits re-embed only the affected parts. Vector search
-- hits this table and joins back to revisions → units → files.
embeddings     (id, unit_revision_id, part_index, content,
                content_hash, embedding vector(768),
                token_span_start, token_span_end)

-- ─── Edges — relations between units (append-only) ────────
-- source ∈ {ast, agent, user}. target_name kept for unresolved
-- or external references (e.g. stdlib, third-party imports).
edges          (id, source_unit_id, target_unit_id, target_name,
                relation, source, evidence, weight, commit_hash,
                observed_at, superseded_at)

-- ─── Annotations — decisions, facts, learnings ────────────
-- Today's "observations", renamed. May link to a unit (`unit_id`)
-- or float free. Temporal primitives (`valid_from`, `valid_until`,
-- `supersedes_id`) unchanged.
annotations    (id, content, embedding vector(768), kind, source,
                project, tags text[], confidence, unit_id,
                session_id, task, commit_hash, valid_from,
                valid_until, supersedes_id, metadata jsonb)

-- ─── Commits — git axis as a first-class citizen ─────────
commits        (hash PK, project, author, committed_at, summary)
```

**Ownership rule.** Parsers own structure (entities, imports, calls, inherits — `source='ast'`); agents own meaning (concepts, patterns, decisions, workarounds — `source='agent'`). `hafiz extract import` rejects AST-territory kinds and relation types. Non-duplication is enforced at the data layer.

**Indexes in place** (post-restructure): `project` / `path` on files; `identity_key` / `kind` / `file_id` on units; `unit_id` / `content_hash` / `commit_hash` on unit_revisions with a partial unique for the current revision; `unit_revision_id` on embeddings with a unique on `(unit_revision_id, part_index)`; `source_unit_id` / `target_unit_id` / `relation` on edges; `kind` / `unit_id` / `commit_hash` on annotations; `project` / `committed_at` on commits.

**Indexes deferred until scale demands them:** `ivfflat` on `embeddings.embedding` and `annotations.embedding`; GIN on `annotations.tags`. Trigger to add them: second-brain corpora (millions of units across domains) or the first noticeable latency regression.

---

## CLI Surface

`hafiz --help` is the authoritative list; [commands.md](commands.md) documents flags, JSON shapes, and when each command is agent-driven vs. human-driven. Current shape:

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

**Output modes.** Human (Rich panels, tables, trees) by default. `--json` on every user-facing command; shapes are stable and documented in [commands.md](commands.md).

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
- [x] [`docs/agents.md`](agents.md) — universal agent integration playbook (stale; see `architecture.md` + `skills.md`)
- [x] [`hafiz/data/agents/skills.md`](../hafiz/data/agents/skills.md) — **Layer 1 stable contract** (see [Two-Layer Stability Model](../CLAUDE.md))

---

## Open work

Items actively on deck — a mix of leftovers from the original roadmap and newer structural work:

- [ ] **Phase 6 — Structural Grounding (AST layer + schema restructure + second-brain scope)** — see [workitems/active/structural-grounding.md](../workitems/active/structural-grounding.md). Splits structure from meaning: AST parsers own code entities/edges, agents own semantic annotations. Greenfield schema (`units` / `unit_revisions` / `edges` / `annotations` / `files` / `commits`). Branch-switching becomes a delta. Agent contract bumps to v2. Opens the pipeline to non-code domains via a Parser Protocol.
- [ ] **One-time migration scripts** — `import_memory.py` (MEMORY.md → observations), `import_knowledgehub.py`, `import_chromadb.py`. No top-level `scripts/` directory exists today.
- [ ] **ChromaDB hard cutover** — not executed; no ChromaDB dependency to decommission in this repo
- [ ] **Retention policies** for stale data — `prune` is on-demand only; no age-based policy
- [ ] **Test coverage for a second agent** (Aider, Codex) beyond the three currently registered

---

## Future (Phase 7+)

Ideas, not committed — most are unblocked by the Structural Grounding work (Phase 6).

- [ ] **Tree-sitter parsers** — Go, TypeScript, Rust, and other languages as drop-in `Parser` Protocol implementations. No schema churn.
- [ ] **Domain parsers — the second-brain build-out** — mail (`mbox`, Gmail export), chat (Slack/Discord exports), clip (read-it-later, Instapaper), research (Zotero), meeting (transcripts), terminal (shell history), OCR'd screenshots. Each is a parser, not a migration.
- [ ] **Code-history reconstruction** — `hafiz history <unit>` to show revisions across commits; `--at-commit <sha>` on queries. Schema already supports it after Phase 6; this is surface work.
- [ ] **Personal-tier privacy** — encryption-at-rest, per-project access modes. Triggered by the first non-code domain landing.
- [ ] **Community Detection** — auto-group related units into "modules" or "domains".
- [ ] **Cross-Domain Learning** — "How did I reason about X in code? Apply that to my research notes." (Generalized from the original *Cross-Project Learning* under second-brain scope.)
- [ ] **Dashboard** — simple web UI showing the knowledge graph visually.
- [ ] **Embedding Model Migration** — install-time model choice + `hafiz embeddings migrate` to re-embed corpus when switching models (enforce one model per DB).
- [ ] **Markdown Dump/Export** — export captures, transcripts, observations, and journal entries as `.md` files for portability, backup, and human review.
- [ ] **DB export/import** — `hafiz export --format json` / `hafiz import` for backup and portability.

---

## Success Criteria

1. **Any agent can query Hafiz in under 2 seconds.**
2. **Hafiz stays fresh automatically** — no manual "re-index" needed (watcher + git hooks).
3. **Zero vendor lock-in** — runs entirely on your machine, your database, your models.
4. **One command to install:** `pipx install hafiz` and it's available everywhere.
5. **Observations persist across sessions** — what Cursor learns today, Claude Code knows tomorrow.
6. **Any agent, any content, one query surface.** A research agent saving papers, a coding agent committing features, and a writing agent drafting notes all read and write through the same Hafiz — no per-domain API, no per-agent adapter beyond `hafiz agent install`.
7. **Branch switches re-index in seconds, not minutes** — proportional to the delta, not the corpus. (Landing with Phase 6.)

---

## Design Principles (Non-Negotiable)

- **Standalone project.** Hafiz is an independent tool — not coupled to any specific agent. Any agent that can run a CLI command can use it.
- **Two-layer stability.** Layer 1 ([skills.md](../hafiz/data/agents/skills.md)) is the stable contract with every agent; Layer 2 (`hafiz review` and friends) is free to iterate. See [CLAUDE.md](../CLAUDE.md).
- **Everything is configurable.** DB connection, embedding model, workspace path — all via `hafiz.toml` or environment variables. No hardcoded values.
- **Library-first, not library-captured.** We lean on LlamaIndex for chunking primitives and fastembed for embeddings, but we own the storage, search, graph, and CLI layers in plain SQL + NetworkX. Wrappers earn their place.
- **No API-key-dependent paths.** Entity/relation extraction is agent-driven by design (direct-LLM path removed in `9440b3f`). The agent *is* the brain.
- **Workspace = Unit of Knowledge.** A "workspace" is a root directory (like a VSCode workspace) that may contain multiple projects. Hafiz indexes the workspace and tags chunks/entities by project. `--workspace` scopes to sibling projects in the parent directory.
- **CLI surface is the product.** Human output must be scannable; `--json` shapes must be stable and documented in [commands.md](commands.md).
- **Identity vs body is a first-class split.** Every fact that can evolve over time has a stable identity and append-only revisions. Branch switches, commit hops, and edits are diffs; nothing is destructively overwritten. Agent annotations reference units by identity, so they survive body changes without orphaning.
- **Parsing is a Protocol, not a monolith.** New content types — new languages, new domains — are additive `Parser` implementations against a stable shape. The schema, ingest pipeline, graph, and query surface are language- and domain-agnostic by construction. Parsers register via Python entry points (`hafiz.parsers` group); installing a parser *is* enabling AST for that language — no config knob, no flag. Unregistered languages fall through to the whole-file fallback.
- **Domain-agnostic by design.** Units don't know they're code. `code.function`, `doc.heading`, `mail.message`, `chat.turn` all live in one table with one query surface. A new domain is a parser, never a schema migration.
- **Non-duplication at the data layer.** Parsers own structural facts (`source='ast'`); agents own semantic facts (`source='agent'`). The import path rejects duplicates; ownership is enforced, not requested.

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
| 7 | Structural extraction | **AST-first for code, agents for meaning.** Parsers own structure (entities, calls, imports, inherits — `source='ast'`); agents own concepts, patterns, decisions, workarounds (`source='agent'`). Non-duplication enforced at the data layer; `hafiz extract import` rejects AST-territory facts. *(Decided 2026-04-21.)* |
| 8 | Schema reshape | **Greenfield, one-shot.** Pre-1.0 grant: old tables (`chunks` / `entities` / `relations` / `observations`) are dropped and replaced with `units` / `unit_revisions` / `embeddings` / `edges` / `annotations` / `files` / `commits`. Identity, body, and embedding are separated (embeddings are 1:N from revisions, enabling partial re-embedding for oversized units). Users re-ingest after upgrade. No backfill shim. *(Decided 2026-04-21.)* |
| 9 | Product scope | **Second brain for the whole system, not just code.** Code is the first workload; mail, chat, research, clip, meeting, terminal, OCR parsers are additive with zero schema change. `units.kind` is namespaced `domain.subtype` by convention (`code.function`, `doc.heading`, `mail.message`, …). *(Decided 2026-04-21.)* |
| 10 | Agent contract | **Hard v2 bump** to match the new schema. `extract export/import` schemas change; `skills.md` is versioned. `hafiz agent install` warns on out-of-date splice. One-release deprecation window. *(Decided 2026-04-21.)* |
| 11 | AST configurability | **Capability-based, not toggleable.** No `[parsers] enabled/disabled` config knob. A file's language is handled by whichever parser is registered; unregistered languages fall through to `WholeFileParser` (file = one unit), and agent extraction fills in structural annotations. Third-party parsers register via the `hafiz.parsers` Python entry-point group — `pip install hafiz-parser-go` is how you turn on Go AST. `hafiz parsers list` exposes what's active. *(Decided 2026-04-21.)* |

---

*Created: 2026-04-14 · Last restructure: 2026-04-21 (structural grounding + second-brain framing)*
*Authors: Irshad Ali
*Status: Phases 1–5 shipped · Phase 6 and open-work items active*
