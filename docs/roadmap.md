# 🧠 Hafiz (حافظ) — The Sovereign Second Brain

> *Named after the tradition of the Hafiz — one who preserves, understands, and recalls with precision.*
> A self-sovereign, CLI-first second brain that any AI agent — or any command — can read from and write to.
> **Your codebase, your research, your decisions, your notes, your conversations — anything you do on your machine, remembered.** Always on, always local, always yours.

---

## Vision

Every tool and agent you use — Claude Code, Cursor, Copilot, Aider, a research assistant, a writing assistant, or anything tomorrow — connects to **one shared second brain**. Not just your code: your notes, decisions, research, conversations, clips, and anything else you want to remember. No more scattered `.md` files, no more "I forgot what we decided last week." Hafiz is always on, always fresh, and always yours.

> **Status (2026-04-25):** Phases 1–6 shipped, plus the temporal/capture/distill layer (journal, note, capture, session, distill, expiry, supersession — see [workitems/done/temporal-session-awareness.md](../workitems/done/temporal-session-awareness.md)) and the tunable registry / self-tuning surface (see [workitems/done/tunable-registry.md](../workitems/done/tunable-registry.md)). Core CLI surface is stable; the product is dogfooded daily. One-time migration scripts are still open. See [Shipped](#shipped) and [Open work](#open-work) below.
>
> **Dropped from scope (as of 2026-04-21):** REST API layer. Hafiz's primary contract is `hafiz` + `--json`; there is no HTTP surface and none planned.
>
> **MCP reinstated (2026-09-06), reversing half of that decision.** `hafiz mcp` serves the knowledge layer to agents that *cannot* shell out — Cursor chat, desktop and hosted clients — over **stdio only**: no HTTP, no SSE, no listening socket. It is 13 tools giving full capability parity of the knowledge surface, generated from one registry keyed to the same core functions the CLI calls, so it cannot drift from the CLI contract. Install/admin, destructive and long-running ingest commands are deliberately absent: an MCP client calls tools without asking anyone, so those stay on the CLI where a human is present. Needs the optional extra — `pip install hafiz[mcp]`.
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
              │  SQLITE+vec  or  POSTGRES   │
              │                             │
              │  ┌─────────┐ ┌───────────┐  │
              │  │  Units  │ │ Revisions │  │
              │  │(identity│ │ (versioned│  │
              │  │ +kind)  │ │  bodies)  │  │
              │  └─────────┘ └───────────┘  │
              │  ┌───────────┐ ┌─────────┐  │
              │  │Annotations│ │  Files  │  │
              │  │ (meaning) │ │ (roots) │  │
              │  └───────────┘ └─────────┘  │
              │  ┌─────────┐ ┌───────────┐  │
              │  │ Commits │ │Embeddings │  │
              │  │(git axis│ │ (vector   │  │
              │  │         │ │  search)  │  │
              │  └─────────┘ └───────────┘  │
              └────────────┬────────────────┘
                           │
                    ┌──────▼──────┐
                    │  BRAIN CORE │  (Python Library)
                    │  pgvector   │  - Vector search
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

**Six knowledge-layer tables**, in one database — an embedded SQLite file with `sqlite-vec` by default, or PostgreSQL with `pgvector` for a shared install. The schema, the CLI surface, the `--json` shapes and the agent contract are identical on both; [hafiz/core/dialect.py](../hafiz/core/dialect.py) is the one place that knows which backend is in use, and the full test suite runs against each. Alembic owns the schema — see [alembic/versions/](alembic/versions/). Move a store between backends with `hafiz migrate-backend`.

Built on two invariants. *Identity is stable, body evolves* — every fact that can change over time is an immutable **unit** with append-only **revisions**; branch switches, commit hops, and edits become diffs, never destructive overwrites. *Bodies are canonical, embeddings are an index* — the vector search layer sits in a separate `embeddings` table (1:N from revisions), so oversized units (long docs, whole-file fallbacks) get multiple embedded parts while small units get one. Agent annotations reference units by stable identity, surviving body changes without orphaning; editing one paragraph of a ten-paragraph note re-embeds one part, not ten.

```sql
-- ─── Files — one row per file ever seen in the project ─────
files          (id, project, path, language, first_seen_commit,
                last_seen_commit, valid_until, metadata jsonb)

-- ─── Units — stable identity of an addressable thing ──────
-- Markdown headings/paragraphs, whole files, mail messages —
-- anything parseable. `kind` is namespaced by convention:
-- doc.heading, doc.paragraph, file.raw, mail.message, …
units          (id, file_id, kind, name, parent_name, identity_key UNIQUE,
                first_seen_commit, last_seen_commit, valid_until)

-- ─── Unit revisions — versioned body (append-only) ─────────
-- Partial unique: ≤1 current revision per unit.
-- source ∈ {ast, parser, agent, user} (ast: a third-party structural parser via the plugin entry point).
unit_revisions (id, unit_id, content, content_hash,
                line_start, line_end, commit_hash, source,
                observed_at, superseded_at, superseded_by)

-- ─── Embeddings — vector search index over revisions ──────
-- 1:N from unit_revisions (UNIQUE on (unit_revision_id, part_index),
-- FK cascade on revision delete). Small units: one row per revision.
-- Oversized units (long docs, whole-file fallback):
-- many rows with token spans. Each part is content-hashed independently,
-- so partial edits re-embed only the affected parts. Vector search
-- hits this table and joins back to revisions → units → files.
embeddings     (id, unit_revision_id, part_index, content,
                content_hash, embedding vector(768),
                token_span_start, token_span_end)

-- ─── Annotations — decisions, facts, learnings ────────────
-- May link to a unit (`unit_id`) or float free. Temporal primitives
-- (`valid_from`, `valid_until`, `supersedes_id`) unchanged.
annotations    (id, content, embedding vector(768), kind, source,
                project, tags text[], confidence, unit_id,
                session_id, task, commit_hash, valid_from,
                valid_until, supersedes_id, metadata jsonb)

-- ─── Commits — git axis as a first-class citizen ─────────
commits        (hash PK, project, author, committed_at, summary)
```

**Ownership rule.** Parsers own indexed content (prose split into `doc.*`; everything else as `file.raw`); agents own meaning (concepts, patterns, decisions, workarounds) via `hafiz observe` / `hafiz note`.

**Indexes in place** (post-restructure): `project` / `path` on files; `identity_key` / `kind` / `file_id` on units; `unit_id` / `content_hash` / `commit_hash` on unit_revisions with a partial unique for the current revision; `unit_revision_id` on embeddings with a unique on `(unit_revision_id, part_index)`; `kind` / `unit_id` / `commit_hash` on annotations; `project` / `committed_at` on commits.

**Indexes deferred until scale demands them:** `ivfflat` on `embeddings.embedding` and `annotations.embedding`; GIN on `annotations.tags`. Trigger to add them: second-brain corpora (millions of units across domains) or the first noticeable latency regression.

---

## CLI Surface

`hafiz --help` is the authoritative list; [commands.md](commands.md) documents flags, JSON shapes, and when each command is agent-driven vs. human-driven. Current shape:

```bash
# ─── SETUP ───
hafiz init                          # create the store (a file, or a Postgres schema)
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
hafiz query   "<text>" --observations [--type] …   # search annotations instead of chunks
hafiz context "<task>" [--project|--workspace] [--json]   # indexed content + observations

# ─── OBSERVE ───
hafiz observe "<text>" [--type] [--source] [--project] [--tags] [--confidence] [--json]

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
- **Later removed (2026-11):** the whole graph + extract surface — see [Removed — Code index](#code-index-2026-11).

### Phase 3 — The Wisdom Layer
- [x] `observe` with `--type`, `--source`, `--project`, `--tags`, `--confidence`
- [x] `query --observations` (semantic search over observations only; shipped as `--recall`, renamed, alias later removed) with `--type` / `--project` filters
- [x] `hafiz context "<task>"` — the killer feature. Markdown + JSON bundle: relevant indexed content + matching observations (graph neighbourhood removed 2026-11)
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

### Phase 6 — Structural Grounding (AST layer + greenfield schema + second-brain scope)
See [workitems/done/structural-grounding.md](../workitems/done/structural-grounding.md) for the full design + closure record.
- [x] **Greenfield schema** — originally seven tables including `edges`; today six knowledge-layer tables after the 2026-11 code-index removal (`files`, `units`, `unit_revisions`, `embeddings`, `annotations`, `commits`). Identity/body/embedding split into three concerns; partial edits to long documents re-embed only the changed parts.
- [x] **Parser Protocol + concrete parsers** — `prose` (Markdown) and `whole_file` fallback remain. Third-party parsers plug in via Python entry points (`hafiz.parsers` group); no config edits required.
- [x] ~~**JavaScript / TypeScript parser**~~ — **removed 2026-11** with the code index (was optional `hafiz[js]` / `tree_sitter_js`). Code files index as `file.raw` via whole-file.
- [x] **Diff-driven ingest** — parser-driven, hash-aligned, skips unchanged units. Re-ingests parse only files changed since `last_seen_commit`.
- [x] ~~**Edge resolver + graph rewire**~~ — **removed 2026-11** with the code index (`hafiz graph` and the `edges` table).
- [x] **Git-axis as first class** — `commits` table populated on every ingest; rewrite resilience (rebase/amend/squash/force-push) via `post-rewrite` hook + reconcile-on-ingest. `commit_hash` advisory, never load-bearing.
- [x] ~~**Agent extraction v2**~~ — **removed 2026-11** (`hafiz extract export/import`). Agents record meaning with `observe` / `note` only.
- [x] **Observability** — `hafiz parsers list`, `hafiz status --diagnose` extended with parser coverage and tombstone counts.
- **Carry-overs (own follow-ups, not Phase 6 debt):** `hafiz review` rewire owned by [self-review-curation-loop](../workitems/active/self-review-curation-loop.md).

---

## Removed

### Code index (2026-11)

Removed the Python AST parser (`python_ast`), the JS/TS tree-sitter parser (`tree_sitter_js`), the call/import/inherits graph they produced, `hafiz graph` (show/deps/impact/path/rank/stats), `hafiz extract` (export/import), the `hafiz_graph` MCP tool, the `edges` table, and deps `networkx` / `scipy` / `anthropic` / optional `hafiz[js]`. Code files are still indexed whole-file as `file.raw` and remain searchable with `hafiz query`. Migration `0011` deletes stored `code.*` units and clears last-seen commits so the next ingest of affected projects re-reads every file once (unchanged docs skipped by hash).

**Why (measured over 120 days of agent sessions):** `graph` and `extract` were run 0 times (versus hundreds of `observe` and dozens of `query`); the instruction to run `graph impact` before every change was never followed; code was about 8% of stored units and no note was ever attached to a code unit; the graph cost every install networkx and scipy. Agents read code with grep/read tools and git, which are always current, while the index went stale fastest exactly where work was happening.

**Reopen only if** agents repeatedly ask structural questions ("who calls X", "what breaks if I change this") that grep cannot answer — and then design from that need rather than restoring the old parsers.

---

## Open work

Items actively on deck — a mix of leftovers from the original roadmap and newer structural work:

- [ ] **One-time migration scripts** — `import_memory.py` (MEMORY.md → observations), `import_knowledgehub.py`, `import_chromadb.py`. No top-level `scripts/` directory exists today.
- [ ] **ChromaDB hard cutover** — not executed; no ChromaDB dependency to decommission in this repo
- [ ] **Retention policies** for stale data — `prune` is on-demand only; no age-based policy
- [ ] **Test coverage for a second agent** (Aider, Codex) beyond the three currently registered

---

## Future (Phase 7+)

Ideas, not committed — most are unblocked by the Structural Grounding work (Phase 6).

- [ ] **Domain parsers — the second-brain build-out** — mail (`mbox`, Gmail export), chat (Slack/Discord exports), clip (read-it-later, Instapaper), research (Zotero), meeting (transcripts), terminal (shell history), OCR'd screenshots. Each is a parser, not a migration.
- [ ] **Code-history reconstruction** — `hafiz history <unit>` to show revisions across commits; `--at-commit <sha>` on queries. Schema already supports it after Phase 6; this is surface work.
- [ ] **Personal-tier privacy** — encryption-at-rest, per-project access modes. Triggered by the first non-code domain landing.
- [ ] **Community Detection** — auto-group related units into "modules" or "domains".
- [ ] **Cross-Domain Learning** — "How did I reason about X in code? Apply that to my research notes." (Generalized from the original *Cross-Project Learning* under second-brain scope.)
- [ ] **Dashboard** — simple web UI showing the knowledge graph visually.
- [ ] **Embedding Model Migration** — install-time model choice + `hafiz embeddings migrate` to re-embed corpus when switching models (enforce one model per DB).
- [x] **Markdown Dump/Export** — `hafiz export` dumps observations (+ optional transcripts) as `.md` for portability, backup, and human review. Code excluded (git is its sovereign copy). The eject complement to `forget`. See [workitems/active/sovereignty-export.md](../workitems/active/sovereignty-export.md).
- [ ] **DB export/import** — `hafiz export --format json` ships (lossless JSONL); the round-trip `hafiz import` is still open — see [workitems/active/sovereignty-export.md](../workitems/active/sovereignty-export.md) Phase 2.
- [ ] **Unify `device_state` onto the Tunable registry** — collapse the GPU/CPU sticky-fallback into the same `Tunable` resolution chain as `embedding.max_part_chars`. Deferred from [workitems/done/tunable-registry.md](../workitems/done/tunable-registry.md) Phase 4 until post-2.0.0; revisit when adding the third tunable forces a decision.

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
- **Library-first, not library-captured.** We lean on fastembed for embeddings, but we own the storage, search, and CLI layers in plain SQL. Wrappers earn their place.
- **No API-key-dependent paths.** Agents record meaning with `hafiz observe` / `hafiz note` (direct-LLM extraction path removed in `9440b3f`). The agent *is* the brain.
- **Workspace = Unit of Knowledge.** A "workspace" is a root directory (like a VSCode workspace) that may contain multiple projects. Hafiz indexes the workspace and tags units by project. `--workspace` scopes to sibling projects in the parent directory.
- **CLI surface is the product.** Human output must be scannable; `--json` shapes must be stable and documented in [commands.md](commands.md).
- **Identity vs body is a first-class split.** Every fact that can evolve over time has a stable identity and append-only revisions. Branch switches, commit hops, and edits are diffs; nothing is destructively overwritten. Agent annotations reference units by identity, so they survive body changes without orphaning.
- **Parsing is a Protocol, not a monolith.** New content types — new domains — are additive `Parser` implementations against a stable shape. The schema, ingest pipeline, and query surface are language- and domain-agnostic by construction. Parsers register via Python entry points (`hafiz.parsers` group). Unregistered types fall through to the whole-file fallback.
- **Domain-agnostic by design.** Units don't know they're code. `doc.heading`, `file.raw`, `mail.message` all live in one table with one query surface. A new domain is a parser, never a schema migration.
- **Ownership at the data layer.** Parsers own indexed content; agents own meaning via annotations.

---

## Decisions (Locked In)

| # | Question | Decision |
|---|---|---|
| 1 | Database | **Dedicated DB** inside the existing `postgres` container (port 5432). Separate database name (`hafiz`). |
| 2 | Embedding model | **nomic-embed-text-v1.5** (768 dims, fastembed ONNX, local). GPU via `fastembed-gpu` extras. |
| 3 | Entity extraction | **Removed 2026-11** with the code index. Agents record meaning with `observe` / `note` only. *(Had been agent-driven `extract export` → agent → `extract import` since 2026-04.)* |
| 4 | Scope | **Workspace-scoped.** One Hafiz instance per workspace; a workspace contains one or more projects. `--workspace` fans out across siblings. |
| 5 | ChromaDB | **Hard cutover** when migration lands. No dual-read. |
| 6 | Stability model | **Two layers.** `skills.md` is the stable agent contract; `review` and friends evolve freely in Layer 2. |
| 7 | Structural extraction | **Superseded 2026-11.** Code AST + call graph removed; see [Removed — Code index](#code-index-2026-11). Parsers own indexed content (prose + whole-file); agents own meaning. *(Original 2026-04-21 decision was AST-first for code.)* |
| 8 | Schema reshape | **Greenfield, one-shot.** Pre-1.0 grant: old tables (`chunks` / `entities` / `relations` / `observations`) dropped for `units` / `unit_revisions` / `embeddings` / `annotations` / `files` / `commits` (plus `edges`, later removed in 2026-11). Identity, body, and embedding are separated. Users re-ingest after upgrade. No backfill shim. *(Decided 2026-04-21.)* |
| 9 | Product scope | **Second brain for the whole system, not just code.** Mail, chat, research, clip, meeting, terminal, OCR parsers are additive with zero schema change. `units.kind` is namespaced `domain.subtype` by convention (`doc.heading`, `file.raw`, `mail.message`, …). *(Decided 2026-04-21.)* |
| 10 | Agent contract | **Versioned `skills.md`.** `hafiz agent install` warns on out-of-date splice. *(Extract v2 contract removed 2026-11.)* |
| 11 | Parser configurability | **Capability-based, not toggleable.** No `[parsers] enabled/disabled` config knob. A file's type is handled by whichever parser is registered; unregistered types fall through to `WholeFileParser` (file = one unit). Third-party parsers register via the `hafiz.parsers` Python entry-point group. `hafiz parsers list` exposes what's active. *(Decided 2026-04-21.)* |
| 12 | Code index | **Removed 2026-11.** See [Removed — Code index](#code-index-2026-11). |

---

*Created: 2026-04-14 · Last restructure: 2026-04-21 (structural grounding + second-brain framing)*
*Authors: Irshad Ali
*Status: Phases 1–5 shipped · Phase 6 and open-work items active*
