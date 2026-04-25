# Hafiz

A sovereign, CLI-first intelligence layer for your workspace. Hafiz indexes your codebase into PostgreSQL + pgvector, extracts entities and relationships, stores observations, and provides semantic search from the terminal or any AI agent.

## Quick Start

```bash
pipx install "hafiz[gpu] @ git+https://github.com/irsali/hafiz.git"  # or without [gpu]
hafiz init
hafiz ingest ./src/ --project my-project
hafiz query "how does authentication work?"
```

See the full setup guide below.

## Prerequisites

- **Python 3.12+** -- `python3 --version`
- **Docker** -- for PostgreSQL + pgvector (or a native PostgreSQL install)
- **NVIDIA GPU + CUDA drivers** (optional) -- for accelerated embeddings (`nvidia-smi` to verify)

## Install

[pipx](https://pipx.pypa.io/) is the recommended way to install Hafiz. It creates an isolated virtual environment and makes the `hafiz` command available globally.

```bash
# Install pipx if you don't have it
sudo apt install pipx   # Debian/Ubuntu
pipx ensurepath          # Add ~/.local/bin to PATH (restart shell after)

# Install Hafiz
pipx install git+https://github.com/irsali/hafiz.git

# With GPU acceleration (requires CUDA drivers)
pipx install "hafiz[gpu] @ git+https://github.com/irsali/hafiz.git"

# Upgrade to latest from GitHub
pipx upgrade hafiz

# Editable install from local clone (changes apply instantly)
pipx install -e ".[gpu]" --force   # or without [gpu]
```

<details>
<summary>Alternative: pip with venv</summary>

```bash
git clone https://github.com/irsali/hafiz.git
cd hafiz
python3 -m venv .venv && source .venv/bin/activate
pip install .          # or pip install ".[gpu]" for GPU support
```

</details>

## Setup

### 1. Start PostgreSQL with pgvector

```bash
docker run -d \
  --name hafiz-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=hafiz \
  -p 5432:5432 \
  --restart unless-stopped \
  pgvector/pgvector:pg17
```

This starts a PostgreSQL 17 container with the pgvector extension pre-installed. Data persists in the container; add `-v hafiz-pgdata:/var/lib/postgresql/data` if you want a named volume.

<details>
<summary>Alternative: native PostgreSQL</summary>

If you prefer a system install instead of Docker:

```bash
# Ubuntu / Debian
sudo apt install postgresql postgresql-17-pgvector
sudo systemctl start postgresql && sudo systemctl enable postgresql
sudo -u postgres psql -c "CREATE DATABASE hafiz;"
sudo -u postgres psql -d hafiz -c "CREATE EXTENSION IF NOT EXISTS vector;"

# macOS (Homebrew)
brew install postgresql@17 pgvector
brew services start postgresql@17
createdb hafiz
psql -d hafiz -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

</details>

### 2. Create the config file

```bash
mkdir -p ~/.config/hafiz
```

Create `~/.config/hafiz/hafiz.toml`:

```toml
[database]
url = "postgresql+asyncpg://postgres:postgres@localhost:5432/hafiz"

[embedding]
model = "nomic-ai/nomic-embed-text-v1.5"
provider = "fastembed"
dimensions = 768

[workspace]
root = "/path/to/your/workspace"          # <-- change this
projects = ["my-project"]                 # <-- change this
ignore = [".git", "node_modules", "__pycache__", ".venv", "dist", "build"]
```

Update `root` to your workspace directory and `projects` to your project names. Adjust the database URL if you changed the credentials above.

### 3. Initialize the database

```bash
hafiz init
```

Creates all tables, indexes, and enables the pgvector extension.

### 4. Verify the setup

```bash
hafiz status --diagnose
```

All checks should pass (database connection, pgvector, embeddings, config).

### 5. Index your first project

```bash
hafiz ingest /path/to/your/project --project my-project
```

### 6. Try it out

```bash
# Semantic search
hafiz query "how does authentication work?"

# Full context for a task
hafiz context "implement rate limiting"

# Cross-project context (sibling projects in parent directory)
hafiz context "implement rate limiting" --workspace

# Store a decision
hafiz observe "JWT preferred over sessions" --type decision

# Explore the knowledge graph
hafiz graph dependents AuthController
```

## The Capture → Distill Loop

Beyond indexing your codebase, Hafiz is a second brain for the work itself — raw thoughts, transcripts, decisions, and what replaced what. The loop is: **capture raw → review → distill into decisions → supersede when things change**.

### Capture

Start a session to auto-tag everything you record next. Sessions are scoped to your terminal (TTY) so two shells don't clobber each other:

```bash
hafiz session start "jwt-migration" --task auth --project my-project
```

Jot raw thoughts as they come (low-bar lane — distill later):

```bash
hafiz note "Wondering if refresh tokens should live in httponly cookies"
```

Pipe a long discussion (Claude Code transcript, meeting notes, whiteboard dump) and Hafiz will chunk it by paragraph / turn:

```bash
cat conversation.md | hafiz capture --title "JWT design meeting"
```

Every observation auto-captures git context (commit hash, branch, dirty state) so you can later ask "what did I decide on `main` at commit `abc123`?".

### Review

See what you recorded, grouped by day — observations and captures together:

```bash
hafiz journal --since 7d
hafiz journal --day yesterday --session jwt-migration
```

### Distill

Turn raw notes into durable decisions. `hafiz distill` **lists** recent notes and transcripts as promotable candidates — it does not auto-promote and does not call an LLM. The agent or human reads the candidates and decides:

```bash
hafiz distill --since 7d
# prints a ready-to-run `hafiz observe ... --derived-from <ids>` scaffold
```

Promote with `--derived-from` to record lineage (non-destructive):

```bash
hafiz observe "Use httponly cookies for refresh tokens" \
  --type decision \
  --derived-from <note-id>,<note-id>
```

### Supersede

When a decision changes, don't delete — **supersede**. The old row stays queryable for audit; the new row records the backref:

```bash
hafiz observe "Use Bearer header, not cookies" \
  --type decision \
  --supersedes <old-decision-id>
```

By default `hafiz query --recall` hides superseded / expired rows. Pass `--include-superseded` to see the full history, dimmed with a `(superseded)` marker.

### Expire

Some learnings become stale by a known date (a version bug, a deadline-bound constraint). Set an explicit lifetime up front:

```bash
hafiz observe "fastembed 0.3 has a GPU bug" --type warning --expires-in 90d
hafiz observe "Migration freeze" --type fact --expires 2026-06-01
```

Recall also shows each row's age (e.g. `3mo ago`) and dims results older than 90 days so fresh decisions win.

## Command Reference

### Search & Query

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz query "<text>"` | Vector similarity search over code and docs | `--type/-t`, `--project/-p`, `--workspace/-w`, `--limit/-l`, `--json/-j`, `--recall` |
| `hafiz query "<text>" --recall` | Search observations (decisions, facts, learnings) | `--type/-t`, `--project/-p`, `--workspace/-w`, `--limit/-l`, `--json/-j` |
| `hafiz context "<task>"` | Synthesize relevant code, graph, and observations for a task | `--project/-p`, `--workspace/-w`, `--json/-j` |

### Knowledge Graph

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz graph show <name>` | Show entity and its direct connections | `--project/-p`, `--json/-j` |
| `hafiz graph deps <name>` | Show outgoing dependencies (what it needs) | `--project/-p`, `--json/-j` |
| `hafiz graph dependents <name>` | Show incoming dependencies (what needs it) | `--project/-p`, `--json/-j` |

### Observations & Capture

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz observe "<text>"` | Store a fact, decision, learning, pattern, warning, or note | `--type/-t`, `--source/-s`, `--project/-p`, `--tags`, `--confidence/-c`, `--expires-in`, `--expires`, `--session`, `--task`, `--supersedes`, `--derived-from`, `--json/-j` |
| `hafiz note "<text>"` | Shortcut for `observe --type note` — low-bar raw-capture lane | Same as `observe` (minus `--type`) |
| `hafiz capture [TEXT]` | Ingest a transcript / multi-page dump (stdin, `--file`, or positional TEXT) | `--title`, `--file/-f`, `--source/-s`, `--project/-p`, `--tags`, `--session`, `--task`, `--json/-j` |

### Journal & Distill

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz journal` | Time-bounded digest of observations **and** captures, grouped by day | `--since`, `--day`, `--project/-p`, `--workspace/-w`, `--source`, `--type/-t`, `--session`, `--task`, `--limit/-l`, `--json/-j` |
| `hafiz distill` | Surface recent notes + transcripts as promotable candidates (scanner — no LLM call) | `--since`, `--project/-p`, `--workspace/-w`, `--session`, `--task`, `--no-transcripts`, `--limit/-l`, `--json/-j` |

### Sessions

Per-TTY named threads that auto-tag subsequent `observe` / `note` / `capture` with a `session_id` and optional `task`. State lives in `~/.cache/hafiz/session-<tty>.json`. Commands still work without a TTY (CI, piped) — writes are just untagged.

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz session start "<name>"` | Start a session for this terminal | `--task`, `--project/-p`, `--json/-j` |
| `hafiz session show` | Show the active session | `--json/-j` |
| `hafiz session end` | Clear the active session | `--json/-j` |

### Ingestion

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz ingest <path>` | Index files into the knowledge base (chunk + embed + store) | `--project/-p`, `--git-hook`, `--prune`, `--json/-j` |
| `hafiz watch <path>` | Real-time file watcher (re-indexes on change) | `--project/-p`, `--json/-j` |
| `hafiz prune` | Remove chunks for deleted files | `--project/-p`, `--dry-run`, `--json/-j` |
| `hafiz extract export` | Export chunks grouped by file as JSON (for agent extraction) | `--project/-p`, `--unextracted`, `--path`, `--limit/-l`, `--offset` |
| `hafiz extract import` | Import extraction results from JSON (file or stdin) | `--file/-f`, `--project/-p` |
| `hafiz hooks install [path]` | Install git hooks (post-commit + post-merge) | `--project/-p` |
| `hafiz agent install <name>` | Install hafiz skills into an AI agent | `--local`, `--path`, `--file` |

### System

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz init` | Create database tables and pgvector extension | |
| `hafiz status` | Show database statistics | `--json/-j`, `--diagnose` |
| `hafiz config show` | Display current configuration | `--json/-j` |
| `hafiz review` | Review knowledge quality and get improvement suggestions | `--project/-p`, `--json/-j` |

### Common Flags

- `--json` / `-j` -- Machine-readable JSON output (for agents)
- `--project` / `-p` -- Filter or tag by project name
- `--type` / `-t` -- Filter by type (varies by command)
- `--limit` / `-l` -- Maximum number of results (default: 10)

### Type Values

- **Chunk types** (for `query --type`): `code`, `doc`, `note`, `decision`, `transcript`
- **Observation types** (for `observe --type`, `journal --type`): `fact`, `decision`, `learning`, `pattern`, `warning`, `note`
- **Entity types**: `class`, `function`, `module`, `api`, `table`, `concept`
- **Relation types**: `calls`, `imports`, `inherits`, `depends_on`, `defines`

## Ignore Rules

Hafiz respects `.gitignore` and `.hafizignore` files at every directory level, including negation patterns (`!important.py`) and subdirectory overrides. The `workspace.ignore` list in `hafiz.toml` provides additional patterns.

Ignore precedence (later overrides earlier):
1. `workspace.ignore` from `hafiz.toml`
2. Root `.gitignore`
3. Root `.hafizignore`
4. Subdirectory `.gitignore` / `.hafizignore` (deeper overrides shallower)

Create a `.hafizignore` file for hafiz-specific exclusions that shouldn't affect git:

```
# Ignore generated code (not in .gitignore because it's tracked)
src/generated/
!src/generated/manifest.json

# Ignore large data files
*.parquet
*.arrow
```

## Configuration

Hafiz looks for `hafiz.toml` in order:
1. Current directory
2. `~/.config/hafiz/hafiz.toml`
3. `/etc/hafiz/hafiz.toml`

See [Setup](#setup) for the full config template. Environment variables override config values using the `HAFIZ_` prefix with double-underscore nesting:

```bash
export HAFIZ_DATABASE__URL="postgresql+asyncpg://user:pass@host:5432/hafiz"
export HAFIZ_EMBEDDING__MODEL="nomic-ai/nomic-embed-text-v1.5"
```

### Embedding device (GPU vs CPU)

Hafiz runs embeddings locally via ONNX. By default (`embedding.device = "auto"`) it probes CUDA on first use, falls back to CPU if that fails, and caches the verdict at `~/.cache/hafiz/device_state.json` so subsequent runs skip the probe.

```bash
hafiz embedding status        # what's being used, and why
hafiz embedding retry         # re-probe after freeing VRAM, upgrading drivers, etc.
```

If another process (LM Studio, Whisper, a training job) is holding VRAM, force CPU for the session:

```bash
HAFIZ_EMBEDDING__DEVICE=cpu hafiz ingest .
```

Or lock it in `hafiz.toml` under `[embedding]`: `device = "cpu"` (never touch the GPU), `"gpu"` (require CUDA, fail loudly otherwise), or `"auto"` (default — probe + cache).

## Architecture

```
Workspace Files
      |
      v
  Chunker (LlamaIndex SentenceSplitter / CodeSplitter)
      |
      v
  Embeddings (nomic-embed-text-v1.5 via fastembed, local ONNX)
      |
      v
  Chunks table (text + 768-dim vectors)
      |
      v (agent-driven extraction)
  chunks export --> Agent (any LLM) --> extract import
                                            |
                                            v
                                  Entities + Relations tables
                                            |
      All tables ----------------> PostgreSQL + pgvector
                                      |
                                      v
                              hafiz CLI (Typer + Rich)
                                      |
              +-----------+-----------+-----------+
              |           |           |           |
           Bilal     Claude Code    Aider     Any Agent
```

### Data Model

- **Chunks**: Raw content split into searchable pieces, each with a 768-dim vector embedding
- **Entities**: Extracted "nouns" (classes, functions, modules, APIs, tables, concepts)
- **Relations**: Extracted "verbs" (calls, imports, inherits, depends_on, defines)
- **Observations**: High-level facts, decisions, learnings, patterns, and warnings

### Tech Stack

| Component | Technology |
|-----------|-----------|
| CLI | Typer + Rich |
| Database | PostgreSQL + pgvector |
| ORM | SQLAlchemy 2.0 (async) |
| Embeddings | fastembed (nomic-embed-text-v1.5, local ONNX) |
| Chunking | LlamaIndex SentenceSplitter / CodeSplitter |
| Extraction | Agent-driven (any LLM in session or piped via CLI) |
| File Watching | watchdog |
| Migrations | Alembic |
| Config | Pydantic + TOML |

## Agent Integration

Hafiz is designed as a standalone tool that any AI agent can use via CLI. Install hafiz skills into your agent so it knows how to use hafiz automatically:

```bash
hafiz agent install claude-code   # Claude Code
hafiz agent install cursor        # Cursor IDE
hafiz agent install github-copilot # GitHub Copilot
```

This writes a skill file to the agent's configuration directory (e.g. `~/.claude/CLAUDE.md` for Claude Code). The skill teaches the agent to use `hafiz context`, `hafiz query`, `hafiz graph`, and `hafiz observe` as part of its workflow.

| Flag | Description |
|------|-------------|
| `--local` | Install into the current project instead of globally |
| `--path <dir>` | Override the destination directory |
| `--file <name>` | Override the filename |

If the target file already exists and was not installed by hafiz, the command skips it to avoid overwriting your work. Files previously installed by hafiz are updated in place.

`hafiz ingest --json` emits newline-delimited JSON progress events, useful for agents and scripts:

```jsonl
{"event":"chunking","status":"done","chunks":71,"files":38}
{"event":"embedding","status":"progress","done":64,"total":71}
{"event":"storing","status":"done","stored":71,"files":38}
{"event":"complete","chunks":71,"files":38,"entities":0,"relations":0}
```

All agents should use `--json` for machine-readable output. The recommended workflow:

1. `hafiz context "<task>" --json` before starting work (or `--workspace` for sibling projects)
2. `hafiz query "<question>" --json` during implementation
3. `hafiz note "<raw thought>"` while working, for anything below decision-grade
4. `hafiz observe "<decision>" --type decision` after deciding (`--supersedes <old-id>` if replacing a prior decision)
5. `hafiz journal --since 7d` / `hafiz distill --since 7d` periodically to review and promote raw notes into decisions

## Development

```bash
git clone https://github.com/irsali/hafiz.git
cd hafiz
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"    # or ".[dev,gpu]" for GPU support
pytest
```

## Project Structure

```
hafiz/
  cli.py              -- Typer CLI entry point
  commands/            -- Command implementations
    agent.py           -- hafiz agent install/uninstall/list
    capture.py         -- hafiz capture (transcripts / multi-page dumps)
    context.py         -- hafiz context
    chunks.py          -- chunk export logic (used by extract export)
    distill.py         -- hafiz distill (promotable-candidate scanner)
    extract.py         -- hafiz extract export/import
    graph.py           -- hafiz graph show/deps/dependents
    hooks.py           -- hafiz hooks install
    ingest.py          -- hafiz ingest (with JSON progress)
    journal.py         -- hafiz journal (time-bounded digest)
    maintenance.py     -- hafiz init/status/config (status --diagnose for diagnostics)
    observe.py         -- hafiz observe / note (recall via query --recall)
    prune.py           -- hafiz prune
    query.py           -- hafiz query
    review.py          -- hafiz review
    session.py         -- hafiz session start/show/end
    watch.py           -- hafiz watch
  core/                -- Business logic
    agents.py          -- Agent registry & file operations
    capture.py         -- Transcript splitter + neighbor expansion
    chunker.py         -- File walking & chunking (.gitignore aware)
    config.py          -- Configuration (TOML + env vars)
    context.py         -- Context synthesis
    database.py        -- SQLAlchemy models
    distill.py         -- Distill-candidate scanner
    durations.py       -- Human-readable duration parser (30d, 2w, 6m, 1y)
    embeddings.py      -- FastEmbed wrapper
    extractor.py       -- Agent-driven entity/relation extraction
    git_context.py     -- Per-observation git HEAD capture
    git_hooks.py       -- Git hook utilities
    journal.py         -- Time-bounded digest assembly
    observations.py    -- Observations store & search (supersession)
    review.py          -- Self-review engine (knowledge quality analysis)
    search.py          -- Vector similarity search
    session.py         -- Per-TTY session state
    store.py           -- Database store operations
    watcher.py         -- File system watcher
  data/agents/         -- Distributable agent skill files
    skills.md          -- Universal hafiz skill (installed by hafiz agent install)
tests/                 -- pytest test suite
alembic/               -- Database migrations
hafiz.toml.example     -- Configuration template
CLAUDE.md              -- Claude Code instructions (project-local)
docs/                  -- All narrative documentation
  README.md            -- Docs index
  architecture.md      -- System, data model, key flows, capture gap analysis
  commands.md          -- Authoritative CLI surface + JSON shapes
  roadmap.md           -- Architecture & vision
  agents.md            -- Universal agent guide (stale; see architecture.md + skills.md)
```

## License

[FSL-1.1-MIT](LICENSE) — Functional Source License. Free for any use except
offering Hafiz as a competing commercial product or service. Each release
converts to the MIT License two years after its release date.
