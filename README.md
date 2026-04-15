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
- **Anthropic API key** (optional) -- for knowledge graph extraction ([console.anthropic.com](https://console.anthropic.com/)). Without it, search and context still work; only `hafiz graph` commands require it.
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

### 2. Set your Anthropic API key (optional)

The API key enables knowledge graph extraction during ingestion (`hafiz graph` commands). Without it, chunking, vector search, and context synthesis all work normally -- ingestion will skip graph extraction automatically.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Add it to your shell profile (`~/.bashrc`, `~/.zshrc`) to persist across sessions.

> **Note:** Claude Code does *not* pass its own API key to shell commands. If you use Hafiz from Claude Code and want graph extraction, you still need the key exported in your shell profile.

### 3. Create the config file

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

[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"

[workspace]
root = "/path/to/your/workspace"          # <-- change this
projects = ["my-project"]                 # <-- change this
ignore = [".git", "node_modules", "__pycache__", ".venv", "dist", "build"]
```

Update `root` to your workspace directory and `projects` to your project names. Adjust the database URL if you changed the credentials above.

### 4. Initialize the database

```bash
hafiz init
```

Creates all tables, indexes, and enables the pgvector extension.

### 5. Verify the setup

```bash
hafiz doctor
```

All checks should pass (database connection, pgvector, embeddings, config).

### 6. Index your first project

```bash
hafiz ingest /path/to/your/project --project my-project
```

### 7. Try it out

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

## Command Reference

### Search & Query

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz query "<text>"` | Vector similarity search over code and docs | `--type/-t`, `--project/-p`, `--workspace/-w`, `--limit/-l`, `--json/-j` |
| `hafiz recall "<query>"` | Search observations (decisions, facts, learnings) | `--type/-t`, `--project/-p`, `--workspace/-w`, `--limit/-l`, `--json/-j` |
| `hafiz context "<task>"` | Synthesize relevant code, graph, and observations for a task | `--project/-p`, `--workspace/-w`, `--json/-j` |

### Knowledge Graph

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz graph show <name>` | Show entity and its direct connections | `--project/-p`, `--json/-j` |
| `hafiz graph deps <name>` | Show outgoing dependencies (what it needs) | `--project/-p`, `--json/-j` |
| `hafiz graph dependents <name>` | Show incoming dependencies (what needs it) | `--project/-p`, `--json/-j` |

### Observations

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz observe "<text>"` | Store a fact, decision, learning, pattern, or warning | `--type/-t`, `--source/-s`, `--project/-p`, `--tags`, `--confidence/-c`, `--json/-j` |

### Ingestion

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz ingest <path>` | Index files into the knowledge base | `--project/-p`, `--no-extract`, `--git-hook`, `--prune`, `--json/-j` |
| `hafiz watch <path>` | Real-time file watcher (re-indexes on change) | `--project/-p`, `--json/-j` |
| `hafiz prune` | Remove chunks for deleted files | `--project/-p`, `--dry-run`, `--json/-j` |
| `hafiz chunks export` | Export indexed chunks as JSON (for agent extraction) | `--project/-p`, `--unextracted`, `--path`, `--limit/-l`, `--offset` |
| `hafiz extract run` | Extract entities from chunks that don't have entities yet | `--project/-p`, `--json/-j` |
| `hafiz extract import` | Import extraction results from JSON (file or stdin) | `--file/-f`, `--project/-p` |
| `hafiz hooks install [path]` | Install git hooks (post-commit + post-merge) | `--project/-p` |
| `hafiz agent install <name>` | Install hafiz skills into an AI agent | `--local`, `--path`, `--file` |

### System

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz init` | Create database tables and pgvector extension | |
| `hafiz status` | Show database statistics | `--json/-j` |
| `hafiz config show` | Display current configuration | `--json/-j` |
| `hafiz doctor` | Run diagnostic checks | `--json/-j` |
| `hafiz review` | Review knowledge quality and get improvement suggestions | `--project/-p`, `--json/-j` |

### Common Flags

- `--json` / `-j` -- Machine-readable JSON output (for agents)
- `--project` / `-p` -- Filter or tag by project name
- `--type` / `-t` -- Filter by type (varies by command)
- `--limit` / `-l` -- Maximum number of results (default: 10)

### Type Values

- **Query types**: `code`, `doc`, `note`, `decision`
- **Observation types**: `fact`, `decision`, `learning`, `pattern`, `warning`
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

## Architecture

```
Workspace Files
      |
      v
  Chunker (LlamaIndex SentenceSplitter / CodeSplitter)
      |
      v
  Embeddings (nomic-embed-text-v1.5 via fastembed, local ONNX)
      |                                |
      v                                v
  Chunks table                  Extractor (Claude LLM)
  (text + 768-dim vectors)            |
                                      v
                              Entities + Relations tables
                                      |
      All tables ---------> PostgreSQL + pgvector
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
| LLM | Anthropic Claude (entity extraction) |
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

This writes a skill file to the agent's configuration directory (e.g. `~/.claude/CLAUDE.md` for Claude Code). The skill teaches the agent to use `hafiz context`, `hafiz query`, `hafiz recall`, `hafiz graph`, and `hafiz observe` as part of its workflow.

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
3. `hafiz observe "<decision>" --type decision` after making decisions
4. `hafiz review --json` periodically to check knowledge quality

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
    chunks.py          -- hafiz chunks export
    context.py         -- hafiz context
    extract.py         -- hafiz extract import/run
    graph.py           -- hafiz graph show/deps/dependents
    hooks.py           -- hafiz hooks install
    ingest.py          -- hafiz ingest (with JSON progress)
    maintenance.py     -- hafiz init/status/doctor/config
    observe.py         -- hafiz observe/recall
    prune.py           -- hafiz prune
    query.py           -- hafiz query
    review.py          -- hafiz review
    watch.py           -- hafiz watch
  core/                -- Business logic
    agents.py          -- Agent registry & file operations
    chunker.py         -- File walking & chunking (.gitignore aware)
    config.py          -- Configuration (TOML + env vars)
    context.py         -- Context synthesis
    database.py        -- SQLAlchemy models
    embeddings.py      -- FastEmbed wrapper
    extractor.py       -- LLM entity/relation extraction
    git_hooks.py       -- Git hook utilities
    observations.py    -- Observations store & search
    review.py          -- Self-review engine (knowledge quality analysis)
    search.py          -- Vector similarity search
    store.py           -- Database store operations
    watcher.py         -- File system watcher
  data/agents/         -- Distributable agent skill files
    skills.md          -- Universal hafiz skill (installed by hafiz agent install)
tests/                 -- pytest test suite
alembic/               -- Database migrations
hafiz.toml.example     -- Configuration template
CLAUDE.md              -- Claude Code instructions (project-local)
BRAIN_AGENT_GUIDE.md   -- Universal agent guide
ROADMAP.md             -- Architecture & vision
```

## License

MIT
