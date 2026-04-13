# Hafiz

A sovereign, CLI-first intelligence layer for your workspace. Hafiz indexes your codebase into PostgreSQL + pgvector, extracts entities and relationships, stores observations, and provides semantic search from the terminal or any AI agent.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Create config
cp hafiz.toml.example hafiz.toml
# Edit hafiz.toml with your database credentials

# Initialize the database
hafiz init

# Index a directory
hafiz ingest ./src/ --project my-project

# Search
hafiz query "how does authentication work?"

# Get full context for a task
hafiz context "implement rate limiting"

# Store a decision
hafiz observe "JWT preferred over sessions" --type decision

# Check what depends on an entity
hafiz graph dependents AuthController
```

## Prerequisites

- Python 3.12+
- PostgreSQL with pgvector extension
- A running PostgreSQL instance (default: `localhost:5432`)

## Install

```bash
# From the project directory
pipx install -e .

# Or with pip
pip install -e ".[dev]"
```

## Command Reference

### Search & Query

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz query "<text>"` | Vector similarity search over code and docs | `--type/-t`, `--project/-p`, `--limit/-l`, `--json/-j` |
| `hafiz recall "<query>"` | Search observations (decisions, facts, learnings) | `--type/-t`, `--project/-p`, `--limit/-l`, `--json/-j` |
| `hafiz context "<task>"` | Synthesize relevant code, graph, and observations for a task | `--project/-p`, `--json/-j` |

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
| `hafiz ingest <path>` | Index files into the knowledge base | `--project/-p`, `--no-extract`, `--git-hook` |
| `hafiz watch <path>` | Real-time file watcher (re-indexes on change) | `--project/-p`, `--json/-j` |
| `hafiz prune` | Remove chunks for deleted files | `--project/-p`, `--dry-run`, `--json/-j` |
| `hafiz hooks install [path]` | Install git post-commit hook | `--project/-p` |

### System

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `hafiz init` | Create database tables and pgvector extension | |
| `hafiz status` | Show database statistics | `--json/-j` |
| `hafiz config show` | Display current configuration | `--json/-j` |
| `hafiz doctor` | Run diagnostic checks | `--json/-j` |

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

## Configuration

Hafiz looks for `hafiz.toml` in:
1. Current directory
2. `~/.config/hafiz/hafiz.toml`
3. `/etc/hafiz/hafiz.toml`

```toml
[database]
url = "postgresql+asyncpg://postgres:password@localhost:5432/hafiz"

[embedding]
model = "nomic-ai/nomic-embed-text-v1.5"
provider = "fastembed"
dimensions = 768

[llm]
provider = "anthropic"
model = "claude-sonnet-4-20250514"

[workspace]
root = "/home/irshad-workstation/workspace"
projects = ["hu-manity", "noble-wave", "irshad"]
ignore = [".git", "node_modules", "__pycache__", ".venv", "dist", "build", ".next", ".cache", "target"]
```

Environment variables override config values using `HAFIZ_` prefix with double-underscore nesting:

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

Hafiz is designed as a standalone tool that any AI agent can use via CLI. See:

- **Claude Code**: `CLAUDE.md` in this repo + `/brain-query` slash command
- **Bilal (OpenClaw)**: `~/.openclaw/skills/hafiz-memory/` skill
- **Any agent**: `BRAIN_AGENT_GUIDE.md` -- universal integration guide with a copy-paste system prompt snippet

All agents should use `--json` for machine-readable output. The recommended workflow:

1. `hafiz context "<task>" --json` before starting work
2. `hafiz query "<question>" --json` during implementation
3. `hafiz observe "<decision>" --type decision` after making decisions

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Project Structure

```
hafiz/
  cli.py              -- Typer CLI entry point
  commands/            -- Command implementations
    context.py         -- hafiz context
    graph.py           -- hafiz graph show/deps/dependents
    hooks.py           -- hafiz hooks install
    ingest.py          -- hafiz ingest
    maintenance.py     -- hafiz init/status/doctor/config
    observe.py         -- hafiz observe/recall
    prune.py           -- hafiz prune
    query.py           -- hafiz query
    watch.py           -- hafiz watch
  core/                -- Business logic
    chunker.py         -- File walking & chunking
    config.py          -- Configuration (TOML + env vars)
    context.py         -- Context synthesis
    database.py        -- SQLAlchemy models
    embeddings.py      -- FastEmbed wrapper
    extractor.py       -- LLM entity/relation extraction
    git_hooks.py       -- Git hook utilities
    observations.py    -- Observations store & search
    search.py          -- Vector similarity search
    store.py           -- Database store operations
    watcher.py         -- File system watcher
tests/                 -- pytest test suite
alembic/               -- Database migrations
hafiz.toml             -- Active configuration
CLAUDE.md              -- Claude Code instructions
BRAIN_AGENT_GUIDE.md   -- Universal agent guide
ROADMAP.md             -- Architecture & vision
```
