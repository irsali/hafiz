# Hafiz

A sovereign, CLI-first intelligence layer for your workspace. Hafiz indexes your codebase into a PostgreSQL + pgvector database, enabling fast vector similarity search from the terminal or any AI agent.

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

## Quick Start

```bash
# 1. Create config (optional — defaults work if postgres is at localhost:5432)
cp hafiz.toml.example hafiz.toml
# Edit hafiz.toml with your database credentials

# 2. Initialize the database
hafiz init

# 3. Index a directory
hafiz ingest ./src/ --project my-project

# 4. Search
hafiz query "how does authentication work?"
hafiz query "auth" --type code --project my-project --json
```

## Commands

| Command | Description |
|---|---|
| `hafiz init` | Create database tables and pgvector extension |
| `hafiz ingest <path>` | Index files into the knowledge base |
| `hafiz query "<text>"` | Vector similarity search |
| `hafiz status` | Show database statistics |
| `hafiz config show` | Display current configuration |

### Flags

- `--json` / `-j` — Machine-readable JSON output (for agents)
- `--project` / `-p` — Filter or tag by project name
- `--type` / `-t` — Filter by chunk type (`code`, `doc`, `note`, `decision`)
- `--limit` / `-l` — Maximum number of results (default: 10)

## Configuration

Hafiz looks for `hafiz.toml` in:
1. Current directory
2. `~/.config/hafiz/hafiz.toml`
3. `/etc/hafiz/hafiz.toml`

Environment variables override config file values using `HAFIZ_` prefix with double-underscore nesting:

```bash
export HAFIZ_DATABASE__URL="postgresql+asyncpg://user:pass@host:5432/hafiz"
export HAFIZ_EMBEDDING__MODEL="nomic-ai/nomic-embed-text-v1.5"
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Architecture

- **Embeddings**: nomic-embed-text-v1.5 via fastembed (local ONNX, no API key needed)
- **Chunking**: LlamaIndex SentenceSplitter / CodeSplitter with auto language detection
- **Storage**: SQLAlchemy 2.0 async + pgvector (direct control, no LlamaIndex vector store)
- **CLI**: Typer + Rich for human-readable output, `--json` for agents
