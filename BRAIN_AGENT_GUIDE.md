# Hafiz Agent Integration Guide

Universal instructions for any AI agent (Aider, Codex, Claude Code, Cursor, etc.) to use
Hafiz as a workspace intelligence layer.

## What is Hafiz?

Hafiz is a CLI tool that indexes your entire workspace into PostgreSQL + pgvector. It provides:
- **Semantic code search** across all projects
- **Entity & relationship graph** (classes, functions, modules, and how they connect)
- **Observations store** for decisions, facts, learnings, warnings, and patterns
- **Context synthesis** that combines all of the above into a single bundle

## Setup

Every command must be run exactly like this:

```bash
cd /home/irshad-workstation/workspace/irshad/hafiz && source .venv/bin/activate && hafiz <command>
```

This ensures the config file (`hafiz.toml`) is found and the correct Python venv is active.

## Command Reference

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz query "<text>"` | Semantic search over code and docs | `--type`, `--project`, `--limit`, `--json` |
| `hafiz context "<task>"` | Full context bundle for a task | `--project`, `--json` |
| `hafiz recall "<query>"` | Search observations only | `--type`, `--project`, `--limit`, `--json` |
| `hafiz observe "<text>"` | Store a decision/fact/learning | `--type`, `--source`, `--project`, `--tags`, `--confidence`, `--json` |
| `hafiz graph show <name>` | Entity and its connections | `--project`, `--json` |
| `hafiz graph deps <name>` | What this entity depends on | `--project`, `--json` |
| `hafiz graph dependents <name>` | What depends on this entity | `--project`, `--json` |
| `hafiz status` | Database statistics | `--json` |
| `hafiz doctor` | System diagnostics | `--json` |

### Type values

- **Query types** (`--type` for `query`): `code`, `doc`, `note`, `decision`
- **Observation types** (`--type` for `observe`/`recall`): `fact`, `decision`, `learning`, `pattern`, `warning`
- **Source** (`--source` for `observe`): `agent:aider`, `agent:claude-code`, `agent:codex`, `user:irshad`

## Copy-Paste System Prompt Snippet

Add this to any agent's system prompt or instructions:

```
You have access to a workspace intelligence tool called `hafiz` that indexes the entire
codebase with semantic search, entity graphs, and an observations store.

IMPORTANT: Every hafiz command must be run as:
cd /home/irshad-workstation/workspace/irshad/hafiz && source .venv/bin/activate && hafiz <command>

Before starting any task, gather context:
  hafiz context "<task description>" --json

To search for relevant code:
  hafiz query "<question>" --json

To check past decisions and known gotchas:
  hafiz recall "<topic>" --json

After making architectural decisions, record them:
  hafiz observe "<decision and reasoning>" --type decision --source agent:<your-name>

To understand code structure and dependencies:
  hafiz graph deps <EntityName> --json
  hafiz graph dependents <EntityName> --json

Always use --json flag when parsing output programmatically.
```

## Recommended Workflows

### Starting a task

1. Run `hafiz context "<task description>" --json` to get a full context bundle
2. Run `hafiz recall "<related topic>" --type decision --json` to check for existing decisions
3. Begin implementation with full context

### During implementation

1. Run `hafiz query "<specific question>" --type code --json` to find relevant code
2. Run `hafiz graph deps <Entity>` to understand what a component depends on
3. Run `hafiz graph dependents <Entity>` before refactoring to assess impact

### After completing work

1. Record significant decisions: `hafiz observe "<decision>" --type decision --source agent:<name>`
2. Record gotchas discovered: `hafiz observe "<gotcha>" --type warning --source agent:<name>`
3. Record learned patterns: `hafiz observe "<pattern>" --type pattern --source agent:<name>`

## JSON Output

All commands support `--json` for machine-readable output. Without it, output uses
Rich terminal formatting (good for human reading, bad for parsing).

## Architecture

```
Workspace Files
      |
      v
  Chunker (LlamaIndex) --> Chunks table (text + vector embeddings)
      |
      v
  Extractor (Claude LLM) --> Entities table + Relations table
      |
      v
  PostgreSQL + pgvector
      |
      v
  hafiz CLI --> Any Agent
```

- **Embeddings**: nomic-embed-text-v1.5 via fastembed (local ONNX, no API key)
- **Database**: PostgreSQL 18 + pgvector (768-dim vectors)
- **LLM**: Anthropic Claude (for entity extraction only)
- **Tables**: chunks, entities, relations, observations
