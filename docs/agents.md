# Hafiz Agent Integration Guide

> **Stale as of 2026-04-25.** This guide predates the Structural Grounding rewrite
> and uses the older four-table vocabulary ("entity & relationship graph,"
> "observations store"). It will be rewritten. Until then, the authoritative
> references are:
>
> - [`../hafiz/data/agents/skills.md`](../hafiz/data/agents/skills.md) — the
>   agent contract (installed by `hafiz agent install` into every agent's
>   config). Source of truth for what agents read/write.
> - [`architecture.md`](architecture.md) — current system, data model, key
>   flows, and capture-coverage analysis.
>
> Kept below for continuity; treat details as possibly outdated.

---

Universal instructions for any AI agent (Aider, Codex, Claude Code, Cursor, etc.) to use
Hafiz as a workspace intelligence layer.

## What is Hafiz?

Hafiz is a CLI tool that indexes your entire workspace into PostgreSQL + pgvector. It provides:
- **Semantic search** across docs and files in all projects
- **Observations store** for decisions, facts, learnings, warnings, and patterns
- **Context synthesis** that combines indexed content and observations into a single bundle

## Setup

Install Hafiz and ensure it's on PATH:
```bash
pip install git+https://github.com/irsali/hafiz.git
```

Create a config file at `~/.config/hafiz/hafiz.toml` (or in the current directory).
See `hafiz.toml.example` for the template.

Every command runs directly:
```bash
hafiz <command>
```

## Command Reference

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz query "<text>"` | Semantic search over docs and files | `--type`, `--project`, `--limit`, `--json` |
| `hafiz context "<task>"` | Full context bundle for a task | `--project`, `--json` |
| `hafiz query "<query>" --observations` | Search observations only | `--type`, `--project`, `--limit`, `--json` |
| `hafiz observe "<text>"` | Store a decision/fact/learning | `--type`, `--source`, `--project`, `--tags`, `--confidence`, `--json` |
| `hafiz ingest <path>` | Index files into knowledge base | `--project`, `--git-hook`, `--json` |
| `hafiz agent install <name>` | Install hafiz skills for an agent | `--local`, `--path`, `--file` |
| `hafiz status` | Database statistics | `--json` |
| `hafiz status --diagnose` | System diagnostics | `--json` |

### Type values

- **Query types** (`--type` for `query`): `doc.heading`, `doc.paragraph`, `file.raw`, …
- **Observation types** (`--type` for `observe` / `query --observations`): `fact`, `decision`, `learning`, `pattern`, `warning`
- **Source** (`--source` for `observe`): `agent:aider`, `agent:claude-code`, `agent:codex`, `user:<name>`

## Copy-Paste System Prompt Snippet

Add this to any agent's system prompt or instructions:

```
You have access to a workspace intelligence tool called `hafiz` that indexes the
workspace with semantic search and an observations store.

When a task needs prior docs or decisions, gather context:
  hafiz context "<task description>" --json

To search indexed content:
  hafiz query "<question>" --json

To check past decisions and known gotchas:
  hafiz query "<topic>" --observations --json

After making architectural decisions, record them:
  hafiz observe "<decision and reasoning>" --type decision --source agent:<your-name>

Always use --json flag when parsing output programmatically.
```

## Recommended Workflows

### Starting a task

1. Run `hafiz context "<task description>" --json` to get a full context bundle
2. Run `hafiz query "<related topic>" --observations --type decision --json` to check for existing decisions
3. Begin implementation with full context

### During implementation

1. Run `hafiz query "<specific question>" --json` to find relevant indexed content
2. Prefer reading the working tree (and git) for the file you are actively editing

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
  Parsers (prose + whole-file) --> units / revisions / embeddings
      |
      v
  Agent observe/note --> annotations
      |
      v
  PostgreSQL + pgvector
      |
      v
  hafiz CLI --> Any Agent
```

- **Embeddings**: nomic-embed-text-v1.5 via fastembed (local ONNX, no API key)
- **Database**: PostgreSQL + pgvector (768-dim vectors)
- **Tables**: files, units, unit_revisions, embeddings, annotations, commits (+ source layer)
