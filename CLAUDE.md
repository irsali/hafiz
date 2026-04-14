# Hafiz -- Workspace Intelligence

This project is Hafiz, a CLI-first intelligence layer backed by PostgreSQL + pgvector.

## Using Hafiz

Hafiz is installed globally. Run from any directory:
```bash
hafiz <command>
```

Configuration is loaded from `hafiz.toml` in the current directory, `~/.config/hafiz/hafiz.toml`, or `/etc/hafiz/hafiz.toml`.

### Before starting work on any file
```bash
hafiz context "<task description>" --json
```
This returns relevant code chunks, entity relationships, and past observations in one call.

### When searching for code or answers
```bash
hafiz query "<question>" --json
hafiz query "<question>" --type code --project <name> --json
```

### When checking past decisions
```bash
hafiz recall "<topic>" --type decision --json
```

### After making architectural decisions
```bash
hafiz observe "<what was decided and why>" --type decision --source agent:claude-code
```

### For cross-project context in multi-project workspaces
```bash
hafiz context "<task description>" --workspace --json
```
Searches all indexed projects at once. Uses `workspace.projects` from hafiz.toml if configured, otherwise discovers from DB.

### When exploring dependencies
```bash
hafiz graph deps <entity>
hafiz graph dependents <entity>
hafiz graph show <entity>
```

### When reviewing knowledge quality
```bash
hafiz review --json
hafiz review --project <name> --json
```
Analyzes observations, graph coverage, staleness, and suggests improvements. This is the self-review mechanism (Layer 2, evolving) — separate from `hafiz agent install` (Layer 1, stable contract).

## Project Structure
- `hafiz/cli.py` -- Typer CLI entry point
- `hafiz/commands/` -- Command implementations (agent, chunks, extract, query, graph, observe, context, ingest, watch, prune, hooks, maintenance, review)
- `hafiz/core/` -- Business logic (agents, chunker, config, database, embeddings, extractor, search, store, watcher, observations, context, git_hooks, review)
- `hafiz/data/agents/skills.md` -- Universal agent skill file (installed by `hafiz agent install`)
- `tests/` -- pytest suite
- `hafiz.toml.example` -- Configuration template
