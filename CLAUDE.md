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

### When exploring dependencies
```bash
hafiz graph deps <entity>
hafiz graph dependents <entity>
hafiz graph show <entity>
```

## Custom Command
Use `/brain-query <question>` to query Hafiz directly.

## Project Structure
- `hafiz/cli.py` -- Typer CLI entry point
- `hafiz/commands/` -- Command implementations (query, graph, observe, context, ingest, watch, prune, hooks, maintenance)
- `hafiz/core/` -- Business logic (chunker, config, database, embeddings, extractor, search, store, watcher, observations, context, git_hooks)
- `tests/` -- pytest suite
- `hafiz.toml.example` -- Configuration template
