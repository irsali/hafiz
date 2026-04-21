# Hafiz — Project Development Guide

This is the Hafiz codebase: a sovereign, CLI-first intelligence layer backed by PostgreSQL + pgvector. When Claude is invoked here, the work is **developing the product** (features, bugfixes, refactors, tests, migrations) — not consuming it.

> For using the `hafiz` CLI as a coding assistant, see the global skills file installed by `hafiz agent install` (already loaded via `~/.claude/CLAUDE.md`). Do **not** duplicate those instructions here.

## Tech Stack

- **Runtime:** Python 3.12+ (3.13 supported; GPU extras pin `onnxruntime-gpu>=1.20`).
- **CLI:** Typer + Rich (subcommand groups via `app.add_typer`).
- **DB:** PostgreSQL + pgvector via SQLAlchemy 2.0 **async** + asyncpg.
- **Embeddings:** fastembed (`nomic-embed-text-v1.5`, 768-dim, ONNX; GPU via `fastembed-gpu`).
- **Indexing:** LlamaIndex core for chunking primitives; custom logic in [hafiz/core/chunker.py](hafiz/core/chunker.py).
- **Migrations:** Alembic ([alembic/versions/](alembic/versions/)).
- **Config:** pydantic-settings reading `hafiz.toml` (cwd → `~/.config/hafiz/` → `/etc/hafiz/`).
- **Tests:** pytest + pytest-asyncio (`asyncio_mode = "auto"`).
- **Lint/format:** ruff (`target-version = py312`, `line-length = 100`).

## Project Layout

```
hafiz/
├── cli.py              # Typer app entrypoint; wires all command groups
├── commands/           # CLI presentation layer (thin — delegate to core/)
│   ├── agent.py        # install/uninstall/list — writes skills.md to agent configs
│   ├── context.py      # context "<task>" — synthesizes chunks + graph + observations
│   ├── extract.py      # export/import — agent-driven entity extraction
│   ├── graph.py        # show / deps / dependents
│   ├── hooks.py        # git post-commit / post-merge installation
│   ├── ingest.py       # chunk + embed + store
│   ├── maintenance.py  # init, status, config, prune
│   ├── observe.py      # persist facts / decisions / learnings / patterns / warnings
│   ├── query.py        # vector search (+ --recall for observations)
│   ├── review.py       # Layer 2 self-review
│   └── watch.py        # long-running file watcher
├── core/               # Business logic; NO Typer/Rich imports here
│   ├── agents.py       # skills.md discovery + agent config paths
│   ├── chunker.py      # File → chunk pipeline
│   ├── config.py       # HafizConfig + loader
│   ├── context.py      # Context bundle assembly
│   ├── database.py     # Async engine, session, ORM models
│   ├── embeddings.py   # fastembed wrapper
│   ├── extractor.py    # Agent-driven extraction helpers
│   ├── git_hooks.py    # Hook templates + install
│   ├── observations.py # CRUD for observations
│   ├── review.py       # Layer 2 review logic
│   ├── search.py       # Vector + hybrid search, scoping
│   ├── store.py        # Chunk / entity / relation persistence
│   └── watcher.py      # watchdog-based file watcher
├── data/agents/
│   └── skills.md       # Layer 1 contract — installed by `hafiz agent install`
└── scripts/            # One-off maintenance scripts

alembic/                # Schema migrations
tests/                  # pytest suite (test_chunker, test_cli, test_config, test_search)
COMMANDS.md             # Source of truth for commands — keep in sync with code
ROADMAP.md              # Vision + data model + future work
BRAIN_AGENT_GUIDE.md    # Agent-integration playbook
hafiz.toml.example      # Config template
```

## Two-Layer Stability Model

A load-bearing architectural invariant — respect the boundary.

- **Layer 1 (stable):** [hafiz/data/agents/skills.md](hafiz/data/agents/skills.md) — the contract between Hafiz and every AI agent. Installed by `hafiz agent install`. Changes here ripple to every user's workflow; treat additions conservatively and never break the documented shape of commands (flags, JSON schema, exit codes).
- **Layer 2 (evolving):** `hafiz review` and related logic in [hafiz/core/review.py](hafiz/core/review.py). Free to iterate; must not be reachable from the Layer 1 contract.

When a change touches Layer 1, call out the cross-cutting impact and prefer a Layer 2 alternative when one exists.

## Development Workflow

### Editable install
```bash
pipx install -e ".[gpu,dev]" --force   # drop [gpu] if no CUDA
```

### Run tests
```bash
pytest                     # full suite
pytest tests/test_cli.py   # single file
pytest -k chunker -x       # one case, stop on first fail
```

### Lint & format
```bash
ruff check .
ruff format .
```

### Database migrations
```bash
alembic revision --autogenerate -m "<summary>"
alembic upgrade head
```
Migrations target the DB configured under `[database].url` in `hafiz.toml`. Never edit a shipped migration — add a new one.

### Dogfooding
Hafiz indexes itself. After non-trivial changes, re-ingest and spot-check:
```bash
hafiz ingest . --project hafiz --prune
hafiz status --diagnose
```

## Conventions

- **Async end-to-end.** Anything touching the DB is `async def`. Command functions in `hafiz/commands/` wrap with `asyncio.run(...)`; never call blocking DB code from a running loop.
- **Core ↔ Commands split.** `hafiz/core/` holds business logic with no Typer/Rich imports. `hafiz/commands/` is presentation only: arg parsing, `--json` vs. rich output, exit codes.
- **`--json` is non-negotiable on user-facing commands.** Agents parse it; keep shapes stable and document new fields in [COMMANDS.md](COMMANDS.md).
- **Scoping flags.** Any new search-ish command takes `--project` and `--workspace` (mutually exclusive), following the resolution rules in [hafiz/core/search.py](hafiz/core/search.py).
- **No API-key-dependent paths.** Extraction is agent-driven only (direct-LLM path removed in `9440b3f`). Don't reintroduce one.
- **Error surfacing.** Human output: Rich panel + `typer.Exit(code=...)`. JSON output: `{"ok": false, "error": "..."}` to stdout + non-zero exit.
- **Paths.** Store and compare **absolute** paths in the DB; accept relative from the CLI and resolve immediately.

## Adding a New Command

1. Implement logic in `hafiz/core/<area>.py` as an async function returning plain data.
2. Add a thin wrapper in `hafiz/commands/<area>.py` that handles `--json` vs. rich output.
3. Register it in [hafiz/cli.py](hafiz/cli.py) (directly or via a sub-`Typer`).
4. Add a row to [COMMANDS.md](COMMANDS.md) — purpose, brain type, agent use, terminal use.
5. Add a Typer `CliRunner` test in [tests/test_cli.py](tests/test_cli.py).
6. If it changes the agent contract, update [hafiz/data/agents/skills.md](hafiz/data/agents/skills.md) and flag it in the PR.

## Reference Docs

- [COMMANDS.md](COMMANDS.md) — command map, brain types, scoping flag semantics.
- [ROADMAP.md](ROADMAP.md) — vision and data model.
- [BRAIN_AGENT_GUIDE.md](BRAIN_AGENT_GUIDE.md) — agent integration playbook.
- [README.md](README.md) — end-user install/setup.
