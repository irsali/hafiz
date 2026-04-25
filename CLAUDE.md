# Hafiz — Project Development Guide

This is the Hafiz codebase: a sovereign, CLI-first intelligence layer backed by PostgreSQL + pgvector. When Claude is invoked here, the work is **developing the product** (features, bugfixes, refactors, tests, migrations) — not consuming it.

> For using the `hafiz` CLI as a coding assistant, see the global skills file installed by `hafiz agent install` (already loaded via `~/.claude/CLAUDE.md`). Do **not** duplicate those instructions here.

## Product Stance

Hafiz is **pre-1.0 and in active development** — there is no legacy to protect. Prefer bold, opinionated decisions over incremental compatibility hacks. Breaking changes are acceptable, and often preferable, when they make the product simpler, safer, or more intuitive.

The north star is a **user-friendly public repo** that scales to thousands of installers and consumers. Evaluate every change through that lens:

- **Install and first-run must be frictionless.** A new user on a fresh machine should reach a working `hafiz status` in minutes, with clear, actionable errors when something is off.
- **Defaults beat flags.** Optimize the common case; reserve flags for genuine optionality.
- **The CLI surface *is* the product.** Human output must be scannable; `--json` shapes must be stable and documented in [commands.md](docs/commands.md).
- **Docs, `--help` text, and error messages are features**, not afterthoughts — treat them as first-class deliverables.
- **Dependencies and system requirements are costs borne by users.** Justify each one; avoid heavyweight additions when a lighter path exists.
- **Product engineering > code archaeology.** When a design has drifted, propose a clean replacement rather than layering workarounds.

When trade-offs arise, favor the experience of the thousandth user over the convenience of the current maintainer. Flag cross-cutting or breaking changes clearly in the plan, but don't shy back from them.

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
workitems/              # Local-only backlog & planning (gitignored) — see "Work Items" below
docs/                   # All narrative documentation
├── README.md           # Index of this directory
├── architecture.md     # System + data model + key flows + capture gap analysis
├── commands.md         # Source of truth for commands — keep in sync with code
├── roadmap.md          # Vision + data model + future work
└── agents.md           # Agent-integration playbook (stale — see architecture.md + skills.md)
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

## Work Items & Backlog Continuity

Multi-step work, deferred decisions, and "let's come back to this" ideas live in `workitems/` as individual markdown files. **Nothing discussed in one session should be lost by the next.**

`workitems/` is gitignored — these are a personal backlog, not a shared artifact. [docs/roadmap.md](docs/roadmap.md) is the public/shared equivalent (one-liners and vision); work items are the private design docs behind them.

### When to create one

- Any task spanning more than one session or one PR.
- Any decision made now that affects future work (so we don't re-litigate).
- Any deferral — "do this after Phase X", "revisit once Y lands".
- Any ROADMAP entry that has a concrete approach sketched — the ROADMAP line is the one-liner; the work item is the design.

In-session todos use the in-conversation task list, not workitems.

### Layout

- `workitems/active/<slug>.md` — proposed or in-progress.
- `workitems/archive/<slug>.md` — deferred (decided not now, but not abandoned). Set `revisit_when:` so a future read knows what would justify reopening.
- `workitems/done/<slug>.md` — shipped or abandoned; kept for audit.
- Slug-based filenames (kebab-case). Move between folders as state changes; **never delete**.

### Standard template

YAML frontmatter for machine-parseable state, then body:

```markdown
---
title: Short title
description: One-line hook — what this is, in a sentence. Required.
status: active            # proposed | active | blocked | deferred | done | abandoned
created: 2026-04-21
updated: 2026-04-21
owner: irshad
# revisit_when: <signal that would justify reopening>   # required when status: deferred
related:
  - docs/roadmap.md#open-work
  - workitems/active/other-item.md
---

## Objective
One paragraph — what and why. A cold reader should be able to judge whether this is still worth doing.

## Scope
- **In:** …
- **Out:** … (explicit, to prevent creep)

## Decisions
Locked-in choices with rationale — written to stop re-litigation.
- **Chose X over Y** because … (date)
- **Rejected Z** because …

## Plan
Phased checklist. Each phase independently shippable.
- [ ] Phase 1 — name — deliverable
- [ ] Phase 2 — …

## Open questions

## Notes
Append-only session log, latest on top.
- 2026-04-21 — initial plan.
```

### Lifecycle

1. Create with `status: proposed` when a plan crystallizes.
2. Flip to `active` when work starts.
3. Update `Notes` + `Plan` checkboxes at the end of each session that touches it; bump `updated`.
4. On completion, move to `done/`, set `status: done` or `abandoned`, append a short outcome.
5. To shelve without abandoning, move to `archive/`, set `status: deferred`, fill in `revisit_when:`, and append a Notes entry explaining the deferral. Reopening = move back to `active/` and bump status.

### Session handover

Before planning any non-trivial task, check `workitems/active/`. At session end, update whichever work item you touched.

## Conventions

- **Async end-to-end.** Anything touching the DB is `async def`. Command functions in `hafiz/commands/` wrap with `asyncio.run(...)`; never call blocking DB code from a running loop.
- **Core ↔ Commands split.** `hafiz/core/` holds business logic with no Typer/Rich imports. `hafiz/commands/` is presentation only: arg parsing, `--json` vs. rich output, exit codes.
- **`--json` is non-negotiable on user-facing commands.** Agents parse it; keep shapes stable and document new fields in [commands.md](docs/commands.md).
- **Scoping flags.** Any new search-ish command takes `--project` and `--workspace` (mutually exclusive), following the resolution rules in [hafiz/core/search.py](hafiz/core/search.py).
- **No API-key-dependent paths.** Extraction is agent-driven only (direct-LLM path removed in `9440b3f`). Don't reintroduce one.
- **Error surfacing.** Human output: Rich panel + `typer.Exit(code=...)`. JSON output: `{"ok": false, "error": "..."}` to stdout + non-zero exit.
- **Paths.** Store and compare **absolute** paths in the DB; accept relative from the CLI and resolve immediately.

## Adding a New Command

1. Implement logic in `hafiz/core/<area>.py` as an async function returning plain data.
2. Add a thin wrapper in `hafiz/commands/<area>.py` that handles `--json` vs. rich output.
3. Register it in [hafiz/cli.py](hafiz/cli.py) (directly or via a sub-`Typer`).
4. Add a row to [commands.md](docs/commands.md) — purpose, brain type, agent use, terminal use.
5. Add a Typer `CliRunner` test in [tests/test_cli.py](tests/test_cli.py).
6. If it changes the agent contract, update [hafiz/data/agents/skills.md](hafiz/data/agents/skills.md) and flag it in the PR.

## Reference Docs

- [docs/architecture.md](docs/architecture.md) — system levels, data model, key flows, capture gap analysis.
- [docs/commands.md](docs/commands.md) — command map, brain types, scoping flag semantics.
- [docs/roadmap.md](docs/roadmap.md) — vision and data model.
- [docs/agents.md](docs/agents.md) — agent integration playbook (stale; authoritative refs are `skills.md` + `architecture.md`).
- [README.md](README.md) — end-user install/setup.
