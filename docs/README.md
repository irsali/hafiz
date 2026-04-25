# Hafiz — Documentation

Narrative documentation for Hafiz lives here. Project-root files cover install,
conventions, and legal; this directory covers the **what, how, and where it's
going**.

| File | What it is | Read when… |
|---|---|---|
| [architecture.md](architecture.md) | Five-level view of the system (context, architecture+ingest, module map, seven-table data model, key flows), plus a code-grounded analysis of what Hafiz captures today and where the gaps are. | You want to understand how Hafiz works end-to-end, or decide where to fill a capability gap. |
| [commands.md](commands.md) | Authoritative command reference — flags, `--json` shapes, brain types, agent-vs-human surfaces. | You're adding a command, wiring an agent, or parsing `--json` output. |
| [roadmap.md](roadmap.md) | Product vision, shipped phases, open work, design principles. | You want to know *why* Hafiz is shaped the way it is, or what's coming next. |
| [agents.md](agents.md) | Legacy agent-integration playbook. **Stale** — predates the Structural Grounding rewrite. | Only for historical context. The live agent contract is [`../hafiz/data/agents/skills.md`](../hafiz/data/agents/skills.md); the live architecture is [architecture.md](architecture.md). |

## What lives elsewhere

- [`../README.md`](../README.md) — install, first-run, quick-start.
- [`../CLAUDE.md`](../CLAUDE.md) — project development guide (conventions, layout, how to add a command, work-item lifecycle).
- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) — issue reporting + PR flow.
- [`../TRADEMARK.md`](../TRADEMARK.md) — name and mark policy.
- [`../hafiz/data/agents/skills.md`](../hafiz/data/agents/skills.md) — **the Layer 1 agent contract** shipped in the wheel and installed by `hafiz agent install`. Path-pinned; do not move.
- [`../workitems/`](../workitems/) — personal backlog and design docs (gitignored). `roadmap.md` is the public/shared equivalent.

## Conventions for docs in this directory

- Lowercase filenames (`commands.md`, not `COMMANDS.md`).
- Link targets are **repo-relative**: `../CLAUDE.md`, `../hafiz/core/store.py`, siblings just by name (`architecture.md`).
- Cite `file:line` when referencing code, so links survive refactors better than paraphrases.
- When a fact is grounded in code, link the code — don't re-narrate it here.
