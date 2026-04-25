<!-- Installed by hafiz — workspace intelligence layer -->
<!-- SKILLS_VERSION: 8 -->
# Hafiz — Workspace Intelligence (v8)

IMPORTANT: You have access to `hafiz`, a CLI tool that is the
user's **sovereign second brain** — not just code indexing. It tracks
code structure via AST parsers, attaches agent-authored meaning via
annotations, and preserves a git-aware history across branches and
rewrites. Always use `--json` when parsing output programmatically.

## Two storage layers (load-bearing)

Hafiz separates **knowledge** (curated, identity-stable, mid-volume)
from **source** (firehose, immutable, retention-bounded). They live
side-by-side; agents must treat them differently.

| Layer | Tables | Examples | Default visibility |
|---|---|---|---|
| **Knowledge** | `units` · `unit_revisions` · `embeddings` · `edges` · `annotations` · `files` · `commits` | code, docs, decisions, learnings, patterns, runbooks | surfaced by default |
| **Source** | `sessions` · `communications` · `communication_messages` · `annotation_targets` | imported agent transcripts, future events | **hidden by default; opt-in via `hafiz recall` or `--include-transcripts`** |

**Rule of thumb:** the wisdom layer is primary. Source rows surface
only when you explicitly ask for them. Don't dilute default queries
with raw transcripts.

## Ownership rule (load-bearing)

Parsers own structure. Agents own meaning.

- **Structural facts** — classes, functions, modules, imports, calls,
  inheritance. Produced by AST parsers at ingest time. You do not
  write these; `hafiz extract import` rejects them.
- **Semantic facts** — decisions, learnings, patterns, warnings,
  concepts, workarounds. Produced by you, via `hafiz observe` for
  free-floating wisdom or `hafiz extract import` for unit-bound
  annotations + semantic edges.

Units are namespaced by kind (`code.function`, `code.class`,
`doc.heading`, `mail.message`, `file.raw`, …). Note that `chat.turn`
is no longer a unit kind — agent transcripts live in the source layer
(`communications` + `communication_messages`).

Annotation kinds: `fact` · `decision` · `learning` · `pattern` ·
`warning` · `note` · `concept` · `service`.

## Required Behaviors

You MUST follow these rules in every session:

1. **Before starting ANY coding task**, gather context first:
   ```bash
   hafiz context "<task description>" --json
   ```

2. **Before refactoring or modifying a unit**, check what depends on it:
   ```bash
   hafiz graph impact <UnitName> --json
   ```

3. **After making an architectural decision**, record it:
   ```bash
   hafiz observe "<what was decided and why>" --type decision --source agent:<your-name>
   ```

4. **After discovering a gotcha or non-obvious behavior**, record it:
   ```bash
   hafiz observe "<the gotcha>" --type warning --source agent:<your-name>
   ```

5. **Before implementing a significant change, record the decision driving it.**
   Trigger: you've decided on an approach and are about to start editing. For
   agents with a pre-implementation review step (multi-perspective gate, plan
   approval, etc.), fire this right after the user approves the synthesized
   plan and before the first edit.

   ```bash
   hafiz observe "<what we're doing and why — include the alternative rejected>" \
     --type decision --source agent:<your-name>
   ```

   When the approach shifts mid-task, observe the pivot. When it replaces an
   earlier recorded decision, use `--supersedes <old-id>`.

   Carve-out: mechanical edits the user explicitly scoped ("rename X to Y",
   "apply this diff") don't need an observe — no real decision is being made.

6. **Route "remember" writes to hafiz — the sovereign memory layer.**
   When the user says "remember X", "save this", "don't forget", or similar,
   write it to hafiz (never to an agent-local memory store):

   ```bash
   hafiz observe "<what to remember>" \
     --type <decision|fact|learning|warning> --source user:<name>
   ```

   Pick the kind from content: preference/rule → `learning`; claim/truth →
   `fact`; architectural choice → `decision`; gotcha → `warning`.

   At session start, pull durable user-scoped preferences so prior
   "remembers" shape your behavior without the user re-asking:

   ```bash
   hafiz query --recall --source user:<name> --limit 50 --json
   ```

   Carve-out: truly harness-specific preferences (slash commands, hook
   configs, IDE settings) may stay in whatever agent-local store exists —
   they won't port across agents anyway.

## Core Commands

| Command | When to use |
|---------|-------------|
| `hafiz context "<task>"` | **First thing** — bundle of relevant units, graph neighborhood, and annotations |
| `hafiz context "<task>" --include-transcripts` | Same as above, plus matching turns from imported transcripts (opt-in source layer) |
| `hafiz query "<text>" --json` | Semantic search over indexed content (knowledge layer) |
| `hafiz query "<text>" --include-transcripts --json` | Add matching transcript turns to results, tagged ``layer="source"`` |
| `hafiz query "<text>" --include-domain code --json` | Restrict to a data domain (``code``, ``doc``, ``chat``, …). Inverse: ``--exclude-domain``. Comma-separated for multiple. |
| `hafiz query "<topic>" --recall --type decision --json` | Search annotations (decisions, facts, learnings, patterns, warnings) |
| `hafiz recall <session-or-comm-id>` | List ordered messages from a session/communication. Add ``--query "<text>"`` for similarity search across the session's turns. Source-layer access — use deliberately. |
| `hafiz graph deps <name> --json` | What this unit depends on (outgoing edges) |
| `hafiz graph impact <name> --json` | Blast radius — what depends on this unit (incoming edges) |
| `hafiz observe "<text>" --type <kind> --source agent:<name>` | Record a decision / warning / pattern / learning / fact |
| `hafiz observe "<text>" --derived-from <id>,<id>` | Cite annotations OR messages OR sessions — polymorphic via the ``annotation_targets`` pivot |
| `hafiz note "<text>" --source agent:<name>` | Capture a raw thought — anything below decision-grade |
| `hafiz journal --since 7d --json` | "What did I record recently?" — annotations grouped by day |
| `hafiz distill --since 7d --json` | Promotable notes + transcript turns with a ready observe scaffold |
| `hafiz import claude-code --project <name>` | Post-hoc importer for Claude Code session JSONL (idempotent) |
| `hafiz forget <comm-or-session-id> [--hard]` | Redact source-layer rows. Soft tombstone by default; ``--hard`` deletes content + messages |
| `hafiz forget --all-expired` | Sweep mode — tombstone every communication past its retention_until |

## Capture → Distill Workflow

The reasoning loop:

1. **Capture raw.** Half-formed thoughts go in without ceremony:
   ```bash
   hafiz note "Wondering if refresh tokens should live in httponly cookies"
   ```

2. **Review.** `hafiz journal --since 7d` groups by day.

3. **Distill.** `hafiz distill --since 7d` lists promotable notes.
   Hafiz does **not** call an LLM — you are the distiller. Promote via:
   ```bash
   hafiz observe "<distilled decision>" --type decision --derived-from <note-id>,<note-id>
   ```

4. **Supersede when things change.** Never silently delete — write the
   new decision with `--supersedes <old-id>` so the old stays auditable:
   ```bash
   hafiz observe "<new decision>" --type decision --supersedes <old-id>
   ```

Sessions (optional) group everything you record in one terminal:
```bash
hafiz session start "jwt-migration" --task auth --project my-project
# subsequent observe / note auto-tag with session_id + task
hafiz journal --session <id>     # pull one thread of work
```

Sessions are now **DB-backed**: the per-TTY JSON is a cursor, the
record lives in the ``sessions`` table. Every annotation written
inside a session links to it via FK; ``hafiz observe --session <slug>``
resolves the slug to the uuid and populates both columns.

## Source layer — agent transcripts (opt-in)

Hafiz can ingest agent-harness transcripts into a dedicated source
layer, so prior conversations are queryable without polluting default
search. **Defaults are conservative**: source rows hide from
``hafiz query`` and ``hafiz context`` until you opt in.

```bash
# Idempotent post-hoc import. Re-running is a no-op.
hafiz import claude-code --project hafiz

# Explicit recall of one session's turns, in order.
hafiz recall <session-slug-or-uuid> --json

# Vector search inside one session.
hafiz recall <session> --query "auth flow" --json

# Add transcript matches to a normal query (clearly tagged in output).
hafiz query "auth flow" --include-transcripts --json

# Cite specific turns when distilling — polymorphic --derived-from.
hafiz observe "<distilled decision>" --type decision \
  --derived-from <message-id>,<message-id>
```

Retention is bounded (default 90 days from started_at). The user's
explicit redaction commands:

```bash
hafiz forget <comm-id-or-session-slug>          # soft tombstone
hafiz forget <comm-id-or-session-slug> --hard   # delete content + messages
hafiz forget --all-expired                      # sweep retention_until
```

**Selective embedding** is enforced at import time: short turns and
pure tool-result echoes don't get embedded. Salient turns are still
written (``content`` is canonical) — just not vector-indexed.

## Agent Extraction (v2)

When you want to attach structured annotations or semantic edges to
the parsed code graph in bulk — beyond the one-shot `hafiz observe` —
use the extract pipeline.

**Step 1 — see what's already parsed:**
```bash
hafiz extract export --project <name> --json > /tmp/units.json
```
Output is the AST-known units (with their stable `identity_key`) and
structural edges. Use it to know which units exist before you
annotate them — don't re-derive structure.

**Step 2 — produce an agent-extraction payload (v2):**
```json
{
  "version": 2,
  "annotations": [
    {
      "content": "Canonical auth entry — all routes funnel through here",
      "kind": "pattern",
      "source": "agent:claude-code",
      "unit_identity_key": "<copy from step 1>",
      "confidence": 0.9,
      "tags": ["auth"]
    }
  ],
  "edges": [
    {
      "source_name": "UserService",
      "source_file": "/abs/path/auth.py",
      "target_name": "SecurityPolicy",
      "target_file": "/abs/path/policy.py",
      "relation": "implements_pattern",
      "evidence": "canonicalized via policy_engine.enforce(...)"
    }
  ]
}
```

Rules:
- **Kinds:** annotations only — `fact` · `decision` · `learning` ·
  `pattern` · `warning` · `note` · `concept` · `service`.
  Anything starting `code.*` is rejected (that's the parser's job).
- **Relations:** semantic only — `implements_pattern` · `is_workaround_for` ·
  `supersedes_approach` · `depends_on_concept` · `related_to` ·
  `documents` · `configures`. Structural relations
  (`calls` / `imports` / `inherits` / `references`) are rejected.
- **Unit references:** prefer `unit_identity_key` (copied from export).
  Fall back to `(unit_name, source_file)` — may be ambiguous when a
  name repeats across files.
- **Source:** always `agent:<your-name>` or `user:<name>`.

**Step 3 — import:**
```bash
cat /tmp/extraction.json | hafiz extract import --project <name>
```

The import surfaces per-row warnings (unknown kind, unresolved unit)
and counts them in the summary. Unresolved references are stored with
null unit_id / target_unit_id; they become resolvable on a later pass
once the parser catches up.

## Ingest (for reference — users usually drive this)

```bash
hafiz ingest <path> --project <name>
```

Hafiz walks the tree (gitignore-aware), picks a parser per file via
the registry, and upserts units / revisions / embeddings. On a git
repo, re-ingests are diff-driven: only files changed since the last
indexed commit are re-parsed. Rebases and amends fire the
`post-rewrite` hook automatically if installed via `hafiz hooks install`.

## Error reporting

Hafiz captures every unhandled exception into a sovereign,
user-scope error log at `~/.cache/hafiz/errors.log` (NDJSON, capped
at 1000 entries). Use it when the user says "hafiz feels broken"
or you want to understand why a recent command misbehaved.

```bash
hafiz errors list --since 1d --json       # newest-first, agent-consumable
hafiz errors show <id> --json             # full traceback + suggested fix
hafiz errors clear                        # reset the log
```

Each record carries: `timestamp`, `command`, `argv`, `exception_type`,
`message`, `traceback`, `cwd`, `hafiz_version`, `git_branch`,
`host_fingerprint`, plus — for recognized error classes — a
`suggested_action` string and structured `context` (e.g.,
`{"missing_module": "scipy", "is_declared_dep": true}`).

The suggestion is informational. Offer it to the user; don't
auto-run remedial commands without explicit opt-in — even safe
ones like `pipx inject` mutate the user's environment.

`hafiz doctor` surfaces the last-24h error count inline so you can
see at a glance whether something's been failing.

## Self-tuning the install

Hafiz has a tunable registry (RAM-sensitive knobs like embedding
part size, policy caps like ingest file-size guard). You can probe
the host and recommend values, and you can persist those
recommendations to a sticky cache so subsequent runs pick them up.

Use this when the user asks "what are the best settings for this
machine?" or reports slowness / OOM / crashes during ingest. Do not
run it unprompted on every session.

```bash
# Read-only: check current config + recommended values for this host.
hafiz doctor --probe --json
```

The `--json` shape is documented in docs/commands.md under the Setup
section; key fields:

- `host.{ram_total_mb, ram_available_mb, cpu_count, onnx_providers, gpu_name, gpu_vram_free_mb, fingerprint}`
- `tuning[i].{key, current, recommended, rationale, confidence, probe_error}`

If the user agrees with the recommendations, persist them to the
sticky cache (user-scope, does not modify `hafiz.toml`):

```bash
hafiz config apply        # or:  hafiz doctor --apply
```

For one-off edits to the checked-in config, use `hafiz config set`
(writes TOML). Unset removes the key and falls through to sticky /
default:

```bash
hafiz config set embedding.max_part_chars 4096
hafiz config unset embedding.max_part_chars
hafiz config get embedding.max_part_chars --json   # shows source layer
hafiz config clear-sticky                          # wipe probed cache
```

**Resolution order** (every tunable reads through this chain):

    env (HAFIZ_*__*)  →  hafiz.toml  →  sticky cache  →  built-in default

Sticky is keyed to a host fingerprint (RAM class + GPU presence +
OS/arch + ORT provider set); moving a laptop-tuned cache to a
workstation is a no-op, not a hazard.

---

## Reference

<details>
<summary>Full command reference and flags</summary>

### Search & Context

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz context "<task>"` | Context bundle (units + graph + annotations) | `--project`, `--workspace`, `--include-domain`, `--exclude-domain`, `--json` |
| `hafiz query "<text>"` | Semantic search over indexed embeddings | `--type` (unit kind), `--project`, `--workspace`, `--include-domain`, `--exclude-domain`, `--limit`, `--json` |
| `hafiz query "<text>" --recall` | Semantic search over annotations | `--type`, `--project`, `--workspace`, `--limit`, `--json` |

- **Domain filter** (on `query` / `context`): `--include-domain code,doc` or `--exclude-domain code` toggle whole data domains. Domain = the part of `kind` before the dot (`code`, `doc`, `chat`, `mail`, `file`). For exact-kind filtering, use `--type code.function`. Mutually exclusive *per-domain*: `--include-domain code --exclude-domain code` errors. `session start --include-domain ...` persists a default for the TTY.

### Knowledge Graph

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz graph show <name>` | Unit and its direct connections | `--project`, `--json` |
| `hafiz graph deps <name>` | What this unit depends on (outgoing) | `--project`, `--json` |
| `hafiz graph impact <name>` | Blast radius — what depends on it | `--project`, `--json` |
| `hafiz graph path <from> <to>` | Shortest directed path | `--project`, `--json` |
| `hafiz graph rank` | Top units by centrality | `--metric`, `--top`, `--project`, `--json` |
| `hafiz graph stats` | Overall graph health | `--project`, `--top-central`, `--json` |

### Annotations (the "wisdom layer")

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz observe "<text>"` | Store a fact / decision / learning / pattern / warning | `--type`, `--source`, `--project`, `--tags`, `--confidence`, `--expires-in`, `--expires`, `--session`, `--task`, `--supersedes`, `--derived-from`, `--json` |
| `hafiz note "<text>"` | Low-bar capture — `kind="note"` | same as `observe` minus `--type` |
| `hafiz journal` | Time-bounded digest grouped by day | `--since`, `--day`, `--project`, `--workspace`, `--source`, `--type`, `--session`, `--task`, `--limit`, `--json` |
| `hafiz distill` | Promotable notes (scanner; no LLM call) | `--since`, `--project`, `--session`, `--task`, `--limit`, `--json` |
| `hafiz session start "<name>"` | Per-TTY session; subsequent writes auto-tag, and `--include-domain`/`--exclude-domain` become defaults for `query`/`context` in this terminal | `--task`, `--project`, `--include-domain`, `--exclude-domain`, `--json` |
| `hafiz session show` / `end` | Inspect / clear | `--json` |

- **Annotation kinds**: `fact` · `decision` · `learning` · `pattern` · `warning` · `note` · `concept` · `service`
- **Source format**: `agent:claude-code`, `agent:cursor`, `agent:copilot`, `user:<name>`
- **Expiration** (`observe` / `note`): `--expires-in 30d|2w|6m|1y` or `--expires 2026-06-01`. Sets `valid_until`; expired rows are hidden from `--recall` by default.
- **Git auto-captured**: `commit_hash` on every write inside a repo; `branch` / `is_dirty` on annotations.
- **Staleness**: `--recall` shows age (`3mo ago`) and dims rows older than 90d.
- **Supersession**: replace a decision with `--supersedes <old-uuid>`; prefer over silent deletion.
- **Lineage**: `--derived-from <ids>` records distillation source without replacing.

### Extraction (agent contract v2)

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz extract export` | Emit the AST-known units + structural edges to attach annotations to | `--project`, `--limit`, `--pretty` |
| `hafiz extract import` | Import v2 payload (annotations + semantic edges) from JSON | `--file`, `--project` |

### Indexing & Maintenance

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz ingest <path>` | Parse + embed + store. Diff-driven on re-runs. | `--project`, `--git-hook`, `--json` |
| `hafiz status` | Counts across all tables; last-indexed commit per project | `--json`, `--diagnose` |
| `hafiz init` | Create schema + pgvector extension | — |
| `hafiz hooks install <repo>` | Write post-commit / post-merge / post-rewrite hooks | `--project` |
| `hafiz agent install` | Splice this skills.md into an agent config | — |

### Source layer (transcripts)

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz import claude-code [path]` | Idempotent import of Claude Code session JSONL into ``communications`` + messages | `--project`, `--limit`, `--since`, `--dry-run`, `--no-embed`, `--json` |
| `hafiz recall <target>` | Ordered messages for a session/communication, optionally vector-searched | `--query`, `--role`, `--from`, `--to`, `--has-tool-call`/`--no-tool-call`, `--limit`, `--json` |
| `hafiz forget <target>` | Targeted redaction (soft tombstone by default). | `--hard`, `--json` |
| `hafiz forget --all-expired` | Sweep mode — tombstone every communication past `retention_until`. | `--dry-run`, `--json` |

### Error reporting

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz errors list` | Recent errors, newest first. Each record includes `suggested_action` + `context` for recognized classes. | `--since`, `--limit`, `--json` |
| `hafiz errors show <id>` | Full structured record: traceback, cwd, git branch, host fingerprint. Accepts a unique-prefix id. | `--json` |
| `hafiz errors clear` | Wipe the log. Returns the count discarded. | `--json` |

### Self-tuning

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz doctor` | Install health + host capabilities + tunable registry. Shape: `{checks, host, tuning, applied}`. | `--json`, `--probe`, `--apply` |
| `hafiz doctor --probe` | Adds measured recommendations to each tunable row (`recommended` / `rationale` / `confidence`). Slow — loads fastembed. | `--json` |
| `hafiz doctor --apply` | Implies `--probe`. Persists recommendations to `~/.cache/hafiz/tuning_state.json`. JSON gains `applied`. | `--json` |
| `hafiz config show` | Current TOML values + per-tunable resolution source (env / toml / sticky / default). | `--json` |
| `hafiz config get <key>` | One tunable's effective value + source. | `--json` |
| `hafiz config set <key> <value>` | Persist to user-scope `~/.config/hafiz/hafiz.toml` (or project `./hafiz.toml` with `--local`). | `--local`, `--json` |
| `hafiz config unset <key>` | Remove from hafiz.toml. Prunes emptied tables. | `--local`, `--json` |
| `hafiz config apply` | Run probers + persist to sticky cache. Same as `doctor --apply`, narrower JSON summary. | `--json` |
| `hafiz config clear-sticky` | Wipe the probed recommendations cache. | `--json` |

### Data model — knowledge layer (the seven tables)

- `files` — one row per file ever seen (tombstoned via `valid_until`).
- `units` — stable identity of an addressable thing (function, heading, …). `kind` is namespaced.
- `unit_revisions` — append-only versioned body; at most one `superseded_at IS NULL` per unit.
- `embeddings` — 1:N vector search index over revisions (oversized bodies split into parts).
- `edges` — append-only relations; `source ∈ {ast, agent, user}`.
- `annotations` — decisions / facts / learnings. May link to a unit or float free. ``session_id`` is now a uuid FK to ``sessions``; the historical text slug is preserved as ``legacy_session_id``.
- `commits` — git axis; populated on ingest. `rewritten_at` marks orphaned commits.

### Data model — source layer

- `sessions` — engineer/agent threads of work. ``slug`` is the human-facing identifier; ``id`` is the canonical uuid that other tables FK against.
- `communications` — agent transcripts, chat threads. Idempotent by ``(agent, external_id)``. ``retention_until`` defaults to ``started_at + 90 days``.
- `communication_messages` — append-only turns. ``content`` is canonical (NOT NULL); ``embedding`` is nullable and populated only when the message clears the selective-embed policy. ``seq`` is monotonic per-communication.
- `annotation_targets` — polymorphic pivot. An annotation may cite units, other annotations, messages, communications, or sessions via ``target_kind`` + ``target_id`` + ``relation``.

</details>

<!-- /Installed by hafiz — do not edit above this block; re-run `hafiz agent install` to update -->
