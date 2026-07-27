<!-- Installed by hafiz — workspace intelligence layer -->
<!-- SKILLS_VERSION: 12 -->
# Hafiz — Workspace Intelligence (v13)

IMPORTANT: You have access to `hafiz`, a CLI tool that is the
user's **sovereign second brain** — not just code indexing. It tracks
code structure via AST parsers, attaches agent-authored meaning via
annotations, and preserves a git-aware history across branches and
rewrites. Always use `--json` when parsing output programmatically.

Wherever you see `<your-name>` in this file (e.g.
`--source agent:<your-name>`), substitute the agent you actually are
— for example `claude-code`, `cursor`, `copilot`, or `aider`. This
tag is how hafiz attributes writes; never leave the literal string
`<your-name>` in a real command.

## Two storage layers (load-bearing)

Hafiz separates **knowledge** (curated, identity-stable, mid-volume)
from **source** (firehose, immutable, retention-bounded). They live
side-by-side; agents must treat them differently.

| Layer | Tables | Examples | Default visibility |
|---|---|---|---|
| **Knowledge** | `units` · `unit_revisions` · `embeddings` · `edges` · `annotations` · `files` · `commits` | code, docs, decisions, learnings, patterns, runbooks | surfaced by default |
| **Source** | `sessions` · `communications` · `communication_messages` · `annotation_targets` · `retrievals` | imported agent transcripts, recorded searches, future events | **hidden by default; opt-in via `hafiz recall` or `--include-transcripts`** |

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
   hafiz query "<what you're about to work on>" --observations \
     --source user:<name> --limit 12 --min-score 0.05 \
     --format compact --with-ids
   ```

   Always pass `--min-score` and `--format compact` on anything you inject
   into your own context. Without them the same query returns ~17k tokens
   of which ~1/3 is uuids and timestamps you'll never read; with them it's
   ~600 tokens of the rows that actually matched. See **Retrieval hygiene**
   below.

   Carve-out: truly harness-specific preferences (slash commands, hook
   configs, IDE settings) may stay in whatever agent-local store exists —
   they won't port across agents anyway.

7. **Never inject an unbounded, unfiltered result set.** See
   **Retrieval hygiene** — this is the difference between hafiz being
   consulted per-task and being ignored as too expensive.

## Retrieval hygiene (read before wiring hafiz into a hook)

Three flags, on both `query` and `context`:

| Flag | Why |
|---|---|
| `--min-score 0.05` | 0–1 relevance floor on the score results are *ranked* by |
| `--format compact` | content + kind + source + age; drops uuids, timestamps, nulls, scores |
| `--with-ids` | re-adds ids — **required** if you might `--supersedes` later |

`--limit` also exists on `context` (caps each section).

**Which score the floor uses.** Annotation recall is cross-encoder reranked by
default, and `--min-score` filters the reranked score, not the raw cosine
`score`. That matters: under reranking `score` is *not* monotonic down the
result list, so a floor on it would drop a higher-ranked row and keep a
lower-ranked one.

**Why useful floors are low.** Measured on a 1,200-annotation store, three
on-topic against three off-topic questions, comparing the best score in each
result set:

| | cosine `score` | `rerank_score` |
|---|---|---|
| on-topic | 0.713 – 0.789 | 0.962 – 0.998 |
| off-topic | 0.480 – 0.523 | 0.0001 – 0.0002 |

Cosine leaves a 0.19 gap you have to calibrate into; the reranker leaves three
orders of magnitude. So one flag does both jobs — `--min-score 0.05` sits ~50×
above the off-topic ceiling and still keeps every genuinely relevant row, and
"nothing came back" *is* the not-relevant signal. No separate hit gate, no
set-max heuristic. `0.4` is aggressive tail-trimming; under `--no-rerank` the
floor applies to cosine, where the useful band is ~`0.5`–`0.65`.

`--json` output carries both `score` (cosine) and `rerank_score` (0–1, or `null`
when reranking didn't run), plus a top-level `reranked` boolean — that's how you
tell reranked output from vector output.

**A blank query is an error, not an empty result.** `query` / `context` exit `2`
with `{"ok": false, "error": ...}` on blank input. If you build a query by
interpolating a variable, check the exit code — an unset variable is a bug in
your hook, and hafiz will no longer paper over it with confidently-scored noise.

**`--format md` on an empty result set prints nothing at all** — no placeholder
line. Under a floor, "no rows" is the ordinary answer to an off-topic prompt, so
a per-task hook can pipe the output straight into context without filtering it.

**Wiring a per-task recall hook.** The whole thing is one command; everything
that used to need a wrapper script — relevance floor, hit gate, compact
rendering, dedup of byte-identical rows — is in the CLI now:

```bash
# Read the harness's prompt from stdin, not from an env var, and only recall
# for prompts substantial enough to have a topic.
PROMPT=$(jq -r '.prompt // empty')
[ ${#PROMPT} -lt 12 ] && exit 0
hafiz query "$PROMPT" --observations --limit 20 --min-score 0.05 --format md \
  2>/dev/null || true
```

Two failure modes worth naming, because both were observed in the wild for 3.5
weeks without anyone noticing:

- **Confirm your harness actually sets the variable you interpolate.** A hook
  built on `hafiz context "$CLAUDE_USER_PROMPT"` never delivered once across
  1,647 prompts — the harness passes the prompt as JSON on stdin and that name
  is unset. hafiz now exits `2` on the resulting blank query instead of
  answering it, but a hook ending in `|| true` still swallows that. Check it.
- **Never let recall fail a turn.** Time it out, swallow errors, exit 0. A
  memory layer that can break the conversation gets removed.

## Code and docs are not the same corpus

`query --observations` searches the wisdom layer. Plain `query` searches the
indexed units — and that index is **~89% documentation**, ~7% whole-file text,
and only ~4% code. Two consequences:

- **Don't dismiss the unit index as "the code index".** Prose questions
  ("how does X decide which rules apply") land on the doc corpus, which is
  exactly what grep can't reach from that wording.
- **Do check the freshness block before trusting code results.** `query` output
  carries `staleness` naming any project whose index trails its repo, with
  `commits_behind` / `is_ancestor`. When it's non-empty, verify against the
  working tree — and for the file you're actively editing, prefer reading the
  file. Staleness concentrates exactly where the work is: on a measured index,
  4.4% of files overall had changed since indexing, but 80.5% had in the one repo
  under active development.

`--include-domain code` / `--exclude-domain code` scope this deliberately.

## Retrieval telemetry

Every search is recorded locally (`retrievals` table) so the store can be
evaluated: what's never recalled, what's recalled constantly, and what was asked
for and **not found**. That last one is the useful one for you:

```bash
hafiz retrievals --since-days 7 --json
```

`unanswered` is a list of questions the store couldn't answer — i.e. a worklist
of things to `observe`. If you see a topic there that you *do* now know the
answer to, record it.

Recording never fails a search, is bounded by retention, and is switchable off
with `[telemetry] retrieval = false`. Query text stays on the machine.

## Core Commands

| Command | When to use |
|---------|-------------|
| `hafiz context "<task>"` | **First thing** — bundle of relevant units, graph neighborhood, and annotations |
| `hafiz context "<task>" --limit 5 --min-score 0.05 --format compact --with-ids` | The form to use when injecting into your own context — capped, filtered, token-lean |
| `hafiz context "<task>" --include-transcripts` | Same as above, plus matching turns from imported transcripts (opt-in source layer) |
| `hafiz query "<text>" --json` | Semantic search over indexed content (knowledge layer) |
| `hafiz query "<text>" --min-score 0.05 --format compact` | Same, with a relevance floor and the token-lean shape. `--format rich\|json\|compact\|md`; `--json` is an alias for `--format json` |
| `hafiz query "<text>" --include-transcripts --json` | Add matching transcript turns to results, tagged ``layer="source"`` |
| `hafiz query "<text>" --include-domain code --json` | Restrict to a data domain (``code``, ``doc``, ``chat``, …). Inverse: ``--exclude-domain``. Comma-separated for multiple. |
| `hafiz retrievals --since-days 7 --json` | What the store was asked for and couldn't answer — a worklist of things to record |
| `hafiz query "<topic>" --observations --type decision --json` | Search the wisdom layer (annotations: decisions, facts, learnings, patterns, warnings). *(Was `--recall`; the flag was renamed to end the collision with the command below. `--recall` still works as a hidden alias for one release, with a deprecation warning.)* |
| `hafiz recall <session-or-comm-id>` | List ordered messages from a session/communication (source layer). Add ``--query "<text>"`` for similarity search across the session's turns. Use deliberately. *(`recall` is the source layer; `query --observations` is the wisdom layer — different jobs.)* |
| `hafiz graph deps <name> --json` | What this unit depends on (outgoing edges) |
| `hafiz graph impact <name> --json` | Blast radius — what depends on this unit (incoming edges) |
| `hafiz observe "<text>" --type <kind> --source agent:<name>` | Record a decision / warning / pattern / learning / fact |
| `hafiz observe "<text>" --derived-from <id>,<id>` | Cite annotations OR messages OR sessions — polymorphic via the ``annotation_targets`` pivot |
| `hafiz note "<text>" --source agent:<name>` | Capture a raw thought — anything below decision-grade |
| `hafiz journal --since 7d --json` | "What did I record recently?" — annotations grouped by day |
| `hafiz distill --since 7d --json` | Promotable notes + transcript turns with a ready observe scaffold |
| `hafiz reconcile --json` | Read-only sweep — clusters near-duplicate live annotations and emits the supersede/retire commands |
| `hafiz import claude-code --project <name>` | Post-hoc importer for Claude Code session JSONL (idempotent) |
| `hafiz forget <comm-or-session-id> [--hard]` | Redact source-layer rows. Soft tombstone by default; ``--hard`` deletes content + messages |
| `hafiz forget --all-expired` | Sweep mode — tombstone every communication past its retention_until |
| `hafiz export --out <dir>` | Sovereignty eject — dump the wisdom layer (observations, +``--include-transcripts``) to plain ``.md`` (``--format json`` for lossless JSONL). Excludes code and forgotten/expired rows. Complements ``forget`` |

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

4. **Handle contradicting knowledge — never overwrite.** When new
   information conflicts with something already recorded, Hafiz's model
   is *write-new, tombstone-old* (the old row is kept for audit, just
   hidden from default recall). Pick the path by what changed:

   | Situation | Action |
   |---|---|
   | New info **contradicts or replaces** an existing annotation | `hafiz observe "<new>" --type <kind> --supersedes <old-id>` — inserts the new row and atomically marks the old one inactive |
   | An existing annotation is **simply wrong**, with no replacement | `hafiz forget <old-id> --annotation` — retires it (soft tombstone; drops from recall, kept for audit) |
   | The info is **net-new** (no conflict) | plain `hafiz observe "<text>" --type <kind>` — no supersession |

   ```bash
   hafiz observe "<new decision>" --type decision --supersedes <old-id>
   hafiz forget <old-id> --annotation          # retire without a replacement
   ```

   **Guardrails:** supersede only on *genuine* contradiction or
   replacement — not on a reword or a related-but-still-true fact, or
   the brain fills with near-duplicates. **Verify the `<old-id>` first**
   (`hafiz query --observations "<topic>"`) so you don't bury a good row
   by superseding the wrong one. Both `--supersedes` and
   `forget --annotation` are *soft* tombstones — the row stays auditable
   and you can read prior beliefs with
   `hafiz query --observations "<topic>" --include-superseded`. Never use
   `hafiz forget --hard` on knowledge-layer annotations; `--hard` is for
   the source layer (transcripts) only.

   **Hafiz helps you catch this.** `hafiz observe` runs near-duplicate
   detection on write: if a similar *live* annotation already exists, the
   `--json` response carries `near_duplicates: [{id, content, kind,
   score}]` (and a hint in human output). When you see one, decide: does
   the new info **replace** it (re-run with `--supersedes <that-id>`), or
   is it genuinely distinct (write as-is)? Detection is surface-only by
   default — it never blocks — and is skipped for `note` and when you
   already passed `--supersedes`. To sweep for drift that predates a write
   or slipped through bulk imports, run `hafiz reconcile` (read-only;
   clusters near-duplicate live annotations and hands you the commands).

   **`reconcile` proposes; you run it.** Each cluster names one `primary`
   member and retires the rest, and `suggested_action` says what happens to
   the primary: `retire` (it's the newest row and no shorter than the ones it
   replaces, so it survives as written) or `merge` (the newest row is under
   80% of the longest — keeping only it would drop text, so the primary
   becomes the *longest* row and you write the merged text). `commands` is
   the ordered resolution, ready to run. It sweeps the **whole store** by
   default; if you pass `--limit` and it bites, the response says
   `truncated: true` rather than quietly reporting fewer clusters.

   **Treat every proposal as a suggestion, not a verdict.** Hafiz measures
   *similarity*, never *contradiction*, and near-duplicates are not always
   restatements — a real 91%-similar pair held a test account's email in one
   row and its password in the other. Read both rows before running anything;
   that is why lengths and dates ride along in the output.

   **Byte-identical writes are handled separately and unconditionally.** If
   `content` + `kind` + `source` + `project` all match a live row, `observe`
   **refuses** — exit `2`, with `existing_id` in the `--json` response. That
   is a prompt to decide: did the belief change (`--supersedes <existing_id>`),
   is this a refinement (edit the text), or do you genuinely want a second
   identical row (`--allow-duplicate`)? `note` instead **succeeds idempotently**:
   exit `0`, `deduped: true`, the existing row returned, nothing written — the
   raw-capture lane is never gated, it just stops storing the same byte twice.
   Blank content is refused on both.

Sessions (optional) group everything you record in one thread of work:
```bash
hafiz session start "jwt-migration" --task auth --project my-project
# subsequent observe / note auto-tag with session_id + task
hafiz journal --session <id>     # pull one thread of work
```

Sessions are **DB-backed**: the on-disk JSON is a cursor, the record
lives in the ``sessions`` table. Every annotation written inside a
session links to it via FK; ``hafiz observe --session <slug>``
resolves the slug to the uuid and populates both columns.

**From a hook or CI step there is no terminal**, so the cursor must be keyed
explicitly — otherwise `session start` fails and nothing you write gets tagged:

```bash
export HAFIZ_SESSION_KEY="$AGENT_SESSION_ID"   # or pass --session-key per call
hafiz session start "jwt-migration" --task auth --project my-project --json
# ...later, in a separate process, same key -> same session:
hafiz observe "chose httponly cookies over localStorage because ..." \
  --type decision --source agent:<your-name> --json
```

Key resolution: `--session-key` → `$HAFIZ_SESSION_KEY` → TTY name → none.
`$HAFIZ_SESSION` is a separate, cursor-free override naming a slug/uuid
directly, for when you already hold the id.

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

Retention is bounded (default 90 days from started_at). `hafiz import`
sweeps expired rows automatically and reports what it tombstoned, but
that trigger stops firing once imports stop — so the backlog is surfaced
as `retention.overdue` in `hafiz status --json` and as a `doctor` check.
**If you see a non-zero overdue count, tell the user** — it's a stated
retention guarantee that isn't currently being met, not a cosmetic nit.

The user's explicit redaction commands:

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
      "source": "agent:<your-name>",
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
hafiz errors list --since 1d --json                          # newest-first, agent-consumable
hafiz errors list --group-by exception_type --since 1d --json # pattern view: counts per class + most_recent
hafiz errors show <id> --json                                 # full traceback + suggested fix
hafiz errors clear                                            # reset the log
```

Each record carries: `timestamp`, `command`, `argv`, `exception_type`,
`message`, `traceback`, `cwd`, `hafiz_version`, `git_branch`,
`host_fingerprint`, plus — for recognized error classes — a
`suggested_action` string and structured `context` (e.g.,
`{"missing_module": "scipy", "is_declared_dep": true}`).

Recognizer set as of v10: `ModuleNotFoundError` (declared-dep aware),
sqlalchemy `OperationalError` (DB connectivity → points at
`hafiz status --diagnose`), pgvector missing (`'extension "vector"
does not exist'` on `ProgrammingError`), pydantic `ValidationError`
raised inside the hafiz config loader (points at `hafiz config show`).

The `--group-by exception_type` shape is distinct from the flat one
— use it for the "what's been failing" lookup:
```json
{
  "since": "1d",
  "grouped_by": "exception_type",
  "total": 7,
  "with_suggestions": 4,
  "most_recent": {"id": "...", "exception_type": "...", "command": "...", "timestamp": "..."},
  "groups": [
    {"exception_type": "ModuleNotFoundError", "count": 3, "with_suggestions": 3,
     "most_recent_id": "...", "most_recent_timestamp": "...",
     "sample_command": "graph stats", "sample_message": "No module named 'scipy'"}
  ]
}
```

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
| `hafiz context "<task>"` | Context bundle (units + graph + annotations) | `--limit`, `--min-score`, `--format`, `--with-ids`, `--project`, `--workspace`, `--include-domain`, `--exclude-domain`, `--json` |
| `hafiz query "<text>"` | Semantic search over indexed embeddings | `--type` (unit kind), `--limit`, `--min-score`, `--format`, `--with-ids`, `--project`, `--workspace`, `--include-domain`, `--exclude-domain`, `--json` |
| `hafiz query "<text>" --observations` | Semantic search over annotations (wisdom layer). Cross-encoder reranked by default. Renamed from `--recall`; that alias still works one release. | `--type`, `--limit`, `--min-score`, `--format`, `--with-ids`, `--no-rerank`, `--project`, `--workspace`, `--json` |

- **Domain filter** (on `query` / `context`): `--include-domain code,doc` or `--exclude-domain code` toggle whole data domains. Domain = the part of `kind` before the dot (`code`, `doc`, `chat`, `mail`, `file`). For exact-kind filtering, use `--type code.function`. Mutually exclusive *per-domain*: `--include-domain code --exclude-domain code` errors. `session start --include-domain ...` persists a default for the cursor.
- **`--min-score FLOAT`** (0–1): relevance floor on the score results are *ranked* by — the reranked score under `--observations`, cosine similarity otherwise. Applied after reranking, before the limit. Reranked scores separate sharply, so useful floors are low (`0.05` default-ish, `0.4` aggressive); cosine floors sit around `0.5`–`0.65`.
- **`--format rich|json|compact|md`**: `--json` is an alias for `--format json` and keeps its exact shape. `compact` emits content + kind + source + age; add `--with-ids` when you might `--supersedes` later. `md` emits raw markdown for prompt injection.
- **Two scores in `--json`**: `score` (cosine) and `rerank_score` (0–1, `null` when reranking didn't run), plus a top-level `reranked` boolean. Under reranking `score` is non-monotonic down the list — don't sort or filter on it.
- **Blank query → exit 2** with `{"ok": false, "error": ...}` on both commands. Check the exit code if you interpolate a variable into the query.

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
| `hafiz observe "<text>"` | Store a fact / decision / learning / pattern / warning. Refuses blank content, and refuses a byte-identical live row (exit 2, `existing_id` returned) unless `--allow-duplicate`. | `--type`, `--source`, `--project`, `--tags`, `--confidence`, `--expires-in`, `--expires`, `--session`, `--task`, `--session-key`, `--supersedes`, `--derived-from`, `--allow-duplicate`, `--json` |
| `hafiz observe ... --supersedes <id>` | Replace a now-wrong annotation: insert the new row, mark the old one inactive (kept for audit) | `--supersedes`, plus all `observe` flags |
| `hafiz forget <id> --annotation` | Retire a wrong annotation with no replacement (soft tombstone; drops from recall, kept for audit) | `--annotation`, `--json` |
| `hafiz query --observations "<topic>" --include-superseded` | Read prior beliefs — superseded/expired annotations, dimmed | `--include-superseded`, plus all `query --observations` flags |
| `hafiz note "<text>"` | Low-bar capture — `kind="note"`. Never gated: a byte-identical repeat returns the existing row with `deduped: true` and exit 0. | same as `observe` minus `--type` |
| `hafiz journal` | Time-bounded digest grouped by day | `--since`, `--day`, `--project`, `--workspace`, `--source`, `--type`, `--session`, `--task`, `--limit`, `--json` |
| `hafiz distill` | Promotable notes (scanner; no LLM call) | `--since`, `--project`, `--session`, `--task`, `--limit`, `--json` |
| `hafiz reconcile` | Read-only sweep: cluster near-duplicate live annotations and emit the supersede/retire commands to run. Scans everything by default; `--limit 0` = all, and a cap that bites reports `truncated: true`. | `--project`, `--type`, `--threshold`, `--limit`, `--json` |
| `hafiz session start "<name>"` | Named session; subsequent writes auto-tag, and `--include-domain`/`--exclude-domain` become defaults for `query`/`context` on the same cursor. Keyed by TTY, or by `--session-key` / `$HAFIZ_SESSION_KEY` when there is no terminal (hooks, CI). | `--task`, `--project`, `--session-key`, `--include-domain`, `--exclude-domain`, `--json` |
| `hafiz session show` / `end` | Inspect / clear | `--session-key`, `--json` |

- **Annotation kinds**: `fact` · `decision` · `learning` · `pattern` · `warning` · `note` · `concept` · `service`
- **Source format**: `agent:claude-code`, `agent:cursor`, `agent:copilot`, `user:<name>`
- **Expiration** (`observe` / `note`): `--expires-in 30d|2w|6m|1y` or `--expires 2026-06-01`. Sets `valid_until`; expired rows are hidden from `query --observations` by default.
- **Git auto-captured**: `commit_hash` on every write inside a repo; `branch` / `is_dirty` on annotations.
- **Staleness**: `query --observations` shows age (`3mo ago`) and dims rows older than 90d.
- **Contradicting knowledge**: never overwrite. Replace with `--supersedes <old-uuid>`, retire (no replacement) with `forget <id> --annotation`, read prior beliefs with `--include-superseded`. See *Capture → Distill* step 4 for the full decision tree.
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
| `hafiz status` | Counts across all tables; per-project index freshness (`staleness.commits_behind`) and `retention.overdue` | `--json`, `--diagnose` |
| `hafiz init` | Create schema + pgvector extension | — |
| `hafiz hooks install <repo>` | Write post-commit / post-merge / post-rewrite hooks. `--project` defaults to the repo directory name and is always pinned into the hook — an untagged hook builds a duplicate untagged index. Refuses if the project is indexed elsewhere. | `--project`, `--force` |
| `hafiz agent install` | Splice this skills.md into an agent config | — |

### Source layer (transcripts)

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz import claude-code [path]` | Idempotent import of Claude Code session JSONL into ``communications`` + messages | `--project`, `--limit`, `--since`, `--dry-run`, `--no-embed`, `--json` |
| `hafiz recall <target>` | Ordered messages for a session/communication, optionally vector-searched | `--query`, `--role`, `--from`, `--to`, `--has-tool-call`/`--no-tool-call`, `--limit`, `--json` |
| `hafiz forget <target>` | Targeted redaction (soft tombstone by default). | `--hard`, `--json` |
| `hafiz forget --all-expired` | Sweep mode — tombstone every communication past `retention_until`. | `--dry-run`, `--json` |

### Sovereignty (export)

The portability complement to `forget`: a one-way dump of the brain's
**wisdom** (annotations) to plain files. Code/AST structure is excluded
(git is its sovereign copy); forgotten and retention-expired rows are
never included. **Not** `extract export` — that emits AST units as an
agent-extraction payload.

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz export` | One-way dump of observations (+ optional transcripts) to a plain-files directory. JSON ``--json`` summary: `{ok, path, format, counts, warning}`. | `--out`/`-o`, `--format`/`-f md\|json`, `--include-transcripts`, `--project`, `--json` |

Round-trip `hafiz import` is not yet implemented; the `json` format is
designed to enable it later.

### Error reporting

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz errors list` | Recent errors, newest first. Each record includes `suggested_action` + `context` for recognized classes. | `--since`, `--limit`, `--group-by`, `--json` |
| `hafiz errors list --group-by exception_type` | Pattern view: per-class `count` + `with_suggestions` + sample fields, plus top-level `total` / `most_recent`. `--limit` does not apply in this mode. | `--since`, `--json` |
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
- `retrievals` — append-only record of every search: ``query_text``, ``result_ids``, ``n_results``, ``top_score``, ``reranked``, ``filters``. Retention-bounded like ``communications``; read it with ``hafiz retrievals``. Rows with ``n_results = 0`` are the store's blind spots.

</details>

<!-- /Installed by hafiz — do not edit above this block; re-run `hafiz agent install` to update -->
