<!-- summary: Full command and flag reference -->
# hafiz guide reference — Full command and flag reference

## Core Commands

| Command | When to use |
|---------|-------------|
| `hafiz context "<task>"` | Pull relevant indexed content (docs + files) and matching observations for a task |
| `hafiz context "<task>" --limit 5 --min-score 0.05 --format compact --with-ids` | The form to use when injecting into your own context — capped, filtered, token-lean |
| `hafiz context "<task>" --include-transcripts` | Same as above, plus matching turns from imported transcripts (opt-in source layer) |
| `hafiz query "<text>" --json` | Semantic search over indexed content (knowledge layer) |
| `hafiz query "<text>" --min-score 0.05 --format compact` | Same, with a relevance floor and the token-lean shape. `--format rich\|json\|compact\|md`; `--json` is an alias for `--format json` |
| `hafiz query "<text>" --include-transcripts --json` | Add matching transcript turns to results, tagged ``layer="source"`` |
| `hafiz query "<text>" --include-domain doc --json` | Restrict to a data domain (``doc``, ``file``, ``chat``, ``mail``). Inverse: ``--exclude-domain``. Comma-separated for multiple. |
| `hafiz retrievals --since-days 7 --json` | What the store was asked for and couldn't answer — a worklist of things to record |
| `hafiz query "<topic>" --observations --type decision --json` | Search the wisdom layer (annotations: decisions, facts, learnings, patterns, warnings) |
| `hafiz recall <session-or-comm-id>` | List ordered messages from a session/communication (source layer). Add ``--query "<text>"`` for similarity search across the session's turns. Use deliberately. *(`recall` is the source layer; `query --observations` is the wisdom layer — different jobs.)* |
| `hafiz observe "<text>" --type <kind> --source agent:<name>` | Record a decision / warning / pattern / learning / fact |
| `hafiz observe "<text>" --derived-from <id>,<id>` | Cite annotations OR messages OR sessions — polymorphic via the ``annotation_targets`` pivot |
| `hafiz note "<text>" --source agent:<name>` | Capture a raw thought — anything below decision-grade |
| `hafiz journal --since 7d --json` | "What did I record recently?" — annotations grouped by day |
| `hafiz distill --since 7d --json` | Promotable backlog — notes + transcript turns grouped into themes, each with a ready observe scaffold |
| `hafiz distill --brief` | Same backlog, token-lean; prints nothing unless it clears its gate — the form to put in a session-start hook |
| `hafiz reconcile --json` | Read-only sweep — clusters near-duplicate live annotations and emits the supersede/retire commands. `suggested_action` is `retire`/`review`/`merge`; only `retire` establishes that no wording is lost, and even that cannot rule out a reversal |
| `hafiz import <claude-code\|cursor\|chatgpt\|codex>` | Post-hoc transcript importers (idempotent; `--dry-run` previews truthfully) |
| `hafiz forget <comm-or-session-id> [--hard]` | Redact source-layer rows. Soft tombstone by default; ``--hard`` deletes content + messages |
| `hafiz forget --all-expired` | Sweep mode — tombstone every communication past its retention_until |
| `hafiz export --out <dir>` | Sovereignty eject — dump the wisdom layer (observations, +``--include-transcripts``) to plain ``.md`` (``--format json`` for lossless JSONL). Excludes code and forgotten/expired rows. Complements ``forget`` |

### Search & Context

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz context "<task>"` | Context bundle (indexed content + annotations) | `--limit`, `--min-score`, `--format`, `--with-ids`, `--project`, `--workspace`, `--include-domain`, `--exclude-domain`, `--json` |
| `hafiz query "<text>"` | Semantic search over indexed embeddings | `--type` (unit kind), `--limit`, `--min-score`, `--format`, `--with-ids`, `--project`, `--workspace`, `--include-domain`, `--exclude-domain`, `--json` |
| `hafiz query "<text>" --observations` | Semantic search over annotations (wisdom layer). Cross-encoder reranked by default. | `--type`, `--limit`, `--min-score`, `--format`, `--with-ids`, `--no-rerank`, `--tags`, `--project`, `--workspace`, `--json` |

- **Domain filter** (on `query` / `context`): `--include-domain doc,file` or `--exclude-domain file` toggle whole data domains. Domain = the part of `kind` before the dot (`doc`, `file`, `chat`, `mail`). For exact-kind filtering, use `--type doc.heading`. Mutually exclusive *per-domain*: `--include-domain doc --exclude-domain doc` errors. `session start --include-domain ...` persists a default for the cursor. There is no `code` domain.
- **`--min-score FLOAT`** (0–1): relevance floor on the score results are *ranked* by — the reranked score under `--observations`, cosine similarity otherwise. Applied after reranking, before the limit. Reranked scores separate sharply, so useful floors are low (`0.05` default-ish, `0.4` aggressive); cosine floors sit around `0.5`–`0.65`.
- **`--format rich|json|compact|md`**: `--json` is an alias for `--format json` and keeps its exact shape. `compact` emits content + kind + source + age; add `--with-ids` when you might `--supersedes` later. `md` emits raw markdown for prompt injection. Both **excerpt** annotations over `[annotations] snippet_chars` (default 480) to the best-matching span, marked with `excerpt: true` / `excerpt of N chars` and `…`; `json` and `rich` always carry full `content`.
- **Two scores in `--json`**: `score` (cosine) and `rerank_score` (0–1, `null` when reranking didn't run), plus a top-level `reranked` boolean. Under reranking `score` is non-monotonic down the list — don't sort or filter on it.
- **Blank query → exit 2** with `{"ok": false, "error": ...}` on both commands. Check the exit code if you interpolate a variable into the query.

### Annotations (the "wisdom layer")

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz observe "<text>"` | Store a fact / decision / learning / pattern / warning. Refuses blank content, and refuses a byte-identical live row (exit 2, `existing_id` returned) unless `--allow-duplicate`. | `--type`, `--source`, `--project`, `--tags`, `--confidence`, `--expires-in`, `--expires`, `--session`, `--task`, `--session-key`, `--supersedes`, `--derived-from`, `--allow-duplicate`, `--json` |
| `hafiz observe ... --supersedes <id>` | Replace a now-wrong annotation: insert the new row, mark the old one inactive (kept for audit) | `--supersedes`, plus all `observe` flags |
| `hafiz forget <id> --annotation` | Retire a wrong annotation with no replacement (soft tombstone; drops from recall, kept for audit) | `--annotation`, `--json` |
| `hafiz query --observations "<topic>" --include-superseded` | Read prior beliefs — superseded/expired annotations, dimmed | `--include-superseded`, plus all `query --observations` flags |
| `hafiz note "<text>"` | Low-bar capture — `kind="note"`. Never gated: a byte-identical repeat returns the existing row with `deduped: true` and exit 0. | same as `observe` minus `--type` |
| `hafiz journal` | Time-bounded digest grouped by day | `--since`, `--day`, `--project`, `--workspace`, `--source`, `--type`, `--session`, `--task`, `--limit`, `--json` |
| `hafiz distill` | Promotable backlog, grouped into themes (scanner; no LLM call). Drains via `observe --derived-from`; decline with `forget --annotation`. | `--since`, `--project`, `--session`, `--task`, `--no-transcripts`, `--include-promoted`, `--limit`, `--message-limit`, `--brief`, `--json` |
| `hafiz reconcile` | Read-only sweep: cluster near-duplicate live annotations and emit the supersede/retire commands to run. Grades each cluster `retire`/`review`/`merge` on word-level text overlap, not on similarity — `retire` means every word survives in the keeper, which is the strongest claim text supports and still not proof of same meaning. Scans everything by default; `--limit 0` = all, and a cap that bites reports `truncated: true`. | `--project`, `--type`, `--threshold`, `--limit`, `--json` |
| `hafiz session start "<name>"` | Named session; subsequent writes auto-tag, and `--include-domain`/`--exclude-domain` become defaults for `query`/`context` on the same cursor. Keyed by TTY, or by `--session-key` / `$HAFIZ_SESSION_KEY` when there is no terminal (hooks, CI). | `--task`, `--project`, `--session-key`, `--include-domain`, `--exclude-domain`, `--json` |
| `hafiz session show` / `end` | Inspect / clear | `--session-key`, `--json` |

- **Annotation kinds**: `fact` · `decision` · `learning` · `pattern` · `warning` · `note` · `concept` · `service`
- **Source format**: `agent:claude-code`, `agent:cursor`, `agent:copilot`, `user:<name>`
- **Expiration** (`observe` / `note`): `--expires-in 30d|2w|6m|1y` or `--expires 2026-06-01`. Sets `valid_until`; expired rows are hidden from `query --observations` by default.
- **Git auto-captured**: `commit_hash` on every write inside a repo; `branch` / `is_dirty` on annotations.
- **Staleness**: `query --observations` shows age (`3mo ago`) and dims rows older than 90d.
- **Contradicting knowledge**: never overwrite. Replace with `--supersedes <old-uuid>`, retire (no replacement) with `forget <id> --annotation`, read prior beliefs with `--include-superseded`. See *Capture → Distill* step 4 for the full decision tree.
- **Lineage**: `--derived-from <ids>` records distillation source without replacing.

### Indexing & Maintenance

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz ingest <path>` | Parse + embed + store. Diff-driven on re-runs. | `--project`, `--git-hook`, `--json` |
| `hafiz status` | Counts across all tables; per-project index freshness (`staleness.commits_behind`), `retention.overdue`, and transcript-capture freshness (`capture[agent].pending_on_disk`) | `--json`, `--diagnose` |
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
**wisdom** (annotations) to plain files. Indexed file bodies are
excluded (git is their sovereign copy); forgotten and retention-expired
rows are never included.

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
