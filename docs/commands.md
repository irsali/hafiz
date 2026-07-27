# Hafiz Command Map

> Source of truth for all hafiz commands. Update this file when commands change.
> Post-structural-grounding: parsers own structure, agents own meaning — see skills.md v4 (adds self-tuning and error reporting to the agent contract).

## Brain Types

| Type | What it is | Cost | Config |
|------|-----------|------|--------|
| **—** | No model needed, pure DB/filesystem operations | Free | — |
| **Embed** | fastembed (nomic-embed-text-v1.5), runs locally via ONNX | Free | `[embedding]` in hafiz.toml |
| **Agent** | The LLM in conversation (Claude Code, Cursor, Copilot) or piped via CLI | Already paying for the session | N/A — agent reads CLI output and acts |
| **Parser** | Deterministic AST / prose parser loaded at ingest time | Free | `[parsers]` via entry points |

## Command Reference

### Setup

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `init` | Create the seven tables + pgvector extension | — | same | same |
| `status` | Count files / units / unit_revisions / embeddings / edges / annotations / commits, broken down by project and kind, plus **index freshness per project** and the **retention-overdue count** | — | `--json` | rich output |
| `status --diagnose` | Config / DB / pgvector / embeddings / parser-registry health | — | `--json` | rich output |
| `doctor` | Install health + host capabilities (RAM, CPU, GPU, onnxruntime) + tunable registry. Stable `--json` shape with `checks` / `host` / `tuning` keys. | — | `--json` | rich tables |
| `doctor --probe` | Same as `doctor`, plus runs per-tunable probers to recommend values for this host. Slow (loads the embedding model, runs several forward passes). | Embed | `--json` | rich tables |
| `doctor --apply [--yes]` | Implies `--probe`; persists recommendations to the sticky tuning cache (`~/.cache/hafiz/tuning_state.json`). **Interactive by default** — prompts `[Y]es / [n]o / [c]ustom` per recommendation. `--yes` skips prompts (CI); `--json` is also non-interactive. JSON response gains `applied` and `interactive` fields. | Embed | `--json --yes` | rich tables |
| `config show` | Display current hafiz.toml settings **and** per-tunable resolution sources (env / toml / sticky / default) | — | `--json` (payload gains a `tunables` array) | rich output |
| `config get <key>` | Print one tunable's effective value + source layer | — | `--json` | rich output |
| `config set <key> <value> [--local]` | Persist a tunable to hafiz.toml. User-scope by default (`~/.config/hafiz/hafiz.toml`); `--local` targets the project's `./hafiz.toml`. Validates and type-coerces the input. | — | `--json` | rich output |
| `config unset <key> [--local]` | Remove a tunable from hafiz.toml so it falls through to sticky / default. Prunes emptied tables. | — | `--json` | rich output |
| `config apply [--yes]` | Runs every prober and prompts to persist each recommendation. **Interactive by default** — `[Y]es / [n]o / [c]ustom` per row, with custom values validated through the tunable's coercer. `--yes` accepts everything; `--json` is non-interactive. JSON gains an `interactive` boolean alongside `applied`. Equivalent to `doctor --apply` with a narrower summary. | Embed | `--json --yes` | rich output |
| `config clear-sticky` | Delete the sticky tuning cache. Re-probe is required to repopulate. | — | `--json` | rich output |
| `errors list [--since] [--limit]` | Recent errors newest-first from `~/.cache/hafiz/errors.log` (NDJSON, 1000-entry cap, FIFO rotation). Each record includes `suggested_action` + `context` for recognized classes. Recognizers (v9): `ModuleNotFoundError` (declared-dep aware), sqlalchemy `OperationalError` (DB connectivity), pgvector missing (`ProgrammingError` with `'extension "vector" does not exist'`), pydantic `ValidationError` from the hafiz config loader. | — | `--json` | rich table |
| `errors list --group-by exception_type` | Pattern view: returns a distinct shape — `{since, grouped_by, total, with_suggestions, most_recent, groups}` — where each group carries `{exception_type, count, with_suggestions, most_recent_id, most_recent_timestamp, sample_command, sample_message}`. `--limit` is ignored in this mode (counts must reflect the full matching window). | — | `--json` | rich table |
| `errors show <id>` | Full structured record: traceback, cwd, git branch, host fingerprint. Accepts unique-prefix ids. | — | `--json` | rich panel |
| `errors clear` | Wipe the log; returns count discarded. | — | `--json` | rich output |
| `hooks install [--project X] [--force]` | Write post-commit + post-merge + post-rewrite git hooks into a repo. `--project` defaults to the repo directory name; the repo path and project are both pinned into the generated hook. **Re-running converges an existing hafiz hook** onto the requested repo/project and keeps the old file as `<hook>.hafiz-bak`. Refuses (exit 2) if the project is already indexed under an unrelated root, or if a hook runs hafiz but wasn't generated by this command — `--force` overrides both. | — | same | same |
| `agent install` | Splice `skills.md` into an agent's config file; warns on version drift | — | same | same |
| `agent uninstall` | Remove the spliced `skills.md` block | — | same | same |
| `agent list` | Show which agents have skills installed | — | same | rich output |
| `parsers list` | List registered parsers (in-tree + entry-point-loaded) and their language coverage. `tree_sitter_js` (JS/TS) appears only when the optional `hafiz[js]` extra is installed. | — | `--json` | rich table |
| `embedding status` | Show current embedding device + provenance (config / sticky cache / probe) | — | `--json` | rich table |
| `embedding retry` | Clear sticky device cache and re-probe | Embed | `--json` | rich output |

**A generated hook is a managed artifact.** `hooks install` used to short-circuit
on "already installed": it discarded `--project`, printed a summary naming the
*new* project, and exited 0. A repo that received a broken hook once could
therefore never be corrected through the CLI — you had to know to delete the file
by hand. That is how four repos ran 30–64 commits stale with hooks firing on
every commit. Re-installing now rewrites the hook, reports `project: old → new`,
and leaves `<hook>.hafiz-bak` behind. A hafiz block appended to somebody else's
hook is replaced in place rather than stacked, so re-running can't produce two
ingests per commit.

**`hafiz doctor --json` shape** (stable — agents parse this):
```json
{
  "checks": [ {"name": "...", "passed": true, "detail": "...", "fix": "..."} ],
  "host": {
    "ram_total_mb": 64000, "ram_available_mb": 45000,
    "cpu_count": 16, "platform": "linux-x86_64",
    "onnx_providers": ["CPUExecutionProvider"],
    "gpu_name": null, "gpu_vram_total_mb": null, "gpu_vram_free_mb": null,
    "onnxruntime_version": "1.24.4",
    "fingerprint": "9c3fa..."
  },
  "tuning": [
    {
      "key": "embedding.max_part_chars",
      "current": 2000, "default": 2000,
      "description": "...", "is_policy": false,
      "recommended": 4000,          // only populated with --probe
      "rationale": "...", "confidence": "high",
      "measured": {"path": "cpu_measured", "budget_mb": 13500, "candidates": [...]},
      "probe_error": null
    }
  ]
}
```
Adding fields is safe; renaming requires a note here.

**`hafiz status --json` — index freshness + retention.** Two keys answer the
questions an operator or agent actually asks of `status`:

```json
{
  "last_commit_per_project": {"Admin Portal": "13367ea0..."},
  "staleness": {
    "Admin Portal": {
      "repo_path": "/repos/Admin Portal",
      "indexed_commit": "13367ea0...",
      "head_commit": "13367ea0...",
      "commits_behind": 0,
      "is_ancestor": true
    }
  },
  "retention": {"overdue": 0, "communications": 0, "retrievals": 0},
  "untagged": {"files": 0}
}
```

- `last_commit_per_project` is ordered by **`commits.committed_at`**, not
  `max(hash)`. Hashes are hex, so a lexicographic max picks an essentially
  random commit — it reported one repo three months stale while it was current.
  Hashes predating the `commits` table have no date and rank below dated ones.
- `commits_behind` is `null` and `is_ancestor` is `false` when the indexed
  commit isn't in HEAD's history (rebased away, force-pushed over). A count
  there would be meaningless: `rev-list A..HEAD` on a missing A yields nothing,
  which reads as "up to date". All fields are `null` when the repo can't be
  located — `status` must still print when something is already wrong.
- `retention.overdue` counts **source-layer** rows past `retention_until` that
  haven't been tombstoned, summed across `communications` and `retrievals` and
  also broken out per table. The sweep runs on `import`; this is how you see the
  backlog when imports have stopped. Clear it with `hafiz forget --all-expired`.
- `untagged.files` counts live `files` rows with `project IS NULL` — the shadow
  index a project-less ingest builds (see `prune --untagged`). Non-zero means
  search is returning some content twice. `doctor` carries the same check as
  "Every file has a project".
- The untagged bucket is deliberately **absent** from `staleness`: its files span
  every repo a broken hook ever walked, so the derived root came out as `/` and
  "how far behind HEAD" is not a meaningful question for it. It still appears in
  `by_project` and `last_commit_per_project` under the `(none)` key.

**Accelerator checks (`doctor`).** `onnxruntime`, `onnxruntime-gpu` and
`onnxruntime-openvino` all install the same `onnxruntime` import package, so only
one can be active — whichever lands last overwrites the others' files while
their metadata survives. An installed extra is therefore **not** proof the
accelerator is in use. `doctor` reports one check per accelerator with hardware
present, distinguishing:

| State | Meaning | Fix |
|---|---|---|
| `active` | provider available | — |
| `shadowed` | accelerator wheel installed, provider absent — the CPU wheel (a `fastembed` dependency) overwrote it | `pipx runpip hafiz uninstall -y onnxruntime`. **Do not** reinstall the extra; it is already installed and would be shadowed again. |
| `missing` | hardware present, no accelerator wheel | install `hafiz[cuda]` or `hafiz[openvino]` |
| `no-hardware` | not reported at all | — |

**Tunable resolution order:** `env (HAFIZ_*__*)` → `hafiz.toml` → sticky tuning cache → built-in default. The sticky layer is written by `doctor --apply` / `config apply` and keyed to a host fingerprint that invalidates itself when the RAM class, GPU presence, OS/arch, or onnxruntime provider set materially changes. Clear it with `config clear-sticky`.

**Probe safety brake.** The `embedding.max_part_chars` prober runs candidate sizes ascending in a subprocess. To stop the *measurement itself* from OOM-ing the host, the subprocess is given a `safety_ceiling_mb` (= the recommendation budget) and (a) extrapolates the next candidate's likely peak from the previous one × 3.0 and skips it if predicted to exceed the ceiling, (b) stops the moment a measured peak crosses the ceiling. Surfaced in `measured.candidates` with `_skipped` / `_stopped` sentinel rows. Without this brake, batched probing can spike RSS far above what the recommendation logic would ever pick.

### Indexing

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `ingest <path>` | Walk the tree, pick a parser per file, upsert units / revisions / embeddings. On a git repo, diff-driven: re-ingests only files changed since last indexed commit | Embed + Parser | `--json` emits NDJSON progress | rich progress |
| `ingest --git-hook` | Same pipeline, designed to run inside the installed post-commit / post-merge / post-rewrite hooks | Embed + Parser | `--json` | rich output |
| `watch <path>` | Long-running: detect file changes, re-ingest automatically. *(Parked — still on the pre-v5 chunk pipeline. Exits 1 with a clean "not yet rewired" message; use `hafiz ingest` + git hooks meanwhile.)* | Embed + Parser | `--json` → `{"ok":false,"error":...}` | rich "not yet rewired" |
| `prune` | Reporting no-op. On-ingest tombstoning (`tombstone_vanished_files`) handles stale files. Kept so existing hooks/scripts don't break. | — | `--json` returns `{"action":"prune","noop":true,...}` | rich "nothing to prune" |
| `prune --untagged` | Tombstone `files` rows with `project IS NULL` — the duplicate shadow index an ingest without `--project` leaves behind. `--dry-run`, `--include-unindexed`, `--under <path>` | — | `--json` → `{"ok":true,"action":"prune-untagged","untagged":N,"duplicated":N,"unindexed":N,"files_tombstoned":N,"units_tombstoned":N,"under":path,"dry_run":bool}` | rich partition + counts |

**Why `--untagged` needs to exist.** `files` is unique on `(project, path)`, so an ingest with no `--project` cannot update a project's rows — it writes a *parallel untagged copy* that search then returns alongside the real one. Those rows are unreachable by any other cleanup: ingest skips `tombstone_vanished_files` outright when `project is None`, and the paths are still on disk anyway, so no walk would call them vanished. Measured on a real deployment: 1,960 untagged rows, 1,782 of them byte-for-byte duplicates of properly-tagged files.

The partition is the safety property. Untagged paths **also** indexed under a project are provably redundant and are tombstoned by default; untagged paths **no project covers** are the only copy of those files, so they are counted, reported, and left alone unless `--include-unindexed` is passed. `--under <path>` narrows the sweep to one subtree, so a repo can be fixed and verified before committing to the rest. Tombstoning is soft — rows stay for audit and drop out of search, which filters on `File.valid_until IS NULL`.

**Race safety:** ingest refuses to run during a rebase / merge / cherry-pick in progress (detects `.git/<marker>` files) and exits with code 2.

**Multi-project ingest.** When indexing a workspace with several projects (e.g. `workspace.projects = ["a", "b", "c"]`), run `hafiz ingest` **one project at a time, sequentially in the same shell** — not in parallel. Each `hafiz ingest` process loads its own ONNX embedding model (~500 MB baseline) and the per-call peak RSS for batched embedding scales with the configured `embedding.max_part_chars`; running N processes in parallel multiplies both, defeating the runtime chunking that bounds peak RSS in `embed_texts`. Concrete pattern: avoid spawning ingest from multiple VSCode tasks, CI matrix shards on the same machine, or git hooks across sibling repos triggered together. If you really need parallelism, install ingest-only on a host with enough RAM headroom for `N × (model + per-call peak)` and confirm with `hafiz doctor`.

### Daemon (warm serving)

A plain `hafiz` call re-pays ~1.3–1.7s of cold start (process launch + embedding-model load + DB connect) before any search runs. The **warm daemon** loads the model + pooled DB engine **once** and answers many requests over a Unix socket, dropping per-call latency to the actual vector op (~15ms recall, ~400ms graph-context) plus cheap local IPC.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `serve` | Run the daemon (foreground). `--detach` backgrounds it; `--idle-timeout <secs>` sets auto-shutdown (default 1800). | Embed + DB | normally auto-spawned; run by hand to pre-warm | rich panel |
| `serve status` | Report whether the daemon is live + its version. | — | `--json` → `{"ok","running","socket","version"}` | rich panel |
| `serve stop` | Stop the running daemon (best-effort: removes the socket). | — | `--json` → `{"ok","was_running","socket_removed"}` | rich line |

- **Transport:** Unix domain socket, **0600**, at `$XDG_RUNTIME_DIR/hafiz/daemon.sock` (falls back to `/tmp/hafiz-<uid>/daemon.sock`). Never a TCP port — a sovereign personal store stays off the network; filesystem permissions gate access. Override the path with `HAFIZ_DAEMON_SOCKET`.
- **On-demand:** clients auto-spawn the daemon if it's absent and **fall back to direct in-process execution** on any daemon error, so the daemon can only make things faster — never break a call that the plain CLI would have served. Disable entirely with `HAFIZ_NO_DAEMON=1`.
- **Idle shutdown:** the daemon exits after `--idle-timeout` seconds of inactivity, so it never lingers forever.
- **Version skew:** every message carries the hafiz version; a client talking to a daemon from a prior hafiz version respawns it automatically.

**Rewrite resilience:** reconcile pass on every ingest marks commits that are no longer reachable in git as `commits.rewritten_at = now`. The installed `post-rewrite` hook triggers a fresh ingest automatically after an amend / rebase.

### Extraction (agent contract v2)

Parsers own structural facts (entities, calls, imports, inherits) — agents no longer write them. The extract pipeline is narrower and semantic-only.

| Step | Command | Brain | What happens |
|------|---------|:-----:|-------------|
| 1. Export | `extract export --project X` | — | Emits the AST-known units (with stable `identity_key`) + structural edges so the agent knows what already exists. |
| 2. Analyze | _(agent reads the output)_ | Agent | Decides which units deserve annotations (decisions / patterns / warnings) and which semantic edges to draw. |
| 3. Import | `extract import --project X` | — | Validates the v2 payload (rejects v1 / AST-territory) and writes annotations + semantic edges. |
| 4. Verify | `status --json` | — | Confirms counts across tables. |

**v2 JSON shape:**
```json
{
  "version": 2,
  "annotations": [
    {
      "content": "Canonical auth entry",
      "kind": "pattern",
      "source": "agent:claude-code",
      "unit_identity_key": "<from step 1>",
      "confidence": 0.9,
      "tags": ["auth"]
    }
  ],
  "edges": [
    {
      "source_name": "UserService", "source_file": "/abs/auth.py",
      "target_name": "SecurityPolicy", "target_file": "/abs/policy.py",
      "relation": "implements_pattern",
      "evidence": "..."
    }
  ]
}
```

Rejected at import time: `kind` starting with `code.*`; relations in `{calls, imports, inherits, references}`. Off-vocabulary kinds / relations produce warnings but are accepted.

| Command | Purpose | Brain | Key Flags |
|---------|---------|:-----:|-----------|
| `extract export` | Emit the AST-known units + edges the agent can attach to | — | `--project`, `--limit`, `--pretty` |
| `extract import` | Import a v2 payload (annotations + semantic edges) from JSON | — | `--file`, `--project` |

### Search

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `query "<text>"` | Vector similarity search over the `embeddings` table (joined back to units + files for context) | Embed | `--json` | rich output |
| `query "<text>" --observations` | Vector search over annotations (wisdom layer), **cross-encoder reranked** by default for precision. `--no-rerank` for pure vector order. Renamed from `--recall` (collided with the top-level `recall` command); `--recall` is a hidden deprecation-warned alias for one release. | Embed + Rerank | `--json` | rich output |
| `context "<task>"` | Synthesize units + graph + annotations for a task. `--limit` caps each section. | Embed | `--json` | rich panel |

**An empty query is an error, not a result set.** A blank query embeds to a
near-zero vector, against which *empty* documents (blank `.scss` files,
whitespace-only headings) score a perfect `1.0`. `query` and `context` therefore
exit **2** with `{"ok": false, "error": ...}` on a blank or whitespace-only
query rather than returning confidently-scored noise. Same guard on the write
side: `observe` / `note` refuse blank content, because such a row would then
score near-perfectly against every future query — a permanent noise magnet.

**Reranking** (on `query --observations`): vector similarity compresses relevant rows and near-random noise into a narrow band; a cross-encoder re-scores the top `limit × rerank.candidate_multiplier` candidates against the query and reorders them, then returns the top `limit`. On by default (`rerank.enabled` config); `--no-rerank` skips it. The reranker model (`Xenova/ms-marco-MiniLM-L-6-v2`, ~80 MB) ships with fastembed — no extra dependency — and loads lazily, cached alongside the embedding model. Reranking is strictly a reordering: if the model is unavailable it falls back to vector order. Warm via the daemon it adds ~300-400ms; the gain is sharp signal/noise separation. Disable on constrained hosts with `hafiz config set rerank.enabled false`.

#### Two scores, and which one to filter on

`query --observations --json` returns **both**:

| Field | Meaning | Range |
|---|---|---|
| `score` | cosine similarity from the vector stage | 0–1 |
| `rerank_score` | cross-encoder relevance, or `null` if reranking didn't run | 0–1 |

`rerank_score` is `null` under `--no-rerank`, with a single-row result set, or
when the model fails — which is how you tell reranked output from vector output.
The payload also carries a top-level `reranked` boolean.

**Filter on the score the results are *ranked* by** — that is what `--min-score`
does. Under reranking, `score` is **not monotonic** down the result list, so a
floor on it fights the ordering. Real series from a deployed index:

```
vector : 0.645 0.573 0.580 0.510 0.551 0.559 0.624   <- NOT monotonic
rerank : 0.881 0.473 0.067 0.013 0.010 0.008 0.005   <- monotonic
```

`--min-score 0.60` against `score` would drop rank 4 (0.510) and keep rank 7
(0.624): neither the top results nor coherently ordered.

The cross-encoder emits **unbounded logits** (measured: `+3.32` relevant,
`-11.35` irrelevant). Those are sigmoid-normalized to 0–1 before being surfaced,
so `--min-score` has one meaning on one scale whether or not reranking ran.
Normalization is monotonic, so ordering is preserved.

**Calibration.** Reranked scores separate sharply, so useful floors are *low*,
not near-1. On the query above (50 candidates): `0.4` → 2 rows, `0.05` → 3 rows,
`0.01` → 5 rows, unset → 50. Under `--no-rerank` the floor applies to cosine
similarity, where the useful band is roughly `0.5`–`0.65`.

Measured across three on-topic and three off-topic questions against a
1,200-annotation store, taking the best score in each result set:

| | cosine `score` | `rerank_score` |
|---|---|---|
| on-topic | 0.713 – 0.789 | 0.962 – 0.998 |
| off-topic | 0.480 – 0.523 | 0.0001 – 0.0002 |

Cosine leaves a 0.19 gap that an integrator has to calibrate into — and the
bands are close enough that one on-topic query ranked a 0.618 row first. The
reranker leaves three orders of magnitude. That collapses two jobs into one
flag: `--min-score 0.05` sits ~50× above the off-topic ceiling while keeping
every genuinely relevant row, so "no results" *is* the not-relevant signal and a
separate hit gate (or a set-max heuristic over `score`) is unnecessary. `0.4` is
aggressive tail-trimming. Re-measure on your own corpus before hard-coding
anything: this is one workload shape, one embedding model, one reranker.

#### Output formats

`--format` on `query` and `context`. `--json` remains as an alias for
`--format json` with its exact existing shape, since installed agent configs and
hooks parse it; new fields are additive.

| Format | For | Notes |
|---|---|---|
| `rich` | terminals | default |
| `json` | existing integrations | today's full shape, field-for-field |
| `compact` | context-window injection | content + kind + source + age only |
| `md` | prompt injection | raw markdown, not Rich-rendered |

`compact` drops uuids, timestamps, null fields, float scores, and (on `context`)
the graph section's edge lists, collapsing units to `name (kind)`. Pass
`--with-ids` to re-add ids — **do this whenever the consumer might write back**,
because an agent that can read a decision but not cite it cannot
`observe --supersedes` it, and the corpus silently accumulates contradictions.

Measured on a real session-start injection (`query --observations --source
user:anjum --limit 50`): 70,272 B of `json` → **2,498 B** of `compact
--min-score 0.05` (~17.5k → ~625 tokens), retaining all three genuinely
relevant rows.

**Scoping flags** (on `context`, `query`):

| Flag | Scope | How it works |
|------|-------|-------------|
| `--project X` | Single named project | Filters queries to `files.project = X` |
| `--workspace` | Sibling projects | Resolves directories in parent of cwd, matches to DB project tags (normalized) |
| _(neither)_ | Everything | No filter |

`--project` and `--workspace` are mutually exclusive.

**Domain filters** (on `context`, `query`): toggle whole data domains on/off without losing the rest of your index.

| Flag | Effect |
|------|--------|
| `--include-domain code,doc` | Restrict results to units whose `kind` starts with one of the listed domains. Comma-separated. |
| `--exclude-domain code` | Drop results whose `kind` starts with one of the listed domains. Comma-separated. |

A "domain" is the prefix of `kind` before the first dot — `code`, `doc`, `mail`, `chat`, `file`, etc. (For exact-kind filters like `code.function`, keep using `--type`.) The two flags are mutually exclusive *per-domain*: `--include-domain code,doc --exclude-domain doc` errors out before hitting the DB. A dotted value (`--include-domain code.function`) also errors — domains are dotless tokens.

When both flags are omitted on the CLI, the active session's defaults (set via `session start --include-domain` / `--exclude-domain`) are inherited. Passing *either* flag explicitly skips inheritance entirely, so a query can flip the filter without first ending the session.

### Knowledge Graph

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `graph show <name>` | Unit and its direct connections | — | `--json` | rich tree |
| `graph deps <name>` | What this unit depends on (outgoing edges) | — | `--json` | rich table |
| `graph impact <name>` | Blast radius — what depends on this unit (incoming edges) | — | `--json` | rich table |
| `graph path <src> <tgt>` | Shortest directed path | — | `--json` | rich tree |
| `graph rank` | Top units by centrality (pagerank / betweenness / degree) | — | `--json` | rich table |
| `graph stats` | Overall graph health (density, components, top-central, kind/relation breakdowns) | — | `--json` | rich tables |

Graph nodes are current units (`valid_until IS NULL`), edges are current edges (`superseded_at IS NULL`) with both endpoints resolved in scope. External references (`target_unit_id IS NULL`, `target_name` set) are excluded from traversal — visible via raw DB if needed.

### Annotations (the "wisdom layer")

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `observe "<text>"` | Embed and store a fact / decision / learning / pattern / warning / note | Embed | `--json` | rich panel |
| `note "<text>"` | Shortcut for `observe --type note` — low-bar raw capture lane (skips *near*-duplicate detection; still collapses byte-identical writes) | Embed | `--json` | rich panel |
| `reconcile` | Read-only sweep: cluster near-duplicate **live** annotations and propose a resolution to run | Embed | `--json` | rich panels |
| `journal` | Time-bounded digest of annotations, grouped by day | — | `--json` | rich tables · `--format mermaid` |
| `distill` | Surface recent notes as promotable candidates (scanner, not promoter) | — | `--json` | rich tables |

- **Annotation kinds**: `fact` · `decision` · `learning` · `pattern` · `warning` · `note` · `concept` · `service`.
- **Auto-captured git context**: `commit_hash` column; `branch` / `is_dirty` in metadata JSONB. Captured when writing inside a git repo.
- **Expiration** (on `observe` / `note`, mutually exclusive): `--expires-in <30d|2w|6m|1y>` or `--expires <ISO-date>`. Sets `valid_until`; `query --observations` hides expired rows by default.
- **Staleness hint**: `query --observations` surfaces age (e.g. `3mo ago`) and dims rows older than 90 days.
- **Supersession** (on `observe` / `note`): `--supersedes <uuid>` atomically marks the target row inactive and records the link. Nothing is deleted.
- **Lineage** (on `observe` / `note`): `--derived-from <uuid>[,<uuid>...]` records distillation source without replacing.
- **Visualize the journal** (`journal --format <rich|json|mermaid>`, default `rich`; `--json`/`-j` is a shortcut for `--format json`): `--format mermaid` emits a copy-pasteable [Mermaid](https://mermaid.js.org) diagram of the window — renders inline in VS Code, GitHub, and Obsidian. `--mermaid-kind supersession` (default) draws the **decision-evolution graph** (`graph LR`: old decision → *superseded by* → new, with superseded nodes dimmed); `--mermaid-kind timeline` draws a month-grouped Mermaid `timeline`. Each entry's body is truncated to ~60 chars in the diagram — the full text stays in `--json` and the rich view. The `--json` entry shape now includes `supersedes_id` (the annotation this one replaced, or `null`). **Note:** a Mermaid diagram you paste elsewhere is a point-in-time snapshot that `hafiz forget` cannot reach — same caveat as `hafiz export`.
- **Blank content is refused** on both `observe` and `note` (exit 1, `{"ok": false, "error": ...}`). An empty annotation embeds to a near-zero vector and then scores near-perfectly against *every* later query — a permanent noise magnet in recall.
- **Exact-duplicate handling** (both lanes, and **not** governed by `[dedup] strict`): a write whose `content` + `kind` + `source` + `project` are all identical to a **live** row is never a legitimate new annotation, so it is never simply appended. Comparison is a plain string equality — no embedding — and is NULL-safe on `source`/`project`, so an untagged rewrite of an untagged row still counts. Superseded/expired rows don't count: re-stating a retired belief is a new assertion. `--supersedes` and `--allow-duplicate` both bypass the check.
  - **`observe` refuses**: exit `2`, `{"ok": false, "error": ..., "existing_id": "<uuid>", "hint": ...}`. The caller may have meant `--supersedes`, and a non-zero exit is what makes them look.
  - **`note` succeeds idempotently**: exit `0`, `{"action": "observe", "deduped": true, "annotation": {...existing row...}}`, nothing written. "Raw capture is never gated" protects the caller from friction; it does not entitle the store to keep identical rows. No caller has to change.
  - `deduped` is present on every `observe`/`note` `--json` response (`false` on a normal write).
- **Near-duplicate detection** (on `observe`, not `note`): before writing, Hafiz cosine-compares the new content against **live** annotations of the same kind+project and surfaces any at/above `[dedup] threshold` (default 0.88). Hafiz detects *similarity*, never *contradiction* — the supersede/refine/distinct call stays with the caller. Unlike the exact check, this one *is* governed by `strict`, because only the author can judge whether a similar row refines, contradicts, or merely resembles the old one.
  - **Surface-only** (default): the write always succeeds; matches ride back in `observe --json` as `near_duplicates: [{id, content, kind, score}]` and as a yellow hint in rich output. Skipped when `--supersedes` is set.
  - **Strict** (`[dedup] strict = true`): a match aborts the write — exit `2`, `{"ok": false, "near_duplicates": [...]}` — unless `--supersedes <id>` or `--allow-duplicate` is given.
  - **`reconcile`** is the after-the-fact backstop: `--project`, `--type`, `--threshold`, `--limit`, `--json`. Never mutates — it proposes, and prints the commands you run yourself.
    - **Scans the whole store by default.** `--limit` caps the scan (newest first); `0` means all. A cap that bites sets `truncated: true` — a partial sweep is never silent. The old `--limit 500` default reported 34 clusters on a 1,099-row store where a full sweep finds 65, with nothing in the output to say so.
    - **Every cluster carries a proposal.** One member is the `primary`; the rest are retired. `suggested_action` says what happens to the primary: `retire` (it is the newest row and no shorter than the ones it replaces — it survives untouched) or `merge` (the newest row is under 80% of the longest, so keeping only it would drop text; the primary becomes the *longest* row and you write text that supersedes it). `commands` is the ordered, runnable resolution — at most one `observe`, since `--supersedes` takes a single id.
    - JSON shape: `{action, scanned, total_live, truncated, threshold, total, clusters: [{kind, project, suggested_action, primary_id, commands: [...], members: [{id, content, score, source, valid_from, chars, primary}]}]}`.
    - The count also rides on `doctor` (and so on `status --diagnose`) as **Knowledge base deduplicated**, because a read-only command nobody remembers to run surfaces nothing. It is not on `status`: with no vector index on `annotations.embedding` the count is a quadratic scan (~310 ms at 1,099 rows), and `status` is on the hot path.
- **Unit binding**: annotations created via `extract import` can link to a unit via `unit_identity_key`. Annotations created via `observe` are unit-free by default (can be linked later via API).

### Captures (transcripts / multi-page dumps)

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `capture [TEXT]` | Ingest a transcript / multi-page dump into the **source layer** as one communication. Reads TEXT, `--file`, or stdin. Splits on blank lines into turns; the selective-embed policy embeds substantive turns. Hidden from default `query`/`context`; surface via `recall <id>` or `--include-transcripts`. | Embed | Pipe a conversation in (`--source agent:<name>`); recall later | `--file`, `--title`, `--project`, `--source`, `--tags`, `--session`, `--task`, `--json` |

**`capture --json` shape:** `{action, communication_id, title, source, project, turn_count, messages_embedded, session_id, task}`. The `agent` column on the stored communication is derived from `--source` (`agent:hermes` → `hermes`), so `hafiz recall --agent <name>` filters to a tool's own captures.

### Sessions

Named threads of work that auto-tag subsequent `observe` / `note` writes with a `session_id` and optional `task`.

The on-disk JSON at `~/.cache/hafiz/session-<key>.json` is a **cursor** — the canonical record lives in the `sessions` table. The cursor carries both `session_uuid` (FK target) and `session_id` (slug) so display continuity is preserved.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `session start "<name>"` | Start a named session: creates a `sessions` row and writes the cursor | — | `--json` | rich panel |
| `session show` | Show the active session | — | `--json` | rich panel |
| `session end` | Clear the cursor and stamp `ended_at` on the DB row | — | `--json` | rich line |

`session start` flags: `--task <name>`, `--project <name>`, `--include-domain <a,b>`, `--exclude-domain <a,b>`, `--session-key <id>`. The two domain flags persist into the cursor JSON (not the DB row) and are inherited by `query` / `context` calls against the same cursor.

#### Cursor identity — using sessions without a terminal

The cursor was originally keyed by TTY name alone, which made sessions
unreachable from the places they matter most: an agent-harness hook or a CI step
has no controlling terminal, and `session start` hard-failed there. Resolution
order is now:

1. an explicit `--session-key <id>`
2. `$HAFIZ_SESSION_KEY`
3. the TTY name — unchanged default for humans
4. nothing: no session, and writes simply don't auto-tag

Separately, **`$HAFIZ_SESSION`** names a session slug or uuid outright and skips
the cursor file entirely, for callers that already hold the id. Write precedence:
`--session` flag → `$HAFIZ_SESSION` → cursor.

`--session-key` is accepted on `session start|show|end` and on `observe` / `note`
/ `capture`, so a caller that can't export env vars still has a path. Keys come
from a harness and are not trusted: anything outside `[A-Za-z0-9._-]` collapses
to `-`, leading dots are stripped, and the result is capped at 64 characters, so
no key can escape `~/.cache/hafiz`.

Typical hook usage — each hafiz call is its own process, so the key is what ties
them together:

```bash
export HAFIZ_SESSION_KEY="$AGENT_SESSION_ID"
hafiz session start "$TASK_NAME" --task refactor --project myproj --json
# ... later, in a different process:
hafiz observe "chose X over Y because ..." --type decision --json
```

With neither a key nor a TTY, `session start` exits 1 with
`{"ok": false, "error": ...}` naming both escape hatches.

**Auto-tagging** on `observe` / `note`: session inherited when no flag is given; explicit `--session` / `--task` always win. ``--session <slug>`` resolves the slug to a uuid via the `sessions` table; both `annotations.session_id` (uuid FK) and `annotations.legacy_session_id` (text slug) are populated, so journal/distill display stays human-readable.

### Source layer (transcripts)

Hafiz can ingest agent-harness transcripts into a dedicated source layer (see [architecture.md "Storage layers"](./architecture.md#storage-layers--knowledge-vs-source)). Source rows are **hidden from default `query` / `context`** — surfacing them is opt-in.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `import claude-code [PATH]` | Idempotent post-hoc import of Claude Code session JSONL into `communications` + messages | — | `--json` | rich table |
| `recall <target>` | Ordered messages for a session (slug or uuid) or communication; or vector search via `--query` | — | `--json` | rich table |
| `query --include-transcripts` | Add matching source-layer turns to results, tagged `layer="source"` | — | `--json` | rich panel (separate transcript section) |
| `context --include-transcripts` | Append matching transcript turns under a `transcripts` field | — | `--json` | rich panel |
| `forget <target>` | Targeted redaction: soft tombstone by default, `--hard` deletes content | — | `--json` | rich line |
| `forget <uuid> --annotation` | Retire a **knowledge-layer** annotation by uuid (soft — sets `valid_until = now`, kept for audit). For wrong/obsolete observations. | — | `--json` → `{ok, action, id, kind, valid_until}` | rich line |
| `forget --all-expired` | Sweep mode — tombstone every communication past its `retention_until` | — | `--json` | rich table |

Defaults:
- `import claude-code` source path: `~/.claude/projects/`. Idempotent by `(agent='claude-code', external_id=<jsonl session uuid>)`.
- `import claude-code` flags: `--project`, `--limit`, `--since 7d`, `--dry-run`, `--no-embed`.
- `recall` flags: `--query`, `--role`, `--from`, `--to`, `--has-tool-call` / `--no-tool-call`, `--limit`.
- `forget` flags: `--hard`, `--all-expired`, `--dry-run`.
- Retention: 90 days from `started_at` unless explicitly overridden at insert time.
- Selective embedding: skip messages under ~30 tokens; skip pure tool-result echoes; embed when `marked_salient=true` regardless of length.

**Retention enforcement.** `import` runs the expiry sweep automatically and
reports it (`retention_sweep: {matched, tombstoned, dry_run}` in `--json`; a row
in the rich table when non-zero). `--dry-run` propagates to the sweep. It is
deliberately *not* wired into `ingest`: that's the code/doc subsystem, it fires
per-commit from a git hook, and its output goes to `/dev/null`, so a sweep there
would be unattributable and unobservable.

An import-bound trigger is not sufficient on its own — it stops firing exactly
when it's needed, since retention keeps ticking after imports stop. **Visibility
is the enforcement mechanism**: the overdue count is a first-class field on
`status --json` (`retention.overdue`, broken down into `communications` and
`retrievals`) and a `doctor` check. Sweep manually with `hafiz forget
--all-expired`, which covers **both** source-layer tables — a retention
guarantee that reaches only part of the layer isn't one. Sweeps are soft
tombstones (`valid_until = now`); rows survive for audit, and `--hard` remains
explicit and per-target.

### Retrieval telemetry

Hafiz could not evaluate itself. Answering "is this earning its keep?" for a live
3.5-week deployment required parsing 169 Claude Code transcripts, because hafiz
kept no record of its own reads — it could not say which annotations had ever
been recalled, which surfaced and were useful, or which had never come up once.
Every quality mechanism that might grow from here (decay dead knowledge, promote
proven knowledge, notice recall quality regressing) needs that data.

One append-only source-layer table, one INSERT per search.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `retrievals` | Never-recalled knowledge, most-recalled rows, and queries that returned nothing. `--since-days`, `--limit` | — | `--json` → `{ok, retrievals, empty_result_rate, never_recalled, blind_before, unanswered[], most_recalled[], telemetry_started, enabled}` | rich tables |

- **Recorded in `hafiz/core/`, not the commands layer.** `hafiz/core/daemon.py` calls `search_annotations` directly, so command-layer telemetry would silently miss every warm request — the exact failure class this audit kept finding. `vector_search` / `search_annotations` default `telemetry_command` to their label, so a *new* caller is recorded without knowing telemetry exists; pass `None` to opt a call out.
- **Never fails a search.** Recording is best-effort and swallows everything. A memory layer that can break the read path gets removed.
- **`n_results = 0` rows are the point.** The gap between what agents ask for and what the store holds is the only signal that says what to write down *next*; nothing else in hafiz produces it.
- **`blind_before`** is the subset of never-recalled rows written before telemetry existed. Without it, "1,094 never recalled" reads as an indictment of knowledge that simply predates the log.
- **Query text is a new data category** for this store: it's what you were *looking for*, not what you concluded. So it lands in the source layer with the source layer's guarantees — `retention_until` (default 90d), swept by `forget --all-expired`, counted in `status`/`doctor` — and it never leaves the machine.
- **Opt out** with `[telemetry] retrieval = false` (or `HAFIZ_TELEMETRY__RETRIEVAL=false`). `retention_days` and `min_query_chars` are tunable in the same block; queries shorter than `min_query_chars` (default 3) aren't recorded, since "ok" / "yes" say nothing about what was asked for.

### Sovereignty (export)

The data-portability complement to `forget`. Dumps the brain's **wisdom layer** — annotations (decisions / facts / learnings / patterns / warnings) — to plain files for backup, human review, or migration. Code and AST structure are **excluded** (git is their sovereign copy). Forgotten (`valid_until`) and retention-expired rows are never included — export reflects only live data.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `export` | One-way dump of annotations (+ optional transcripts) to a plain-files directory | Annotations | `--json` → `{ok, path, format, counts, warning}` | rich table + path + secrets warning |

Defaults & flags:
- `--out` / `-o`: output directory (default `./hafiz-export`). Written atomically (temp dir → rename); an existing directory is replaced.
- `--format` / `-f`: `md` (human-readable tree, default) or `json` (per-table JSONL — `knowledge/annotations.jsonl`, plus `source/*.jsonl` when transcripts are included). Lossless except embeddings.
- `--include-transcripts`: also dump source-layer agent transcripts (opt-in, mirrors `query` / `context`). Off by default.
- `--project`: limit to one project; default is the whole brain.
- Every export writes a `manifest.json` (tool/schema version, format, generated-at, scope, counts).
- **Distinct from `extract export`**, which emits AST units as an agent-extraction payload — not a portability dump.
- `import` (round-trip restore) is **not yet implemented**; the `json` format is designed to enable it later.

### Review

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `review` | Self-review of the knowledge base: counts units / edges / embeddings / annotations, then flags annotation-quality gaps (no decisions, low confidence, staleness), orphan units, and projects with units but no edges. | Graph + Annotations | `--json` emits `{stats, findings, summary}` | rich panel of findings |

## Common Flags

| Flag | Available on | Purpose |
|------|-------------|---------|
| `--json` / `-j` | Most commands | Machine-readable output for agents |
| `--format` / `-f` | `journal` | Output format: `rich` (default), `json`, or `mermaid`. `--json` is a shortcut for `--format json` |
| `--mermaid-kind` | `journal` | With `--format mermaid`: `supersession` (decision-evolution graph, default) or `timeline` |
| `--project` / `-p` | Most commands | Filter or tag by project name |
| `--workspace` / `-w` | `context`, `query`, `journal` | Scope to sibling projects in parent directory |
| `--type` / `-t` | `query`, `observe`, `journal` | Unit kind or annotation kind depending on context |
| `--limit` / `-l` | `query`, `extract export`, `journal` | Maximum results |
| `--observations` | `query` | Search annotations (wisdom layer) instead of content. Renamed from `--recall`, which remains a hidden alias for one release. |
| `--since` | `journal` / `distill` | Duration window ending now (default `7d`) |
| `--day` | `journal` | Specific UTC day (ISO date). Exclusive with `--since` |
| `--expires-in` | `observe`, `note` | Expire after duration. Exclusive with `--expires` |
| `--expires` | `observe`, `note` | Expire at ISO date. Exclusive with `--expires-in` |
| `--source` | `observe`, `note`, `journal` | Origin tag (`agent:<name>`, `user:<name>`) |
| `--session` | `observe`, `note`, `journal`, `distill` | Explicit session id |
| `--task` | `observe`, `note`, `journal`, `distill` | Explicit task label |
| `--supersedes` | `observe`, `note` | UUID of annotation being replaced |
| `--derived-from` | `observe`, `note` | UUIDs this row was distilled from |
| `--include-superseded` | `query --observations` | Return superseded / expired rows |
| `--include-transcripts` | `query`, `context` | Add source-layer transcript matches to results (off by default) |
| `--include-domain` | `query`, `context`, `session start` | Comma-separated data-domain allowlist (`code`, `doc`, `chat`, …). Domain = `kind` prefix before the first dot. |
| `--exclude-domain` | `query`, `context`, `session start` | Comma-separated data-domain denylist. Mutually exclusive *per-domain* with `--include-domain`. |
| `--diagnose` | `status` | Full diagnostic checks including parser registry |

## Architecture Note

Two stability layers:

- **Layer 1 (stable contract):** `skills.md` v2 installed via `hafiz agent install`. Ownership rule (parsers own structure, agents own meaning) is load-bearing. `hafiz agent install` detects version drift and warns when refreshing an older splice.
- **Layer 2 (evolving):** `hafiz review`, `hafiz parsers list`, Phase 7 observability surfaces. Free to iterate; not referenced from `skills.md`.
