# Hafiz Command Map

> Source of truth for all hafiz commands. Update this file when commands change.
> Parsers own indexed content; agents own meaning — see skills.md (Layer 1 agent contract).

## Brain Types

| Type | What it is | Cost | Config |
|------|-----------|------|--------|
| **—** | No model needed, pure DB/filesystem operations | Free | — |
| **Embed** | fastembed (nomic-embed-text-v1.5), runs locally via ONNX | Free | `[embedding]` in hafiz.toml |
| **Agent** | The LLM in conversation (Claude Code, Cursor, Copilot) or piped via CLI | Already paying for the session | N/A — agent reads CLI output and acts |
| **Parser** | Deterministic prose / whole-file parser loaded at ingest time | Free | `[parsers]` via entry points |

## Command Reference

### Setup

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `init` | Write a starter `hafiz.toml` if none exists, then create the knowledge-layer tables (files, units, unit_revisions, embeddings, annotations, commits) plus source-layer tables and the pgvector extension. Never overwrites an existing config; an unreachable database exits 1 with the `docker run` line to fix it. | — | `--json` → `{ok, config_path, config_created, database_url, backend, reingest_projects}` — `reingest_projects` lists projects whose old code units were just removed (re-ingest them so code is searchable as plain text again) | rich output |
| `status` | Count files / units / unit_revisions / embeddings / annotations / commits, broken down by project and kind, plus **index freshness per project**, the **retention-overdue count**, and **transcript-capture freshness** | — | `--json` | rich output |
| `status --diagnose` | Config / DB / pgvector / embeddings / parser-registry health | — | `--json` | rich output |
| `doctor` | Install health + host capabilities (RAM, CPU, GPU, onnxruntime) + tunable registry. Stable `--json` shape with `checks` / `host` / `tuning` keys. | — | `--json` | rich tables |
| `doctor --probe` | Same as `doctor`, plus runs per-tunable probers to recommend values for this host. Slow (loads the embedding model, runs several forward passes). | Embed | `--json` | rich tables |
| `doctor --apply [--yes]` | Implies `--probe`; persists recommendations to the sticky tuning cache (`~/.cache/hafiz/tuning_state.json`). **Interactive by default** — prompts `[Y]es / [n]o / [c]ustom` per recommendation. `--yes` skips prompts (CI); `--json` is also non-interactive. JSON response gains `applied` and `interactive` fields. | Embed | `--json --yes` | rich tables |
| `config show` | Display current hafiz.toml settings **and** per-tunable resolution sources (env / toml / sticky / default) | — | `--json` (payload gains a `tunables` array) | rich output |
| `config get <key>` | Print one key's effective value + source layer. Accepts **config keys** (`database.url`, `embedding.device`) as well as tunables. | — | `--json` → `{key, value, source, default}` | rich output |
| `config set <key> <value> [--local]` | Persist a key to hafiz.toml. Accepts **config keys** as well as tunables — `hafiz config set database.url …` is how you switch backends. User-scope by default (`~/.config/hafiz/hafiz.toml`); `--local` targets the project's `./hafiz.toml`. Validates and type-coerces against the settings model, so an invalid value is refused before the file is written. | — | `--json` | rich output |
| `config unset <key> [--local]` | Remove a key from hafiz.toml so it falls through to sticky / default. Prunes emptied tables. | — | `--json` | rich output |

Config keys are resolved against the settings models rather than an allowlist, so a field added to a settings model is settable without being registered anywhere. An unrecognised key errors with `unknown_key` (renamed from `unknown_tunable` on 2026-09-06, when these commands stopped being tunable-only).
| `config apply [--yes]` | Runs every prober and prompts to persist each recommendation. **Interactive by default** — `[Y]es / [n]o / [c]ustom` per row, with custom values validated through the tunable's coercer. `--yes` accepts everything; `--json` is non-interactive. JSON gains an `interactive` boolean alongside `applied`. Equivalent to `doctor --apply` with a narrower summary. | Embed | `--json --yes` | rich output |
| `config clear-sticky` | Delete the sticky tuning cache. Re-probe is required to repopulate. | — | `--json` | rich output |
| `errors list [--since] [--limit]` | Recent errors newest-first from `~/.cache/hafiz/errors.log` (NDJSON, 1000-entry cap, FIFO rotation). Each record includes `suggested_action` + `context` for recognized classes. Recognizers (v9): `ModuleNotFoundError` (declared-dep aware), sqlalchemy `OperationalError` (DB connectivity), pgvector missing (`ProgrammingError` with `'extension "vector" does not exist'`), pydantic `ValidationError` from the hafiz config loader. | — | `--json` | rich table |
| `errors list --group-by exception_type` | Pattern view: returns a distinct shape — `{since, grouped_by, total, with_suggestions, most_recent, groups}` — where each group carries `{exception_type, count, with_suggestions, most_recent_id, most_recent_timestamp, sample_command, sample_message}`. `--limit` is ignored in this mode (counts must reflect the full matching window). | — | `--json` | rich table |
| `errors show <id>` | Full structured record: traceback, cwd, git branch, host fingerprint. Accepts unique-prefix ids. | — | `--json` | rich panel |
| `errors clear` | Wipe the log; returns count discarded. Also the remedy `doctor` names for a log that predates redaction and still holds a credential on disk. | — | `--json` | rich output |
| `hooks install [--project X] [--force]` | Write post-commit + post-merge + post-rewrite git hooks into a repo. `--project` defaults to the repo directory name; the repo path and project are both pinned into the generated hook. **Re-running converges an existing hafiz hook** onto the requested repo/project and keeps the old file as `<hook>.hafiz-bak`. Refuses (exit 2) if the project is already indexed under an unrelated root, or if a hook runs hafiz but wasn't generated by this command — `--force` overrides both. | — | same | same |
| `agent install` | Splice `skills.md` into an agent's config file; warns on version drift | — | same | same |
| `agent install <agent> --hooks` | Wire hafiz into the agent's settings file: **every-turn memory recall** (`SessionStart` → `hook session`, `UserPromptSubmit` → `hook prompt`) and **automatic transcript capture** (`PreCompact` + `SessionEnd`). Idempotent; preserves user-authored hooks and every other key. Warns (never edits) if you already have a hand-written hook that shells out to `hafiz query`/`hafiz-recall`, which would inject the same rules twice. `--path DIR` / `--file NAME` redirect the settings file (for a trial run). `claude-code` only — see *Agent hooks and pinned rules* below for why not Cursor. | — | same | rich output |
| `agent install codex [--local]` | Splice `skills.md` into Codex's `AGENTS.md` — global `~/.codex/AGENTS.md` (or `$CODEX_HOME/AGENTS.md`), local `./AGENTS.md`. | — | same | same |
| `agent install hermes [--path HOME]` | Install the **Hermes memory plugin** (recall each turn, `hafiz_recall` / `hafiz_remember` tools, session capture) into `$HERMES_HOME/plugins/hafiz/` (default `~/.hermes`; `--path` names the Hermes home) and set `memory.provider: hafiz` in its `config.yaml`. Only that one line is edited (comments and every other key untouched; one-shot `config.yaml.hafiz-backup`). Another provider already set is **never replaced**; a missing or oddly-shaped config is left alone and the exact lines to add are printed. Idempotent. | — | same | rich output |
| `agent uninstall` | Remove the spliced `skills.md` block | — | same | same |
| `agent uninstall hermes [--path HOME]` | Remove only what install wrote (tracked in `plugins/hafiz/.hafiz-managed`) and restore the `memory.provider` line hafiz changed. A hand-installed plugin with no marker is skipped; a `provider: hafiz` hafiz did not set is left and reported. | — | same | rich output |
| `agent uninstall <agent> --hooks` | Remove the recall and capture hooks, leaving user-authored hooks intact. Already-captured transcripts are untouched. | — | same | rich output |
| `hook session` | **Harness entry point** (SessionStart). Reads the harness JSON on stdin (`.cwd`), prints markdown for the model: (1) *Always applies* — every observation tagged `always`, no reranking, no floor, rendered **whole** (never excerpted); (2) *Relevant to <repo> <branch>* — recall for the repo name + branch subject, `--min-score 0.05`, limit 5, deduped against the pins. **Never fails**: any error prints nothing, exit 0. | Embed (daemon-first) | markdown only — not a `--json` command | markdown |
| `hook prompt` | **Harness entry point** (UserPromptSubmit). Reads `.prompt` from the harness JSON on stdin and prints *Relevant prior decisions* — `--min-score 0.05`, limit 6, long records excerpted and marked. Skips (prints nothing) prompts under 12 characters, `/loop` and `<<autonomous-loop` ticks, and harness messages (`<task-notification>`); editor preambles (`<ide_opened_file>`, `<ide_selection>`, `<system-reminder>`) are stripped first. When observation recall returns **nothing**, falls back to up to 2 turns from *earlier* agent conversations (cosine ≥ 0.70, never the current session, never forgotten or past-retention ones), under *From earlier conversations (hafiz — unverified, not a recorded decision)* — see *Agent hooks and pinned rules*. **Never fails**: exit 0 always. | Embed (daemon-first); transcript fallback runs in-process | markdown only | markdown |
| `agent list` | Show which agents have skills installed | — | same | rich output |
| `parsers list` | List registered parsers (in-tree + entry-point-loaded) and their language coverage. In-tree: prose (Markdown) and whole-file fallback. | — | `--json` | rich table |
| `embedding status` | Show current embedding device + provenance (config / sticky cache / probe). `--json` `sticky` object carries `probe_fingerprint`, `stale`, and `stale_reason`. | — | `--json` | rich table |
| `embedding retry` | Clear sticky device cache and re-probe | Embed | `--json` | rich output |

**First run is two commands, not a checklist.** `hafiz init` used to assume a
config already existed and told you to `cp hafiz.toml.example hafiz.toml` — a
file a `pipx install hafiz` user does not have — and an unreachable database
surfaced as a traceback. It now writes the config itself and turns a failed
connection into the exact remedy. The written file carries the *built-in*
defaults, never the current environment: a throwaway `HAFIZ_DATABASE__URL` in
your shell must not be frozen into your config permanently.

**Resolution order is `env → hafiz.toml → sticky → default`, and now actually
is.** TOML values are passed to pydantic-settings as *init* arguments, which
outrank environment variables — so `HAFIZ_EMBEDDING__DEVICE=cpu hafiz ingest .`
(advertised in the README) was silently ignored for any key the config file also
set. `load_settings` now drops keys that a `HAFIZ_<SECTION>__<KEY>` variable
addresses before handing the rest over, letting the env source supply them. The
defect was known: the test harness had been working around it with a patched
`load_settings`, and its docstring named the cause.

**Credentials never appear in diagnostic output.** `hafiz doctor`,
`hafiz status --diagnose` and `hafiz config show` are the commands whose output
gets pasted into a bug report, and `skills.md` tells agents to read them — so a
database password in them is not cosmetic. All three printed one verbatim,
human and `--json` alike, until [redaction.py](../hafiz/core/redaction.py)
landed: one stdlib-only `redact_credentials()`, shared by every display site
*and* by the error log's capture path, replacing the password in
`scheme://user:secret@host` with `***` while keeping the username, host and
database name that make the value diagnostic.

Two things are deliberately **not** redacted, and are commented as such in
place so they aren't "fixed" later. `fix=` hints like
`postgresql+asyncpg://user:pass@host/db` are placeholders teaching the format
and contain no secret. `migrate-backend`'s target URL is the argument the user
just typed, printed inside a `hafiz config set database.url …` command meant to
be copy-pasted — masking it would write a literal `***` as the password.

The error log gets the same treatment at both ends: scrubbed on write so
nothing new lands on disk, and on read so records written before that stop
leaking through `errors list` / `errors show` without anyone having to destroy
their own audit trail first. Recognizers still receive the *unredacted* text —
they match on error wording, and `OperationalError` is recognised by the very
message that carries the URL. `append()` also chmods the log `0600`
unconditionally, which tightens logs created before that was true; a traceback
from a failed `INSERT` carries the statement's parameters, so the file holds
more than stack frames. `hafiz doctor`'s **Error log is private** check reports
both halves — leftover credentials (as a *count*, never the value) and a
group- or world-readable mode — and names the remedy rather than clearing
anything itself.

**The sticky device verdict expires when its cause changes, not on a timer.**
`hafiz` remembers a GPU probe's outcome in `~/.cache/hafiz/device_state.json` so
later runs skip the probe. That cache used to invalidate only when
`onnxruntime.__version__` changed — and every rung of the GPU remediation ladder
leaves the version untouched. Uninstalling the CPU wheel that was shadowing
`onnxruntime-gpu` (the fix `hafiz doctor` prints) and `pip install tensorrt` (the
fix the non-finite-output error prints) both changed nothing, so hafiz ignored
its own advice and stayed on CPU. Staleness now compares a `probe_fingerprint`
of the facts that can change a probe's outcome — ORT version, CUDA provider
present, TensorRT provider present — rendered as `1.27.0|cuda|trt`.
`embedding status --json` exposes it alongside `stale` and a human
`stale_reason`; a state file written before the field existed loads with
`probe_fingerprint: null` and re-probes once.

A *timer* was rejected deliberately. Probing is neither cheap nor
side-effect-free — it builds a GPU session and runs a real embed — so a timed
re-probe would slow an arbitrary later command and, in the out-of-memory case,
take VRAM from whatever process holds it, to re-learn something that cannot have
changed. `out_of_memory` is the one exception and does also expire on time (24h):
it means another process held the VRAM, which no fingerprint can observe and
which resolves on its own.

**Hardware you installed a wheel for but aren't using is announced, not
inferred.** `onnxruntime`, `onnxruntime-gpu` and `onnxruntime-openvino` all
install the same `onnxruntime` import package, so the last one installed wins
and the others' providers vanish while their `.dist-info` remains. The result is
indistinguishable from "no GPU on this host", which is why the CPU-fallback path
was silent — correct for someone who never asked for GPU, wrong for someone who
installed the extra and silently didn't get it. A fresh probe now runs the same
diagnosis `hafiz doctor` does and prints a panel (stderr) with the remedy when it
finds a *shadowed* accelerator; the detail is stored in the sticky `reason`, so
`embedding status` can still explain the verdict days later. Only the `shadowed`
state is announced — `missing` is an unexercised opportunity, not a broken
install, and telling that user on every fresh probe would be noise.

**Agent hooks and pinned rules.** `agent install claude-code --hooks` replaces the
hand-written recall script everyone ended up with. By convention **a pinned rule
is an observation tagged `always`**: `hafiz observe "<rule>" --type learning
--tags always`. Pins are addressed by tag because "always applies" is not a
semantic property — no query text retrieves it. An unscoped pin shows in every
session; a pin with `--project X` shows only in sessions whose **workspace**
includes X (the session's own indexed project plus its sibling repos, resolved
the way `--workspace` does) — so a rule scoped to one repo of a multi-repo
workspace still reaches its siblings but never leaks into an unrelated
workspace. When the working directory cannot be resolved, only unscoped pins
show.

*The 10,000-character cliff.* Claude Code truncates hook output over 10,000
characters: it saves the text to a file and the model sees only a ~2 KB
preview, so a pin set that grew past the limit silently stopped delivering its
tail (a real 14-pin store was ~11.9k). Output is therefore hard-capped at
`[hooks] max_chars` (default 9000). Pins fill the budget first, then topical
rows; anything dropped is counted in one closing line —
`_hafiz: N pinned rule(s) not shown (hook output capped at 9000 chars). List
them: hafiz query "always" --observations --tags always --no-rerank --limit 60
--json_` — so truncation is never silent. `[hooks] timeout_seconds` (default
20) bounds the whole call, daemon start and model load included; on overrun the
hook prints nothing. Both hooks call the warm daemon first (falling back
in-process), so the embedding model stays loaded between prompts.

*Earlier-conversation fallback.* When `hook prompt` recalls **no observation**
for a prompt, it searches the source layer for the best matching turns from
**earlier** agent sessions and prints them under `## From earlier conversations
(hafiz — unverified, not a recorded decision)`, each trimmed to ~400 characters
with its role, age and agent. It never fires when an observation matched. A
replay of 40 empty-recall prompts found the best earlier turn clearly useful 10
times, partly useful 7 and noise 23; a cosine floor of 0.70 kept all 10 useful
hits and let 4 noise hits through. Because transcripts are retention-bounded and
user-redactable, the search **never** returns a message from a communication that
was forgotten (`hafiz forget`) or is past `retention_until` (even if the sweep
has not run), nor from the session in progress (the harness's `session_id`,
matched against the communication's `external_id`). The same live-only filter
now applies to `hafiz recall --query` and `--include-transcripts`; before, those
could return turns of a forgotten or expired communication. The search runs
in-process (there is no daemon op for message search), costing one embedding on
a prompt that already found nothing; any failure prints nothing. Config, under
`[hooks]`:

| Key | Default | Meaning |
|---|---|---|
| `transcript_fallback` | `true` | Set `false` to turn the fallback off. Env: `HAFIZ_HOOKS__TRANSCRIPT_FALLBACK` |
| `transcript_min_score` | `0.70` | Cosine floor for a turn to be shown |
| `transcript_limit` | `2` | Most snippets per prompt (`0` also disables) |

*Cursor* is instructions-only. Its hooks API
([cursor.com/docs/agent/hooks](https://cursor.com/docs/agent/hooks)) can inject
model context only from `sessionStart` (`additional_context`) and after tool
calls; `beforeSubmitPrompt` can only allow/block, so per-prompt recall is not
possible, and a session-start-only half would deliver pins without topical
recall. Revisit if Cursor adds prompt-time context injection.

**Agent hooks are managed structurally, not by markers.** Instruction files are
Markdown and use paired sentinel comments; an agent's `settings.json` is JSON, so
hafiz-owned entries are instead identified by the `# hafiz-managed` sentinel
inside the hook *command*. Install strips its own entries and re-adds them, so
re-running updates in place rather than stacking duplicates; uninstall removes
only those entries and drops any event left empty, restoring the file
byte-for-byte. Everything without the sentinel — the user's hooks, permissions,
env, model — is never touched. A settings file that cannot be parsed is
**refused, not overwritten** (exit 1), and the first modification keeps a
one-shot `settings.json.hafiz-backup` that later runs never clobber.

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

**`hafiz status --json` — index freshness, retention, capture.** These keys
answer the questions an operator or agent actually asks of `status`:

```json
{
  "last_commit_per_project": {"Billing Portal": "13367ea0..."},
  "staleness": {
    "Billing Portal": {
      "repo_path": "/repos/Billing Portal",
      "indexed_commit": "13367ea0...",
      "head_commit": "13367ea0...",
      "commits_behind": 0,
      "is_ancestor": true
    }
  },
  "retention": {"overdue": 0, "communications": 0, "retrievals": 0},
  "capture": {
    "claude-code": {
      "last_captured_at": "2026-06-30T12:44:56+00:00",
      "days_since": 66,
      "communications": 467,
      "live": 0,
      "tombstoned": 467,
      "pending_on_disk": 77,
      "active_on_disk": 1
    }
  },
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
- `capture` answers **"is anything still arriving?"**, which `retention.overdue`
  cannot: a store receiving nothing passes the overdue check trivially, because
  there is nothing left to sweep. That gap was not theoretical — transcript
  capture for `claude-code` stopped on 2026-06-30 and went unnoticed until
  2026-09-04 while `status` reported `overdue: 0` throughout. **Silence is not
  health.**
- `pending_on_disk` counts transcripts whose **session id is not in the store at
  all**, read from each file's head rather than by parsing it (~0.2 s over 201
  files, since `status` is on the hot path). For `hermes`, whose sessions are
  rows rather than files, a session the memory plugin captured counts as stored,
  and a session past retention or with nothing an import would keep is not
  counted — importing it would change nothing. Two rejected cheaper proxies, both
  measured wrong on a real store: comparing file *mtime* against
  `last_captured_at` (which is `max(started_at)` — a session that ran for more
  than a moment has an mtime past its own start, so it reported as pending while
  fully captured), and using the **filename stem** as the session id (it matches
  only a session's *first* file — 124 of 200 files disagreed, because Claude Code
  reuses one `sessionId` across resumed/forked files).
- `active_on_disk` counts transcripts touched within the last 30 minutes, held
  back from `pending_on_disk`. The session you are running right now is unknown
  to the store until first capture, so without this the warning would fire on
  every `status` during any active session — and a warning that is always on is
  a warning nobody reads, which is the failure mode this signal exists to fix.
  In-progress sessions are not uncaptured work.
- Only `pending_on_disk > 0` is warned about in human output, since it is the
  only condition with a remedy (`hafiz import <agent>`). A long-quiet agent with
  nothing pending is either one you stopped using or one whose files already
  rotated away — warning about it on every `status` just teaches the operator to
  skim past the block. Agents with no discoverable on-disk store omit the field.
- Agent harnesses rotate their own transcripts, so uncaptured sessions are lost
  **permanently** — this is the one staleness signal in hafiz that is not
  recoverable by re-running an import later. Stop the drift with
  `hafiz agent install <agent> --hooks`.
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

### MCP (stdio server)

For agents that **cannot shell out** — Cursor chat, desktop and hosted clients. Everything else should use the CLI: it is the primary contract, and `--json` is stable.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `mcp` | Serve the knowledge layer over MCP on **stdio**. Normally launched by the client, not by hand. | Embed + DB | 13 tools, see below | — (speaks JSON-RPC on stdout) |
| `mcp --list` | Print the tool surface and exit. Exits **1** on registry drift. | — | `--json` → `{"ok","transport","count","drift","tools":[{"name","summary","writes","target","input_schema"}]}` | rich table |

Client config:

```json
{"mcpServers": {"hafiz": {"command": "hafiz", "args": ["mcp"]}}}
```

Needs the optional extra — `pipx inject hafiz mcp`, or `pip install 'hafiz[mcp]'`. It is an extra because the SDK pulls 19 transitive packages (starlette, uvicorn, sse-starlette, a compiled `cryptography`) — an entire HTTP server stack this stdio-only transport never touches.

**The 13 tools.** Read: `hafiz_context`, `hafiz_query`, `hafiz_recall_observations`, `hafiz_recall_session`, `hafiz_journal`, `hafiz_distill`, `hafiz_reconcile`, `hafiz_retrievals`, `hafiz_status`. Write: `hafiz_observe`, `hafiz_note`, `hafiz_capture`, `hafiz_session`.

`hafiz_status` defaults to `verbose=false`, which reports the same fields as `hafiz status --json` but names only the projects actually behind their repo, plus a `staleness_summary` count of what was checked — so an empty staleness map cannot be misread as "never checked". `verbose=true` returns the full CLI payload. `retention`, `capture` and `embed_coverage` are never trimmed: they are the "is retention being enforced", "is anything still arriving" and "is what arrives still being indexed" signals. `embed_coverage` reports `{window_days, messages, embedded, ratio}` over live communications in the window; `ratio` is `null` rather than `0.0` on an empty window, because "no turns arrived" and "turns arrived and none were embedded" are different problems. A low ratio is normal — selective embedding declines short turns and tool-result echoes by design, leaving roughly 8% — so the human warning fires only on zero-with-traffic, which is unambiguously an embedder fault.

Parity is of **capability**, not of command count: every flag on a covered command survives as a tool parameter, so `hafiz_query(observations=true, kind="decision")` covers `query --observations --type decision`. Schemas are generated from the same core function signatures the CLI calls, so the two surfaces cannot drift — a parameter added to a core function fails the build until it is described.

**Deliberately absent**, and staying CLI-only: install/admin (`init`, `hooks`, `agent`, `config set`, `embedding`, `serve`, `watch`), destructive (`forget`, `prune`) and long-running ingest paths (`ingest`, `import`, `review`). An MCP client calls tools without asking anyone; anything irreversible or environment-mutating needs a human present.

**Attribution.** A write with no explicit `source` is attributed to the connecting client — `agent:cursor-ide`, from what it sends at `initialize` — so the audit trail records which client wrote what rather than a uniform `agent:mcp`.

**Transport.** stdio only. No HTTP, no SSE, no listening socket. The server is spawned by the client and dies with it.

### Daemon (warm serving)

A plain `hafiz` call re-pays ~1.3–1.7s of cold start (process launch + embedding-model load + DB connect) before any search runs. The **warm daemon** loads the model + pooled DB engine **once** and answers many requests over a Unix socket, dropping per-call latency to the actual vector op (~15ms recall, ~400ms context) plus cheap local IPC.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `serve` | Run the daemon (foreground). `--detach` backgrounds it; `--idle-timeout <secs>` sets auto-shutdown (default 1800). Idempotent — if one is already serving it reports that instead of starting a second. `--json` → `{"ok","started","already_running","pid","socket"}`. | Embed + DB | normally auto-spawned; run by hand to pre-warm | rich panel |
| `serve status` | Report whether the daemon is live, its version, its **pid**, and whether it *would* spawn. | — | `--json` → `{"ok","running","socket","version","pid","enabled","idle_timeout"}` | rich panel |
| `serve stop` | SIGTERM the daemon and **confirm** it stopped; unlinking the socket is only the fallback. | — | `--json` → `{"ok","was_running","pid","signalled","stopped","socket_removed"}` | rich line |

**Which commands use it:** `hafiz query`, `hafiz query --observations` and
`hafiz context`. All return identical results with or without a daemon — same
objects, same `--json`, same ordering. The daemon also picks the `compact`/`md`
excerpts (under its warm cross-encoder) and computes `query`'s `staleness`, so a
served call loads neither fastembed nor the DB layer client-side. Writes
(`observe` / `note` / `capture`) and `--include-transcripts` still run in-process.

**Measured** on a 2,100-annotation / 83k-embedding store, daemon warm:

| Call | Before | After |
|---|---|---|
| `query --observations --format md` | 1.35s, 324 MB | 0.67s, 44 MB |
| `query --format compact` (units) | 1.86s, 1 GB | 0.30s, 32 MB |

Most of what remains of a recall is the cross-encoder rerank (~0.3s).

- **Transport:** Unix domain socket, **0600**, at `$XDG_RUNTIME_DIR/hafiz/daemon.sock` (falls back to `/tmp/hafiz-<uid>/daemon.sock`). Never a TCP port — a sovereign personal store stays off the network; filesystem permissions gate access. Override the path with `HAFIZ_DAEMON_SOCKET`.
**One daemon per socket, enforced.** Observed on the author's machine: **four**
daemons, 4.1 GB resident between them, all started within one second, all bound
to the same socket path — so three were unreachable and had each paid a ~1 GB
embedding-model load to become so. Nothing serialized startup, and three places
assumed sole ownership: the client's spawn path had no lock, so N concurrent
callers spawned N daemons; `serve` unlinked the socket before binding, stealing
the path from a working daemon; and `serve`'s shutdown unlinked *the path*, so a
loser's idle timeout deleted the **winner's** socket and left a live daemon
nothing could reach.

`serve` now takes an exclusive `flock` on `<socket>.lock` **before** loading any
model, and holds it for the process lifetime. Losing costs milliseconds instead
of a gigabyte, and both unlinks become correct because only the lock holder can
reach them. `flock` rather than a pidfile deliberately: the kernel releases it
when the holder exits for any reason, including SIGKILL, so there is no
stale-lock recovery path to get wrong. The lock file also carries the holder's
pid, which is what makes `serve stop` a real signal rather than a hopeful
socket unlink.

Because `flock` guards an *inode* and not a name, deleting the lock file while a
daemon holds it would leave the holder locking an orphan and let a newcomer lock
a fresh file at the same path. So after winning the lock, `serve` also pings the
socket and exits if anything answers. Do not delete `<socket>.lock`; `serve
stop` deliberately leaves it in place and empties it instead.

- **On-demand:** clients auto-spawn the daemon if it's absent and **fall back to direct in-process execution** on any daemon error, so the daemon can only make things faster — never break a call that the plain CLI would have served. Disable entirely with `HAFIZ_NO_DAEMON=1`; `serve status --json` reports that as `enabled: false`.
- **Idle shutdown:** the daemon exits after `idle_timeout` seconds of inactivity. Precedence is `--idle-timeout` → `HAFIZ_DAEMON_IDLE` → `[daemon] idle_timeout` in `hafiz.toml` → 1800. **Set it to `0` to keep the daemon hot indefinitely** — the right choice when another process consumes hafiz continuously, where a 30-minute shutdown means repaying the model-load cost after every quiet spell.
- **Caching:** the daemon caches the embedding model and the DB pool, **never results**. Every op re-runs the query, so `forget` and `observe` from another process are visible immediately, and a redacted row can never be served from memory.
- **Recall hooks use it.** `hook session` / `hook prompt` call the daemon's `query_recall` op (which accepts `snippet_budget` to return the same marked excerpts as `--format md`), so the embedding and rerank models stay warm across prompts instead of reloading per hook call.
- **Version skew:** every message carries the hafiz version; a client talking to a daemon from a prior hafiz version respawns it automatically.

**Rewrite resilience:** reconcile pass on every ingest marks commits that are no longer reachable in git as `commits.rewritten_at = now`. The installed `post-rewrite` hook triggers a fresh ingest automatically after an amend / rebase.

### Search

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `query "<text>"` | Vector similarity search over the `embeddings` table (joined back to units + files for context) | Embed | `--json` | rich output |
| `query "<text>" --observations` | Vector **and keyword** search over annotations (wisdom layer), merged and **cross-encoder reranked** by default for precision. `--no-rerank` for pure vector order. | Embed + Rerank | `--json` | rich output |
| `context "<task>"` | Synthesize relevant indexed content (docs + files) plus matching observations for a task. `--limit` caps each section. `--json` has `chunks` + `annotations` (no `entities` key). `--format compact` collapses to content + kind + source + age (no `units` key). | Embed | `--json` | rich panel |

**An empty query is an error, not a result set.** A blank query embeds to a
near-zero vector, against which *empty* documents (blank `.scss` files,
whitespace-only headings) score a perfect `1.0`. `query` and `context` therefore
exit **2** with `{"ok": false, "error": ...}` on a blank or whitespace-only
query rather than returning confidently-scored noise. Same guard on the write
side: `observe` / `note` refuse blank content, because such a row would then
score near-perfectly against every future query — a permanent noise magnet.

**Harness messages are skipped, not searched.** Agent harnesses put text into the
conversation that the user never typed, and a per-prompt recall hook forwards it.
A prompt that *starts with* `<task-notification>` (a sub-agent's final report —
an answer, not a question; 88% of such recalls came back empty) returns an empty
result set with exit `0`, is not embedded, and is logged in `retrievals` as the
marker `[skipped] <task-notification>` rather than the full text. Editor
preambles at the front of a real prompt — `<ide_opened_file>`, `<ide_selection>`,
`<system-reminder>` blocks — are stripped and the rest is searched; a prompt that
is *only* preamble is skipped the same way. Only these leading tags count: a
prompt that merely mentions one, or starts with any other tag, is searched as-is.

**Reranking** (on `query --observations`): vector similarity compresses relevant rows and near-random noise into a narrow band; a cross-encoder re-scores the top `limit × rerank.candidate_multiplier` candidates against the query and reorders them, then returns the top `limit`. On by default (`rerank.enabled` config); `--no-rerank` skips it. The reranker model (`Xenova/ms-marco-MiniLM-L-6-v2`, ~90 MB) ships with fastembed — no extra dependency — and loads lazily, cached alongside the embedding model. Reranking is strictly a reordering: if the model is unavailable it falls back to vector order. Warm via the daemon it adds ~300-400ms; the gain is sharp signal/noise separation. Disable on constrained hosts with `hafiz config set rerank.enabled false`.

**Query condensing** (on `query --observations`, so on every recall hook): a cross-encoder scores one (query, memory) pair, so a long prompt that covers several topics matches none of them, and dictated filler ("uh", "or something") or a pasted element-picker dump dilutes a short one. Before searching, hafiz cleans the query — spoken fillers, Cursor `DOM Path` / `Position` / `HTML Element` dumps (keeping the element's visible text), code fences, stack traces, and URLs (reduced to host and path words). A query still over 300 chars is searched as its cleaned whole **plus** up to 4 of its most specific sentences, and each row keeps its best score across them, so condensing can add hits but not lose what the whole query found. A short, clean query is searched exactly as given. The retrieval log records the original query, with `filters.segments` = how many texts were searched when that wasn't one. Measured by replaying 180 real long hook prompts: hit rate 47% → 79%, nothing lost, about two in three new hits on target (misses mostly score under 0.1); short prompts unchanged. Cost: up to 5 reranks for a long prompt — p95 search time 1.1s → 2.9s on CPU, median unchanged. Off with `hafiz config set rerank.condense_queries false`.

**Convention: pinned = tag `always`.** The recall hooks (`hook session`) deliver every observation tagged `always` at the start of every session — see *Agent hooks and pinned rules* above.

**Keyword channel** (on `query --observations`, so on `context` and every recall hook): an embedding smears exact strings — a file name, a port, an error constant, a flag, a ticket id — so a question that leans on one used to miss the note that contains it verbatim. Beside the vector search, hafiz now runs a full-text search over annotation content and adds the best few matches to the vector candidates **before** the cross-encoder; the reranker and `--min-score` still decide what survives (there is no score fusion). How a prompt becomes a lookup:

- **Terms.** Each whitespace-separated chunk of the prompt is a *term*, split into alphanumeric words that must appear adjacent and in order: `agents.py` → `agents py`, `--min-score` → `min score`, `ERR_X` → `err x`, `5432` → `5432`. A path also yields its last segment (`src/core/agents.py` also looks up `agents.py`). Plain English stop words and single letters are not looked up. Terms are OR'd — a natural-language prompt matches through *some* of its words, never all.
- **Same words on both backends.** Postgres keeps `src/core/agents.py` as one token under its default parser, so the index text is flattened to alphanumeric words first (`to_tsvector('simple', …)` over a generated `content_tsv` column + GIN index, migration `0012`); SQLite uses an FTS5 table (`annotations_fts`, `unicode61`, kept in sync by triggers). There is no stemming and no stop-word list in the index: `agent` does not match `agents`, so the two backends agree and exact strings survive.
- **Only distinctive terms.** A term that matches more than 1% of annotations (floor of 5 rows) is vocabulary, not a lookup, and is dropped; a term that matches nothing is dropped. Surviving terms are ranked by rarity (Postgres: summed `ln(1 + N/df)`; SQLite: `bm25()`), so the one rare token in "where do I set ERR_X" outranks the common words around it.
- **Same filters as the vector channel.** `--project`/`--workspace`, `--kind`, `--source`, `--tags`, and the superseded/retired/expired exclusion (and the requirement that a row has an embedding) are one shared clause, so a keyword hit can never return a row the vector path would have refused.
- **The reranker sees the match.** The cross-encoder reads only the first 400 characters of a row. When a looked-up term first appears beyond that, it is shown the row's opening plus a window around the term instead, so a row found *because* of text past the cut is not scored on an unrelated opening.
- **Only when reranking runs.** Under `--no-rerank` (or `rerank.enabled = false`) the keyword channel is skipped: with no cross-encoder to arbitrate, keyword rows would displace vector rows on a lexical-rank guess, and `--no-rerank` is the mode that promises pure vector order — which keeps `--tags` pins exact.
- **Never fails a recall.** A store that has not run migration `0012`, or a query the engine rejects, costs the extra candidates and logs a warning; recall continues vector-only.

Config: `[recall] keyword_channel` (default `true`; turn it off with `hafiz config set recall.keyword_channel false`); `keyword_candidates = 6` caps the keyword-only rows added per searched text (each costs one more row through the cross-encoder). Measured by replaying a scratch copy of a real 3,270-annotation store (`scripts/replay_keyword_recall.py`): 40 exact-string questions (a file name, flag, port, constant or ticket id plus ordinary words): the note containing the token was in the top 6 for 16 of 40 with the channel off and 40 of 40 on (35% → 90% once the `0.05` floor a recall hook applies is counted). 60 ordinary prompts sampled from the retrieval log: a recall hook shows 157 rows off and 166 on, 95% of the rows it showed before are still shown, the top row changes for 5 of 60, and 8 of 60 lose a row to a better-scoring newcomer. At a hook's `--limit 20` the same 60 prompts show 338 rows off and 357 on, 99% of the old rows are still shown, and 2 of 60 lose one. Cost: the keyword lookup itself is ~10 ms (p50); the rest is up to 6 more rows through the cross-encoder per searched text — median search time +9% at `--limit 20` (1.19 s → 1.29 s, p95 within run-to-run noise on a shared CPU), and up to about a quarter at `--limit 6` on a long condensed prompt. The retrieval log is unchanged.

**Pinning vs. ranking** (`--tags` on `query --observations`): some sets are
defined by curation, not by content. "The rules that always apply" is the
canonical case — it is not a semantic property, so no query text retrieves it.
Measured on a 1,528-row store, the best hand-tuned query for the topic-independent
hard rules recovered 2 of ~6; the rest sat below any usable floor. `--tags`
closes that gap: tag the rows once, then address them by tag. The filter runs in
SQL alongside the vector scan, so `--limit` counts only matching rows. For a
stable pin, pair it with `--no-rerank` and **no** `--min-score` — a floor would
re-introduce the ranking judgement the pin exists to bypass. Tags are still
free-form; hafiz does not reserve any name.

**`--limit` still truncates, and silently.** `--tags` picks *which* rows are
eligible; it does not exempt them from the limit, which defaults to **10** and
cuts with no signal that anything was dropped — in reranked order by default,
cosine order only under `--no-rerank` (`total` in
`--json` is the post-truncation count). So a pin is only exact when `--limit` is
at least the size of the tagged set — set it deliberately above the expected
size, and treat a result count equal to the limit as "possibly truncated".
Adding an ANN index (`hnsw`/`ivfflat`) to `annotations.embedding` would break
the guarantee differently and more quietly, by bounding recall before the tag
filter is applied; there is no such index today.

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
| `compact` | context-window injection | content + kind + source + age only; long annotations excerpted |
| `md` | prompt injection | raw markdown, not Rich-rendered; long annotations excerpted |

`compact` drops uuids, timestamps, null fields, and float scores. Pass
`--with-ids` to re-add ids — **do this whenever the consumer might write back**,
because an agent that can read a decision but not cite it cannot
`observe --supersedes` it, and the corpus silently accumulates contradictions.

#### Snippet extraction (`compact` / `md` only)

Annotations have no schema beyond `kind`, so they arrive as prose blobs —
median 951 chars, p90 1,589 on a measured store — and were injected whole.
`compact` and `md` now carry the **best-matching excerpt** of any annotation
over `[annotations] snippet_chars` (default 480); shorter records pass through
untouched. Measured on a `--observations --limit 20` recall: **18,839 → 6,495
bytes (−65%)** for **~+0.5 s**.

- **Marked, never silent.** `compact` sets `excerpt: true` and `full_chars`;
  `md` appends `excerpt of N chars` to the metadata line; both elide with `…`.
  A decision read without its scope qualifier is worse than one never
  retrieved, so truncation is always visible.
- **Nothing becomes unreachable.** `json` and `rich` always carry full
  `content`. Set `snippet_chars = 0` to disable extraction entirely.
- **Two-stage selection**, mirroring the search path: a free lexical pass
  shortlists a few spans per record (always including the head), then the same
  cross-encoder that ranked the results picks among them. Cross-encoding every
  span cost ~1.0 s; shortlisting first cuts that to ~0.5 s.
- **Degrades, never fails.** If the cross-encoder is unavailable the lexical
  winner is used, and failing that the head of the record.

Measured on a real session-start injection (`query --observations --source
user:anjum --limit 50`): 70,272 B of `json` → **2,498 B** of `compact
--min-score 0.05` (~17.5k → ~625 tokens), retaining all three genuinely
relevant rows.

**Scoping flags** (on `context`, `query`):

| Flag | Scope | How it works |
|------|-------|-------------|
| `--project X` | Single named project | Resolves `X` to a project on file (below), then filters to it |
| `--workspace` | Sibling projects | Resolves directories in parent of cwd, matches to DB project tags (normalized) |
| _(neither)_ | Everything | No filter |

`--project` and `--workspace` are mutually exclusive.

**How a project name resolves** — on every command, the daemon and MCP alike (`hafiz/core/projects.py`):

1. The exact name on file, or a **declared synonym** (`hafiz projects alias`).
2. The same name ignoring case, spaces, `_` and `-` (`ledger api` → `Ledger API`).
3. A near-certain typo (similarity ≥ 0.9, clear of the runner-up) — corrected on read with a one-line notice on stderr. Names differing only in a number (`…-v3` / `…-v4`) are siblings, never typos.
4. Several projects still fit → the one not **retired** wins; if that doesn't settle it, or nothing fits, exit `2` with `{"ok": false, "error": ..., "candidates": [...]}`. A misspelt project used to return an empty result indistinguishable from "nothing known".

Writes (`observe`, `note`, `capture`) apply 1–2 and file under the main name. A **new** name that is a near-certain typo of an existing one is refused unless `--new-project`; anything else is a new project. `ingest` and the importers never refuse — an unclear name is kept as given, because git hooks pin it. Agent writes with no `--project` (`--source agent:*`, and MCP `hafiz_observe`/`hafiz_note`) file under the project of the working directory: the closer of the indexed tree that holds it and its git repo's folder name. A user's own notes stay unscoped.

Spellings that merely look alike are never merged automatically — two repos can be one key apart (a retired v1 and its replacement). See `projects` below.

**Domain filters** (on `context`, `query`): toggle whole data domains on/off without losing the rest of your index.

| Flag | Effect |
|------|--------|
| `--include-domain doc,file` | Restrict results to units whose `kind` starts with one of the listed domains. Comma-separated. |
| `--exclude-domain file` | Drop results whose `kind` starts with one of the listed domains. Comma-separated. |

A "domain" is the prefix of `kind` before the first dot — `doc`, `file`, `mail`, `chat`. There is no `code` domain (`--include-domain code` matches nothing). (For exact-kind filters like `doc.heading`, keep using `--type`.) The two flags are mutually exclusive *per-domain*: `--include-domain doc,file --exclude-domain doc` errors out before hitting the DB. A dotted value (`--include-domain doc.heading`) also errors — domains are dotless tokens.

When both flags are omitted on the CLI, the active session's defaults (set via `session start --include-domain` / `--exclude-domain`) are inherited. Passing *either* flag explicitly skips inheritance entirely, so a query can flip the filter without first ending the session.

### Annotations (the "wisdom layer")

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `observe "<text>"` | Embed and store a fact / decision / learning / pattern / warning / note | Embed | `--json` | rich panel |
| `note "<text>"` | Shortcut for `observe --type note` — low-bar raw capture lane (skips *near*-duplicate detection; still collapses byte-identical writes) | Embed | `--json` | rich panel |
| `reconcile` | Read-only sweep: cluster near-duplicate **live** annotations and propose a resolution to run | Embed | `--json` | rich panels |
| `projects` | Every project name with its row count, synonyms and retirement, plus undeclared look-alikes (same name ignoring case/spacing, different project) | — | `--json` → `{"projects":[{"name","rows","retired","aliases":[...],"unmerged_alias_rows"}],"look_alikes":[[name,...]]}` | rich table |
| `projects alias ALIAS PROJECT` | Declare a synonym: reads and writes through `ALIAS` land on `PROJECT` from now on. Stored in the `project_names` table, so every entry point sees it. | — | `--json` → `{"ok":true,"alias","project"}` | one line |
| `projects unalias ALIAS` | Remove a synonym; rows already merged stay put | — | `--json` → `{"ok":true,"alias","removed":bool}` | one line |
| `projects retire PROJECT [--undo]` | Mark a project retired: still searchable, but loses to an active project when a name fits both | — | `--json` → `{"ok":true,"project","retired":bool}` | one line |
| `projects merge [--apply]` | Move rows filed under a declared synonym to its main name — notes, transcripts, sessions. Never the file index (git hooks pin its name; the synonym already routes reads to it), and never an undeclared look-alike. Preview unless `--apply`. | — | `--json` → `{"ok":true,"applied":bool,"moves":[{"alias","project","rows":{table:n}}],"skipped":[{"alias","project","reason"}]}` | per-synonym lines |
| `journal` | Time-bounded digest of annotations, grouped by day | — | `--json` | rich tables · `--format mermaid` |
| `distill` | Surface the promotable backlog — recent notes + transcript turns, grouped into themes (scanner, not promoter) | Embed | `--json` · `--brief` | rich panels |

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
    - **Cosine clusters; text overlap decides.** These were the same judgement until a full sweep showed they diverge below ~0.97. Median word-level overlap of cluster members, by cosine band, on a 1,540-row store:

      | cosine | ≥0.99 | 0.97–0.99 | 0.95–0.97 | 0.92–0.95 | 0.88–0.92 |
      |---|---:|---:|---:|---:|---:|
      | same words | 0.93 | 0.61 | 0.48 | 0.19 | **0.09** |

      At the default `0.88`, members share under a tenth of their words — same *subject*, different *statement*. One cluster paired the Shop Front v2 **stage** pipeline path with the **prod** one, where the prod row says in as many words "NOT front/v2 like stage". So `threshold` still governs what gets grouped (and what `observe` surfaces on write, where "you may already have said this" is useful), but it no longer governs what gets recommended.
    - **Every cluster carries a proposal, and the proposal is made on text.** One member is the `primary`; the rest are retired. Each non-primary member reports `overlap` (word-level ratio vs the primary), `unique_words` (every word it has and the primary lacks) and `unique_fragments` (those words as contiguous runs, longest first). `suggested_action` has three tiers:

      | tier | means | your move |
      |---|---|---|
      | `retire` | `unique_words == 0` on every sibling — every word of theirs appears in the primary, in order | no *wording* is lost; still read the pair, since a subsequence can invert a claim |
      | `review` | words differ, but none form a run of ≥4 | read the stray words; usually a contraction or spelling |
      | `merge` | a sibling carries whole claims the primary lacks | write text covering `unique_fragments`, then retire |

      `commands` is the ordered, runnable resolution — at most one `observe`, since `--supersedes` takes a single id. `review` gets the **merge** commands, not a ready-made `forget`: merging mere rewording costs a redundant rewrite, retiring real text costs the fact. The terminal names the cheaper path in prose instead of pasting it.
    - **The tiers grade evidence, not meaning.** Whether an unmatched word is a rewording or a reversal is a semantic judgement, and this command deliberately does not guess — two earlier versions did, and each had a case where it proposed destroying a fact. So a total meaning change carried by one short word (`stage` vs `prod`) lands in `review`, not `merge`. `review` still means a human looks, which is the property that matters. Only `retire` makes a promise, and it is the one promise text can actually keep.
    - **Judge safety on `unique_words == 0`, never on `overlap` being high.** `overlap` is `2·matches/total`, so length asymmetry drags it down: a row wholly contained in a keeper four times its length scores ~0.40 while losing nothing (`2M/T`: 2× → 0.67, 3× → 0.50, 4× → 0.40). Terminal output prints the verdict ("nothing only here" / "N word(s) only here") next to the ratio for exactly this reason.
    - **Primary is the superset, not the newest.** Newest-unless-materially-shorter read plausibly and inverted in practice: on a same-day pair it proposed retiring the only row carrying "supersedes the earlier same-task decision", because the row lacking that sentence was a few characters longer. Containment is checked directly; newest wins only among rows that already say the same things.
    - JSON shape: `{action, scanned, total_live, truncated, threshold, total, clusters: [{kind, project, suggested_action, primary_id, commands: [...], members: [{id, content, score, source, valid_from, chars, primary, overlap, unique_words, unique_fragments}]}]}`. `suggested_action` is `retire` | `review` | `merge` — `review` is new, so a consumer switching on the value needs a branch for it.
    - The count also rides on `doctor` (and so on `status --diagnose`) as **Knowledge base deduplicated**, because a read-only command nobody remembers to run surfaces nothing. It is not on `status`: with no vector index on `annotations.embedding` the count is a quadratic scan (~310 ms at 1,099 rows), and `status` is on the hot path.
- **Unit binding**: annotations created via `observe` are unit-free by default (can be linked later via API).

#### The distillation backlog (`distill`)

`distill` is still a scanner and still never calls an LLM — the synthesis is yours. What is automatic is the *backlog*: it drains, it groups itself, and it can come find you.

- **It drains, using state that already existed.** There is no "distilled" flag and no proposals table. A capture leaves the queue when `observe --derived-from <id>` cites it (**promoted** — that citation in `annotation_targets` is the receipt) or when `forget <id> --annotation` retires it (**declined** — the candidate query only ever returned live rows). A citation from an annotation that was itself retired or expired doesn't count: the distilled belief is gone, so the note is promotable again. `--include-promoted` shows the drained rows, flagged `promoted: true`, for auditing.
- **Themes, not a flat list.** Candidates are grouped by embedding similarity using the same single-linkage clustering as `reconcile`. Notes and transcript turns cluster **together**, so a note lands with the turns it came from and one `observe` can cite both. Each theme carries a `scaffold` citing **every** member — a truncated citation list would strand the uncited members in the queue permanently.
- **Two corpora, two thresholds.** `distill.cluster_threshold` (default `0.65`) governs note↔note pairs; `distill.cluster_threshold_messages` (default `0.78`) governs turn↔turn pairs; a note/turn pair takes whichever is **stricter**. The note floor is deliberately far below `dedup.threshold` (`0.88`): dedup asks "is this the same claim restated?", clustering asks "are these about the same thing?" — at `0.88` essentially nothing groups. Turns need a higher bar because they are far more self-similar: they come from the same work sessions and share vocabulary and framing, so under single-linkage almost every adjacent pair clears a note-tuned floor and one chain absorbs the corpus. Measured on a real 30-day window at `0.65`, one theme held **366 of 400** turns and produced ~nothing of a readable size — and it did so at every corpus size tried (39 of 50, 105 of 120), so it was never a corpus-size problem. At `0.78`: 21 themes of a readable 2–12 members. Cross-kind pairs take the stricter value because a cross-kind link is exactly the one that can *bridge* the two populations — the looser value would let one note glue together turns the turn threshold correctly held apart.
- **Near-identical repeats collapse onto one representative.** `distill.collapse_threshold` (default 0.98) folds *message* members of a theme that are the same text, so `size` counts distinct captures. This matters because themes sort by size: with themes anchored on your own turns, the largest ones turned out to be harness-injected skill preambles ("Base directory for this skill: …") at similarity 0.99–1.00 — `role=user`, but the harness said them — measured at 57 of 197 multi-member members, and twelve copies of one string outranked a real three-capture topic. The repeats are **still cited** by the scaffold (`cited_ids`, not `members`): they are hidden from the reader, not from the drain, since hiding them from both is how a capture strands. `backlog.collapsed` counts them, and `total_size` reports the uncollapsed count. Well above `dedup.threshold` (0.88) on purpose — that asks "the same claim restated?", this asks the narrower "the same text?". **Notes are exempt**: a note is deliberate, `hafiz note` already refuses a byte-identical repeat, and a genuinely repeated note is a signal.
- **Oversized themes are split, never truncated.** `distill.max_theme_size` (default 12) caps members per theme; a bigger cluster becomes consecutive parts, each with its own complete scaffold. This is correctness, not cosmetics: a scaffold cites **every** member because citing is what drains the queue, so dropping ids to shorten the command would strand those captures permanently. An 80-uuid `--derived-from` strands them just as effectively by being too long to run. Chaining means a big cluster is not one topic anyway — at 80 members its ends share nothing — so splitting is closer to the truth than the merge was. Parts split on `ts`, so each reads as a stretch of one conversation, and each is rescored against its own members rather than inheriting the parent's similarity. `0` disables splitting.
- **The clustering corpus prefers your own turns.** Themes were 91% assistant turns and clustered on the *shape of a progress report* — "Done.", "committed", "N passed, M skipped" — rather than on topic. A user turn is irreplaceable: nothing else records what was asked and why, whereas an assistant status report duplicates what `git log` and your work items already hold. So the corpus query puts `role='user'` turns at the head of its ordering. Measured on a 30-day window the embedded pool is 3,500 assistant turns to 355 user ones against a cap of 400, which moved the user share of theme members from 11.5% to 88.8% and doubled readable themes from 28 to 56. It is a **preference, not a filter** — assistant turns carry the rationale, so they stay eligible and fill whatever capacity your turns don't, and a window with no user turns still fills to the cap. The readable turn list is **not** reordered: it stays recency-ordered so conversational flow, including that rationale, survives.
- **Two caps, because reading and clustering are different jobs.** `distill.message_limit` (default 50) bounds the **readable** turn list, which keeps its unembedded turns. `distill.theme_corpus_limit` (default 400) bounds the embedded-only corpus themes are grouped from. These shared one cap until it was measured: only ~8% of turns carry a vector, so a 50-row candidate slice offered clustering ~4 rows. Filtering the readable list instead would have been a regression — an unembedded turn is perfectly distillable by reading it. Clustering is O(n²) in the corpus size, which is why it is capped at all.
- **An unembedded turn is not a candidate.** Selective embedding already declined it at import (short turns, pure tool-result echoes). Re-offering it here contradicts that judgement: on a real 90-day window it buried 2 useful themes under ~26 singletons of browser chatter and raw `<toolCall>` echoes. The drop is counted in `backlog.skipped_unembedded`, never silent. Notes are exempt — a note is a deliberate capture, so an unembedded one stays as a singleton.
- **Retention is a filter, not a nicety.** Turns whose communication is past `retention_until` are excluded. Distilling one would mint a fresh annotation carrying its content, and an annotation has no `retention_until` — the source data would outlive the window it was promised to expire in. The `forget --all-expired` sweep can lag; this must not.
- **`--brief` is the automatic surface.** Token-lean markdown for a session-start hook, printed with plain `print` (Rich soft-wrap would insert hard newlines mid-command, and a wrapped scaffold is a broken scaffold). It prints **nothing at all** unless the backlog clears `distill.brief_min_pending` (default 3) **or** `distill.brief_min_age_days` (default 2.0) — either fires it. Silence has to be the ordinary answer or the hook gets removed. Rendering is capped by `brief_max_themes` / `brief_max_members`, and anything dropped is stated in the output.
- **Ordering:** biggest theme first, then most recent — the head of the list is where distillation pays best. Members inside a theme run oldest-first, so reading one is reading how the thought developed.
- **Caps:** `--limit` on notes (default 200), `--message-limit` on turns (default `distill.message_limit`, 50). Turns are taken salient-first then newest; the previous hardcoded 50 under `ts ASC` surfaced the *oldest* fifty in a window and ignored `marked_salient` entirely. The clustering corpus has its own cap, `distill.theme_corpus_limit` — see "Two caps" above.
- **`--json` shape:** the existing keys are unchanged, plus `notes[].promoted`, and two new blocks:
  - `backlog: {pending, promoted, oldest_pending_age_days, themes, clustered, skipped_unembedded}`. `pending` is derived from the themes, not the raw candidate lists, so "30 pending in 14 themes" cannot disagree with itself and the `--brief` gate counts only work a reader can act on.
  - `themes: [{size, score, oldest, newest, scaffold, members: [{id, kind, label, ts, content}]}]` where `kind` is `note` or `message`.

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
| `import claude-code --from-hook` | Import only the session an agent-harness hook names, reading the hook payload as JSON on **stdin**. Always exits 0. | — | `--json` → `{ok, action, transcript_path, project, summary}` | silent |
| `import cursor [PATH]` | Import Cursor chat conversations from its SQLite store (**read-only**). Defaults to this platform's Cursor storage. | — | `--json` | rich table |
| `import chatgpt <PATH>` | Import a ChatGPT data export. Accepts the `.zip`, its unzipped directory, or `conversations.json`. Path is required. | — | `--json` | rich table |
| `import codex [PATH]` | Import Codex CLI session rollouts. Defaults to `$CODEX_HOME/sessions` **and** `archived_sessions`. | — | `--json` | rich table |
| `import hermes [PATH]` | Import Hermes agent sessions from its SQLite `state.db` (**read-only**). Defaults to `$HERMES_HOME/state.db` (`~/.hermes/state.db`). Skips sessions the Hermes memory plugin already captured, and sessions already past the 90-day retention. | — | `--json` | rich table |
| `recall <target>` | Ordered messages for a session (slug or uuid) or communication; or vector search via `--query` | — | `--json` | rich table |
| `query --include-transcripts` | Add matching source-layer turns to results, tagged `layer="source"` | — | `--json` | rich panel (separate transcript section) |
| `context --include-transcripts` | Append matching transcript turns under a `transcripts` field | — | `--json` | rich panel |
| `forget <target>` | Targeted redaction: soft tombstone by default, `--hard` deletes content | — | `--json` | rich line |
| `forget <uuid> --annotation` | Retire a **knowledge-layer** annotation by uuid (soft — sets `valid_until = now`, kept for audit). For wrong/obsolete observations. | — | `--json` → `{ok, action, id, kind, valid_until}` | rich line |
| `forget --all-expired` | Sweep mode — tombstone every communication past its `retention_until` | — | `--json` | rich table |

Defaults:
- `import claude-code` source path: `~/.claude/projects/`. Idempotent by `(agent='claude-code', external_id=<jsonl session uuid>)` at the communication level, and by **`source_message_id`** (Claude Code's own per-record `uuid`) at the turn level.
- `import claude-code` flags: `--project`, `--limit`, `--since 7d`, `--dry-run`, `--no-embed`, `--from-hook`.
- `import claude-code [PATH]` accepts **either** a directory of session JSONL **or** a single `.jsonl` file — the latter is what `--from-hook` uses, so capture cost is proportional to the session that just ended rather than to every session ever had.

**`import <agent> --json` shape** (all five importers):
`{action, root, project, since, dry_run, embed, summary, retention_sweep}`, where
`summary` is `{files_seen, files_skipped, skipped_expired, communications_created,
communications_existing, messages_written, messages_embedded, sessions_created, errors}`.
`skipped_expired` counts sessions an importer refused because they were already
past the 90-day retention (only `hermes` sets it today; it is `0` elsewhere).

**`--dry-run` is a real preview.** It resolves the same idempotency keys a real
run would and reports what *would* be written, split into
`communications_created` / `communications_existing` with `messages_written`
counting only turns not yet stored. A session imported mid-flight therefore
reports as *existing* while still showing pending turns — the case hook-driven
capture hits constantly. (Through v11 this returned flat zeros for every dry
run, which is part of how a two-month capture outage stayed invisible.)

**Automatic capture (`--from-hook`).** Parsing the harness payload inside hafiz
— rather than in the hook body — is what keeps the installed hook a portable
one-liner: no `jq`, no shell JSON handling, and the extraction is unit-tested.
Two invariants hold on every path: it **imports one file, not the store**, and
it **never fails the turn** (empty stdin, malformed JSON, absent
`transcript_path`, missing file, or any exception all exit 0 silently). The
session's `cwd` is resolved to an indexed project by longest-prefix match, so
captured transcripts land project-scoped rather than untagged; no match yields
`None` rather than a guess.
- `recall` flags: `--query`, `--role`, `--from`, `--to`, `--has-tool-call` / `--no-tool-call`, `--limit`.
- `forget` flags: `--hard`, `--all-expired`, `--dry-run`.
- Retention: 90 days from `started_at` unless explicitly overridden at insert time.
- Selective embedding: skip messages under ~30 tokens; skip pure tool-result echoes; embed when `marked_salient=true` regardless of length.

**Turn identity is the source's, not ours** (schema `0008`). A harness spreads
one session across several files — resumed sessions, sidechains — and each file
restarts its positional `seq` at 0. Deduping turns on `(communication_id, seq)`
therefore made every file after the first collide with the first and lose its
turns: **11,214 of 38,249 (29.3%)** on a real store, one session spanning 25
files. It also caused the opposite fault, writing the same turn twice under
different seqs when a resumed file shifted positions.

`communication_messages.source_message_id` holds the id the source assigns each
turn, under a partial unique index over `(communication_id, source_message_id)`.
Consequences worth knowing:

- **`seq` is now an append-only ordering key** within a communication, allocated
  from the current maximum, not taken from the file. `ts` remains the time axis.
- **Re-import heals.** Turns lost to the old collision are recognised as absent
  and finally land. Migration `0008` backfills `source_message_id` from the
  `metadata->>'claude_uuid'` the importer had been storing all along, which is
  what makes an existing store recoverable rather than merely fixed going
  forward.
- **The column is nullable on purpose.** `hafiz capture` and hand-built rows
  have no source-native id and keep deduping positionally, so
  `uq_messages_comm_seq` is retained as their guarantee.
- Importers **must** set `MessageInput.source_message_id` whenever the source
  has one. An importer that omits it silently inherits the positional bug, so
  the shared writer appends a warning to `summary.errors` when a parser hands
  over turns with no identity at all.

#### Per-importer notes

All five share one options set (`--project`, `--limit`, `--since`, `--dry-run`,
`--no-embed`, `--json`) and one summary shape, because they share one writer
(`hafiz/core/importers/base.py`); only parsing differs.

| Importer | Source | Turn identity | Project scoping |
|---|---|---|---|
| `claude-code` | `~/.claude/projects/**/*.jsonl` | record `uuid` | session `cwd` |
| `cursor` | `state.vscdb` (SQLite, read-only) | `bubbleId` | workspace folder |
| `chatgpt` | export `conversations.json` | node `id` | `--project` only |
| `codex` | `$CODEX_HOME/sessions/**/*.jsonl` | record `id`, else `<session>:<line>` | session `cwd` |
| `hermes` | `$HERMES_HOME/state.db` (SQLite, read-only) | `messages.id` | `git_repo_root`, else `cwd` |

- **Cursor** is opened **read-only** (`file:…?mode=ro`) — it is a running
  application's database. Conversations are Cursor *composers*; turns are
  *bubbles*, ordered by `composerData.fullConversationHeadersOnly` rather than
  key order, since bubbles are independent rows. `workspaceStorage/<id>/workspace.json`
  maps a workspace to its folder, so conversations land project-scoped. Drafts
  and composers with no bubbles are skipped (on a real store, 70 headers yielded
  26 conversations).
- **ChatGPT** exports are a message **tree**, not a list: regenerated answers and
  edited prompts create branches. The importer walks parent pointers back from
  `current_node`, so it imports the conversation you actually had rather than
  every abandoned draft. There is no default path — an export is a file you
  download, so `PATH` is required and its absence is a clean exit 1.
- **Codex** appends to one rollout file when a session resumes, so unlike
  claude-code there are no sibling files and a record's line number is a stable
  identity where no native `id` exists. `CODEX_HOME` is honoured. Reasoning
  records and `event_msg` telemetry are skipped; unknown record types are
  ignored rather than treated as errors, because Codex is explicitly tolerant of
  its own schema drift. `developer` messages (Codex's injected permission /
  agent-role prompts) are stored with role `system`, and the re-injected
  `# AGENTS.md instructions for …` user item is dropped. **OpenAI does not
  document the rollout schema; this parser follows community references and has
  not been verified against a real Codex install — run `--dry-run` first.**
  For hands-free capture Codex offers `SessionEnd` / `Stop` hooks
  (`$CODEX_HOME/hooks.json`, payload carries `session_id` and `transcript_path`);
  hafiz does not wire them yet.

- **Hermes** keeps all sessions in one SQLite `state.db` (WAL mode), opened
  **read-only**. `PATH` may be the file or the Hermes home directory;
  `$HERMES_HOME` is honoured. Sessions are visited newest first, and `--limit N`
  stops after N sessions have been looked at (expired and plugin-captured ones
  included). A session whose last activity — its end, else its newest turn — is
  under 30 minutes old is left alone: the plugin captures it at session end, and
  importing it first would store it twice. Optional columns are probed with `PRAGMA table_info`, so an
  older Hermes with fewer columns still imports. Hermes soft-deletes turns with
  two flags, and the importer follows them:
  - `active = 0, compacted = 0` — the user rewound the turn. **Skipped.**
  - `active = 0, compacted = 1` — archived by context compaction; the only full
    record of the session. **Imported.**
  - Compaction re-inserts the surviving tail as live rows, and a later
    compaction archives those copies too, so one turn can exist many times (355
    copies among 592 rows on a real session). A copy keeps the original's
    timestamp, so a row with the same role, content, tool call and timestamp as
    an earlier one is skipped; a message genuinely repeated later has a new
    timestamp and is kept. (An older Hermes without timestamps: a live row that
    matches an archived one is skipped, each archived row cancelling at most
    one.)
  - **A rewind after an import is not undone.** Turns already imported stay;
    remove them with `hafiz forget <session>`.

  An assistant turn's tool calls become `tool_use` entries and each `tool` row a
  `tool_result` entry (same shape as the other importers); reasoning is kept in
  message metadata, never in `content`. Two interactions with the rest of hafiz:
  - **The Hermes memory plugin already captures sessions** at their end via
    `hafiz capture`, under a random `external_id`, recording the Hermes session
    id in `metadata.session_slug` and the title `hermes:<id>`. A session found
    there is counted in `communications_existing` and **not imported again**
    (tombstoned captures count too, so a forgotten capture is not resurrected).
  - **Expired sessions are not written.** A session whose `started_at` + 90 days
    has passed is skipped and counted in `skipped_expired`, so no chat content
    past retention is ever stored. (The other importers write first and rely on
    the next sweep to tombstone.)

**Retention enforcement.** `import` runs the expiry sweep automatically and
reports it (`retention_sweep: {matched, tombstoned, dry_run}` in `--json`; a row
in the rich table when non-zero). `--dry-run` propagates to the sweep. It is
deliberately *not* wired into `ingest`: that's the file/doc subsystem, it fires
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
| `retrievals` | Never-recalled knowledge, most-recalled rows, per-entry-point usage, queries that returned nothing, and **delivery**: what recall hooks showed agents and whether agents appear to have used it. Skipped harness messages are counted in `skipped` and left out of every other figure. `--since-days`, `--limit` | — | `--json` → `{ok, retrievals, skipped, by_command[], empty_result_rate, never_recalled, blind_before, unanswered[], most_recalled[], delivery, telemetry_started, enabled}` | rich tables |

- **Recorded in `hafiz/core/`, not the commands layer.** `hafiz/core/daemon.py` calls `search_annotations` directly, so command-layer telemetry would silently miss every warm request — the exact failure class this audit kept finding. `vector_search` / `search_annotations` default `telemetry_command` to their label, so a *new* caller is recorded without knowing telemetry exists; pass `None` to opt a call out.
- **Never fails a search.** Recording is best-effort and swallows everything. A memory layer that can break the read path gets removed.
- **`by_command` answers "is anyone calling this?"** — `{command, calls, empty, avg_top_score}` per entry point. Each label is one search against one layer, so a `context` bundle records **two** rows: `context` (units) and `context --observations` (annotations). Those labels exist because they didn't: `context` used to record under `query` / `query --observations`, which made the command the docs open with indistinguishable from a plain query — a 30-day audit could not tell "never used" from "never instrumented". The `enabled` flag and `retrievals` total say whether recording works; `by_command` says whether the *product* is being used.
  - `avg_top_score` is a SQL `AVG`, so it skips rows with a NULL `top_score` — i.e. it averages the **non-empty** calls only. It is not `calls`-weighted; don't multiply the two.
  - It covers the two instrumented layers, not every command. `recall` (source layer) records nothing — `telemetry.RECALL` is defined but unwired — so its absence from `by_command` is "not instrumented", not "not used". That's the same ambiguity these labels fixed for `context`, still open one layer down.
- **`delivery` answers "did it help?"** — `{injections, sessions, by_event[], most_used[], shown_never_used[], method}`, or `{error}` if only this section failed. `by_event[]` is `{event, injections, memories_shown, scorable, likely_used, used_rate, avg_chars, p50_ms, p95_ms}` per hook event (`SessionStart`, `UserPromptSubmit`); the memory lists are `{preview, shown, scorable, used}`.
  - **Source.** `hafiz import claude-code` keeps the stdout of any hook whose command mentions `hafiz` as a `role='system'` message with `metadata.kind = "hafiz_injection"`, `hook_event` and `duration_ms`. Those rows are never embedded and are excluded from `distill` and from embed coverage, so the store's own memories don't come back as transcript hits or distill proposals. Re-run the import to backfill existing sessions.
  - **Method (rough, stated in `method`).** A memory counts as *likely used* when the agent's following turns — its text and tool inputs, until the next prompt; the whole session for `SessionStart` — reuse ≥3 (and ≥20%) of the memory's specific terms that the session had not already seen. Memories with fewer than 3 such terms are *unscorable* and left out of `used_rate`. It **under-counts**: a memory that changes behaviour without changing vocabulary, or is paraphrased, scores as unused. Read it as a lower bound and a trend line, not a verdict on one memory.
  - **`shown_never_used`** lists memories judged ≥3 times and never picked up — candidates to reword, retarget, or unpin.
- **`n_results = 0` rows are the point.** The gap between what agents ask for and what the store holds is the only signal that says what to write down *next*; nothing else in hafiz produces it.
- **`blind_before`** is the subset of never-recalled rows written before telemetry existed. Without it, "1,094 never recalled" reads as an indictment of knowledge that simply predates the log.
- **Query text is a new data category** for this store: it's what you were *looking for*, not what you concluded. So it lands in the source layer with the source layer's guarantees — `retention_until` (default 90d), swept by `forget --all-expired`, counted in `status`/`doctor` — and it never leaves the machine.
- **Opt out** with `[telemetry] retrieval = false` (or `HAFIZ_TELEMETRY__RETRIEVAL=false`). `retention_days` and `min_query_chars` are tunable in the same block; queries shorter than `min_query_chars` (default 3) aren't recorded, since "ok" / "yes" say nothing about what was asked for.

### Switching backends

`export` is a one-way eject to plain files. This is the transport: every table, both layers, into a database on the other backend.

It exists because "just re-ingest" only half works. Docs and files can be re-derived — git is their source. **Annotations cannot.** Someone who starts on SQLite, accumulates two years of decisions and then wants Postgres for a second machine has nothing to re-ingest from.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `migrate-backend --to <url>` | Copy the whole store to another backend. `--dry-run` reports without writing; `--yes` skips the prompt. | All tables | `--json` → `{ok, dry_run, source_backend, target_backend, target, total_rows, vector_check, tables:[{table, source_rows, copied, back_references, ok}]}` | rich table + verification line |

```bash
hafiz migrate-backend --to "sqlite:///~/.local/share/hafiz/hafiz.db" --dry-run
hafiz migrate-backend --to "sqlite:///~/.local/share/hafiz/hafiz.db"
hafiz config set database.url sqlite:///~/.local/share/hafiz/hafiz.db
```

Guarantees, in the order they matter:

- **Nothing is recomputed.** Every column copies verbatim, including `retention_until`, `valid_until` and `superseded_at`. Tombstoned rows migrate *as tombstoned* — they are the audit trail, not junk.
- **The source is opened read-only** and is never modified.
- **It refuses rather than resumes.** The target must be empty; this copies, it does not merge. A run that dies halfway leaves a database the next run declines to touch — delete it and retry.
- **It verifies itself.** Per-table row counts are compared and a vector is round-tripped, because "exited zero" is not evidence that 400,000 rows arrived. A mismatch is a failure, not a warning.

Measured on a real store: **421,732 rows, Postgres → SQLite, 81s**, 71,194 self-references restored, vectors round-tripping with a maximum element delta of 0.

### Sovereignty (export)

The data-portability complement to `forget`. Dumps the brain's **wisdom layer** — annotations (decisions / facts / learnings / patterns / warnings) — to plain files for backup, human review, or migration. Indexed file bodies are **excluded** (git is their sovereign copy). Forgotten (`valid_until`) and retention-expired rows are never included — export reflects only live data.

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `export` | One-way dump of annotations (+ optional transcripts) to a plain-files directory | Annotations | `--json` → `{ok, path, format, counts, warning}` | rich table + path + secrets warning |

Defaults & flags:
- `--out` / `-o`: output directory (default `./hafiz-export`). Written atomically (temp dir → rename); an existing directory is replaced.
- `--format` / `-f`: `md` (human-readable tree, default) or `json` (per-table JSONL — `knowledge/annotations.jsonl`, plus `source/*.jsonl` when transcripts are included). Lossless except embeddings.
- `--include-transcripts`: also dump source-layer agent transcripts (opt-in, mirrors `query` / `context`). Off by default.
- `--project`: limit to one project; default is the whole brain.
- Every export writes a `manifest.json` (tool/schema version, format, generated-at, scope, counts).
- `import` (round-trip restore) is **not yet implemented**; the `json` format is designed to enable it later.

### Review

| Command | Purpose | Brain | Agent use | Terminal use |
|---------|---------|:-----:|-----------|-------------|
| `review` | Self-review of the knowledge base: counts units / embeddings / annotations, then flags annotation-quality gaps (no decisions, low confidence, staleness). | Annotations | `--json` emits `{stats, findings, summary}` | rich panel of findings |

## Common Flags

| Flag | Available on | Purpose |
|------|-------------|---------|
| `--json` / `-j` | Most commands | Machine-readable output for agents |
| `--format` / `-f` | `journal` | Output format: `rich` (default), `json`, or `mermaid`. `--json` is a shortcut for `--format json` |
| `--mermaid-kind` | `journal` | With `--format mermaid`: `supersession` (decision-evolution graph, default) or `timeline` |
| `--project` / `-p` | Most commands | Filter or tag by project name |
| `--workspace` / `-w` | `context`, `query`, `journal` | Scope to sibling projects in parent directory |
| `--type` / `-t` | `query`, `observe`, `journal` | Unit kind or annotation kind depending on context |
| `--limit` / `-l` | `query`, `journal` | Maximum results |
| `--observations` | `query` | Search annotations (wisdom layer) instead of content |

| `--since` | `journal` / `distill` | Duration window ending now (default `7d`) |
| `--day` | `journal` | Specific UTC day (ISO date). Exclusive with `--since` |
| `--expires-in` | `observe`, `note` | Expire after duration. Exclusive with `--expires` |
| `--expires` | `observe`, `note` | Expire at ISO date. Exclusive with `--expires-in` |
| `--source` | `observe`, `note`, `journal` | Origin tag (`agent:<name>`, `user:<name>`) |
| `--session` | `observe`, `note`, `journal`, `distill` | Explicit session id |
| `--task` | `observe`, `note`, `journal`, `distill` | Explicit task label |
| `--supersedes` | `observe`, `note` | UUID of annotation being replaced |
| `--derived-from` | `observe`, `note` | UUIDs this row was distilled from — also what drains those ids from the `distill` backlog |
| `--include-promoted` | `distill` | Show captures a live annotation already cites (hidden by default) |
| `--message-limit` | `distill` | Cap on source-layer turns (default `distill.message_limit`, 50) |
| `--brief` | `distill` | Token-lean markdown for a session hook; silent unless the backlog clears its gate |
| `--include-superseded` | `query --observations` | Return superseded / expired rows |
| `--tags` | `query --observations` | Comma-separated tag allowlist; keeps rows carrying **any** of them (array overlap). Filters in SQL, so `--limit` counts only matching rows. Refused without `--observations` — only annotations carry tags. |
| `--include-transcripts` | `query`, `context` | Add source-layer transcript matches to results (off by default) |
| `--include-domain` | `query`, `context`, `session start` | Comma-separated data-domain allowlist (`doc`, `file`, `chat`, `mail`). Domain = `kind` prefix before the first dot. No `code` domain. |
| `--exclude-domain` | `query`, `context`, `session start` | Comma-separated data-domain denylist. Mutually exclusive *per-domain* with `--include-domain`. |
| `--diagnose` | `status` | Full diagnostic checks including parser registry |

## Architecture Note

Two stability layers:

- **Layer 1 (stable contract):** `skills.md` installed via `hafiz agent install`. Ownership rule (parsers own indexed content, agents own meaning) is load-bearing. `hafiz agent install` detects version drift and warns when refreshing an older splice.
- **Layer 2 (evolving):** `hafiz review`, `hafiz parsers list`, Phase 7 observability surfaces. Free to iterate; not referenced from `skills.md`.
