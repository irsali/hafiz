<!-- Installed by hafiz — workspace intelligence layer -->
<!-- SKILLS_VERSION: 22 -->
# Hafiz — Workspace Intelligence (v22)

IMPORTANT: You have access to `hafiz`, a CLI tool that is the user's
**sovereign second brain**. It indexes docs and files for semantic search
and stores agent-authored meaning (decisions, warnings, learnings) as
annotations. Use `--json` when parsing output programmatically.

Replace `<your-name>` below with the agent you are (`claude-code`,
`cursor`, `copilot`, `aider`, …). Never leave the literal string in a command.

**This is the short version.** Details live behind `hafiz guide <topic>`;
read a topic when the task needs it, not up front:

| Topic | Read it when |
|---|---|
| `retrieval` | wiring recall into a hook; `--tags` pinning; excerpts; score floors |
| `distill` | promoting notes, superseding/retiring rows, `reconcile`, sessions |
| `transcripts` | importing, recalling or forgetting agent transcripts; retention |
| `model` | how knowledge vs. source layers and unit kinds fit together |
| `mcp` | you cannot run a shell and talk to hafiz over MCP |
| `ops` | ingest, `hafiz errors`, tuning the install for this machine |
| `reference` | the full command and flag table |

`hafiz guide` with no topic lists them.

## Required Behaviors

You MUST follow these rules in every session:

1. **When a task needs prior docs or decisions** and no recall hook already
   injected them, pull a context bundle:
   ```bash
   hafiz context "<task>" --limit 5 --min-score 0.05 --format compact --with-ids
   ```

2. **After an architectural decision**, record it:
   `hafiz observe "<what was decided and why>" --type decision --source agent:<your-name>`

3. **After discovering a gotcha or non-obvious behavior**, record it:
   `hafiz observe "<the gotcha>" --type warning --source agent:<your-name>`

4. **Before implementing a significant change, record the decision driving
   it** — including the alternative you rejected. If there is a plan-approval
   step, fire it right after approval and before the first edit. Observe a
   mid-task pivot too; if it replaces an earlier decision, add
   `--supersedes <old-id>`. Mechanical edits the user explicitly scoped
   ("rename X to Y", "apply this diff") are exempt.

5. **Route "remember" writes to hafiz, never to an agent-local memory store.**
   When the user says "remember X", "save this", "don't forget":
   ```bash
   hafiz observe "<what to remember>" --type <kind> --source user:<name>
   ```
   Kind from content: preference/rule → `learning`; claim → `fact`;
   architectural choice → `decision`; gotcha → `warning`. A rule that must
   apply in **every** session whatever the topic gets `--tags always` (add
   `--project <name>` to scope it). Pin sparingly — each pin costs context in
   every session. Harness-only preferences (slash commands, hook config, IDE
   settings) may stay in the agent's own store.

   At session start, if no hook already did it, pull durable preferences:
   ```bash
   hafiz query "<what you're about to work on>" --observations \
     --source user:<name> --limit 12 --min-score 0.05 --format compact --with-ids
   ```

6. **Never inject an unbounded, unfiltered result set.** Anything you put in
   your own context gets `--min-score 0.05 --format compact` (plus
   `--with-ids` if you might supersede). Unfiltered, one query is ~17k tokens;
   filtered, ~600.

7. **Never overwrite knowledge — supersede or retire it.**

   | Situation | Action |
   |---|---|
   | New info replaces an existing row | `hafiz observe "<new>" --type <kind> --supersedes <old-id>` |
   | Existing row is simply wrong | `hafiz forget <old-id> --annotation` |
   | Net-new, no conflict | plain `hafiz observe` |

   Verify the old id first. Supersede only on genuine contradiction, not a
   reword. When `observe --json` returns `near_duplicates`, decide whether to
   supersede or keep both. Never use `forget --hard` on annotations.

8. **Write one claim per record.** Put a rejected alternative or a scope caveat
   in its own record linked with `--derived-from <id>`.

9. **Tell the user immediately** when `hafiz status --json` shows either:
   - `capture.<agent>.pending_on_disk` > 0 — transcripts not yet captured are
     lost for good when the harness rotates them. Fix:
     `hafiz import <agent>` then `hafiz agent install <agent> --hooks`.
   - `retention.overdue` > 0 — a stated retention guarantee isn't being met.
     Fix: `hafiz forget --all-expired`.

## Reading results correctly

- **Excerpts.** `compact`/`md` trim long annotations (`excerpt: true` /
  `excerpt of N chars`). Don't act on a partial *decision* — re-read it with
  `--format json`.
- **Exit 2 is a bug, not "no results".** Blank query, unknown `--project`
  (retry with the returned `candidates`), or `--tags ""` all exit 2.
- **No rows under `--min-score 0.05` means "not relevant"**, not an error. One
  very close match may still come back below the floor, flagged `strong_match: true`.
- **Exact sets come from tags, not similarity:** `query --observations --tags
  <a,b> --no-rerank --limit <above set size>`, with no `--min-score`.
- **Code and docs are one corpus.** Plain `query` searches indexed docs and
  files (there is no `code` domain); `query --observations` searches the
  wisdom layer. For a file you're editing, read the file instead.
- Transcripts are hidden by default; opt in with `--include-transcripts` or
  `hafiz recall <session>`.

## Core Commands

| Command | Use |
|---|---|
| `hafiz context "<task>"` | Docs + files + matching observations for a task |
| `hafiz query "<text>"` | Semantic search over indexed docs and files |
| `hafiz query "<topic>" --observations [--type decision]` | Search decisions, warnings, learnings, facts |
| `hafiz observe "<text>" --type <kind> --source agent:<your-name>` | Record a decision / warning / learning / fact / pattern |
| `hafiz note "<text>"` | Raw thought, below decision-grade |
| `hafiz journal --since 7d` | What was recorded recently |
| `hafiz distill --since 7d` | Notes waiting to be promoted (see `hafiz guide distill`) |
| `hafiz recall <session>` | One imported transcript's turns, in order |
| `hafiz status --json` | Health, freshness, retention and capture |
| `hafiz guide [topic]` | The details this file leaves out |

Annotation kinds: `fact` · `decision` · `learning` · `pattern` · `warning` ·
`note` · `concept` · `service`.

<!-- /Installed by hafiz — do not edit above this block; re-run `hafiz agent install` to update -->
