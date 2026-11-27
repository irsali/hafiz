<!-- summary: Retrieval hygiene, recall hooks and telemetry -->
# hafiz guide retrieval — Retrieval hygiene, recall hooks and telemetry

## Retrieval hygiene (read before wiring hafiz into a hook)

Three flags, on both `query` and `context`:

| Flag | Why |
|---|---|
| `--min-score 0.05` | 0–1 relevance floor on the score results are *ranked* by |
| `--format compact` | content + kind + source + age; drops uuids, timestamps, nulls, scores |
| `--with-ids` | re-adds ids — **required** if you might `--supersedes` later |

`--limit` also exists on `context` (caps each section).

**When you need an exact set rather than the best matches, pin it by tag.**
`query --observations --tags <a,b>` keeps only rows carrying any of those tags
(filtered in SQL, so `--limit` counts matching rows). This is the escape hatch
for sets that curation defines and content does not — "the rules that always
apply" is not a semantic property, so no query text retrieves it reliably. For a
stable pin, pair `--tags` with `--no-rerank` and **no** `--min-score`; a floor
would re-introduce exactly the ranking judgement the pin exists to bypass.

Two things to get right, or the pin quietly stops being exact:

- **Raise `--limit` above the tagged set.** `--tags` chooses what's eligible, not
  what survives; the limit still cuts, with no signal (default 10, in reranked
  order — cosine order only under `--no-rerank`). A
  result count equal to your limit means "possibly truncated".
- **Pass a non-empty value.** `--tags ""` or an unset shell variable exits **2**
  rather than running an unpinned recall you'd mistake for a pinned one. Same
  for `--tags` without `--observations` (exit 1) — only annotations carry tags.

**`compact` and `md` give you an excerpt of a long annotation, not the whole
record.** Anything over `[annotations] snippet_chars` (default 480) is trimmed
to its best-matching span — measured at **−65%** on a `--observations --limit
20` recall. Short records are untouched. You can always tell which is which:
`compact` sets `excerpt: true` plus `full_chars`, `md` appends `excerpt of N
chars`, and both elide with `…`.

**When you see that marker and the excerpt doesn't settle the question, go get
the whole record** — `--format json` (or `rich`) never trims. That matters most
for decisions: an excerpt can land on the conclusion and miss the scope it only
applies under. Don't act on a partial decision record; re-read it in full.

**Which score the floor uses.** Annotation recall is cross-encoder reranked by
default, and `--min-score` filters the reranked score, not the raw cosine
`score`. That matters: under reranking `score` is *not* monotonic down the
result list, so a floor on it would drop a higher-ranked row and keep a
lower-ranked one.

One exception, marked so you can see it: the single closest vector match is
kept under the floor when its cosine is very high (`[recall] strong_match`,
default 0.70), because the reranker can underrate a plainly worded memory
against a chatty prompt about exactly it. `--json` flags that row
`strong_match: true`; at most one row per search carries it, after the rows
that cleared the floor and only if `--limit` leaves room.

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

**An unknown project is an error, not an empty result.** `--project` accepts
synonyms and case/spacing variants (`ledger-api` finds `Ledger API`); a name
that fits nothing, or several projects, exits `2` with `candidates` to retry
with. On writes, a new name that is a near-certain typo of an existing project
is refused unless `--new-project`. Omit `--project` on your own writes and the
current repo is used. `hafiz projects --json` lists the names on file.

**A blank query is an error, not an empty result.** `query` / `context` exit `2`
with `{"ok": false, "error": ...}` on blank input. If you build a query by
interpolating a variable, check the exit code — an unset variable is a bug in
your hook, and hafiz will no longer paper over it with confidently-scored noise.

**Harness messages are skipped, not searched.** A prompt that *is* a harness
message (`<task-notification>` — a sub-agent's final report) returns no results
and exits `0`; editor preambles at the front of a prompt (`<ide_opened_file>`,
`<ide_selection>`, `<system-reminder>`) are stripped and the rest is searched.
A recall hook can forward the harness prompt as-is.

**`--format md` on an empty result set prints nothing at all** — no placeholder
line. Under a floor, "no rows" is the ordinary answer to an off-topic prompt, so
a per-task hook can pipe the output straight into context without filtering it.

**Recall hooks ship with hafiz.** `hafiz agent install claude-code --hooks`
wires every-turn memory into Claude Code: `hafiz hook session` (SessionStart)
injects the pinned rules (tag `always`, rendered whole) plus a recall for the
repo and branch, and `hafiz hook prompt` (UserPromptSubmit) recalls what is
relevant to each prompt. Both cap their output (`[hooks] max_chars`, default
9000 — Claude Code truncates hook output over 10,000 characters), never fail a
turn, and end with a one-line count plus the command to list anything that did
not fit. You do not need to hand-write this for Claude Code. For any other
harness, the whole thing is one command; everything that used to need a
wrapper script — relevance floor, hit gate, compact rendering, dedup of
byte-identical rows — is in the CLI now:

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
indexed units — docs as `doc.heading` / `doc.paragraph`, and everything else
(including code) as whole-file `file.raw` units. Domains today are `doc`,
`file`, `chat`, `mail` — there is no `code` domain (`--include-domain code`
matches nothing).

- **Don't dismiss the unit index as "the code index".** Prose questions
  ("how does X decide which rules apply") land on the doc corpus, which is
  exactly what grep can't reach from that wording.
- **For the file you're actively editing, prefer reading the file** (and use
  git for history). The index is for semantic discovery across docs and
  files; check the `staleness` block on `query` output before trusting an
  indexed file that may trail the working tree.

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
