<!-- summary: Capture, distill, supersede, reconcile and sessions -->
# hafiz guide distill — Capture, distill, supersede, reconcile and sessions

## Capture → Distill Workflow

The reasoning loop:

1. **Capture raw.** Half-formed thoughts go in without ceremony:
   ```bash
   hafiz note "Wondering if refresh tokens should live in httponly cookies"
   ```

2. **Review.** `hafiz journal --since 7d` groups by day.

3. **Distill.** `hafiz distill --since 7d` groups the promotable backlog
   into **themes** — notes and transcript turns clustered together by
   similarity, so a note arrives alongside the turns it came from. Each
   theme prints the one `observe` that would drain it. Hafiz does **not**
   call an LLM — you are the distiller. Promote via:
   ```bash
   hafiz observe "<distilled decision>" --type decision --derived-from <note-id>,<note-id>
   ```

   **The backlog drains itself, and `--derived-from` is what drains it.**
   Citing a capture is the promotion receipt, so there is no separate
   bookkeeping step — and no way for the queue to keep re-offering work
   you already did. Cite **every** id in the theme, not just the one you
   found most useful; an uncited member stays in the queue forever. To
   reject a capture instead of promoting it, retire it:

   ```bash
   hafiz forget <note-id> --annotation      # declined, not promoted
   hafiz distill --include-promoted --json  # audit what's already drained
   ```

   **Sleep-time distillation.** `hafiz distill --brief` renders the
   backlog as token-lean markdown and prints **nothing at all** unless it
   is big enough or old enough to be worth interrupting a session for.
   That makes it safe in a session-start hook:

   ```bash
   # Never let it fail a turn: bound it, swallow errors, exit 0.
   timeout 10s hafiz distill --since 30d --brief 2>/dev/null || true
   ```

   When it does fire, you are the sleep-time agent: read the themes,
   write the observations. Hafiz assembles and surfaces the queue; it
   never synthesizes a belief on its own. Gate and clustering tunables
   live under `[distill]` (`brief_min_pending`, `brief_min_age_days`,
   `cluster_threshold`).

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
   member and retires the rest, and `suggested_action` is one of three tiers:

   - `retire` — every word of the siblings also appears in the primary, in
     order. No *wording* is lost. Still not proof the primary means the same:
     a subsequence can invert a claim ("consent is not required" sits inside
     "consent is not optional and consent is required"). Cheapest to check,
     not safe to skip checking.
   - `review` — words differ but none form a run of 4+. Read them: usually a
     contraction or spelling, occasionally a reversal.
   - `merge` — a sibling carries whole claims the primary lacks. Write text
     covering `unique_fragments`, then retire.

   `commands` is the ordered resolution, ready to run; `review` gets the
   *merge* commands on purpose, because merging mere rewording is cheap and
   retiring real text is not. It sweeps the **whole store** by default; if
   you pass `--limit` and it bites, the response says `truncated: true`.

   **Cosine groups the cluster; text decides the action — read the text
   fields, not the score.** Each non-primary member carries `overlap`,
   `unique_words` and `unique_fragments`. At the default `0.88` threshold
   members typically share under a tenth of their words: two facts about one
   subject, not one fact twice. Judge safety on `unique_words == 0`, never on
   `overlap` being high — `overlap` is `2·matches/total`, so a row fully
   contained in a keeper four times its length scores ~0.40 while losing
   nothing (`2M/T`: 2x -> 0.67, 3x -> 0.50, 4x -> 0.40). And note the tiers grade *evidence*, not *meaning*: a total
   meaning change carried by one short word (`stage` vs `prod`) lands in
   `review`. Only `retire` makes a promise.

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

5. **Write one claim per record.** `kind` is the only schema an annotation
   has, so there is nothing stopping you from packing a decision, its
   rationale, its scope and its rejected alternatives into one blob — and
   that is exactly what agents do. On a real store the mean annotation grew
   from 746 to 1,127 characters in eight weeks.

   It costs you three ways, all at recall time:

   - **You get the whole record back.** There is no snippet extraction, so a
     2,700-char blob spends ~700 tokens of your context to deliver the two
     sentences that matched.
   - **It retrieves worse.** One embedding covers the entire record, so a blob
     spanning four topics is found by the *average* of four topics and matches
     none of them sharply.
   - **Duplicate detection goes blind.** Similarity is computed blob-to-blob,
     so two records that state the same decision with different rationale
     don't look alike, and near-duplicates accumulate unseen.

   So: one claim per `observe`. Put the rejected alternative in its own record
   and link it with `--derived-from <id>`; the same for a scope caveat worth
   recalling on its own. Above `[annotations] max_recommended_chars` (default
   1,500) `observe` says so — a hint in human output, `oversized: {chars,
   limit, hint}` in `--json`. **It is advisory: the write always succeeds**,
   exit `0`, and `oversized` is `null` the rest of the time. Never fires on
   `note`.

   Long records are sometimes right. Splitting a genuinely single claim into
   fragments that each recall badly is worse than one long record, so treat
   the warning as a prompt to check, not an instruction to chop.

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
