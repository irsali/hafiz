# Recall eval

A small eval of what a coding agent actually feels from Hafiz: *does the right
memory come back, does a retired decision stay buried, and how much does it cost
in context window?* Anyone can rerun it on a laptop in about ten minutes.

It measures **Hafiz only**. Vendor benchmarks for agent memory (LongMemEval,
LoCoMo, BEAM) are mostly self-run, use different setups and are not comparable
with each other, so this page does not pretend to rank Hafiz against anything.
It exists so that a change to retrieval can be shown to help, or to hurt, with
numbers anyone can reproduce.

## What is measured

All three measures run against one fixed, synthetic memory store and use the
retrieval the per-prompt recall hook uses:

```bash
hafiz query "<prompt>" --observations --min-score 0.05 --limit 6 --format md    # what gets injected
hafiz query "<prompt>" --observations --min-score 0.05 --limit 6 --format json  # what gets scored
```

| # | Measure | Question it answers | How it is scored |
|---|---|---|---|
| 1 | **Exact-string recall** | When I ask about a port, env var, error code, file path, CLI flag or ticket id, does the memory holding it come back? | `hit@1` and `hit@6` by expected memory id; plus whether the exact string is visible in the text that would be injected. |
| 2 | **Decision recall** | Do I get the *current* decision, and never the one it replaced? | `hit@1` / `hit@6` on the current memory. **Stale-belief leak rate**: share of questions about a superseded decision whose top 6 contains a superseded memory. |
| 3 | **Tokens injected per turn** | What does recall cost me on every prompt, and does it stay quiet when the prompt has nothing to do with memory? | Size of the injected markdown (`ceil(chars / 4)` — an estimate, not a tokenizer). **Off-topic silence rate**: share of unrelated prompts that inject nothing. |

`hit@k` means *an expected memory is among the first k results*. Where several
memories would be a fair answer the question lists them all.

## The data

Everything is invented. The "team" is a fictional parcel-routing company with
made-up services (`gatewayd`, `scanlink`, `manifestd`, `routekit`, `beacon`,
`ledgerbox`), ports, error codes and tickets. No real project, person, prompt or
customer data is in it, so it is safe to publish.

| File | Contents |
|---|---|
| [`eval/data/memories.jsonl`](../eval/data/memories.jsonl) | 103 memories: 32 facts, 12 warnings, 10 learnings, 8 patterns, 41 decisions. 16 supersede chains (one of three links), so 17 memories are retired and must not come back. |
| [`eval/data/questions.jsonl`](../eval/data/questions.jsonl) | 95 questions (50 exact-string, 45 decision) and 20 off-topic prompts. |

Deliberate difficulty: near-distractors share vocabulary (`MANIFESTD_RETRY_LIMIT`
vs `SCANLINK_RETRY_LIMIT`, `E4012` vs `E4021`, `PW-1042` vs `PW-1024`, port 8443
as a fact and as a warning), questions are phrased the way a person asks rather
than echoing the stored wording, and some off-topic prompts share words with the
memories (`what is the time zone of Tokyo`, `what is the difference between a
process and a thread`).

Both files are plain JSONL with stable ids, so a change to the dataset shows up
as a readable diff and changes the `dataset` hashes recorded with each run.

## Running it

```bash
python eval/run.py                              # writes eval/results/<commit>.json and .md
python eval/run.py --label after-keyword --update-docs   # name the run, refresh the table below
```

Use an interpreter that has Hafiz's dependencies installed (a venv with
`pip install -e ".[dev]"`, or the pipx venv's `python`). There is no Makefile in
this repo; that one command is the whole interface. Useful flags: `--jobs N`
(parallel queries, default 2), `--keep` (leave the sandbox for inspection),
`--no-write`, `--quiet`.

The run is sandboxed so it can never touch your real memory:

- a fresh temp SQLite store (`HAFIZ_DATABASE__URL=sqlite:///<tmp>/eval.db`), a temp
  `HOME` / `XDG_*` tree and an empty working directory — so no `hafiz.toml`,
  session cursor or error log of yours is read or written;
- every inherited `HAFIZ_*` variable is dropped, and `HAFIZ_NO_DAEMON=1` keeps a
  running daemon out of it;
- before writing anything the runner asks `hafiz config get database.url --json`
  and **aborts** unless it reports `"source": "env"` and the temp file;
- it runs the `hafiz` package from *this checkout* (it pins `PYTHONPATH` and
  checks which file was imported), not whatever is installed elsewhere;
- embeddings run on CPU so results do not depend on your GPU. The only thing
  shared with your machine is the downloaded embedding-model cache, symlinked
  read-only in effect so the run does not re-download ~500 MB.

Memories are loaded through the public CLI (`hafiz observe … --supersedes <id>`),
in file order, and the runner maps dataset ids to the ids Hafiz returns. A run
is deterministic: the same checkout and dataset give the same numbers.

## Results

Each run is saved in [`eval/results/`](../eval/results/) (full per-question
detail in the `.json`). The table below is regenerated from those files by
`python eval/run.py --update-docs`; do not edit it by hand.

<!-- eval:results:start -->
<!-- eval:results:end -->

## How to read it

Findings from the first (baseline) run, which any later run should be read
against:

- **Exact strings are the strong suit.** Almost every port, env var, error code
  and ticket id comes back at rank 1, including next to near-identical
  distractors. The misses are a retrieval-order slip (`E4021` ranked behind a
  learning *about* `E4021`) and one outright absence (the test-database env var).
- **Decision recall is the weak spot, and the cause is the relevance floor, not
  the embedding.** Every decision miss is also a `hit@6` miss, and most of them
  inject *nothing at all*. In debugging, the right memory was the top vector
  match (cosine 0.55–0.63), but the cross-encoder reranker scored short,
  declarative decisions near zero (0.0005–0.05), and `--min-score 0.05` cut them.
  A silent miss is worse than a wrong answer, because the agent does not know it
  should have been told something.
- **The stale-belief leak is 0% — by construction.** Superseded memories are
  hidden from default recall, so a retired decision cannot come back through the
  vector path. The number is a regression guard rather than a score: a new
  retrieval channel (for example keyword search) that forgets to filter retired
  memories will show up here immediately.
- **Cost is small and mostly quiet.** An on-topic prompt injects about 60 tokens
  on average (p95 about 120) — but that average *includes* the silent misses
  above, so it falls as recall improves. All off-topic prompts but one inject
  nothing; the exception (`a process and a thread`) pulled in the unrelated
  "solver is not thread-safe" warning on a shared word.

## Limits — please read before quoting a number

- **Small.** 95 scored questions. One question is about one point of a
  percentage; differences of a few points are noise-level, and a single run has
  no confidence interval.
- **Synthetic and self-authored.** The same people who build Hafiz wrote both
  the memories and the questions, which tends to flatter the system. Real
  prompts are messier, longer and mention things that were never saved.
- **One topic area.** A single fictional codebase in English. Nothing here tests
  multilingual text, very long memories, thousands of memories, or recall across
  many projects.
- **Retrieval only.** It does not check whether an agent *uses* what was
  injected, or whether the right memory was ever written in the first place —
  that is usually the bigger problem.
- **Token counts are an estimate** (`chars / 4`) of the recall text alone, not of
  any hook framing around it.
- **Hafiz only.** No competitor is run, so there is no claim about relative
  quality.
- **CPU embeddings.** GPU runs may differ in the last decimal of a score, which
  can in principle reorder a near-tie.

## Changing the eval

Add memories or questions to the JSONL files (the runner validates them: unique
ids, supersede targets defined earlier, expected ids exist, exact strings really
appear in the expected memory, off-topic prompts at least 12 characters because
the hook ignores shorter ones). Re-run, and commit the new
`eval/results/*.json` together with the dataset change. Old runs keep their
dataset hashes, so a table column made on an older dataset is visibly
not-quite-comparable.
