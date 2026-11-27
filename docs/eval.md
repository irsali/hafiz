# Recall eval

A small eval of what a coding agent actually feels from Hafiz: *does the right
memory come back, does a retired decision stay buried, and how much does it cost
in context window?* Anyone can rerun it on a laptop in under half an hour (a run
is ~560 short hafiz processes; it took about 15 minutes on an idle 20-core box).

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
| [`eval/data/memories.jsonl`](../eval/data/memories.jsonl) | 177 memories: 50 facts, 24 warnings, 19 learnings, 13 patterns, 71 decisions. 25 supersede chains (two of three links), so 27 memories are retired and must not come back. 103 are short one-liners (ids `f01`…`s08`); 74 are long, multi-topic records of 720–1,800 characters (ids `L01`…`L74`). |
| [`eval/data/questions.jsonl`](../eval/data/questions.jsonl) | 160 questions (90 exact-string, 70 decision) and 30 off-topic prompts. 46 of the decision questions are about a superseded decision and name the retired memory. |

Deliberate difficulty: near-distractors share vocabulary (`MANIFESTD_RETRY_LIMIT`
vs `SCANLINK_RETRY_LIMIT`, `E4012` vs `E4021`, `PW-1042` vs `PW-1024`, port 8443
as a fact and as a warning), questions are phrased the way a person asks rather
than echoing the stored wording, and some off-topic prompts share words with the
memories (`what is the time zone of Tokyo`, `what is the difference between a
process and a thread`).

The long set (`L01`…`L74`, questions `ex51`–`ex90`, `dc46`–`dc70`, `off21`–`off30`)
is there because the short set is too easy. Real memories are written by agents:
several topics in one record, dated, with rationale, file paths and commands, and
the string you later ask for sits somewhere in the middle. So here the memories
average about 1,000 characters, 34 of the 40 new exact-string questions target a
string that appears after character 400 (past the 400 characters the reranker
reads, and diluted in the single embedding), and the other six are kept near the
start as a control. Near-distractors are everywhere: `SCANLINK_BATCH_FLUSH_MS` vs
`SCANLINK_BATCH_FLUSH_BYTES`, `SL-317` vs `SL-371`, `MF_SEAL_409` vs
`MF_SEAL_490`, `BC-88` vs `BC-808`, `E5103` vs `E5130`, ports `9150` / `9105` and
`7401` / `7410`, `8713` vs `8731`, tickets `PW-2412` vs `PW-2421` and `OPS-518` vs
`OPS-581`, two migrations numbered `0047`. Nine more supersede chains (one of three
links) keep the stale-belief measure honest on long text. The new decision
questions are written the way people actually type: 130–220 characters, chatty,
with a side remark or a half-remembered detail ("I think we changed it after the
battery complaints?"), about both new and old decisions. The ten new off-topic
prompts are equally long and share words with the store (parcels, depots, a
gateway router, ledgers, gRPC, a mutex "at a shipping depot") while asking
nothing it answers.

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

All three runs use the current dataset, CPU embeddings and the SQLite backend;
each is the recall code of the commit named, with this checkout's `eval/`:

- **before-keyword** — `ad97fdc`, recall as it was before every-turn hooks and
  the keyword channel (`ad97fdc` adds only the first eval on top of `421ad3c`).
- **keyword-channel** — adds the keyword (full-text) search beside the vector
  search (migration `0012`) and the every-turn hooks.
- **strong-match** — adds the strong-match safety net under the floor
  (`[recall] strong_match = 0.70`). Run at `8016a2e`; the follow-up `ba0bbca`
  only changes prompts over 300 characters, which this dataset does not have.

Runs on the first, short-only dataset are kept in
[`eval/results/v1/`](../eval/results/v1/) and are not comparable with these.

<!-- eval:results:start -->
| Measure | before-keyword (`ad97fdc`) | keyword-channel (`52849e4`) | strong-match (`8016a2e`) |
|---|---|---|---|
| Exact-string recall hit@1 (n=90) | 74.4% | 84.4% | 84.4% |
| Exact-string recall hit@6 | 78.9% | 91.1% | 91.1% |
| Exact string visible in injected text | 76.7% | 88.9% | 88.9% |
| Decision recall hit@1 (n=70) | 77.1% | 75.7% | 77.1% |
| Decision recall hit@6 | 80.0% | 80.0% | 81.4% |
| Stale-belief leak rate (n=46) | 0.0% | 0.0% | 0.0% |
| Tokens injected, on-topic prompts, mean (n=160) | 140 | 167 | 169 |
| Tokens injected, on-topic, p95 | 396 | 428 | 428 |
| Off-topic prompts that inject nothing (n=30) | 96.7% | 96.7% | 96.7% |
| Tokens injected, off-topic prompts, mean | 2.0 | 2.0 | 2.0 |
<!-- eval:results:end -->

## How to read it

- **Keyword search earns its place on long memories.** With the needle buried
  past the 400 characters the reranker reads, before-keyword found the right
  memory in the top 6 for 71 of 90 exact-string questions; the keyword channel
  finds 82 (78.9% → 91.1%), and the exact string reaches the injected text for
  80 instead of 69. Cost: three questions slip from rank 1 to a lower rank in
  the top 6 (`ex83`, `ex87`, and one decision, `dc29`), and an on-topic prompt
  injects ~27 more tokens on average, because it now brings back long memories
  it used to miss. The first dataset could not show any of this (identical
  numbers before and after): short memories leave the vector search nothing to
  miss.
- **Decisions are still the weak spot, and the cause is still the relevance
  floor.** Nine decision questions inject nothing at all — a silent miss, worse
  than a wrong answer because the agent does not know it should have been told
  something. In the five examined closely (`dc10`, `dc11`, `dc24`, `dc28`,
  `dc42`) the right memory was the top vector match (cosine 0.55–0.65), but the
  cross-encoder scored the short declarative decision near zero against the
  question and `--min-score 0.05` cut it.
- **The strong-match safety net helps less here than on a real store.** It keeps
  the closest vector match under the floor when its cosine is at least 0.70. On
  this synthetic store it recovers one question (`dc60`), because the silent
  misses examined sit at cosine 0.55–0.65: one-line invented memories embed less
  closely to a question than real, dense notes do. On a real 3,270-annotation
  store, replaying 515 real prompts that recalled nothing, it recovered 38
  (about 6 in 10 on topic) and kept all 40 off-topic prompts silent — see
  `[recall] strong_match` in [commands.md](commands.md). Lowering the bar to
  reach 0.55–0.65 is not safe: real off-topic prompts reach 0.63.
- **The stale-belief leak is 0% — by construction.** Superseded memories are
  hidden from default recall, and the keyword channel applies the same filter,
  so a retired decision cannot come back through either path. The number is a
  regression guard rather than a score.
- **Off-topic prompts stay quiet.** 29 of 30 inject nothing, including all ten
  long chatty ones that share the store's vocabulary; the exception is still
  `a process and a thread`, which pulls in the "solver is not thread-safe"
  warning on a shared word.
- **Not exercised here:** the hook's earlier-conversation fallback (the eval
  stores no transcripts).

## Limits — please read before quoting a number

- **Small.** 160 scored questions (90 exact-string, 70 decision). One question
  is one to two points of a percentage; differences of a few points are
  noise-level, and a single run has no confidence interval.
- **Synthetic and self-authored.** The same people who build Hafiz wrote both
  the memories and the questions, which tends to flatter the system. Real
  prompts are messier, longer and mention things that were never saved.
- **One topic area.** A single fictional codebase in English. Nothing here tests
  multilingual text, memories much longer than ~1,800 characters, thousands of
  memories, or recall across many projects.
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
