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

**baseline** is `dev` at `421ad3c` (before every-turn hooks and the keyword
search channel) plus this eval; `ad97fdc` adds only files under `eval/`, `docs/`
and `tests/`, so its recall code is `421ad3c`'s. CPU embeddings, SQLite backend,
two runs gave identical numbers and identical failures.

<!-- eval:results:start -->
| Measure | baseline (`ad97fdc`) | after-improvements (`8031aea`) |
|---|---|---|
| Exact-string recall hit@1 (n=50) | 96.0% | 96.0% |
| Exact-string recall hit@6 | 98.0% | 98.0% |
| Exact string visible in injected text | 98.0% | 98.0% |
| Decision recall hit@1 (n=45) | 77.8% | 77.8% |
| Decision recall hit@6 | 77.8% | 77.8% |
| Stale-belief leak rate (n=32) | 0.0% | 0.0% |
| Tokens injected, on-topic prompts, mean (n=95) | 62 | 62 |
| Tokens injected, on-topic, p95 | 117 | 117 |
| Off-topic prompts that inject nothing (n=20) | 95.0% | 95.0% |
| Tokens injected, off-topic prompts, mean | 3.0 | 3.0 |
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

**after-improvements** adds the keyword channel (migration `0012`) and the
every-turn hooks, and moves nothing — every number and every failure is
identical. That is a statement about this dataset, not about the channel: the
keyword index is built and used here (verified), but with 103 memories the
vector path already ranks almost every exact string first, so there is no
headroom for keyword search to win. On a 3,270-annotation real store, replaying
40 exact-string questions, the same channel took hit@6 from 16/40 to 40/40. The
decision misses are unchanged because their cause is the relevance floor, which
neither change touches. Two consequences: the next lever for this eval is the
floor on short decisions, and the dataset needs a harder exact-string set (more
memories, needles past the reranker's 400-character window) before it can show
keyword gains. The hook's earlier-conversation fallback is not exercised here at
all — the eval stores no transcripts.

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
