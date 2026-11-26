#!/usr/bin/env python
"""Recall eval runner. See docs/eval.md.

    python eval/run.py                       # run, write eval/results/<commit>.{json,md}
    python eval/run.py --label baseline --update-docs

What it does, in order:

1. Builds a throwaway sandbox: a temp SQLite store, a temp HOME / XDG tree, an
   empty cwd, and a scrubbed environment (no ``HAFIZ_*`` leaks in, the daemon
   off). Your real memory store is never opened.
2. Proves the isolation took (``hafiz config get database.url`` must report
   ``source: env`` and our temp file) and aborts if it did not.
3. ``hafiz init``, then loads ``eval/data/memories.jsonl`` through the public
   CLI (``hafiz observe ... --supersedes``).
4. Asks every question in ``eval/data/questions.jsonl`` exactly as the shipped
   per-prompt recall hook does, scores it, and writes a JSON + markdown report.

The ``hafiz`` it drives is the one in *this checkout* (PYTHONPATH is pinned to
the repo root), run with whatever interpreter you launch this script with — it
must have hafiz's dependencies installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # `python eval/run.py` puts eval/ first, not the repo root
    sys.path.insert(0, str(REPO))

from eval import scoring  # noqa: E402

DATA = REPO / "eval" / "data"
RESULTS = REPO / "eval" / "results"
DOCS = REPO / "docs" / "eval.md"
DOCS_START = "<!-- eval:results:start -->"
DOCS_END = "<!-- eval:results:end -->"

#: What the shipped per-prompt recall hook passes after the query text.
HOOK_ARGS = ["--observations", "--min-score", "0.05", "--limit", str(scoring.K)]

#: The module has no ``__main__``; this is what the ``hafiz`` console script does.
_BOOT = "import sys;from hafiz.cli import main;sys.argv[0]='hafiz';main()"

SOURCE = "agent:eval"
TIMEOUT_S = 600


class EvalError(RuntimeError):
    """Something that makes the run untrustworthy; abort rather than continue."""


# ── Sandbox ─────────────────────────────────────────────────────────────


class Sandbox:
    """A hermetic place to run hafiz: own store, own HOME, scrubbed env."""

    def __init__(self, python: str, keep: bool) -> None:
        self.python = python
        self.keep = keep
        self.root = Path(tempfile.mkdtemp(prefix="hafiz-eval-"))
        self.db = self.root / "eval.db"
        self.cwd = self.root / "work"
        self.cwd.mkdir()
        self.url = f"sqlite:///{self.db}"
        self.env = self._build_env()

    def _build_env(self) -> dict[str, str]:
        home = self.root / "home"
        xdg_cache = self.root / "xdg-cache"
        (xdg_cache / "hafiz").mkdir(parents=True)
        home.mkdir()
        # Share the (read-only in practice) embedding-model cache so a run does not
        # re-download ~500 MB. It holds models only — no user data.
        real_base = os.environ.get("XDG_CACHE_HOME")
        real_cache = (Path(real_base) if real_base else Path.home() / ".cache") / "hafiz" / "models"
        real_cache.mkdir(parents=True, exist_ok=True)
        (xdg_cache / "hafiz" / "models").symlink_to(real_cache, target_is_directory=True)

        env = {k: v for k, v in os.environ.items() if not k.startswith("HAFIZ_")}
        env.update(
            HOME=str(home),
            XDG_CACHE_HOME=str(xdg_cache),
            XDG_CONFIG_HOME=str(self.root / "xdg-config"),
            XDG_DATA_HOME=str(self.root / "xdg-data"),
            PYTHONPATH=str(REPO),
            PYTHONHASHSEED="0",
            HAFIZ_NO_DAEMON="1",
            HAFIZ_DATABASE__URL=self.url,
            # CPU embeddings: identical on every machine, so numbers reproduce.
            HAFIZ_EMBEDDING__DEVICE="cpu",
        )
        return env

    def hafiz(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            [self.python, "-c", _BOOT, *args],
            cwd=self.cwd,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        if check and proc.returncode != 0:
            raise EvalError(
                f"`hafiz {' '.join(args)[:80]}` exited {proc.returncode}\n"
                f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
            )
        return proc

    def cleanup(self) -> None:
        if self.keep:
            print(f"(kept sandbox at {self.root})")
        else:
            shutil.rmtree(self.root, ignore_errors=True)


def verify_isolation(sb: Sandbox) -> dict[str, str]:
    """Abort unless hafiz demonstrably points at the sandbox store and checkout."""
    probe = subprocess.run(
        [sb.python, "-c", "import hafiz;print(hafiz.__file__)"],
        cwd=sb.cwd,
        env=sb.env,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise EvalError(
            f"{sb.python} cannot import hafiz (are its dependencies installed?):\n{probe.stderr}"
        )
    hafiz_file = Path(probe.stdout.strip()).resolve()
    if REPO not in hafiz_file.parents:
        raise EvalError(f"imported hafiz from {hafiz_file}, not this checkout ({REPO})")

    out = sb.hafiz("config", "get", "database.url", "--json").stdout
    cfg = json.loads(out)
    expected = f"sqlite:///{sb.db}"
    if cfg.get("source") != "env" or cfg.get("value") != expected:
        raise EvalError(
            "isolation check FAILED — refusing to write anything.\n"
            f"  expected source=env value={expected}\n  got      {out.strip()}"
        )
    if (sb.cwd / "hafiz.toml").exists():
        raise EvalError("sandbox cwd unexpectedly contains a hafiz.toml")
    return {"hafiz_file": str(hafiz_file), "database_url": cfg["value"], "source": cfg["source"]}


# ── Phases ──────────────────────────────────────────────────────────────


def load_memories(sb: Sandbox, memories: list[dict[str, Any]]) -> dict[str, str]:
    """Write every memory through ``hafiz observe``; return dataset id → real id."""
    id_map: dict[str, str] = {}
    for i, m in enumerate(memories, 1):
        args = ["observe", m["content"], "--type", m["kind"], "--source", SOURCE, "--json"]
        if m.get("supersedes"):
            args += ["--supersedes", id_map[m["supersedes"]]]
        payload = json.loads(sb.hafiz(*args).stdout)
        id_map[m["id"]] = payload["annotation"]["id"]
        if i % 20 == 0 or i == len(memories):
            print(f"  loaded {i}/{len(memories)} memories")
    return id_map


def ask(sb: Sandbox, query: str) -> tuple[list[str], str]:
    """One recall, exactly as the hook runs it: markdown to inject, json to score."""
    md = sb.hafiz("query", query, *HOOK_ARGS, "--format", "md").stdout
    js = sb.hafiz("query", query, *HOOK_ARGS, "--format", "json").stdout
    return scoring.returned_ids(json.loads(js)), md


def run_questions(
    sb: Sandbox, questions: list[dict[str, Any]], id_map: dict[str, str], jobs: int
) -> list[dict[str, Any]]:
    done = 0

    def one(q: dict[str, Any]) -> dict[str, Any]:
        nonlocal done
        returned, md = ask(sb, q["query"])
        done += 1
        if done % 20 == 0 or done == len(questions):
            print(f"  asked {done}/{len(questions)} questions")
        return scoring.score_question(q, returned, md, id_map)

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(one, questions))


# ── Reporting ───────────────────────────────────────────────────────────


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_meta(args: argparse.Namespace, sb: Sandbox, started: float) -> dict[str, Any]:
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    version = sb.hafiz("--version", check=False).stdout.strip()
    return {
        "label": args.label or commit,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--", ".", ":!eval/results")),
        "hafiz_version": version,
        "python": sys.version.split()[0],
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_s": round(time.monotonic() - started, 1),
        "backend": "sqlite",
        "embedding_device": "cpu",
        "query_args": HOOK_ARGS,
        "token_estimate": "ceil(chars / 4)",
        "dataset": {
            "memories_sha256": _sha256(DATA / "memories.jsonl"),
            "questions_sha256": _sha256(DATA / "questions.jsonl"),
        },
    }


def update_docs(results_dir: Path, docs: Path) -> None:
    """Regenerate the comparison table inside docs/eval.md from every saved run."""
    runs = [json.loads(p.read_text()) for p in results_dir.glob("*.json")]
    runs.sort(key=lambda r: r["meta"]["finished_at"])
    text = docs.read_text()
    if DOCS_START not in text or DOCS_END not in text:
        raise EvalError(f"{docs} has no {DOCS_START} … {DOCS_END} block to update")
    head, rest = text.split(DOCS_START, 1)
    _, tail = rest.split(DOCS_END, 1)
    table = scoring.render_comparison(runs)
    docs.write_text(f"{head}{DOCS_START}\n{table}{DOCS_END}{tail}")
    print(f"updated {docs.relative_to(REPO)} ({len(runs)} run(s))")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the hafiz recall eval in an isolated sandbox.")
    ap.add_argument("--label", help="name for this run (default: short commit hash)")
    ap.add_argument("--python", default=sys.executable, help="interpreter that has hafiz installed")
    ap.add_argument("--jobs", type=int, default=2, help="parallel queries (default 2)")
    ap.add_argument("--update-docs", action="store_true", help="refresh the table in docs/eval.md")
    ap.add_argument("--no-write", action="store_true", help="do not write eval/results/ files")
    ap.add_argument("--keep", action="store_true", help="keep the temp sandbox for inspection")
    ap.add_argument("--quiet", action="store_true", help="do not list failing questions")
    args = ap.parse_args(argv)

    memories = scoring.load_jsonl(DATA / "memories.jsonl")
    questions = scoring.load_jsonl(DATA / "questions.jsonl")
    problems = scoring.validate_dataset(memories, questions)
    if problems:
        print("dataset is invalid:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 2

    started = time.monotonic()
    sb = Sandbox(args.python, args.keep)
    try:
        print(f"sandbox: {sb.root}")
        iso = verify_isolation(sb)
        print(f"isolation ok: database.url source={iso['source']}, hafiz from {iso['hafiz_file']}")
        sb.hafiz("init")
        print(f"loading {len(memories)} memories …")
        id_map = load_memories(sb, memories)
        print(f"asking {len(questions)} questions …")
        rows = run_questions(sb, questions, id_map, max(1, args.jobs))
        meta = build_meta(args, sb, started)
    except EvalError as exc:
        print(f"eval aborted: {exc}", file=sys.stderr)
        return 1
    finally:
        sb.cleanup()

    result = {
        "meta": meta,
        "aggregate": scoring.aggregate(rows),
        "questions": [{k: v for k, v in r.items() if k != "had_forbid"} for r in rows],
    }
    reverse = {real: ds for ds, real in id_map.items()}
    for r in result["questions"]:
        r["returned"] = [reverse.get(i, i) for i in r["returned"]]
        r["leaked"] = [reverse.get(i, i) for i in r["leaked"]]

    print()
    print(scoring.render_summary(result))
    if not args.quiet:
        qmap = {q["id"]: q for q in questions}
        bad = scoring.failures(rows, qmap, reverse)
        print(f"{len(bad)} question(s) not fully correct:")
        for line in bad:
            print("  " + line)

    if not args.no_write:
        RESULTS.mkdir(exist_ok=True)
        stem = RESULTS / _safe(meta["label"])
        stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        stem.with_suffix(".md").write_text(scoring.render_summary(result))
        print(f"\nwrote {stem.relative_to(REPO)}.json and .md  ({meta['elapsed_s']}s)")
        if args.update_docs:
            update_docs(RESULTS, DOCS)
    return 0


def _safe(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in label)


if __name__ == "__main__":
    sys.exit(main())
