"""Hafiz memory plugin — MemoryProvider interface.

Hafiz is a sovereign, local CLI-first intelligence layer backed by
PostgreSQL + pgvector. Unlike API-backed providers (mem0, honcho), this
provider shells out to the ``hafiz`` CLI — no network, no API key, all
data stays on the user's machine.

Two storage layers, treated differently per Hafiz's design:
  * KNOWLEDGE layer (curated, surfaced by default): code, decisions,
    learnings, warnings. We READ this for prefetch/recall and WRITE to
    it only on an EXPLICIT hafiz_remember tool call.
  * SOURCE layer (firehose, hidden by default, retention-bounded): we
    write the full conversation here at session end via ``hafiz capture``.

Design decisions:
  * Auto-write policy = explicit writes + session transcripts. sync_turn
    is a deliberate NO-OP so per-turn chatter never pollutes the curated
    knowledge layer (Hafiz's "agents own meaning" ownership rule).
  * prefetch is KNOWLEDGE-ONLY ('hafiz context', no --include-transcripts).
  * project tag derives from agent_workspace -> git repo name -> cwd
    basename -> 'hermes', mirroring the honcho provider's precedent.

Installed by ``hafiz agent install hermes`` into ``$HERMES_HOME/plugins/hafiz/``.

Config: none required. Optional $HERMES_HOME/hafiz.json keys:
  bin          — path to the hafiz executable (default: 'hafiz' on PATH)
  project      — force a fixed --project tag (default: auto-derived)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive subprocess failures, pause
# hafiz calls for _BREAKER_COOLDOWN_SECS to avoid hammering a down DB.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120

# Subprocess timeouts (seconds). prefetch/recall load fastembed cold and can
# be slow; keep generous but bounded so a turn never wedges.
_TIMEOUT_FAST = 8  # status checks
_TIMEOUT_QUERY = 30  # context / query (fastembed cold start)
_TIMEOUT_WRITE = 30  # observe / capture

_SOURCE = "agent:hermes"

# Relevance gate for AUTO-prefetch only (never the explicit hafiz_recall tool).
# nomic-embed cosine scores: on-topic ~0.47-0.57, off-topic noise floor ~0.45.
# Those overlap, so an absolute cutoff is brittle - use a RELATIVE gate: require
# the best hit to clear a floor, then keep only rows within a small delta of it.
# This drops the noise floor while keeping a tight on-topic cluster, and adapts
# per query. Override the floor with HAFIZ_RECALL_MIN_SCORE.
# Chunks (code) score systematically higher than annotations under nomic-embed
# (measured: off-topic chunks top out ~0.58, on-topic start ~0.66; off-topic
# annotations ~0.50, on-topic ~0.63). So gate them with SEPARATE floors.
_PREFETCH_FLOOR_ANNOTATION = 0.52
_PREFETCH_FLOOR_CHUNK = 0.62
_PREFETCH_SCORE_DELTA = 0.06
_PREFETCH_MAX_LINES = 8  # total injected lines (annotations + chunks)

# Cold-start: how many user-preference + project rows to pin at session start,
# before background prefetch has warmed.
_COLDSTART_USER_LIMIT = 6
_COLDSTART_PROJECT_LIMIT = 4
_COLDSTART_SINCE = "180d"


def _score_floor(default: float) -> float:
    """The prefetch relevance floor: HAFIZ_RECALL_MIN_SCORE override, else default."""
    raw = os.environ.get("HAFIZ_RECALL_MIN_SCORE", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def _gate_by_relevance(rows: list, *, floor: float) -> list:
    """Relative relevance gate: keep the on-topic cluster, drop the noise floor.

    Returns rows (assumed each to have a numeric ``score``) sorted strongest
    first, keeping only those where the *top* score clears ``floor`` and each
    row is within ``_PREFETCH_SCORE_DELTA`` of that top. Empty list when the
    best hit doesn't even clear the floor - i.e. nothing relevant, inject
    nothing rather than dilute the prompt.
    """
    scored = sorted(
        (r for r in rows if isinstance(r.get("score"), (int, float))),
        key=lambda r: r["score"],
        reverse=True,
    )
    if not scored:
        return []
    top = scored[0]["score"]
    if top < _score_floor(floor):
        return []
    cutoff = top - _PREFETCH_SCORE_DELTA
    return [r for r in scored if r["score"] >= cutoff]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _load_config() -> dict:
    """Load optional config from $HERMES_HOME/hafiz.json. All keys optional."""
    config = {
        "bin": os.environ.get("HAFIZ_BIN", "hafiz"),
        "project": os.environ.get("HAFIZ_PROJECT", ""),
    }
    try:
        from hermes_constants import get_hermes_home

        config_path = get_hermes_home() / "hafiz.json"
        if config_path.exists():
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
    except Exception:
        pass
    return config


# ---------------------------------------------------------------------------
# Tool schemas — explicit recall + explicit write only
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "hafiz_recall",
    "description": (
        "Search the user's sovereign knowledge base (Hafiz) for relevant "
        "decisions, facts, learnings, warnings, and patterns recorded across "
        "all their work. Use when you need prior context, past decisions, or "
        "'what do we know about X'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "type": {
                "type": "string",
                "description": "Optional annotation kind filter.",
                "enum": ["decision", "fact", "learning", "pattern", "warning", "note"],
            },
            "limit": {"type": "integer", "description": "Max results (default 8, max 25)."},
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "hafiz_remember",
    "description": (
        "Store a durable piece of knowledge in Hafiz (the curated knowledge "
        "layer). Use for explicit decisions, facts, learnings, warnings, or "
        "user preferences worth keeping across sessions. Stored verbatim — "
        "write a clear, self-contained statement."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The knowledge to store, self-contained."},
            "type": {
                "type": "string",
                "description": "Kind of knowledge.",
                "enum": ["decision", "fact", "learning", "pattern", "warning", "note"],
            },
        },
        "required": ["content", "type"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------


class HafizMemoryProvider(MemoryProvider):
    """Local Hafiz CLI-backed memory: knowledge recall + source-layer capture."""

    def __init__(self):
        self._config: dict = {}
        self._bin = "hafiz"
        self._project = ""  # resolved project tag for writes
        self._session_id = ""
        self._session_turns: list[dict[str, str]] = []
        self._turns_lock = threading.Lock()
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._coldstart_block = ""  # pinned turn-1 context (built in initialize)
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "hafiz"

    # -- config / availability ----------------------------------------------

    def get_config_schema(self):
        return [
            {
                "key": "bin",
                "description": "Path to the hafiz executable",
                "default": "hafiz",
                "env_var": "HAFIZ_BIN",
            },
            {
                "key": "project",
                "description": "Fixed --project tag (blank = auto-derive from workspace/cwd)",
                "default": "",
                "env_var": "HAFIZ_PROJECT",
            },
        ]

    def save_config(self, values, hermes_home):
        config_path = Path(hermes_home) / "hafiz.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def is_available(self) -> bool:
        """PATH check + a fast `hafiz status` to confirm the DB is reachable.

        The ABC says is_available should be cheap. `hafiz status` is a quick
        count query — fast enough, and the only honest way to know the DB is up.
        """
        cfg = _load_config()
        binary = cfg.get("bin", "hafiz")
        if not shutil.which(binary) and not Path(binary).is_file():
            return False
        try:
            proc = subprocess.run(
                [binary, "status", "--json"],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_FAST,
            )
            return proc.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    # -- circuit breaker -----------------------------------------------------

    def _is_breaker_open(self) -> bool:
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Hafiz circuit breaker tripped after %d failures. Pausing %ds.",
                self._consecutive_failures,
                _BREAKER_COOLDOWN_SECS,
            )

    # -- subprocess helper ---------------------------------------------------

    def _run(self, args: list[str], *, timeout: int, stdin: str | None = None) -> dict | None:
        """Run `hafiz <args> --json`, return parsed JSON or None on any failure.

        Never raises — a hafiz failure degrades to None so the caller can
        return empty recall / a tool error rather than wedging the turn.
        """
        try:
            proc = subprocess.run(
                [self._bin, *args],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0:
                self._record_failure()
                logger.debug(
                    "hafiz %s failed (rc=%s): %s", args[:1], proc.returncode, proc.stderr[:200]
                )
                return None
            self._record_success()
            return json.loads(proc.stdout) if proc.stdout.strip() else {}
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            self._record_failure()
            logger.debug("hafiz %s errored: %s", args[:1], e)
            return None

    # -- warm-daemon socket client (stdlib only; CLI is the fallback) --------

    def _socket_path(self) -> str | None:
        """Resolve the hafiz daemon socket path without importing hafiz.

        Mirrors hafiz.core.daemon.socket_path: $HAFIZ_DAEMON_SOCKET override,
        else $XDG_RUNTIME_DIR/hafiz/daemon.sock, else /tmp/hafiz-<uid>/...
        """
        override = os.environ.get("HAFIZ_DAEMON_SOCKET")
        if override:
            return override
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if xdg and os.path.isdir(xdg):
            return os.path.join(xdg, "hafiz", "daemon.sock")
        return os.path.join(f"/tmp/hafiz-{os.getuid()}", "daemon.sock")

    def _daemon_request(self, payload: dict, *, timeout: float, spawn: bool = True) -> dict | None:
        """Send one JSON request to the warm daemon over its Unix socket.

        Returns the parsed response dict, or None if the daemon couldn't serve
        it (caller falls back to the CLI). Optionally auto-spawns the daemon
        (via `hafiz serve --detach`) on connection refusal and retries once.
        Pure stdlib — the socket protocol is the contract, so the provider
        never imports hafiz across venvs.
        """
        import socket as _socket

        sock_path = self._socket_path()

        def _attempt() -> dict | None:
            try:
                s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
                s.settimeout(timeout)
                s.connect(sock_path)
            except OSError:
                return None
            try:
                s.sendall((json.dumps(payload) + "\n").encode("utf-8"))
                buf = b""
                while b"\n" not in buf:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                if not buf:
                    return None
                return json.loads(buf.split(b"\n", 1)[0])
            except (OSError, json.JSONDecodeError):
                return None
            finally:
                s.close()

        resp = _attempt()
        if resp is None and spawn:
            # No live daemon — spawn one (detached) and poll for readiness.
            try:
                subprocess.Popen(
                    [self._bin, "serve", "--detach"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError:
                return None
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                time.sleep(0.3)
                ping = _attempt()
                if ping is not None:
                    break
            resp = _attempt()
        return resp

    # -- project-tag derivation (mirrors honcho's cwd/git precedent) ---------

    @staticmethod
    def _git_repo_name(cwd: str) -> str | None:
        try:
            root = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=5,
            )
            if root.returncode == 0 and root.stdout.strip():
                return Path(root.stdout.strip()).name
        except (OSError, subprocess.TimeoutExpired):
            pass
        return None

    def _resolve_project(self, **kwargs) -> str:
        """agent_workspace -> git repo name -> cwd basename -> 'hermes'."""
        forced = self._config.get("project")
        if forced:
            return forced
        ws = kwargs.get("agent_workspace")
        if ws and ws != "hermes":
            return ws
        cwd = os.getcwd()
        repo = self._git_repo_name(cwd)
        if repo:
            return repo
        base = Path(cwd).name
        return base or "hermes"

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._bin = self._config.get("bin", "hafiz")
        self._session_id = session_id
        self._project = self._resolve_project(**kwargs)
        self._coldstart_block = ""
        with self._turns_lock:
            self._session_turns = []
        # Build the pinned cold-start block once, in the background, so turn 1
        # has durable context before the per-turn prefetch has warmed. Failure
        # is silent — an empty block just means turn 1 leans on hafiz_recall.
        threading.Thread(target=self._build_coldstart, daemon=True, name="hafiz-coldstart").start()

    def _build_coldstart(self) -> None:
        """Pin recent user-scoped preferences + project decisions for turn 1.

        Uses `hafiz journal` (recency-ordered, no query needed — recall is
        vector-only and can't list). Best-effort; warms the daemon as a side
        effect of the first CLI call.
        """
        lines: list[str] = []
        # Durable user preferences. `--source` is an EXACT match, not a prefix,
        # and the username isn't known here — so pull a recent window unscoped
        # and keep `user:*` rows client-side. Portable, one call.
        recent = self._run(
            ["journal", "--since", _COLDSTART_SINCE, "--json", "--limit", "120"],
            timeout=_TIMEOUT_QUERY,
        )
        user_rows = [
            e for e in self._journal_entries(recent) if (e.get("source") or "").startswith("user:")
        ]
        for e in user_rows[:_COLDSTART_USER_LIMIT]:
            c = (e.get("content") or "").strip().replace("\n", " ")
            if c:
                lines.append(f"- [pref] {c[:280]}")
        # Recent decisions, preferring the active project but falling back to
        # recent decisions in any project — in practice decisions are often
        # recorded untagged (project=None), so a strict project filter would
        # usually starve this half. Prefer-then-fallback keeps it useful.
        decisions: list[dict] = []
        if self._project:
            proj = self._run(
                [
                    "journal",
                    "--project",
                    self._project,
                    "--type",
                    "decision",
                    "--since",
                    _COLDSTART_SINCE,
                    "--json",
                    "--limit",
                    str(_COLDSTART_PROJECT_LIMIT),
                ],
                timeout=_TIMEOUT_QUERY,
            )
            decisions = self._journal_entries(proj)
        if not decisions:
            recent_dec = self._run(
                [
                    "journal",
                    "--type",
                    "decision",
                    "--since",
                    _COLDSTART_SINCE,
                    "--json",
                    "--limit",
                    str(_COLDSTART_PROJECT_LIMIT),
                ],
                timeout=_TIMEOUT_QUERY,
            )
            decisions = self._journal_entries(recent_dec)
        for e in decisions[:_COLDSTART_PROJECT_LIMIT]:
            c = (e.get("content") or "").strip().replace("\n", " ")
            if c:
                lines.append(f"- [decision] {c[:280]}")
        if lines:
            self._coldstart_block = "\n".join(lines)

    @staticmethod
    def _journal_entries(data: dict | None) -> list[dict]:
        """Pull the annotation entries out of a `hafiz journal --json` payload."""
        if not isinstance(data, dict):
            return []
        return data.get("entries") or []

    def system_prompt_block(self) -> str:
        head = (
            "# Hafiz Memory\n"
            f"Active (local knowledge base, project: {self._project}). "
            "Relevant prior knowledge is auto-recalled below when available. "
            "Use hafiz_recall to search deeper, hafiz_remember to store durable "
            "decisions/facts/learnings."
        )
        block = self._coldstart_block
        if block:
            return f"{head}\n\n## Pinned context\n{block}"
        return head

    # -- prefetch (knowledge-only) ------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            # Brief join only: with the warm daemon a background fetch is
            # ~tens of ms, so don't stall the turn waiting on a slow one.
            self._prefetch_thread.join(timeout=0.5)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return f"## Hafiz Memory\n{result}" if result else ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open() or not query.strip():
            return

        def _run():
            # Daemon-first (warm, ~hundreds of ms); CLI fallback (cold, ~1.7s).
            dresp = self._daemon_request({"op": "context", "query": query}, timeout=_TIMEOUT_QUERY)
            if dresp and dresp.get("ok"):
                data = dresp.get("bundle") or {}
            else:
                data = self._run(["context", query, "--json"], timeout=_TIMEOUT_QUERY)
            if not data:
                return
            # (A) Relevance gate: keep only the on-topic cluster, drop the
            # nomic-embed noise floor so we never dilute the prompt with
            # near-random rows. Annotations and chunks are gated independently
            # (their score scales differ), then merged strongest-first.
            anns = _gate_by_relevance(
                data.get("annotations") or [], floor=_PREFETCH_FLOOR_ANNOTATION
            )
            chunks = _gate_by_relevance(data.get("chunks") or [], floor=_PREFETCH_FLOOR_CHUNK)
            lines: list[str] = []
            for a in anns:
                content = (a.get("content") or "").strip().replace("\n", " ")
                if content:
                    kind = a.get("kind", "note")
                    lines.append((a["score"], f"- [{kind}] {content[:300]}"))
            for c in chunks:
                name = c.get("name") or c.get("unit_name") or c.get("source_file") or "code"
                snippet = (c.get("content") or c.get("body") or "").strip().replace("\n", " ")
                if snippet:
                    lines.append((c["score"], f"- [code] {name}: {snippet[:200]}"))
            # (C) Lean: strongest-first, capped — a handful of high-signal facts
            # beats a wall the model reads past.
            lines.sort(key=lambda t: t[0], reverse=True)
            rendered = [text for _, text in lines[:_PREFETCH_MAX_LINES]]
            if rendered:
                with self._prefetch_lock:
                    self._prefetch_result = "\n".join(rendered)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="hafiz-prefetch")
        self._prefetch_thread.start()

    # -- per-turn: buffer for end-of-session capture, NO knowledge writes ----

    def sync_turn(
        self, user_content: str, assistant_content: str, *, session_id: str = "", messages=None
    ) -> None:
        """Deliberately does NOT write to the knowledge layer.

        We only buffer the turn so on_session_end can capture the whole
        conversation to the source layer. Auto-extracting per-turn would
        pollute the curated knowledge store (Hafiz ownership rule).
        """
        with self._turns_lock:
            self._session_turns.append({"user": user_content, "assistant": assistant_content})

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [RECALL_SCHEMA, REMEMBER_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps(
                {
                    "error": (
                        "Hafiz temporarily unavailable (consecutive failures). "
                        "Will retry automatically."
                    )
                }
            )

        if tool_name == "hafiz_recall":
            query = (args.get("query") or "").strip()
            if not query:
                return tool_error("Missing required parameter: query")
            limit = min(int(args.get("limit", 8)), 25)
            # Daemon-first (warm ~15ms); CLI fallback (cold ~1.3s).
            dpayload = {"op": "query_recall", "query": query, "limit": limit}
            if args.get("type"):
                dpayload["kind"] = args["type"]
            dresp = self._daemon_request(dpayload, timeout=_TIMEOUT_QUERY)
            if dresp and dresp.get("ok"):
                results = dresp.get("results", [])
            else:
                cmd = ["query", query, "--observations", "--json", "--limit", str(limit)]
                if args.get("type"):
                    cmd += ["--type", args["type"]]
                data = self._run(cmd, timeout=_TIMEOUT_QUERY)
                if data is None:
                    return tool_error("Hafiz recall failed (is the DB running?).")
                results = data.get("results", [])
            if not results:
                return json.dumps({"result": "No relevant knowledge found."})
            items = [
                {
                    "content": r.get("content", ""),
                    "kind": r.get("kind"),
                    "source": r.get("source"),
                    "age_days": r.get("age_days"),
                    "score": r.get("score"),
                }
                for r in results
            ]
            return json.dumps({"results": items, "count": len(items)})

        if tool_name == "hafiz_remember":
            content = (args.get("content") or "").strip()
            kind = args.get("type")
            if not content:
                return tool_error("Missing required parameter: content")
            if not kind:
                return tool_error("Missing required parameter: type")
            # Daemon-first; CLI fallback.
            dpayload = {"op": "observe", "content": content, "kind": kind, "source": _SOURCE}
            if self._project:
                dpayload["project"] = self._project
            dresp = self._daemon_request(dpayload, timeout=_TIMEOUT_WRITE)
            if dresp and dresp.get("ok"):
                ann = dresp.get("annotation") or {}
            else:
                cmd = ["observe", content, "--type", kind, "--source", _SOURCE, "--json"]
                if self._project:
                    cmd += ["--project", self._project]
                data = self._run(cmd, timeout=_TIMEOUT_WRITE)
                if data is None:
                    return tool_error("Hafiz write failed (is the DB running?).")
                ann = data.get("annotation") or {}
            return json.dumps({"result": "Stored.", "id": ann.get("id"), "kind": kind})

        return tool_error(f"Unknown tool: {tool_name}")

    # -- session boundary: capture full transcript to the SOURCE layer -------

    def _capture_session(self, reason: str) -> None:
        if self._is_breaker_open():
            return
        with self._turns_lock:
            turns = list(self._session_turns)
            self._session_turns = []
        if not turns:
            return
        parts: list[str] = []
        for t in turns:
            u = (t.get("user") or "").strip()
            a = (t.get("assistant") or "").strip()
            if u:
                parts.append(f"User: {u}")
            if a:
                parts.append(f"Assistant: {a}")
        transcript = "\n\n".join(parts).strip()
        if not transcript:
            return
        title = f"hermes:{self._session_id or 'session'}"
        # Daemon-first capture (warm, persistent connection reuse). The daemon
        # has no parked-stub problem — store_transcript is wired there directly.
        dpayload = {
            "op": "capture",
            "text": transcript,
            "source": _SOURCE,
            "title": title,
            "tags": ["hermes"],
        }
        if self._session_id:
            dpayload["session_id"] = self._session_id
        if self._project:
            dpayload["project"] = self._project
        dresp = self._daemon_request(dpayload, timeout=_TIMEOUT_WRITE)
        if dresp and dresp.get("ok"):
            return
        # CLI fallback (cold start, but needs no daemon).
        cmd = ["capture", "--source", _SOURCE, "--title", title, "--tags", "hermes", "--json"]
        if self._session_id:
            cmd += ["--session", self._session_id]
        if self._project:
            cmd += ["--project", self._project]
        self._run(cmd, timeout=_TIMEOUT_WRITE, stdin=transcript)

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        self._capture_session("session_end")

    def on_session_switch(
        self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, **kwargs
    ) -> None:
        # A genuine new conversation: flush the prior one to the source layer
        # before the buffer is cleared, then re-scope to the new session.
        if reset:
            self._capture_session("session_switch")
        self._session_id = new_session_id
        with self._turns_lock:
            self._session_turns = []

    def shutdown(self) -> None:
        # Flush anything still buffered (e.g. process exit without a clean
        # session-end signal), then join the prefetch thread.
        self._capture_session("shutdown")
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)


def register(ctx) -> None:
    """Register Hafiz as a memory provider plugin."""
    ctx.register_memory_provider(HafizMemoryProvider())
