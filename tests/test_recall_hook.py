"""`hafiz hook session|prompt` — every-turn recall, and the contract around it.

DB-free by construction: retrieval (`_recall`) and project resolution are
replaced, so these tests pin what the hook *does with* rows — what it skips,
what it pins, how it caps — not what the vector index returns.

The contracts that matter, because a violation is invisible until it hurts:

* **Never fail a turn.** Exit 0 and empty output on every failure path.
* **Never exceed the cap.** Claude Code cuts hook output over 10,000 chars and
  the model then sees a ~2 KB preview; a pin set that grew past that quietly
  stopped delivering its tail.
* **Pins render whole.** An excerpt of a rule can open at its rationale and
  elide the rule.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core import recall_hook
from hafiz.core.config import HooksSettings

NOW = datetime.now(UTC)


def row(
    n: int,
    *,
    content: str | None = None,
    project: str | None = None,
    snippet: str | None = None,
    kind: str = "learning",
) -> dict:
    return {
        "id": f"id-{n}",
        "content": content if content is not None else f"rule number {n}",
        "kind": kind,
        "source": "user:test",
        "project": project,
        "tags": ["always"],
        "valid_from": NOW.isoformat(),
        "snippet": snippet,
    }


class FakeStore:
    """Stands in for `_recall`: pins come back for the tag query, topical otherwise."""

    def __init__(self, pins=(), topical=()):
        self.pins, self.topical, self.calls = list(pins), list(topical), []

    async def __call__(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return list(self.pins if kwargs.get("tags") else self.topical)

    def pin_calls(self):
        return [c for c in self.calls if c[1].get("tags")]

    def topical_calls(self):
        return [c for c in self.calls if not c[1].get("tags")]


@pytest.fixture
def hook(monkeypatch):
    """Install a FakeStore and budgets; returns a function to invoke the CLI."""
    state = {"settings": HooksSettings(max_chars=9000, timeout_seconds=5)}

    def configure(store: FakeStore, *, max_chars=9000, timeout=5, projects=(), topic="repo topic"):
        state["settings"] = HooksSettings(max_chars=max_chars, timeout_seconds=timeout)
        monkeypatch.setattr(recall_hook, "_recall", store)

        async def resolve(cwd):
            return frozenset(projects)

        monkeypatch.setattr(recall_hook, "_session_projects", resolve)
        monkeypatch.setattr(recall_hook, "repo_context", lambda cwd: topic)

    monkeypatch.setattr(recall_hook, "_settings", lambda: state["settings"])
    configure(FakeStore())

    def invoke(mode: str, payload) -> object:
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        return CliRunner().invoke(app, ["hook", mode], input=raw)

    invoke.configure = configure
    return invoke


# ── prompt ────────────────────────────────────────────────────────────────


def test_prompt_recalls_and_renders_under_its_heading(hook):
    store = FakeStore(topical=[row(1, content="use httponly cookies for refresh tokens")])
    hook.configure(store)

    result = hook("prompt", {"prompt": "how should refresh tokens be stored?"})

    assert result.exit_code == 0
    assert result.output.startswith("## Relevant prior decisions (hafiz)\n")
    assert "httponly cookies" in result.output
    query, kwargs = store.calls[0]
    assert query == "how should refresh tokens be stored?"
    assert kwargs["min_score"] == 0.05 and kwargs["limit"] == 6 and kwargs["snippet"] is True


def test_prompt_marks_excerpts(hook):
    hook.configure(FakeStore(topical=[row(1, content="x" * 900, snippet="… the claim …")]))

    out = hook("prompt", {"prompt": "something long enough"}).output

    assert "… the claim …" in out
    assert "excerpt of 900 chars" in out


@pytest.mark.parametrize("prompt", ["", "yes", "go on", "continue pl"])
def test_prompt_under_twelve_chars_is_skipped(hook, prompt):
    store = FakeStore(topical=[row(1)])
    hook.configure(store)

    result = hook("prompt", {"prompt": prompt})

    assert result.exit_code == 0 and result.output == ""
    assert store.calls == []


@pytest.mark.parametrize(
    "prompt",
    [
        "/loop 5m check the deploy",
        "<<autonomous-loop-dynamic>> keep going with the migration",
        "tick <<autonomous-loop>> run the next step of the plan",
        "<task-notification>agent finished: all tests pass</task-notification>",
    ],
)
def test_loop_and_harness_prompts_are_skipped(hook, prompt):
    store = FakeStore(topical=[row(1)])
    hook.configure(store)

    result = hook("prompt", {"prompt": prompt})

    assert result.exit_code == 0 and result.output == ""
    assert store.calls == []


def test_editor_preambles_are_stripped_before_the_prompt_is_judged(hook):
    store = FakeStore(topical=[row(1)])
    hook.configure(store)
    preamble = "<ide_opened_file>The user opened /a/b/c.py in the IDE.</ide_opened_file>\n"

    hook("prompt", {"prompt": preamble + "why does consent fail on reload?"})
    assert store.calls[0][0] == "why does consent fail on reload?"

    store.calls.clear()
    hook("prompt", {"prompt": preamble + "ok"})  # nothing left after stripping
    assert store.calls == []


def test_prompt_with_nothing_recalled_prints_nothing(hook):
    hook.configure(FakeStore(topical=[]))
    result = hook("prompt", {"prompt": "a perfectly good question here"})
    assert result.exit_code == 0 and result.output == ""


def test_prompt_cap_is_enforced_with_a_not_shown_line(hook):
    rows = [row(i, content="y" * 600) for i in range(6)]
    hook.configure(FakeStore(topical=rows), max_chars=1500)

    out = hook("prompt", {"prompt": "a perfectly good question here"}).output

    assert len(out) <= 1500
    shown = out.count("- **[")
    assert 0 < shown < 6
    assert f"{6 - shown} more recalled row(s) not shown" in out
    assert "hafiz query" in out and "--observations" in out


# ── session: pins ─────────────────────────────────────────────────────────


def test_session_pins_are_fetched_by_tag_unranked_and_rendered_whole(hook):
    whole = "Never merge feat/iab-tcf-2.3. " + ("Because the audit trail depends on it. " * 40)
    store = FakeStore(pins=[row(1, content=whole, snippet="… truncated …")])
    hook.configure(store)

    out = hook("session", {"cwd": "/work/repo"}).output

    assert "## Always applies (hafiz — pinned)" in out
    assert whole.strip() in out  # all of it, not the snippet
    assert "truncated" not in out
    query, kwargs = store.pin_calls()[0]
    assert kwargs["tags"] == ["always"]
    assert kwargs["rerank"] is False
    assert "min_score" not in kwargs  # a pin is the absence of a ranking judgement
    assert not kwargs.get("snippet")


def test_session_pin_scoped_to_a_sibling_repo_still_applies_but_an_unrelated_one_does_not(hook):
    """Scope is the workspace: alpha + beta are siblings, gamma is another workspace."""
    pins = [
        row(1, content="global rule"),
        row(2, content="rule for alpha", project="alpha"),
        row(3, content="rule for beta", project="beta"),
        row(4, content="rule for gamma", project="gamma"),
    ]
    hook.configure(FakeStore(pins=pins), projects={"alpha", "beta"})

    out = hook("session", {"cwd": "/work/alpha"}).output

    assert "global rule" in out and "rule for alpha" in out
    assert "rule for beta" in out  # sibling repo, same workspace
    assert "rule for gamma" not in out  # unrelated workspace


def test_session_in_an_unresolvable_dir_hides_scoped_pins_but_keeps_global_ones(hook):
    pins = [row(1, content="global rule"), row(2, content="rule for alpha", project="alpha")]
    hook.configure(FakeStore(pins=pins), projects=())

    out = hook("session", {"cwd": "/somewhere/else"}).output

    assert "global rule" in out
    assert "rule for alpha" not in out


def test_session_topical_uses_repo_context_and_skips_what_is_pinned(hook):
    pinned = row(1, content="pinned rule")
    store = FakeStore(pins=[pinned], topical=[pinned, row(2, content="topical decision")])
    hook.configure(store, topic="hafiz feat recall")

    out = hook("session", {"cwd": "/work/hafiz"}).output

    assert out.count("pinned rule") == 1
    assert "topical decision" in out
    assert "## Relevant to hafiz feat recall (hafiz)" in out
    query, kwargs = store.topical_calls()[0]
    assert query == "hafiz feat recall"
    assert kwargs["limit"] == 5 and kwargs["min_score"] == 0.05


def test_session_uses_the_payload_cwd_not_the_process_cwd(hook, monkeypatch):
    seen = []
    monkeypatch.setattr(recall_hook, "repo_context", lambda cwd: seen.append(cwd) or "t")
    store = FakeStore(topical=[row(1)])
    hook.configure(store)
    monkeypatch.setattr(recall_hook, "repo_context", lambda cwd: seen.append(cwd) or "t")

    hook("session", {"cwd": "/from/the/payload"})

    assert seen == ["/from/the/payload"]


def test_session_with_no_topic_skips_the_topical_query(hook):
    store = FakeStore(pins=[row(1)])
    hook.configure(store, topic="")

    out = hook("session", {"cwd": "/gone"}).output

    assert "rule number 1" in out
    assert store.topical_calls() == []


# ── the cap ───────────────────────────────────────────────────────────────


def test_session_cap_fills_pins_first_then_topical_and_says_what_was_dropped(hook):
    pins = [row(i, content="p" * 300) for i in range(10)]
    topical = [row(100 + i, content="t" * 300) for i in range(3)]
    hook.configure(FakeStore(pins=pins, topical=topical), max_chars=1800)

    out = hook("session", {"cwd": "/w"}).output

    assert len(out) <= 1800
    assert out.count("p" * 300) > 0
    assert "t" * 300 not in out  # topical is what yields when the budget is short
    last = out.strip().splitlines()[-1]
    assert "not shown" in last and "capped at 1800 chars" in last
    assert 'hafiz query "always" --observations --tags always' in last
    pins_shown = out.count("p" * 300)
    assert f"{(10 - pins_shown) + 3} pinned rule(s)" not in last  # mixed drop says "rule(s)"
    assert f"{(10 - pins_shown) + 3} rule(s) not shown" in last


def test_a_pin_set_over_ten_thousand_chars_is_capped_below_the_harness_limit(hook):
    """The live bug: 14 pins ≈ 11.9k chars, and the harness cut the output at 10k."""
    pins = [row(i, content=("r" * 850) + f" #{i}") for i in range(14)]
    hook.configure(FakeStore(pins=pins), max_chars=9000)

    out = hook("session", {"cwd": "/w"}).output

    assert 8000 < len(out) <= 9000 < 10_000
    shown = out.count("- **[")
    assert shown < 14
    assert f"{14 - shown} pinned rule(s) not shown" in out


def test_nothing_is_dropped_and_no_footer_when_everything_fits(hook):
    hook.configure(FakeStore(pins=[row(1), row(2)], topical=[row(3)]), max_chars=9000)

    out = hook("session", {"cwd": "/w"}).output

    assert "not shown" not in out
    assert out.count("- **[") == 3


def test_assemble_never_exceeds_the_cap_whatever_the_sizes():
    footer = lambda d: f"_hafiz: {sum(d)} not shown_"  # noqa: E731
    for cap in (60, 120, 500, 2000):
        for sizes in ([10, 10, 10], [200, 5, 5], [90] * 20, [3000], []):
            sections = [("## A", ["x" * n for n in sizes]), ("## B", ["y" * n for n in sizes])]
            assert len(recall_hook.assemble(sections, max_chars=cap, footer=footer)) <= cap


# ── never fail a turn ─────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["prompt", "session"])
def test_a_failing_recall_exits_zero_and_prints_nothing(hook, monkeypatch, mode):
    async def boom(*a, **k):
        raise RuntimeError("database is on fire")

    hook.configure(FakeStore())
    monkeypatch.setattr(recall_hook, "_recall", boom)

    result = hook(mode, {"prompt": "a perfectly good question here", "cwd": "/w"})

    assert result.exit_code == 0
    assert result.output == ""


@pytest.mark.parametrize("mode", ["prompt", "session"])
@pytest.mark.parametrize("payload", ["", "not json", "[1,2]", '"str"', "null", "{}"])
def test_garbage_stdin_exits_zero(hook, mode, payload):
    hook.configure(FakeStore(topical=[row(1)]))
    result = hook(mode, payload)
    assert result.exit_code == 0, result.output


def test_a_hung_recall_is_cut_off_at_the_time_budget(hook, monkeypatch):
    async def hang(*a, **k):
        await asyncio.sleep(30)

    hook.configure(FakeStore(), timeout=0.2)
    monkeypatch.setattr(recall_hook, "_recall", hang)

    result = hook("prompt", {"prompt": "a perfectly good question here"})

    assert result.exit_code == 0 and result.output == ""


def test_a_broken_config_degrades_to_defaults(monkeypatch):
    import hafiz.core.config as config

    def boom():
        raise ValueError("bad toml")

    monkeypatch.setattr(config, "load_settings", boom)
    cfg = recall_hook._settings()
    assert (cfg.max_chars, cfg.timeout_seconds) == (9000, 20.0)


# ── surface ───────────────────────────────────────────────────────────────


def test_hook_help_lists_both_entry_points():
    result = CliRunner().invoke(app, ["hook", "--help"])
    assert result.exit_code == 0
    assert "session" in result.output and "prompt" in result.output


def test_hooks_defaults_are_below_the_harness_cut():
    cfg = HooksSettings()
    assert cfg.max_chars < 10_000
    assert cfg.timeout_seconds == 20


# ── workspace resolution (real directories, stubbed index) ────────────────


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """ws/{alpha,beta,notes}, with alpha and beta indexed; other/gamma indexed too."""
    import subprocess

    for d in ("ws/alpha/src", "ws/beta", "ws/notes", "other/gamma"):
        (tmp_path / d).mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(tmp_path / "ws" / "alpha")], check=True)

    async def indexed():
        return {"alpha", "beta", "gamma"}

    async def own(path):
        return {"alpha": "alpha", "gamma": "gamma"}.get(
            next(
                (
                    p.name
                    for p in [tmp_path / "ws" / "alpha", tmp_path / "other" / "gamma"]
                    if str(path).startswith(str(p))
                ),
                "",
            )
        )

    monkeypatch.setattr("hafiz.core.context._all_indexed_projects", indexed)
    monkeypatch.setattr("hafiz.core.store.project_for_path", own)
    return tmp_path


async def test_session_projects_include_own_project_and_indexed_siblings(workspace):
    got = await recall_hook._session_projects(str(workspace / "ws" / "alpha"))
    assert got == {"alpha", "beta"}  # `notes` is a sibling but not indexed; gamma is elsewhere


async def test_session_projects_resolve_siblings_from_the_repo_root_not_a_subdirectory(workspace):
    got = await recall_hook._session_projects(str(workspace / "ws" / "alpha" / "src"))
    assert got == {"alpha", "beta"}


async def test_session_projects_exclude_an_unrelated_workspace(workspace):
    got = await recall_hook._session_projects(str(workspace / "other" / "gamma"))
    assert "alpha" not in got and "beta" not in got


async def test_an_unindexed_repo_and_its_unindexed_git_siblings_count_by_folder_name(workspace):
    """A write from an unindexed repo is scoped to its folder name, so its pins
    must reach sessions there and in sibling repos (the 2026-11-26 regression:
    pins scoped to an unindexed hub repo showed in no session at all)."""
    import subprocess

    for d in ("ws/hub", "ws/notes"):
        subprocess.run(["git", "init", "-q", str(workspace / d)], check=True)
    in_hub = await recall_hook._session_projects(str(workspace / "ws" / "hub"))
    in_alpha = await recall_hook._session_projects(str(workspace / "ws" / "alpha"))
    in_gamma = await recall_hook._session_projects(str(workspace / "other" / "gamma"))

    assert {"hub", "alpha", "beta", "notes"} <= in_hub
    assert "hub" in in_alpha
    assert "hub" not in in_gamma


def test_pin_scope_matches_spelling_variants():
    projects = frozenset({"control-room"})
    assert recall_hook.pin_visible({"project": "Control Room"}, projects)
    assert recall_hook.pin_visible({"project": None}, projects)
    assert not recall_hook.pin_visible({"project": "elsewhere"}, projects)


@pytest.mark.parametrize("cwd", [None, "", "/definitely/not/a/real/dir"])
async def test_session_projects_are_empty_when_cwd_is_unusable(workspace, cwd):
    assert await recall_hook._session_projects(cwd) == frozenset()
