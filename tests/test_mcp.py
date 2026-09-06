"""The MCP surface, and the guards that keep it honest.

MCP is a **second agent contract** beside ``skills.md``. The whole reason it
was worth building rather than avoiding is the claim that it cannot drift
from the CLI, because its schemas are derived from the same core functions.
A claim like that is worth exactly as much as the test that enforces it.

So the load-bearing tests here are not "does a tool call work" — they are:

- :func:`test_registry_has_not_drifted`, which fails when a core function
  gains, loses or renames a parameter, and
- :func:`test_the_drift_check_actually_catches_drift`, which proves that
  check can fail at all. A drift detector that cannot detect drift certifies
  the surface while missing the thing it exists to find; this repo has
  shipped two of those already (the dead daemon, the vacuous skip exemption).

Most of this module runs without the optional ``mcp`` extra installed.
``hafiz.core.mcp_server.call_tool`` deliberately imports nothing from the
SDK — only ``build_server`` and ``serve_stdio`` do — so the surface is
testable on a plain install, which is also the CI default.
"""

from __future__ import annotations

import inspect

import pytest
from sqlalchemy import text

from hafiz.core import mcp_registry
from hafiz.core.database import close_engine, get_session_factory
from hafiz.core.mcp_registry import TOOLS, ToolSpec, by_name, check_drift, to_jsonable
from hafiz.core.mcp_server import call_tool

# ── The contract: what is exposed, and what must never be ────────────────

EXPECTED_TOOLS = {
    "hafiz_context",
    "hafiz_query",
    "hafiz_recall_observations",
    "hafiz_recall_session",
    "hafiz_graph",
    "hafiz_journal",
    "hafiz_distill",
    "hafiz_observe",
    "hafiz_note",
    "hafiz_capture",
    "hafiz_session",
    "hafiz_reconcile",
    "hafiz_retrievals",
    "hafiz_status",
}

#: Core functions that must never become reachable from MCP. An MCP client
#: calls tools without asking anyone, so these are the operations whose blast
#: radius requires a human: irreversible deletion, environment mutation, and
#: anything that reads arbitrary paths off the user's disk into the store.
FORBIDDEN_TARGETS = (
    "forget",
    "prune",
    "tombstone",
    "config_set",
    "config_unset",
    "run_init",
    "index_file",
    "ingest",
    "install",
    "uninstall",
)


def test_the_exposed_surface_is_the_declared_one():
    """Adding or removing a tool must be a deliberate edit in two places.

    Not ceremony: the tool list is what every connected client loads into its
    system prompt on every request. It should not grow as a side effect of
    someone adding a core function.
    """
    assert {t.name for t in TOOLS} == EXPECTED_TOOLS
    assert len({t.name for t in TOOLS}) == len(TOOLS), "duplicate tool name"


def test_no_destructive_or_administrative_target_is_reachable():
    """The exclusions are a safety property, so assert them rather than trusting the list."""
    for spec in TOOLS:
        target = spec.target.lower()
        for forbidden in FORBIDDEN_TARGETS:
            assert forbidden not in target, (
                f"{spec.name} targets {spec.target!r}, which matches the forbidden "
                f"pattern {forbidden!r}. Destructive and administrative operations "
                f"stay on the CLI, where a human is present."
            )


def test_registry_has_not_drifted():
    """Every described parameter exists, and every exposed parameter is described."""
    problems = check_drift()
    assert not problems, "MCP registry no longer matches the core functions:\n  " + "\n  ".join(
        problems
    )


def test_the_drift_check_actually_catches_drift():
    """Guard the guard, in both directions.

    Constructed rather than mutated: a spec describing a parameter its target
    does not have, and one whose target has a parameter it does not describe.
    If ``check_drift`` shrugs at either, it is decoration.
    """
    stale = ToolSpec(
        name="fake_stale",
        summary="describes a parameter that does not exist",
        target="hafiz.core.telemetry:retrieval_report",
        params={"since_days": "d", "limit": "l", "invented_parameter": "not real"},
    )
    problems = _drift_of(stale)
    assert any("invented_parameter" in p for p in problems), (
        "check_drift did not notice a described parameter that the target rejects"
    )

    undescribed = ToolSpec(
        name="fake_undescribed",
        summary="omits a parameter the target accepts",
        target="hafiz.core.telemetry:retrieval_report",
        params={"since_days": "d"},  # `limit` exists but is undescribed
    )
    problems = _drift_of(undescribed)
    assert any("limit" in p for p in problems), (
        "check_drift did not notice an exposed parameter with no description"
    )


def _drift_of(spec: ToolSpec) -> list[str]:
    """Run check_drift against one constructed spec, restoring TOOLS after."""
    original = mcp_registry.TOOLS
    mcp_registry.TOOLS = (spec,)
    try:
        return mcp_registry.check_drift()
    finally:
        mcp_registry.TOOLS = original


def test_schemas_are_well_formed():
    for spec in TOOLS:
        schema = spec.input_schema()
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        signature = inspect.signature(spec.resolve()).parameters
        for name, prop in schema["properties"].items():
            assert prop.get("description"), f"{spec.name}.{name} has no description"
            assert prop.get("type"), f"{spec.name}.{name} has no type"
        # Required must mean "no default", or a client will omit something the
        # function cannot run without and get a TypeError instead of a prompt.
        for name in schema["required"]:
            assert signature[name].default is inspect.Parameter.empty


def test_pinned_and_hidden_parameters_never_reach_the_schema():
    """A forced value the caller could override would not be forced at all."""
    for spec in TOOLS:
        exposed = set(spec.input_schema()["properties"])
        assert not (exposed & set(spec.force)), f"{spec.name} exposes a pinned parameter"
        assert not (exposed & set(spec.exclude)), f"{spec.name} exposes an excluded parameter"
        assert "telemetry_command" not in exposed


def test_note_cannot_supersede():
    """`note` is the raw-capture lane; it must not be able to retire a considered belief."""
    note = by_name()["hafiz_note"]
    schema = note.input_schema()["properties"]
    assert "supersedes_id" not in schema
    assert note.force["kind"] == "note"
    assert "kind" not in schema


def test_write_tools_are_flagged_as_such():
    """Clients apply confirmation policy off the read-only hint, so it must be right."""
    writers = {t.name for t in TOOLS if t.writes}
    assert writers == {"hafiz_observe", "hafiz_note", "hafiz_capture", "hafiz_session"}


def test_to_jsonable_survives_the_shapes_core_returns():
    import dataclasses
    import datetime
    import uuid as _uuid

    @dataclasses.dataclass
    class Nested:
        when: datetime.datetime
        ident: _uuid.UUID

    value = to_jsonable(
        {
            "rows": [Nested(datetime.datetime(2026, 9, 6, 12, 0), _uuid.UUID(int=1))],
            "count": 1,
            "missing": None,
        }
    )
    assert value["rows"][0]["when"] == "2026-09-06T12:00:00"
    assert value["rows"][0]["ident"] == "00000000-0000-0000-0000-000000000001"
    assert value["missing"] is None


# ── Dispatch ─────────────────────────────────────────────────────────────

#: Every source string these tests write under, so teardown can find its own
#: rows and nothing else. Deleting by content would be fragile; deleting
#: unscoped would eat the developer's real store on a misconfigured run.
ROUND_TRIP_SOURCE = "agent:test-mcp"
CLIENT_SOURCE = "agent:some-ide"
EXPLICIT_SOURCE = "user:anjum-mcp-test"
TEST_SOURCES = (ROUND_TRIP_SOURCE, CLIENT_SOURCE, EXPLICIT_SOURCE)


async def _wipe() -> None:
    factory = get_session_factory()
    async with factory() as session:
        for source in TEST_SOURCES:
            await session.execute(text("DELETE FROM annotations WHERE source = :s"), {"s": source})
        await session.commit()


@pytest.fixture(autouse=True)
async def _engine_per_test():
    """Dispose the shared engine after every test in this module.

    ``pytest-asyncio`` in auto mode gives each test its own event loop, while
    ``get_session_factory`` caches one engine for the process. A test that
    touches the DB therefore leaves an engine bound to a loop that is closed
    by the time the next test runs — asyncpg reports that as "attached to a
    different loop" or "another operation is in progress", and only on the
    Postgres leg. Every other DB-touching module here closes per test for the
    same reason; a no-op when no engine was created.
    """
    yield
    await close_engine()


@pytest.fixture
async def written_rows_cleaned():
    """Clean before and after, and close the engine only at teardown.

    The engine is a module-level singleton shared by the whole session.
    Disposing it *inside* a test body — which an earlier version of these
    tests did, in a ``finally`` — leaves later operations bound to a closed
    loop. SQLite shrugged; asyncpg raised "another operation is in progress"
    and "attached to a different loop", and only on the Postgres leg. Hence
    a fixture, matching what every other DB-touching module here does.
    """
    await _wipe()
    yield
    await _wipe()


async def test_unknown_tool_is_a_clean_error():
    with pytest.raises(KeyError):
        await call_tool("hafiz_not_a_tool", {})


async def test_unknown_arguments_are_dropped_not_forwarded():
    """A newer client must degrade to 'that filter was ignored', not TypeError.

    Same contract as ``daemon._supported_kwargs`` on the warm path — a client
    built against a later hafiz should still work against this one.
    """
    result = await call_tool(
        "hafiz_retrievals", {"since_days": 1, "limit": 1, "a_filter_from_the_future": "x"}
    )
    assert isinstance(result, dict)


async def test_pinned_telemetry_command_is_applied():
    """`hafiz_query` must record its searches under the same command the CLI uses."""
    spec = by_name()["hafiz_query"]
    from hafiz.core import telemetry

    assert spec.force["telemetry_command"] == telemetry.QUERY


async def test_a_write_and_a_read_round_trip(written_rows_cleaned):
    """The real path: write through a tool, read it back through another."""
    marker = "mcp round trip marker phrase"
    written = await call_tool(
        "hafiz_note", {"content": marker, "source": ROUND_TRIP_SOURCE, "tags": ["mcp-test"]}
    )
    assert written["kind"] == "note"
    assert written["source"] == ROUND_TRIP_SOURCE

    found = await call_tool("hafiz_recall_observations", {"query": marker, "limit": 5})
    assert any(row.get("content") == marker for row in found), (
        "a note written through MCP was not readable through MCP"
    )


async def test_client_name_becomes_the_source_when_unset(written_rows_cleaned):
    """Attribution: an unattributed write would make every client look identical."""
    written = await call_tool("hafiz_note", {"content": "attribution test"}, client="some-ide")
    assert written["source"] == CLIENT_SOURCE


async def test_an_explicit_source_beats_the_client_name(written_rows_cleaned):
    written = await call_tool(
        "hafiz_note",
        {"content": "explicit source test", "source": EXPLICIT_SOURCE},
        client="some-ide",
    )
    assert written["source"] == EXPLICIT_SOURCE


# ── SDK-dependent ────────────────────────────────────────────────────────


def test_server_builds_with_every_tool():
    """Schemas must satisfy the SDK's own validation, not just ours."""
    pytest.importorskip("mcp", reason="the optional 'mcp' extra is not installed")
    from hafiz.core.mcp_server import build_server

    server = build_server()
    assert server is not None


def test_missing_extra_explains_itself():
    """The failure a user without the extra actually hits must name the fix."""
    from hafiz.core.mcp_server import MissingDependencyError

    assert issubclass(MissingDependencyError, RuntimeError)
    message = MissingDependencyError.__doc__ or ""
    assert "extra" in message


# ── Status (the 14th tool) ───────────────────────────────────────────────


async def test_status_trims_only_the_uninformative_half(_engine_per_test):
    """`verbose=False` must drop up-to-date projects and nothing else.

    The trim exists to surface signal, not to save bytes — measured on a real
    store where 8 of 11 projects were stale it saved only 12%. So the test
    asserts the *semantics*: what survives is exactly what is actionable, and
    the compliance-relevant fields are never among the casualties.
    """
    from hafiz.core.health import collect_status, is_stale

    full = await collect_status(verbose=True)
    trimmed = await collect_status(verbose=False)

    # Same contract, minus nothing, plus one summary.
    assert set(trimmed) - set(full) == {"staleness_summary"}
    assert not set(full) - set(trimmed)

    # Only genuinely stale projects survive, and all of them do.
    assert set(trimmed["staleness"]) == {n for n, e in full["staleness"].items() if is_stale(e)}

    summary = trimmed["staleness_summary"]
    assert summary["projects_checked"] == len(full["staleness"])
    assert summary["stale"] == len(trimmed["staleness"])
    assert summary["trimmed"] == summary["projects_checked"] - summary["stale"]


async def test_status_never_trims_the_retention_and_capture_signals(_engine_per_test):
    """These two are why status is worth exposing at all.

    `retention.overdue` is a stated guarantee that can quietly stop being
    met; `capture` answers "is anything still arriving", whose silence let
    transcript capture die unnoticed for two months. A payload-shrinking
    change that dropped either would be a regression disguised as an
    optimisation.
    """
    from hafiz.core.health import collect_status

    trimmed = await collect_status(verbose=False)
    assert set(trimmed["retention"]) == {"overdue", "communications", "retrievals"}
    assert "capture" in trimmed


async def test_an_empty_staleness_is_never_ambiguous(_engine_per_test):
    """Trimmed output must distinguish "all fresh" from "never checked".

    An empty `staleness` with nothing beside it reads as health whether or
    not anything was examined — the exact shape of failure this repo has
    already paid for once.
    """
    from hafiz.core.health import collect_status

    trimmed = await collect_status(verbose=False)
    assert "staleness_summary" in trimmed, (
        "trimmed status omits the summary, so an empty staleness map cannot be "
        "told apart from freshness never having been checked"
    )


async def test_status_is_reachable_as_a_tool(_engine_per_test):
    payload = await call_tool("hafiz_status", {})
    assert payload["staleness_summary"]["projects_checked"] >= 0
    assert "files" in payload and "units" in payload

    verbose = await call_tool("hafiz_status", {"verbose": True})
    assert "staleness_summary" not in verbose


def test_collect_status_does_not_touch_engine_lifecycle():
    """A long-lived server must not have its pool disposed by a status call.

    The code this was extracted from ended in ``finally: await
    close_engine()``. Correct for a one-shot CLI process; for the MCP server
    it would dispose the pool on every call. Asserted statically because the
    failure is a *slow* one — the next call simply reconnects, so nothing
    looks broken until connection churn shows up under load.

    Parsed rather than grepped: the first version searched the source text
    and failed on the module docstring, which explains the very rule it is
    enforcing. A check that cannot tell code from prose about code has to be
    silenced, and a silenced check protects nothing.
    """
    import ast as _ast
    import inspect as _inspect

    from hafiz.core import health

    called = {
        getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        for node in _ast.walk(_ast.parse(_inspect.getsource(health)))
        if isinstance(node, _ast.Call)
    }
    assert "close_engine" not in called, (
        "hafiz.core.health calls close_engine; engine lifecycle belongs to "
        "whichever caller owns the process, not to the shared data collector. "
        "The MCP server would be disposing its own pool on every status call."
    )
