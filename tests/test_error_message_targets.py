"""An error message must point somewhere the reader can actually go.

`workitems/` is **gitignored** (.gitignore:48) — it is the owner's private
design record, and `docs/roadmap.md` is its public equivalent. So a runtime
error that says "see workitems/done/structural-grounding.md" sends a
`pipx install hafiz` user to a file they will never have. Three did:

  - core/extractor.py   ExtractContractError on a v1 extraction payload
  - core/database.py    the _RemovedInV5 stub, on any use of a dropped model
  - core/dialect.py     an unimplemented dialect branch

The project stance is that error messages are features, not afterthoughts,
and that the thousandth user's experience beats the current maintainer's
convenience. A pointer into a directory only the maintainer has fails both.

This guard is deliberately about **runtime strings**, not comments. A comment
saying "see workitems/..." is a note between maintainers reading the source;
it is mildly useless to a stranger but it never surfaces in anyone's
terminal. A message string does, and that is the difference between untidy
and misleading.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "hafiz"

#: Paths that exist only for this repo's maintainer.
#:
#: The criterion is "can the person reading this message go there?", not "is
#: it in the wheel?" — a first draft of this list included `.claude/` and the
#: test immediately failed on `~/.claude/settings.json`, which is exactly
#: right: that is the user's own agent config, and `hafiz agent install`
#: writes it. Those paths are useful to name. `workitems/` is gitignored and
#: personal, so nobody but the owner can follow it.
UNSHIPPED = ("workitems/",)


def _string_constants(tree: ast.AST) -> list[str]:
    """Every string literal in the module, docstrings excluded.

    Docstrings are documentation for a reader of the source, which puts them
    on the comment side of the line this test draws. Only strings that can
    reach a terminal are in scope.
    """
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_no_runtime_string_points_at_an_unshipped_path():
    """The class-level guard, not just the three instances that were found.

    Fixing the three messages without this test would leave the next one free
    to appear — and it would be invisible, because the author has the file the
    message names.
    """
    offenders: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for value in _string_constants(tree):
            for needle in UNSHIPPED:
                if needle in value:
                    rel = path.relative_to(PACKAGE.parent)
                    offenders.append(f"{rel}: {needle!r} in {value[:70]!r}")

    assert not offenders, (
        "these runtime strings name a path only this repo's maintainer has, "
        "so a user following them finds nothing — point at docs/, the README, "
        "or the issue tracker instead:\n  " + "\n  ".join(offenders)
    )


def test_the_guard_would_catch_a_reintroduction():
    """Guard the guard.

    A checker that cannot be shown to catch its own bug certifies the codebase
    while missing the thing it exists to find. Two earlier guards in this repo
    passed on the broken tree before this lesson stuck.
    """
    tree = ast.parse('raise ValueError("see workitems/done/whatever.md")')
    values = _string_constants(tree)
    assert any("workitems/" in v for v in values)


def test_a_docstring_reference_is_allowed():
    """The line the guard draws, stated as a test.

    Comments and docstrings are for someone reading the source. Only strings
    that can reach a terminal are in scope, so a docstring pointing at a work
    item is untidy rather than misleading and must not fail the build.
    """
    tree = ast.parse('"""Design notes: see workitems/active/thing.md."""\nx = 1\n')
    assert _string_constants(tree) == []


# ── cold-start: the errors a brand-new user actually meets ───────────
#
# Found by walking a genuinely empty environment (clean HOME + XDG, a cwd
# outside the repo so config discovery finds nothing). The first thing a new
# user does is `hafiz status` before `hafiz init`, and what they got was a raw
# SQLAlchemy traceback plus advice to check that a Postgres server they had
# never installed was reachable — because the recognizer predated the embedded
# store becoming the default.


def test_a_store_with_no_schema_is_told_to_run_init():
    """Not "check whether Postgres is reachable" — there is no Postgres.

    This is the single most likely error on a fresh install, so its message is
    the one that decides whether someone concludes hafiz is broken.
    """
    from sqlalchemy.exc import OperationalError

    from hafiz.core.error_log import _recognize_db_connectivity

    exc = OperationalError("SELECT count(*) FROM files", {}, Exception("no such table: files"))
    got = _recognize_db_connectivity(exc, argv=["status"], traceback_text="")
    assert got is not None
    message, context = got
    assert "hafiz init" in message
    assert context.get("missing_schema") is True
    assert "Postgres" not in message, (
        "a SQLite user with no schema must not be sent to check a Postgres server"
    )


def test_a_genuine_connectivity_failure_still_points_at_diagnose():
    """The narrowing must not swallow the case the recognizer was built for."""
    from sqlalchemy.exc import OperationalError

    from hafiz.core.error_log import _recognize_db_connectivity

    exc = OperationalError("SELECT 1", {}, Exception("connection refused"))
    got = _recognize_db_connectivity(exc, argv=["status"], traceback_text="")
    assert got is not None
    message, _ = got
    assert "--diagnose" in message
    assert "hafiz init" in message, "and should still mention the no-server path"


def test_the_json_contract_holds_on_the_unhandled_error_path():
    """`--json` is a contract, and failure paths are where it matters most.

    An agent that asked for JSON and received a Rich-formatted traceback has
    no way to read what went wrong — it just fails to parse. The backstop
    handler used to exempt itself from the documented
    `{"ok": false, "error": ...}` shape.
    """
    from hafiz.cli import _wants_json

    assert _wants_json(["status", "--json"])
    assert _wants_json(["query", "x", "--format", "json"])
    assert _wants_json(["query", "x", "--format=json"])
    assert not _wants_json(["status"])
    assert not _wants_json(["query", "x", "--format", "compact"])
    # A value that merely looks like the flag must not trigger it.
    assert not _wants_json(["observe", "the --json flag is documented"])


def test_the_getting_started_path_matches_the_readme():
    """Three steps: init, ingest, query.

    It used to route a brand-new user through `status --diagnose` and
    `doctor --probe` between init and ingest. `doctor --probe`'s own help calls
    it slow — it loads fastembed — so that was a stall recommended as setup,
    and it contradicted the README's documented happy path.
    """
    import hafiz.cli as cli

    help_text = cli.app.info.help or ""
    started = help_text.split("Getting started:")[1].split("\n")[0]
    assert "hafiz init" in started
    assert "hafiz ingest" in started
    assert "--probe" not in started, "a slow diagnostic is not a setup step"
    assert "--diagnose" not in started, "diagnostics belong under 'When stuck'"
