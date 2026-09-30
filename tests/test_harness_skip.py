"""Harness-generated prompts are not searched as if the user typed them.

A per-prompt recall hook forwards whatever the harness hands it. When a
sub-agent finishes, that is its whole final report wrapped in
``<task-notification>`` — an answer, not a question. On the deployment that
surfaced this, 468 such recalls were recorded and 88% came back empty, making
them the largest source of noise in ``hafiz retrievals`` → ``unanswered``.

Editor context (``<ide_opened_file>``) is different: it is glued to the front of
a real prompt, so it is stripped and the prompt behind it is searched.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from hafiz.core import telemetry
from hafiz.core.search import strip_harness_wrappers

NOTIFICATION = (
    "<task-notification>\n<task-id>a0d8</task-id>\n<status>completed</status>\n"
    "<result>Only the two intended files changed.</result>\n</task-notification>"
)
IDE = (
    "<ide_opened_file>The user opened the file /repo/readme.txt in the IDE. "
    "This may or may not be related to the current task.</ide_opened_file>"
)


# ── The filter ──────────────────────────────────────────────────────────


def test_a_task_notification_is_skipped():
    assert strip_harness_wrappers(NOTIFICATION) == ("", "<task-notification>")


def test_a_truncated_notification_is_still_skipped():
    """The closing tag is not required: a hook may cap the prompt it forwards."""
    assert strip_harness_wrappers(NOTIFICATION[:60])[1] == "<task-notification>"


def test_editor_preamble_is_stripped_and_the_prompt_kept():
    assert strip_harness_wrappers(f"{IDE} fix readme.txt") == ("fix readme.txt", None)


def test_several_preambles_are_all_stripped():
    selection = "<ide_selection>def f(): ...</ide_selection>"
    reminder = "<system-reminder>be brief</system-reminder>"
    assert strip_harness_wrappers(f"{IDE}\n{selection}{reminder} explain f") == (
        "explain f",
        None,
    )


def test_a_prompt_that_is_only_preamble_is_skipped():
    assert strip_harness_wrappers(IDE) == ("", "<ide_opened_file>")


@pytest.mark.parametrize(
    "prompt",
    [
        "why does <task-notification> show up in recall?",  # mentions, doesn't lead
        "<div>unknown tags are ordinary text</div> keep me",
        "<ide_opened_file>never closed, so not a recognisable block",
        "plain question",
    ],
)
def test_everything_else_is_searched_verbatim(prompt):
    assert strip_harness_wrappers(prompt) == (prompt, None)


# ── The search paths ────────────────────────────────────────────────────


async def _no_embedding(_query):
    raise AssertionError("a skipped prompt must not be embedded")


async def test_annotation_search_returns_nothing_without_embedding(monkeypatch):
    from hafiz.core import annotations

    monkeypatch.setattr(annotations, "embed_query", _no_embedding)
    assert await annotations.search_annotations(NOTIFICATION, telemetry_command=None) == []


async def test_unit_search_returns_nothing_without_embedding(monkeypatch):
    from hafiz.core import search

    monkeypatch.setattr(search, "embed_query", _no_embedding)
    assert await search.vector_search(NOTIFICATION, telemetry_command=None) == []


async def test_a_skip_logs_the_marker_not_the_message(monkeypatch):
    seen = []

    async def fake_record(**kw):
        seen.append(kw)

    monkeypatch.setattr(telemetry, "record_retrieval", fake_record)
    from hafiz.core.search import prepare_query

    assert await prepare_query(NOTIFICATION, telemetry_command=telemetry.OBSERVATIONS) is None
    assert seen[0]["query"] == "[skipped] <task-notification>"
    assert seen[0]["command"] == telemetry.OBSERVATIONS
    assert "Only the two intended files" not in repr(seen)


# ── The report ──────────────────────────────────────────────────────────


async def _db_available() -> bool:
    from hafiz.core.database import get_session_factory

    try:
        async with get_session_factory()() as s:
            await s.execute(text("SELECT 1 FROM retrievals LIMIT 1"))
        return True
    except Exception:
        return False


async def _wipe() -> None:
    from hafiz.core.database import get_session_factory

    async with get_session_factory()() as s:
        await s.execute(
            text(
                "DELETE FROM retrievals WHERE query_text LIKE '[skipped] %' "
                "OR query_text LIKE '<task-notification>%'"
            )
        )
        await s.commit()


@pytest.fixture
async def db():
    from hafiz.core.config import reset_settings
    from hafiz.core.database import close_engine

    if not await _db_available():
        pytest.skip("Postgres not reachable")
    reset_settings()
    await _wipe()
    yield
    await _wipe()
    reset_settings()
    await close_engine()


async def test_skips_are_counted_apart_from_real_searches(db):
    before = await telemetry.retrieval_report(since_days=1)
    await telemetry.record_skip(command=telemetry.OBSERVATIONS, wrapper="<task-notification>")
    # A row from before skipping existed: the whole message as the query.
    await telemetry.record_retrieval(command=telemetry.OBSERVATIONS, query=NOTIFICATION)

    report = await telemetry.retrieval_report(since_days=1)
    assert report["skipped"] == before["skipped"] + 2
    assert report["retrievals"] == before["retrievals"]
    unanswered = [row["query"] for row in report["unanswered"]]
    assert not any(q.startswith(("[skipped]", "<task-notification>")) for q in unanswered)
