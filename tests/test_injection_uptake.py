"""Was recalled knowledge used? — capture of hook injections + the uptake estimate.

A 2026-11 audit could not answer "is hafiz helping?": the memories a recall hook
injected were in Claude Code's own log but dropped on import. These tests pin
the two halves of the fix: the importer keeps hafiz hook output, and
:mod:`hafiz.core.delivery` estimates whether the agent went on to use it.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core.communications import should_embed_message
from hafiz.core.database import close_engine
from hafiz.core.delivery import (
    INJECTION_KIND,
    _Msg,
    delivery_report,
    judge_session,
    parse_bullets,
    specific_terms,
)
from hafiz.core.importers.claude_code import import_claude_code, parse_jsonl_file

MEMORY = (
    "- **[decision | user:anjum]** Braintree settlement for ft2479 runs via "
    "submitForSettlement in billing/settle.py after the 30-day authorization window"
)
INJECTION = f"## Relevant prior decisions (hafiz)\n\n{MEMORY}\n"


def _msg(role: str, content: str, at: int, **meta) -> _Msg:
    base = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
    return _Msg(base + timedelta(seconds=at), role, content, None, meta)


def _inj(at: int, event: str = "UserPromptSubmit", text: str = INJECTION) -> _Msg:
    return _msg("system", text, at, kind=INJECTION_KIND, hook_event=event)


# ── Parsing an injection ────────────────────────────────────────────────


def test_bullets_drop_headings_and_labels_and_keep_continuations():
    text = "## Always applies\n\n- **[learning | user:x]** first rule\n  more of it\n- second\n"
    assert parse_bullets(text) == ["first rule more of it", "second"]


def test_specific_terms_keep_identifiers_and_drop_stock_words():
    terms = specific_terms("Use shop-front v2.4 for #47471 because something merged")
    assert {"shop-front", "v2.4", "47471"} <= terms
    assert not {"because", "something", "merged", "use"} & terms


# ── Judging uptake ──────────────────────────────────────────────────────


def test_a_memory_the_agent_acts_on_is_likely_used():
    (inj, [verdict]) = judge_session(
        [
            _msg("user", "charge the pending ones", 0),
            _inj(1),
            _msg("assistant", "Running submitForSettlement for ft2479 in billing/settle.py", 2),
        ]
    )[0]
    assert verdict.scorable and verdict.likely_used


def test_a_memory_the_agent_ignores_is_not_used():
    (_, [verdict]) = judge_session(
        [_msg("user", "charge the pending ones", 0), _inj(1), _msg("assistant", "Done.", 2)]
    )[0]
    assert verdict.scorable and not verdict.likely_used


def test_terms_the_session_already_had_prove_nothing():
    # The user named every specific term; repeating them is not uptake.
    prompt = "settle ft2479 with submitForSettlement in billing/settle.py, 30-day authorization"
    (_, [verdict]) = judge_session(
        [_msg("user", prompt, 0), _inj(1), _msg("assistant", prompt, 2)]
    )[0]
    assert not verdict.scorable


def test_a_prompt_memory_is_judged_on_that_turn_only():
    (_, [verdict]) = judge_session(
        [
            _msg("user", "charge the pending ones", 0),
            _inj(1),
            _msg("assistant", "Done.", 2),
            _msg("user", "next thing", 3),
            _msg("assistant", "submitForSettlement ft2479 billing/settle.py", 4),
        ]
    )[0]
    assert not verdict.likely_used


def test_a_session_start_memory_is_judged_on_the_whole_session():
    (_, [verdict]) = judge_session(
        [
            _inj(0, event="SessionStart"),
            _msg("user", "hello there", 1),
            _msg("assistant", "Done.", 2),
            _msg("user", "next thing", 3),
            _msg("assistant", "submitForSettlement ft2479 billing/settle.py", 4),
        ]
    )[0]
    assert verdict.likely_used


def test_time_not_input_order_decides_before_and_after():
    # A backfilled injection arrives with a later seq than the turns around it.
    msgs = [
        _msg("assistant", "Running submitForSettlement for ft2479 in billing/settle.py", 2),
        _msg("user", "charge the pending ones", 0),
        _inj(1),
    ]
    (_, [verdict]) = judge_session(msgs)[0]
    assert verdict.likely_used


# ── Importing injections ────────────────────────────────────────────────


def _line(sid: str, ts: datetime, **rec) -> str:
    return json.dumps(
        {
            "uuid": str(uuid.uuid4()),
            "sessionId": sid,
            "timestamp": ts.isoformat(),
            "cwd": "/tmp/test",
            "isSidechain": False,
            **rec,
        }
    )


def _hook(command: str, stdout: str, event: str = "UserPromptSubmit") -> dict:
    return {
        "type": "attachment",
        "attachment": {
            "type": "hook_success",
            "hookEvent": event,
            "command": command,
            "stdout": stdout,
            "content": "<persisted-output> preview only",
            "durationMs": "712",
        },
    }


def _session(tmp_path: Path, *, sid: str | None = None, base: datetime | None = None) -> Path:
    sid = sid or str(uuid.uuid4())
    base = base or datetime.now(UTC) - timedelta(minutes=5)
    user = {"type": "user", "message": {"role": "user", "content": "charge the pending ones"}}
    reply = {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Running submitForSettlement for ft2479 in billing/settle.py",
                }
            ],
        },
    }
    lines = [
        _line(sid, base, **user),
        _line(
            sid, base + timedelta(seconds=1), **_hook('python3 "$HOME/hafiz-recall.py"', INJECTION)
        ),
        _line(sid, base + timedelta(seconds=1), **_hook("echo unrelated", "not ours")),
        _line(sid, base + timedelta(seconds=1), **_hook("hafiz import claude-code", "")),
        _line(sid, base + timedelta(seconds=2), **reply),
    ]
    path = tmp_path / f"{sid}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_only_hafiz_hook_output_is_kept_as_a_system_turn(tmp_path: Path):
    parsed = parse_jsonl_file(_session(tmp_path))
    system = [m for m in parsed.messages if m.role == "system"]
    assert len(system) == 1  # unrelated hook and empty stdout are dropped
    assert system[0].content == INJECTION.strip()  # stdout, not the truncated preview
    assert system[0].metadata["kind"] == INJECTION_KIND
    assert system[0].metadata["hook_event"] == "UserPromptSubmit"
    assert system[0].metadata["duration_ms"] == 712


def test_injections_are_never_embedded():
    assert not should_embed_message(role="system", content=INJECTION * 20)


# ── End to end ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
async def _isolate_engine():
    await close_engine()
    yield
    await close_engine()


async def test_an_imported_injection_shows_up_as_used(tmp_path: Path):
    await import_claude_code(root=_session(tmp_path), resolve_project=False, embed=False)
    report = await delivery_report(since_days=1, limit=50)
    assert any("ft2479" in r["preview"] for r in report["most_used"])
    prompt_row = next(r for r in report["by_event"] if r["event"] == "UserPromptSubmit")
    assert prompt_row["likely_used"] >= 1
    assert report["method"].startswith("rough")


async def test_reimport_does_not_duplicate_injections(tmp_path: Path):
    path = _session(tmp_path)
    await import_claude_code(root=path, resolve_project=False, embed=False)
    before = (await delivery_report(since_days=1, limit=50))["injections"]
    await import_claude_code(root=path, resolve_project=False, embed=False)
    assert (await delivery_report(since_days=1, limit=50))["injections"] == before


def test_retrievals_json_carries_the_delivery_section():
    result = CliRunner().invoke(app, ["retrievals", "--since-days", "1", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "error" not in payload["delivery"], payload["delivery"]
    assert "by_event" in payload["delivery"]
