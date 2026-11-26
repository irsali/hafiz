"""Codex rollout record shapes beyond the basic fixture in test_importers.py.

Shapes follow community references to the rollout JSONL (OpenAI documents
AGENTS.md, notify and hooks but not the rollout schema itself); see the
``hafiz.core.importers.codex`` docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

from hafiz.core.importers.codex import parse_rollout_file


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def _msg(role: str, kind: str, text: str) -> dict:
    return {
        "type": "response_item",
        "payload": {"type": "message", "role": role, "content": [{"type": kind, "text": text}]},
    }


def test_developer_messages_are_system_not_assistant(tmp_path: Path):
    """Codex injects permission/agent-role prompts as ``developer`` messages."""
    path = _write(
        tmp_path / "rollout-a.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "sid-dev"}},
            _msg("developer", "input_text", "<permissions instructions>"),
            _msg("assistant", "output_text", "done"),
        ],
    )
    assert [m.role for m in parse_rollout_file(path).messages] == ["system", "assistant"]


def test_reinjected_agents_md_preamble_is_dropped(tmp_path: Path):
    path = _write(
        tmp_path / "rollout-b.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "sid-agents"}},
            _msg("user", "input_text", "# AGENTS.md instructions for /repo\n\n..."),
            _msg("user", "input_text", "fix the bug"),
        ],
    )
    assert [m.content for m in parse_rollout_file(path).messages] == ["fix the bug"]


def test_custom_tool_call_keeps_its_input(tmp_path: Path):
    path = _write(
        tmp_path / "rollout-c.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "sid-ct"}},
            {
                "type": "response_item",
                "payload": {
                    "type": "custom_tool_call",
                    "name": "exec",
                    "input": "ls -la",
                    "call_id": "c1",
                },
            },
        ],
    )
    call = parse_rollout_file(path).messages[0].tool_calls[0]
    assert call["name"] == "exec" and call["input"] == "ls -la"


def test_event_msg_mirrors_are_not_double_counted(tmp_path: Path):
    """``event_msg`` user/agent messages mirror response_item messages."""
    path = _write(
        tmp_path / "rollout-d.jsonl",
        [
            {"type": "session_meta", "payload": {"id": "sid-mirror"}},
            {"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}},
            _msg("user", "input_text", "hi"),
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "yo"}},
            _msg("assistant", "output_text", "yo"),
        ],
    )
    assert [m.content for m in parse_rollout_file(path).messages] == ["hi", "yo"]
