"""Tests for `hafiz guide` and the lean skills.md contract it backs."""

import json
import re

from typer.testing import CliRunner

from hafiz.cli import app
from hafiz.core.agents import load_skills_content
from hafiz.core.guide import list_topics

runner = CliRunner()

# skills.md is spliced into every request of every installed agent. Raising this budget is a
# product decision, not a test fix — move reference material into a guide topic instead.
SKILLS_CHAR_BUDGET = 8000


def test_guide_lists_topics_as_json():
    result = runner.invoke(app, ["guide", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    names = [t["name"] for t in payload["topics"]]
    assert "distill" in names and "reference" in names
    assert all(t["summary"] for t in payload["topics"])


def test_guide_without_topic_prints_index():
    result = runner.invoke(app, ["guide"])
    assert result.exit_code == 0
    assert "retrieval" in result.stdout and "hafiz guide <topic>" in result.stdout


def test_guide_topic_prints_markdown():
    result = runner.invoke(app, ["guide", "distill"])
    assert result.exit_code == 0
    assert result.stdout.startswith("# hafiz guide distill")
    assert "<!-- summary:" not in result.stdout


def test_guide_topic_json_shape():
    result = runner.invoke(app, ["guide", "retrieval", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["topic"] == "retrieval"
    assert payload["summary"]
    assert "--min-score" in payload["content"]


def test_guide_unknown_topic_exits_2_with_candidates():
    result = runner.invoke(app, ["guide", "distil", "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["candidates"][0] == "distill"


def test_guide_unknown_topic_human_output():
    result = runner.invoke(app, ["guide", "nonsense"])
    assert result.exit_code == 2
    assert "hafiz guide" in result.stdout


def test_skills_topic_table_matches_shipped_topics():
    """Every topic skills.md points to must exist, and every shipped topic must be indexed."""
    skills = load_skills_content()
    table_topics = set(re.findall(r"^\| `([a-z]+)` \|", skills, flags=re.MULTILINE))
    assert table_topics == {t["name"] for t in list_topics()}


def test_skills_core_stays_lean():
    assert len(load_skills_content()) <= SKILLS_CHAR_BUDGET
