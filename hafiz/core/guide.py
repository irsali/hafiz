"""On-demand agent guide — the detail the always-loaded skills.md leaves out.

skills.md is spliced into an agent's instructions and paid for on every request, so it
carries only the rules an agent must always follow. Everything else ships as packaged
markdown under ``hafiz/data/agents/guide/<topic>.md`` and is read with ``hafiz guide``.
Each topic file starts with a ``<!-- summary: ... -->`` line used for the index.
"""

from __future__ import annotations

import difflib
import importlib.resources
import re

_SUMMARY_RE = re.compile(r"\A<!--\s*summary:\s*(.+?)\s*-->\n?")


class UnknownTopicError(LookupError):
    def __init__(self, name: str, candidates: list[str]):
        super().__init__(f"unknown guide topic {name!r}")
        self.candidates = candidates


def _guide_dir():
    return importlib.resources.files("hafiz.data.agents").joinpath("guide")


def _read(name: str) -> tuple[str, str]:
    """Return (summary, content) for a topic file, with the summary marker stripped."""
    raw = _guide_dir().joinpath(f"{name}.md").read_text(encoding="utf-8")
    match = _SUMMARY_RE.match(raw)
    return (match.group(1), raw[match.end() :]) if match else ("", raw)


def topic_names() -> list[str]:
    return sorted(p.name[:-3] for p in _guide_dir().iterdir() if p.name.endswith(".md"))


def list_topics() -> list[dict]:
    return [{"name": n, "summary": _read(n)[0]} for n in topic_names()]


def load_topic(name: str) -> dict:
    names = topic_names()
    if name not in names:
        close = difflib.get_close_matches(name, names, n=3, cutoff=0.5)
        raise UnknownTopicError(name, close or names)
    summary, content = _read(name)
    return {"topic": name, "summary": summary, "content": content}


def read_guide(topic: str | None = None) -> dict:
    """The guide as plain data: the topic index, one topic, or the closest names for a miss.

    An unknown name is a result, not an exception, so an agent (over MCP or `--json`)
    can correct a typo from the returned ``candidates``.
    """
    if topic is None:
        return {"ok": True, "topics": list_topics()}
    try:
        return {"ok": True, **load_topic(topic)}
    except UnknownTopicError as exc:
        return {"ok": False, "error": str(exc), "candidates": exc.candidates}
