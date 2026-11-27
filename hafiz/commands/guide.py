"""hafiz guide — print an on-demand section of the agent guide."""

from __future__ import annotations

import json

import typer

from hafiz.core.guide import UnknownTopicError, list_topics, load_topic


def run_guide(topic: str | None, *, output_json: bool = False) -> None:
    if topic is None:
        topics = list_topics()
        if output_json:
            typer.echo(json.dumps({"ok": True, "topics": topics}))
            return
        width = max(len(t["name"]) for t in topics)
        typer.echo("Read a topic with: hafiz guide <topic>\n")
        for t in topics:
            typer.echo(f"  {t['name']:<{width}}  {t['summary']}")
        return

    try:
        payload = load_topic(topic)
    except UnknownTopicError as exc:
        if output_json:
            body = {"ok": False, "error": str(exc), "candidates": exc.candidates}
            typer.echo(json.dumps(body))
        else:
            typer.echo(f"Error: {exc}. Try: hafiz guide {' | '.join(exc.candidates)}")
        raise typer.Exit(2) from None

    if output_json:
        typer.echo(json.dumps({"ok": True, **payload}))
    else:
        typer.echo(payload["content"], nl=False)
