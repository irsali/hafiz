"""hafiz guide — print an on-demand section of the agent guide."""

from __future__ import annotations

import json

import typer

from hafiz.core.guide import read_guide


def run_guide(topic: str | None, *, output_json: bool = False) -> None:
    payload = read_guide(topic)

    if output_json:
        typer.echo(json.dumps(payload))
    elif not payload["ok"]:
        candidates = " | ".join(payload["candidates"])
        typer.echo(f"Error: {payload['error']}. Try: hafiz guide {candidates}")
    elif topic is None:
        topics = payload["topics"]
        width = max(len(t["name"]) for t in topics)
        typer.echo("Read a topic with: hafiz guide <topic>\n")
        for t in topics:
            typer.echo(f"  {t['name']:<{width}}  {t['summary']}")
    else:
        typer.echo(payload["content"], nl=False)

    if not payload["ok"]:
        raise typer.Exit(2)
