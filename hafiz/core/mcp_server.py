"""stdio MCP server exposing the tools declared in :mod:`hafiz.core.mcp_registry`.

Transport is stdio only — no HTTP, no SSE, no listening socket. The server is
spawned by the client, talks over the pipe it was given, and dies with it.
That is a different threat model from the always-warm auto-spawning daemon,
which is why the no-TCP rule there does not force a different answer here.

**stdout is the protocol channel.** Anything written to it that is not a
JSON-RPC message corrupts the stream, and the failure looks like a client-side
parse error a long way from its cause. So this module routes logging to
stderr explicitly rather than trusting whatever the ambient logging config
happens to be, and no code path here prints.

The ``mcp`` SDK is an optional extra. It is imported inside functions rather
than at module scope so that ``hafiz mcp`` can fail with an actionable
message, and so the registry's drift tests run on an install without it.
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
from datetime import datetime
from typing import Any

from hafiz.core.mcp_registry import TOOLS, ToolSpec, by_name, to_jsonable

logger = logging.getLogger(__name__)

SERVER_NAME = "hafiz"

INSTRUCTIONS = """\
Hafiz is the user's sovereign second brain: a local store of code structure, \
documents, and — most importantly — accumulated decisions, learnings, patterns \
and warnings.

Two habits make it useful rather than decorative:

1. Call `hafiz_context` BEFORE starting a task. It returns prior decisions on \
the topic, so you do not re-litigate something already settled.
2. Call `hafiz_observe` AFTER deciding an approach or discovering a gotcha. \
Nothing else preserves it once the conversation ends.

`hafiz_recall_observations` searches curated belief; `hafiz_query` searches \
indexed content; `hafiz_recall_session` searches raw transcripts. They are \
different corpora — pick deliberately.

Destructive and administrative operations are deliberately absent from this \
surface; they live on the `hafiz` CLI, where a human is present.\
"""


class MissingDependencyError(RuntimeError):
    """Raised when the optional ``mcp`` extra is not installed."""


def _require_sdk():
    try:
        import mcp.server as sdk
        import mcp.types as types
        from mcp.server.stdio import stdio_server
    except ImportError as e:  # pragma: no cover — exercised by hand, not in CI
        raise MissingDependencyError(
            "The MCP server needs the optional 'mcp' extra.\n\n"
            "  pipx inject hafiz mcp          # if hafiz was installed with pipx\n"
            "  pip install 'hafiz[mcp]'       # otherwise\n\n"
            "It is optional because the SDK pulls a full HTTP server stack "
            "(starlette, uvicorn, cryptography) that a stdio-only server never uses."
        ) from e
    return types, sdk.Server, stdio_server


def _coerce(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    """Turn JSON argument values into what the core function expects.

    Only datetimes need real work — JSON has no date type, so a client sends
    an ISO string where the signature wants a ``datetime``. Unknown keys are
    dropped rather than forwarded: a newer client talking to an older hafiz
    should degrade to "that filter was ignored", not ``TypeError``. That is
    the same contract ``daemon._supported_kwargs`` provides on the warm path.
    """
    exposed = {p.name: p for p in spec.exposed()}
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        param = exposed.get(key)
        if param is None or value is None:
            continue
        annotation = param.annotation if isinstance(param.annotation, str) else ""
        if "datetime" in annotation and isinstance(value, str):
            out[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            out[key] = value
    return out


async def call_tool(name: str, arguments: dict[str, Any], *, client: str | None = None) -> Any:
    """Run one tool and return its JSON-shaped result.

    Separated from the protocol layer so the whole surface is testable without
    standing up a transport.
    """
    spec = by_name().get(name)
    if spec is None:
        raise KeyError(f"unknown tool {name!r}")

    kwargs = {**_coerce(spec, arguments or {}), **spec.force}

    # Attribution. A write with source "agent:mcp" tells the audit trail
    # nothing about who actually wrote it, and every MCP client would look
    # identical in the store. The connecting client's own name is the only
    # honest default available.
    if spec.writes and "source" in {p.name for p in spec.exposed()}:
        if not kwargs.get("source") and client:
            kwargs["source"] = f"agent:{client}"

    fn = spec.resolve()
    result = fn(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return to_jsonable(result)


def client_name(ctx) -> str | None:
    """Name of the connected client, from what it sent at ``initialize``.

    Read off ``ctx.session.client_params.client_info.name``. Wrapped
    defensively because it is a convenience, not a contract: the path is
    internal to the SDK, and a rename there must degrade to an unattributed
    write rather than a failed tool call.
    """
    try:
        params = ctx.session.client_params
        return getattr(getattr(params, "client_info", None), "name", None)
    except Exception:  # noqa: BLE001 — attribution must never break a call
        return None


def build_server():
    """Construct the MCP ``Server`` with the registry wired into it."""
    types, server_cls, _stdio = _require_sdk()

    def _tool(spec: ToolSpec):
        return types.Tool(
            name=spec.name,
            description=spec.summary,
            input_schema=spec.input_schema(),
            annotations=types.ToolAnnotations(
                read_only_hint=not spec.writes,
                destructive_hint=False,
                idempotent_hint=not spec.writes,
            ),
        )

    async def on_list_tools(_ctx, _params):
        return types.ListToolsResult(tools=[_tool(spec) for spec in TOOLS])

    async def on_call_tool(ctx, params):
        client = client_name(ctx)
        try:
            payload = await call_tool(params.name, params.arguments or {}, client=client)
        except Exception as e:  # noqa: BLE001 — a bad call must not kill the server
            logger.exception("tool %r failed", params.name)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"{type(e).__name__}: {e}")],
                is_error=True,
            )
        return types.CallToolResult(
            content=[
                types.TextContent(type="text", text=json.dumps(payload, indent=2, default=str))
            ],
            structured_content={"result": payload},
        )

    return server_cls(
        SERVER_NAME,
        version=_hafiz_version(),
        instructions=INSTRUCTIONS,
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


def _hafiz_version() -> str:
    try:
        from importlib.metadata import version

        return version("hafiz")
    except Exception:  # noqa: BLE001
        return "0"


async def serve_stdio() -> None:
    """Run the server over stdin/stdout until the client disconnects."""
    _types, _server_cls, stdio_server = _require_sdk()

    # Protocol hygiene: stdout belongs to JSON-RPC. Force every handler onto
    # stderr before anything can log, and do it by replacing handlers rather
    # than adding one, so an inherited stdout handler cannot survive.
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(stderr_handler)

    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
