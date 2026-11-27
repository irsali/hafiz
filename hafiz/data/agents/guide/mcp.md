<!-- summary: Using hafiz over MCP when you cannot shell out -->
# hafiz guide mcp — Using hafiz over MCP when you cannot shell out

## If you cannot shell out: the MCP server

Everything above assumes you can run `hafiz`. If you cannot — a chat
surface, a desktop or hosted client with no shell — the same knowledge
layer is served over MCP:

```bash
pipx install "hafiz[mcp]"    # the extra is not installed by default
hafiz mcp                    # stdio server; no port, no listener
```

Client config is one entry, `command: hafiz`, `args: ["mcp"]`.

**Use it exactly as you would the CLI.** The tools take structured
parameters instead of flags, so `hafiz_query(observations=true,
kind="decision")` is `query --observations --type decision`. Every
**Required Behavior** above still applies unchanged: pull context when
a task needs prior docs or decisions, and record decisions and gotchas
as you make them.

Four things worth knowing, because they are deliberate:

- **Read and write are both available** — `hafiz_observe` and
  `hafiz_note` are there, so the memory layer is not read-only over MCP.
- **Install, destructive and long-running commands are absent.** No
  `init`, `hooks`, `agent`, `config set`, `forget`, `prune`, `ingest`,
  `import`. An MCP client calls tools without asking anyone, and those
  belong where a human is present. Reach for the CLI, or ask the user.
- **Your writes are attributed to you automatically.** The server reads
  your client identity from the MCP handshake, so an observation with no
  explicit `--source` records `agent:<your-client>` rather than a
  uniform `agent:mcp`. You may still pass a source explicitly.
- **This guide is a tool too.** `hafiz_guide()` lists the topics and
  `hafiz_guide(topic="distill")` returns one, so the details the CLI
  agents read with `hafiz guide` are yours as well.

The tool list is **not duplicated here on purpose** — it is generated
from the same core functions the CLI calls, and a hand-copied list in
this file would drift from it silently. `tools/list` is authoritative;
`docs/commands.md` carries the prose reference.
