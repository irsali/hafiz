<!-- Installed by hafiz — workspace intelligence layer -->
# Hafiz — Workspace Intelligence

IMPORTANT: You have access to `hafiz`, a CLI tool that indexes the entire
codebase with semantic search, an entity graph, and an observations store.
Always use `--json` when parsing output programmatically.

## Required Behaviors

You MUST follow these rules in every session:

1. **Before starting ANY coding task**, gather context first:
   ```bash
   hafiz context "<task description>" --json
   ```

2. **Before refactoring or modifying a class/function**, check what depends on it:
   ```bash
   hafiz graph dependents <EntityName> --json
   ```

3. **After making an architectural decision**, record it:
   ```bash
   hafiz observe "<what was decided and why>" --type decision --source agent:<your-name>
   ```

4. **After discovering a gotcha or non-obvious behavior**, record it:
   ```bash
   hafiz observe "<the gotcha>" --type warning --source agent:<your-name>
   ```

## Core Commands

| Command | When to use |
|---------|-------------|
| `hafiz context "<task>"` | **First thing** — synthesizes code, graph, and past observations |
| `hafiz query "<text>" --json` | Searching for specific code or answers |
| `hafiz recall "<topic>" --type decision --json` | Checking past decisions or known gotchas |
| `hafiz graph deps <name> --json` | Understanding what an entity depends on |
| `hafiz graph dependents <name> --json` | Assessing impact before changing an entity |
| `hafiz observe "<text>" --type <type> --source agent:<name>` | Recording decisions, warnings, patterns, learnings |

## Ingest Workflow

When the user asks to ingest, index, or re-index a codebase, you act as the
extraction engine — no external API key is needed.

**Step 1** — Chunk and embed files:
```bash
hafiz ingest <path> --no-extract
```

**Step 2** — Export chunks for analysis:
```bash
hafiz chunks export --limit 200
```
If `total` exceeds the batch, repeat with `--offset` to get all chunks.

**Step 3** — Extract entities and relationships from the chunk content.
Produce a JSON object with this schema:

```json
{
  "entities": [
    {
      "name": "ExactNameFromCode",
      "entity_type": "<class|function|module|api_endpoint|database_table|concept|config|service>",
      "description": "Brief description",
      "source_file": "/absolute/path/to/file.py",
      "chunk_id": "uuid-from-chunks-export"
    }
  ],
  "relations": [
    {
      "source_name": "Caller",
      "source_type": "function",
      "target_name": "Callee",
      "target_type": "function",
      "relation_type": "<calls|imports|inherits|depends_on|defines|reads|writes|configures|implements>",
      "evidence": "actual_code_snippet(proving_this)"
    }
  ]
}
```

Rules: use exact names from code, provide real code as evidence, do not invent
entities or relations, set `chunk_id` from the export, one entity per definition.

**Step 4** — Import results:
```bash
cat /tmp/hafiz_extraction.json | hafiz extract import
```

**Step 5** — Verify:
```bash
hafiz status --json
```

---

## Reference

<details>
<summary>Full command reference and flags</summary>

### Search & Context

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz context "<task>"` | Full context bundle (chunks + graph + observations) | `--project`, `--workspace`, `--json` |
| `hafiz query "<text>"` | Semantic search over indexed code and docs | `--type`, `--project`, `--limit`, `--json` |
| `hafiz recall "<query>"` | Search observations only | `--type`, `--project`, `--limit`, `--json` |

### Knowledge Graph

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz graph show <name>` | Entity and its direct connections | `--project`, `--json` |
| `hafiz graph deps <name>` | What this entity depends on (outgoing) | `--project`, `--json` |
| `hafiz graph dependents <name>` | What depends on this entity (incoming) | `--project`, `--json` |

### Observations

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz observe "<text>"` | Store a decision, fact, or learning | `--type`, `--source`, `--project`, `--tags`, `--confidence`, `--json` |

- **Observation types**: `fact`, `decision`, `learning`, `pattern`, `warning`
- **Source format**: `agent:claude-code`, `agent:cursor`, `agent:copilot`, `user:<name>`

### Indexing & Maintenance

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz ingest <path>` | Index files (chunk + embed + extract) | `--project`, `--no-extract`, `--git-hook`, `--json` |
| `hafiz chunks export` | Export indexed chunks as JSON | `--project`, `--path`, `--limit`, `--offset` |
| `hafiz extract import` | Import entity/relation extraction from JSON | `--file`, `--project` |
| `hafiz watch <path>` | Watch directory and re-index on change | `--project`, `--json` |
| `hafiz prune` | Remove chunks for deleted files | `--project`, `--dry-run`, `--json` |
| `hafiz status` | Database statistics and index health | `--json` |
| `hafiz doctor` | System diagnostics | `--json` |
| `hafiz review` | Review knowledge quality, suggest improvements | `--project`, `--json` |

### Type Values

- **Query types** (`--type` for `query`): `code`, `doc`, `note`, `decision`

</details>
