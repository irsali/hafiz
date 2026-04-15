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
| `hafiz query "<topic>" --recall --type decision --json` | Checking past decisions or known gotchas |
| `hafiz graph deps <name> --json` | Understanding what an entity depends on |
| `hafiz graph dependents <name> --json` | Assessing impact before changing an entity |
| `hafiz observe "<text>" --type <type> --source agent:<name>` | Recording decisions, warnings, patterns, learnings |

## Ingest Workflow

When the user asks to ingest, index, or re-index a codebase, you act as the
extraction engine — no external API key is needed. You ARE the brain.

**Step 1** — Chunk and embed files:
```bash
hafiz ingest <path> --project <name>
```

**Step 2** — Export chunks grouped by file for analysis:
```bash
hafiz extract export --unextracted --project <name> --limit 200
```
Output is grouped by `source_file` with chunks ordered by line number.
If `total` exceeds the batch, repeat with `--offset` to get all chunks.

**Step 3** — Two-phase extraction from the chunk content.

*Phase 1 — Entities (per file):* Read each file's chunks together. Identify
entities **defined** in that file. Entities are file-scoped — a class, function,
or config is defined in one file.

*Phase 2 — Relations (cross-file):* With all entities identified, find
relationships across files. Look at imports, function calls, type references,
inheritance. Relations are project-scoped — they cross file boundaries.

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
Only declare a relation if you can see both sides (the caller and the callee).

**Step 4** — Import results:
```bash
cat /tmp/hafiz_extraction.json | hafiz extract import --project <name>
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
| `hafiz query "<text>"` | Semantic search over indexed code and docs | `--type`, `--project`, `--workspace`, `--limit`, `--json` |
| `hafiz query "<text>" --recall` | Search observations (decisions, facts, learnings) | `--type`, `--project`, `--workspace`, `--limit`, `--json` |

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
| `hafiz ingest <path>` | Index files (chunk + embed + store) | `--project`, `--git-hook`, `--prune`, `--json` |
| `hafiz extract export` | Export chunks grouped by file as JSON | `--project`, `--unextracted`, `--path`, `--limit`, `--offset` |
| `hafiz extract import` | Import entity/relation extraction from JSON | `--file`, `--project` |
| `hafiz watch <path>` | Watch directory and re-index on change | `--project`, `--json` |
| `hafiz prune` | Remove chunks for deleted files | `--project`, `--dry-run`, `--json` |
| `hafiz status` | Database statistics and index health | `--json`, `--diagnose` |
| `hafiz review` | Review knowledge quality, suggest improvements | `--project`, `--json` |

### Type Values

- **Query types** (`--type` for `query`): `code`, `doc`, `note`, `decision`

</details>
