<!-- Installed by hafiz — workspace intelligence layer -->
# Hafiz — Workspace Intelligence

You have access to `hafiz`, a CLI tool that indexes the entire codebase into
PostgreSQL + pgvector. It provides semantic code search, an entity/relationship
graph, and an observations store for decisions and learnings.

Run from any directory: `hafiz <command>`. Always use `--json` when parsing output.

## When to Use Hafiz

| Situation | Action |
|-----------|--------|
| Before starting ANY task | `hafiz context "<task description>" --json` |
| Searching for code or answers | `hafiz query "<question>" --json` |
| Checking past decisions or gotchas | `hafiz recall "<topic>" --type decision --json` |
| Before refactoring an entity | `hafiz graph dependents <Name> --json` |
| After making architectural decisions | `hafiz observe "<what and why>" --type decision --source agent:<your-name>` |
| After discovering gotchas | `hafiz observe "<gotcha>" --type warning --source agent:<your-name>` |
| User asks to ingest/index code | Follow the **Ingest Workflow** below |

## Command Reference

### Search & Context

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz context "<task>"` | Full context bundle (chunks + graph + observations) | `--project`, `--json` |
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

**Observation types** (`--type`): `fact`, `decision`, `learning`, `pattern`, `warning`
**Source format** (`--source`): `agent:claude-code`, `agent:cursor`, `agent:copilot`, `user:<name>`

### Indexing & Maintenance

| Command | Purpose | Key Flags |
|---------|---------|-----------|
| `hafiz ingest <path>` | Index files (chunk + embed + extract) | `--project`, `--no-extract`, `--git-hook` |
| `hafiz chunks export` | Export indexed chunks as JSON | `--project`, `--path`, `--limit`, `--offset` |
| `hafiz extract import` | Import entity/relation extraction from JSON | `--file`, `--project` |
| `hafiz watch <path>` | Watch directory and re-index on change | `--project`, `--json` |
| `hafiz prune` | Remove chunks for deleted files | `--project`, `--dry-run`, `--json` |
| `hafiz status` | Database statistics and index health | `--json` |
| `hafiz doctor` | System diagnostics | `--json` |

### Query Type Values

- **Query types** (`--type` for `query`): `code`, `doc`, `note`, `decision`

## Workflows

### Starting a Task

```bash
hafiz context "<task description>" --json
hafiz recall "<related topic>" --type decision --json
```

Read the output, then begin implementation with full context.

### During Implementation

```bash
hafiz query "<specific question>" --type code --json
hafiz graph deps <EntityName> --json
hafiz graph dependents <EntityName> --json    # before refactoring — assess impact
```

### After Completing Work

```bash
hafiz observe "<what was decided and why>" --type decision --source agent:<your-name>
hafiz observe "<gotcha discovered>" --type warning --source agent:<your-name>
hafiz observe "<useful pattern>" --type pattern --source agent:<your-name>
```

## Ingest Workflow (Agent-Driven Extraction)

When the user asks to ingest, index, or re-index a codebase, follow these steps.
You act as the extraction engine — no external API key is needed.

### Step 1 — Chunk & Embed

```bash
hafiz ingest <path> --no-extract
```

This chunks files and generates embeddings locally. Report the chunk count.

### Step 2 — Export Chunks

```bash
hafiz chunks export --limit 200
```

If `total` in the output exceeds the batch, repeat with `--offset` to get remaining chunks.

### Step 3 — Extract Entities & Relationships

Analyse the exported chunks and produce a JSON object with this schema:

```json
{
  "entities": [
    {
      "name": "ExactNameFromCode",
      "entity_type": "<class|function|module|api_endpoint|database_table|concept|config|service>",
      "description": "Brief description of what this entity does",
      "source_file": "/absolute/path/to/file.py",
      "chunk_id": "uuid-from-chunks-export"
    }
  ],
  "relations": [
    {
      "source_name": "CallerEntity",
      "source_type": "function",
      "target_name": "CalleeEntity",
      "target_type": "function",
      "relation_type": "<calls|imports|inherits|depends_on|defines|reads|writes|configures|implements>",
      "evidence": "the_actual_code_snippet(proving_this)"
    }
  ]
}
```

**Extraction rules:**
- Only extract entities that are **defined** or **clearly referenced** in the code
- Use the **exact name** as it appears in the code (e.g. `MyClass`, `get_user`)
- For relations, provide the actual code snippet as `evidence`
- Do not invent entities or relationships not present in the code
- Set `chunk_id` to the chunk's ID from the export so entities link back to their source
- One entity per definition — pick the chunk where it is defined

### Step 4 — Import Results

Write the JSON to a temp file and import:

```bash
cat /tmp/hafiz_extraction.json | hafiz extract import
```

Report the entity and relation counts.

### Step 5 — Verify

```bash
hafiz status --json
```

Confirm entities and relations are populated.
