Ingest files into the Hafiz knowledge base with full graph extraction. Path: $ARGUMENTS (default: current directory)

## Step 1 — Chunk & Embed

Run the ingest pipeline without API-based extraction:

```bash
hafiz ingest ${ARGUMENTS:-.} --no-extract
```

Report the chunk count from the output.

## Step 2 — Export chunks for extraction

Fetch the indexed chunks as JSON so you can analyse them:

```bash
hafiz chunks export --limit 200
```

If `total` exceeds the batch, repeat with `--offset` to get remaining chunks.

## Step 3 — Extract entities & relationships

You ARE the extraction engine. Analyse every chunk and produce a single JSON object with this exact schema:

```json
{
  "entities": [
    {
      "name": "ExactNameFromCode",
      "entity_type": "<one of: class, function, module, api_endpoint, database_table, concept, config, service>",
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
      "relation_type": "<one of: calls, imports, inherits, depends_on, defines, reads, writes, configures, implements>",
      "evidence": "the_code_snippet(proving_this)"
    }
  ]
}
```

### Extraction rules
- Only extract entities that are **defined** or **clearly referenced** in the code.
- Use the **exact name** as it appears in the code (e.g. `MyClass`, `get_user`).
- For relations, provide the actual code snippet as `evidence`.
- Be precise — do not invent entities or relationships not in the code.
- If a chunk has no meaningful entities, skip it.
- Set `chunk_id` to the chunk's `chunk_id` from the export so entities link back to their source chunk.
- Prefer **one entity per definition** — don't duplicate an entity across chunks; pick the chunk where it is defined.

## Step 4 — Store results

Write the JSON to a temp file and import it:

```bash
cat /tmp/hafiz_extraction.json | hafiz extract import
```

Report the entity and relation counts from the output.

## Step 5 — Verify

Run `hafiz status --json` and confirm entities and relations are populated.
