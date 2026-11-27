<!-- summary: Storage layers, ownership and data model -->
# hafiz guide model — Storage layers, ownership and data model

## Two storage layers (load-bearing)

Hafiz separates **knowledge** (curated, identity-stable, mid-volume)
from **source** (firehose, immutable, retention-bounded). They live
side-by-side; agents must treat them differently.

| Layer | Tables | Examples | Default visibility |
|---|---|---|---|
| **Knowledge** | `units` · `unit_revisions` · `embeddings` · `annotations` · `files` · `commits` | docs, files, decisions, learnings, patterns, runbooks | surfaced by default |
| **Source** | `sessions` · `communications` · `communication_messages` · `annotation_targets` · `retrievals` | imported agent transcripts, recorded searches, future events | **hidden by default; opt-in via `hafiz recall` or `--include-transcripts`** |

**Rule of thumb:** the wisdom layer is primary. Source rows surface
only when you explicitly ask for them. Don't dilute default queries
with raw transcripts.

## Ownership rule (load-bearing)

Parsers own indexed content. Agents own meaning.

- **Indexed content** — Markdown/prose is split into `doc.heading` /
  `doc.paragraph` units; everything else (including code) is one
  `file.raw` unit per file. Produced by parsers at ingest time. You
  do not write these.
- **Meaning** — decisions, learnings, patterns, warnings, concepts,
  workarounds. Produced by you, via `hafiz observe` or `hafiz note`.

Units are namespaced by kind (`doc.heading`, `doc.paragraph`,
`file.raw`, `mail.message`, …). Note that `chat.turn` is no longer a
unit kind — agent transcripts live in the source layer
(`communications` + `communication_messages`). There is no `code`
domain any more.

Annotation kinds: `fact` · `decision` · `learning` · `pattern` ·
`warning` · `note` · `concept` · `service`.

### Data model — knowledge layer (six tables)

- `files` — one row per file ever seen (tombstoned via `valid_until`).
- `units` — stable identity of an addressable thing (heading, whole file, …). `kind` is namespaced.
- `unit_revisions` — append-only versioned body; at most one `superseded_at IS NULL` per unit.
- `embeddings` — 1:N vector search index over revisions (oversized bodies split into parts).
- `annotations` — decisions / facts / learnings. May link to a unit or float free. ``session_id`` is now a uuid FK to ``sessions``; the historical text slug is preserved as ``legacy_session_id``.
- `commits` — git axis; populated on ingest. `rewritten_at` marks orphaned commits.

### Data model — source layer

- `sessions` — engineer/agent threads of work. ``slug`` is the human-facing identifier; ``id`` is the canonical uuid that other tables FK against.
- `communications` — agent transcripts, chat threads. Idempotent by ``(agent, external_id)``. ``retention_until`` defaults to ``started_at + 90 days``.
- `communication_messages` — append-only turns. ``content`` is canonical (NOT NULL); ``embedding`` is nullable and populated only when the message clears the selective-embed policy. ``seq`` is monotonic per-communication.
- `annotation_targets` — polymorphic pivot. An annotation may cite units, other annotations, messages, communications, or sessions via ``target_kind`` + ``target_id`` + ``relation``.
- `retrievals` — append-only record of every search: ``query_text``, ``result_ids``, ``n_results``, ``top_score``, ``reranked``, ``filters``. Retention-bounded like ``communications``; read it with ``hafiz retrievals``. Rows with ``n_results = 0`` are the store's blind spots.
