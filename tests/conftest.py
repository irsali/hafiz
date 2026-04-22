"""Pytest configuration.

A handful of test files exercise modules that still depend on the old
chunks/entities/relations/observations schema. Those modules will be
rewired as the structural-grounding work continues (see
workitems/active/structural-grounding.md). Until then, their tests
can't even be collected — quarantined below.

Remove an entry from ``collect_ignore`` when its module is rewired.
When the list is empty, delete this file.
"""

collect_ignore = [
    # Uses the old chunker API (ChunkResult, chunk_file, LANGUAGE_MAP). The
    # new chunker is walk_files + prepare_embedding_parts; a fresh test
    # module will replace this.
    "test_chunker.py",
    # hafiz/core/capture.py still imports ChunkResult from chunker and
    # Chunk from database. Un-quarantine when capture is rewired.
    "test_capture.py",
    # hafiz/core/context.py still imports from the old observations module
    # and indirectly from graph_analysis → Entity/Relation.
    "test_context_graph.py",
    # hafiz/core/graph_analysis.py uses Entity/Relation at module scope.
    # Un-quarantine when Phase 4 (edge resolver + graph) rewires it.
    "test_graph_analysis.py",
]
