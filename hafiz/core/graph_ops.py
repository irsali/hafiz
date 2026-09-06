"""One entry point onto the code graph, callable with plain JSON arguments.

``graph_analysis`` is written for callers that already hold a graph: every
function there takes an ``nx.MultiDiGraph`` as its first argument, because
the CLI loads the graph once and then runs several analyses against it. That
shape is right for the CLI and unusable from a registry whose arguments
arrive as JSON — no caller across a socket can pass a NetworkX object.

So this module owns the loading and the dispatch, and nothing else. It adds
no analysis of its own; every branch delegates to ``graph_analysis``, which
stays the single implementation.
"""

from __future__ import annotations

from typing import Any

from hafiz.core import graph_analysis as ga


class UnknownGraphOperationError(ValueError):
    """Raised for an unrecognised ``operation``, listing what is valid."""


OPERATIONS = ("show", "deps", "impact", "path", "rank", "stats")


async def graph_op(
    operation: str,
    name: str | None = None,
    target: str | None = None,
    project: str | None = None,
    depth: int = 1,
    metric: str = "pagerank",
    limit: int = 20,
) -> dict[str, Any]:
    """Run one graph operation and return a JSON-shaped result.

    Args:
        operation: One of :data:`OPERATIONS`.
        name: Unit name — required for show, deps, impact and path.
        target: Destination unit name — required for path.
        project: Restrict the graph to one project.
        depth: Hops to walk for deps/impact.
        metric: Centrality metric for rank.
        limit: Cap on returned rows.
    """
    if operation not in OPERATIONS:
        raise UnknownGraphOperationError(
            f"unknown graph operation {operation!r}; expected one of {', '.join(OPERATIONS)}"
        )

    graph, meta = await ga.get_cached_graph(project)

    if operation == "stats":
        return {"operation": operation, "stats": ga.graph_stats(graph, top_central=limit)}

    if operation == "rank":
        ranked = ga.rank_nodes(graph, metric=metric, top_n=limit)
        return {
            "operation": operation,
            "metric": metric,
            "ranked": [{"unit": n, "score": s} for n, s in ranked],
        }

    if name is None:
        raise ValueError(f"graph operation {operation!r} requires 'name'")

    matches = ga.find_nodes_by_name(graph, name, project=project)
    if not matches:
        return {"operation": operation, "name": name, "found": False, "matches": []}
    node = matches[0]

    if operation == "path":
        if target is None:
            raise ValueError("graph operation 'path' requires 'target'")
        target_matches = ga.find_nodes_by_name(graph, target, project=project)
        if not target_matches:
            return {"operation": operation, "name": name, "target": target, "found": False}
        route = ga.shortest_path_between(graph, node, target_matches[0])
        return {
            "operation": operation,
            "name": name,
            "target": target,
            "found": route is not None,
            "path": route or [],
        }

    # show / deps / impact all walk from one node; only direction differs.
    # "show" is both directions at depth 1 — the neighbourhood, not a lineage.
    direction = {"deps": "out", "impact": "in", "show": "both"}[operation]
    walked = ga.walk(graph, node, depth=1 if operation == "show" else depth, direction=direction)
    neighbours = [
        {"unit": other, "distance": distance, "edges": ga.edges_between(graph, node, other)}
        for other, distance in sorted(walked.items(), key=lambda kv: (kv[1], kv[0]))
        if other != node
    ][:limit]
    return {
        "operation": operation,
        "name": name,
        "found": True,
        "resolved": node,
        "ambiguous": matches[1:] if len(matches) > 1 else [],
        "neighbours": neighbours,
        "graph_built_at": getattr(meta, "built_at", None),
    }
