"""hafiz graph — explore the knowledge graph via NetworkX.

Commands:
    show    Entity and its N-hop neighborhood (undirected walk)
    deps    Transitive dependencies (outgoing walk)
    impact  Blast radius (incoming walk — what breaks if the entity changes)
    path    Shortest directed path between two entities
    rank    Top-N entities by a centrality metric
    stats   Overall graph health (counts, density, components, top-central nodes)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import networkx as nx
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from hafiz.core import graph_analysis as ga

console = Console()


# ── Shared helpers ──────────────────────────────────────────────────────────


def _node_payload(G: nx.MultiDiGraph, node_id: str) -> dict[str, Any]:
    """Serialize a node's attributes to a plain dict for JSON output."""
    attrs = G.nodes[node_id]
    return {
        "id": node_id,
        "name": attrs.get("name"),
        "kind": attrs.get("kind"),
        "parent_name": attrs.get("parent_name"),
        "project": attrs.get("project"),
        "source_file": attrs.get("source_file"),
    }


def _entity_label(G: nx.MultiDiGraph, node_id: str) -> str:
    """Rich-formatted one-line label for a unit node."""
    attrs = G.nodes[node_id]
    name = attrs.get("name") or "(unnamed)"
    kind = attrs.get("kind") or "?"
    return f"[bold]{name}[/bold] [dim]({kind})[/dim]"


def _resolve_or_exit(
    G: nx.MultiDiGraph,
    name: str,
    *,
    project: str | None,
    label: str = "entity",
) -> str:
    """Resolve `name` to a single node ID, or exit with an informative message.

    Ambiguity (multiple entities share the name) is reported; the first match is
    used so the user can still get output, but a warning is printed to stderr.
    """
    matches = ga.find_nodes_by_name(G, name, project=project)
    if not matches:
        console.print(
            f"[red]{label.capitalize()} not found:[/red] {name}"
            + (f" [dim](project: {project})[/dim]" if project else "")
        )
        raise SystemExit(1)
    if len(matches) > 1:
        alternatives = ", ".join(
            f"{G.nodes[n].get('name')} ({G.nodes[n].get('kind')})" for n in matches
        )
        console.print(
            f"[yellow]Warning: {len(matches)} units share the name "
            f"{name!r} — using the first. Candidates: {alternatives}[/yellow]",
            highlight=False,
        )
    return matches[0]


# ── show ────────────────────────────────────────────────────────────────────


def run_graph_show(
    name: str,
    *,
    depth: int = 1,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    """Show an entity and every node reachable within `depth` hops (undirected)."""
    G, _ = ga.get_cached_graph_sync(project=project)
    node = _resolve_or_exit(G, name, project=project)

    distances = ga.walk(G, node, depth=depth, direction="both")

    if output_json:
        payload = {
            "entity": _node_payload(G, node),
            "depth": depth,
            "neighbors": [
                {
                    **_node_payload(G, n),
                    "distance": d,
                }
                for n, d in sorted(distances.items(), key=lambda kv: (kv[1], kv[0]))
                if n != node
            ],
        }
        console.print_json(json.dumps(payload, default=str))
        return

    # Rich display
    console.print()
    attrs = G.nodes[node]
    info = f"[bold cyan]{attrs.get('name')}[/bold cyan] [dim]({attrs.get('kind')})[/dim]"
    if attrs.get("parent_name"):
        info += f"\n[dim]Parent: {attrs['parent_name']}[/dim]"
    if attrs.get("source_file"):
        info += f"\n[dim]Source: {attrs['source_file']}[/dim]"
    console.print(Panel(info, title="Unit", border_style="cyan"))

    neighbors = [(n, d) for n, d in distances.items() if n != node]
    if not neighbors:
        console.print("[dim]No connections within the requested depth.[/dim]")
        console.print()
        return

    # Group by distance
    by_dist: dict[int, list[str]] = {}
    for n, d in neighbors:
        by_dist.setdefault(d, []).append(n)

    tree = Tree(f"[bold]Neighborhood[/bold] (depth ≤ {depth}, {len(neighbors)} nodes)")
    for d in sorted(by_dist):
        branch = tree.add(f"[yellow]{d} hop{'s' if d != 1 else ''}[/yellow]")
        for n in sorted(by_dist[d], key=lambda x: (G.nodes[x].get("name") or "").lower()):
            branch.add(_entity_label(G, n))
    console.print(tree)
    console.print()


# ── deps / impact (directional walks) ───────────────────────────────────────


def _walk_and_render(
    name: str,
    *,
    direction: str,
    depth: int,
    project: str | None,
    output_json: bool,
    title_verb: str,
    empty_msg: str,
) -> None:
    """Shared implementation for deps (direction='out') and impact (direction='in')."""
    G, _ = ga.get_cached_graph_sync(project=project)
    node = _resolve_or_exit(G, name, project=project)

    distances = ga.walk(G, node, depth=depth, direction=direction)

    # Drop self, sort by distance then name
    reachable = [(n, d) for n, d in distances.items() if n != node]
    reachable.sort(key=lambda kv: (kv[1], (G.nodes[kv[0]].get("name") or "").lower()))

    if output_json:
        key = "dependencies" if direction == "out" else "dependents"
        payload = {
            "unit": _node_payload(G, node),
            "direction": direction,
            "depth": depth,
            key: [
                {
                    **_node_payload(G, n),
                    "distance": d,
                    "relations": _direct_relations(G, node, n, direction),
                }
                for n, d in reachable
            ],
        }
        console.print_json(json.dumps(payload, default=str))
        return

    console.print()
    if not reachable:
        console.print(f"[dim]{empty_msg}[/dim]")
        console.print()
        return

    unit_name = G.nodes[node].get("name")
    table = Table(
        title=f"{title_verb} {unit_name} (depth ≤ {depth}, {len(reachable)} results)",
        border_style="cyan",
    )
    table.add_column("Unit", style="bold")
    table.add_column("Kind", style="dim")
    table.add_column("Distance", justify="right")
    table.add_column("Direct relations", style="yellow")

    for n, d in reachable:
        rels = _direct_relations(G, node, n, direction)
        rel_cell = ", ".join(rels) if rels else "[dim]—[/dim]"
        table.add_row(
            G.nodes[n].get("name") or "(unnamed)",
            G.nodes[n].get("kind") or "?",
            str(d),
            rel_cell,
        )

    console.print(table)
    console.print()


def _direct_relations(
    G: nx.MultiDiGraph,
    source: str,
    other: str,
    direction: str,
) -> list[str]:
    """Return relation names of the DIRECT edge(s) between ``source`` and
    ``other``.

    For indirect paths (distance > 1) this returns [] — multi-hop
    relation labels are the ``path`` command's job.
    """
    if direction == "out":
        edges = ga.edges_between(G, source, other)
    elif direction == "in":
        edges = ga.edges_between(G, other, source)
    else:
        edges = ga.edges_between(G, source, other) + ga.edges_between(G, other, source)
    seen: set[str] = set()
    result: list[str] = []
    for e in edges:
        rel = e.get("relation")
        if rel and rel not in seen:
            seen.add(rel)
            result.append(rel)
    return result


def run_graph_deps(
    name: str,
    *,
    depth: int = 1,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    """Walk outgoing edges from `name` up to `depth` hops — things it depends on."""
    _walk_and_render(
        name,
        direction="out",
        depth=depth,
        project=project,
        output_json=output_json,
        title_verb="Dependencies of",
        empty_msg=f"{name} has no outgoing dependencies.",
    )


def run_graph_impact(
    name: str,
    *,
    depth: int = 1,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    """Walk incoming edges from `name` up to `depth` hops — blast radius."""
    _walk_and_render(
        name,
        direction="in",
        depth=depth,
        project=project,
        output_json=output_json,
        title_verb="Impact (dependents) of",
        empty_msg=f"Nothing depends on {name}.",
    )


# ── path ────────────────────────────────────────────────────────────────────


def run_graph_path(
    source_name: str,
    target_name: str,
    *,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    """Shortest directed path from `source_name` → `target_name`."""
    G, _ = ga.get_cached_graph_sync(project=project)
    source = _resolve_or_exit(G, source_name, project=project, label="source entity")
    target = _resolve_or_exit(G, target_name, project=project, label="target entity")

    path = ga.shortest_path_between(G, source, target)

    if output_json:
        if path is None:
            payload = {
                "source": _node_payload(G, source),
                "target": _node_payload(G, target),
                "path": None,
            }
        else:
            payload = {
                "source": _node_payload(G, source),
                "target": _node_payload(G, target),
                "length": len(path) - 1,
                "path": [
                    {
                        **_node_payload(G, path[i]),
                        "next_relations": (
                            _direct_relations(G, path[i], path[i + 1], "out")
                            if i + 1 < len(path)
                            else []
                        ),
                    }
                    for i in range(len(path))
                ],
            }
        console.print_json(json.dumps(payload, default=str))
        return

    console.print()
    if path is None:
        console.print(
            f"[yellow]No directed path found from "
            f"[bold]{source_name}[/bold] to [bold]{target_name}[/bold].[/yellow]"
        )
        console.print(
            "[dim]The `path` command walks directed edges only — "
            "try `graph show` for undirected neighborhoods.[/dim]"
        )
        console.print()
        return

    tree = Tree(f"[bold]Path[/bold] ({len(path) - 1} hop{'s' if len(path) - 1 != 1 else ''})")
    for i, nid in enumerate(path):
        label = _entity_label(G, nid)
        if i + 1 < len(path):
            rels = _direct_relations(G, nid, path[i + 1], "out")
            rel_str = f" [yellow]--({', '.join(rels)})-->[/yellow]" if rels else " -->"
            label = f"{label}{rel_str}"
        tree.add(label)
    console.print(tree)
    console.print()


# ── rank ────────────────────────────────────────────────────────────────────


def run_graph_rank(
    *,
    metric: str = "pagerank",
    top_n: int = 20,
    project: str | None = None,
    output_json: bool = False,
) -> None:
    """List the top-`top_n` most central entities by the chosen metric."""
    G, _ = ga.get_cached_graph_sync(project=project)
    if G.number_of_nodes() == 0:
        console.print("[yellow]No entities in graph.[/yellow]")
        return

    try:
        ranked = ga.rank_nodes(G, metric=metric, top_n=top_n)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc

    if output_json:
        payload = {
            "metric": metric,
            "top_n": top_n,
            "project": project,
            "results": [
                {
                    **_node_payload(G, nid),
                    "score": score,
                    "rank": idx + 1,
                }
                for idx, (nid, score) in enumerate(ranked)
            ],
        }
        console.print_json(json.dumps(payload, default=str))
        return

    console.print()
    scope = f" (project: {project})" if project else ""
    table = Table(
        title=f"Top {len(ranked)} by {metric}{scope}",
        border_style="cyan",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Unit", style="bold")
    table.add_column("Kind", style="dim")
    table.add_column("Project", style="dim")
    table.add_column("Score", justify="right")

    for idx, (nid, score) in enumerate(ranked, start=1):
        attrs = G.nodes[nid]
        table.add_row(
            str(idx),
            attrs.get("name") or "(unnamed)",
            attrs.get("kind") or "?",
            attrs.get("project") or "—",
            f"{score:.4f}",
        )
    console.print(table)
    console.print()


# ── stats ───────────────────────────────────────────────────────────────────


def run_graph_stats(
    *,
    project: str | None = None,
    top_central: int = 5,
    output_json: bool = False,
) -> None:
    """Show overall graph-level metrics for the (optionally project-scoped) graph."""
    G, meta = ga.get_cached_graph_sync(project=project)
    stats = ga.graph_stats(G, top_central=top_central)

    if output_json:
        payload = asdict(stats)
        payload["project"] = project
        payload["top_by_pagerank"] = [
            {**_node_payload(G, nid), "score": score} for nid, score in stats.top_by_pagerank
        ]
        payload["cache_built_at"] = meta.built_at.isoformat()
        console.print_json(json.dumps(payload, default=str))
        return

    console.print()
    scope = f" — project: [bold]{project}[/bold]" if project else ""
    summary = Table(title=f"Graph stats{scope}", show_header=False, border_style="cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Nodes", str(stats.node_count))
    summary.add_row("Edges", str(stats.edge_count))
    summary.add_row("Density", f"{stats.density:.4f}")
    summary.add_row("Weakly connected components", str(stats.weakly_connected_components))
    summary.add_row("Largest component size", str(stats.largest_component_size))
    summary.add_row("Isolated nodes", str(stats.isolated_nodes))
    console.print(summary)

    if stats.kind_counts:
        etable = Table(title="Unit kinds", border_style="cyan")
        etable.add_column("Kind")
        etable.add_column("Count", justify="right")
        for t, c in stats.kind_counts.items():
            etable.add_row(t, str(c))
        console.print()
        console.print(etable)

    if stats.relation_counts:
        rtable = Table(title="Edge relations", border_style="cyan")
        rtable.add_column("Relation")
        rtable.add_column("Count", justify="right")
        for t, c in stats.relation_counts.items():
            rtable.add_row(t, str(c))
        console.print()
        console.print(rtable)

    if stats.top_by_pagerank:
        ttable = Table(
            title=f"Top {len(stats.top_by_pagerank)} by PageRank",
            border_style="cyan",
        )
        ttable.add_column("Unit", style="bold")
        ttable.add_column("Kind", style="dim")
        ttable.add_column("Project", style="dim")
        ttable.add_column("Score", justify="right")
        for nid, score in stats.top_by_pagerank:
            attrs = G.nodes[nid]
            ttable.add_row(
                attrs.get("name") or "(unnamed)",
                attrs.get("kind") or "?",
                attrs.get("project") or "—",
                f"{score:.4f}",
            )
        console.print()
        console.print(ttable)

    console.print()
