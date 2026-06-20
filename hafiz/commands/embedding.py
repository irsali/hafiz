"""hafiz embedding — inspect and retry the embedding device selection."""

from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from hafiz.core import device_state as dstate
from hafiz.core import embeddings
from hafiz.core.config import get_settings

console = Console()


def _describe(state: dstate.DeviceState | None, configured: str) -> dict:
    """Machine-readable summary of the current device state + its provenance."""
    if configured in ("cpu", "gpu"):
        source = "config"
        effective = configured
    elif state is not None:
        source = "sticky"
        effective = state.device
    else:
        source = "not-probed"
        effective = "unknown"

    return {
        "configured": configured,
        "source": source,
        "effective_device": effective,
        "sticky": {
            "device": state.device,
            "reason": state.reason,
            "reason_category": state.reason_category,
            "probed_at": state.probed_at,
            "onnxruntime_version": state.onnxruntime_version,
            "gpu_name": state.gpu_name,
            "stale": dstate.is_stale(state),
        }
        if state
        else None,
    }


def run_embedding_status(*, output_json: bool = False) -> None:
    """Show the currently selected embedding device and its provenance."""
    settings = get_settings()
    state = dstate.load_state()
    info = _describe(state, settings.embedding.device)

    if output_json:
        console.print_json(json.dumps(info))
        return

    table = Table(title="Embedding device", show_header=False, border_style="cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("config (embedding.device)", info["configured"])
    table.add_row("source", info["source"])
    table.add_row("effective device", info["effective_device"])

    if state is not None:
        table.add_row("", "")
        table.add_row("sticky device", state.device)
        table.add_row("sticky reason", state.reason or "(GPU probe clean)")
        if state.reason_category:
            table.add_row("reason category", state.reason_category)
        table.add_row("probed at", state.probed_at)
        if state.gpu_name:
            table.add_row("gpu", state.gpu_name)
        if state.onnxruntime_version:
            table.add_row("onnxruntime", state.onnxruntime_version)
        if dstate.is_stale(state):
            table.add_row(
                "stale",
                "[yellow]yes — ORT version changed; will reprobe on next use[/yellow]",
            )

    console.print()
    console.print(table)
    console.print()
    console.print(
        "[dim]Override:  set embedding.device in hafiz.toml or "
        "HAFIZ_EMBEDDING__DEVICE=[auto|cpu|gpu][/dim]"
    )
    console.print("[dim]Retry:     hafiz embedding retry[/dim]")
    console.print()


def run_embedding_retry(*, output_json: bool = False) -> None:
    """Clear sticky state and re-probe the embedding device."""
    settings = get_settings()
    configured = settings.embedding.device
    existed = dstate.clear_state()
    embeddings.reset_cache()

    if configured in ("cpu", "gpu"):
        result = {
            "ok": True,
            "device": configured,
            "cleared_prior": existed,
            "message": (
                f"embedding.device = '{configured}' is set explicitly — "
                "sticky state is not consulted in this mode."
            ),
        }
        if output_json:
            console.print_json(json.dumps(result))
            return
        console.print()
        console.print(
            f"[yellow]Config sets device = '{configured}' explicitly.[/yellow] "
            "Sticky state is not consulted in this mode."
        )
        if existed:
            console.print("[dim]Prior sticky state was cleared.[/dim]")
        console.print("[dim]Set embedding.device = 'auto' to re-enable sticky probing.[/dim]")
        console.print()
        return

    try:
        _model, state = embeddings.probe_device("auto", settings.embedding.model, persist=True)
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "cleared_prior": existed}
        if output_json:
            console.print_json(json.dumps(result))
            raise SystemExit(1)
        console.print(f"[red]Probe failed:[/red] {exc}")
        raise SystemExit(1)

    result = {
        "ok": True,
        "cleared_prior": existed,
        "device": state.device,
        "reason": state.reason,
        "reason_category": state.reason_category,
        "gpu_name": state.gpu_name,
        "probed_at": state.probed_at,
    }

    if output_json:
        console.print_json(json.dumps(result))
        return

    console.print()
    if state.device == "gpu":
        console.print(
            f"[green]GPU probe succeeded[/green] — using [cyan]{state.gpu_name or 'CUDA'}[/cyan]."
        )
    else:
        console.print(f"[yellow]Using CPU.[/yellow] {state.reason or ''}")
    if existed:
        console.print("[dim]Prior sticky state was cleared.[/dim]")
    console.print()
