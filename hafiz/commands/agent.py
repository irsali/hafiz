"""hafiz agent — install/uninstall/list agent configurations."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from hafiz.core.agents import (
    AGENTS,
    install_file,
    is_hafiz_managed,
    load_skills_content,
    resolve_target,
    uninstall_file,
)

console = Console()


def run_agent_install(
    name: str | None = None,
    *,
    local: bool = False,
    path_override: str | None = None,
    file_override: str | None = None,
) -> None:
    """Install hafiz skills for an agent."""
    try:
        target, agent = resolve_target(
            name, local=local, path_override=path_override, file_override=file_override
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    display_name = agent.display_name if agent else (name or "custom agent")
    console.print(f"Installing hafiz skills for [bold]{display_name}[/bold]...")

    # Load and optionally transform content
    content = load_skills_content()
    if agent and agent.wrapper:
        content = agent.wrapper(content)

    status = install_file(target, content)

    icons = {
        "created": "[green]+[/green]",
        "updated": "[yellow]~[/yellow]",
        "skipped": "[dim]-[/dim]",
    }
    console.print(f"  {icons.get(status, '?')} {target}  [dim]({status})[/dim]")

    if status == "skipped":
        console.print(
            "\n[yellow]File exists and was not installed by hafiz — skipping to avoid overwrite.[/yellow]"
            "\nDelete it manually or use a different --file name."
        )
    else:
        console.print(f"\n[green]Done.[/green] {display_name} is configured to use hafiz.")


def run_agent_uninstall(
    name: str | None = None,
    *,
    local: bool = False,
    path_override: str | None = None,
    file_override: str | None = None,
    force: bool = False,
) -> None:
    """Uninstall hafiz skills for an agent."""
    try:
        target, agent = resolve_target(
            name, local=local, path_override=path_override, file_override=file_override
        )
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    display_name = agent.display_name if agent else (name or "custom agent")
    console.print(f"Uninstalling hafiz skills for [bold]{display_name}[/bold]...")

    status = uninstall_file(target, force=force)

    icons = {
        "removed": "[red]x[/red]",
        "skipped": "[yellow]![/yellow]",
        "not_found": "[dim]-[/dim]",
    }
    console.print(f"  {icons.get(status, '?')} {target}  [dim]({status})[/dim]")

    if status == "skipped":
        console.print(
            "\n[yellow]File was not installed by hafiz — skipping.[/yellow]"
            "\nUse --force to remove it anyway."
        )
    elif status == "removed":
        console.print(f"\n[green]Done.[/green] Hafiz skills removed for {display_name}.")


def run_agent_list() -> None:
    """List available agents and their installation status."""
    table = Table(title="Hafiz Agent Integration", border_style="cyan")
    table.add_column("Agent", style="bold")
    table.add_column("Global", justify="center")
    table.add_column("Local", justify="center")

    for agent in AGENTS.values():
        # Check global status
        if agent.supports_global:
            global_path = Path(agent.instructions["global"]).expanduser()
            if is_hafiz_managed(global_path):
                global_str = f"[green]installed[/green] [dim]({global_path})[/dim]"
            else:
                global_str = f"[dim]available[/dim] [dim]({global_path})[/dim]"
        else:
            global_str = "[dim]n/a[/dim]"

        # Check local status
        if agent.supports_local:
            local_path = Path.cwd() / agent.instructions["local"]
            if is_hafiz_managed(local_path):
                local_str = f"[green]installed[/green] [dim]({agent.instructions['local']})[/dim]"
            else:
                local_str = f"[dim]available[/dim] [dim]({agent.instructions['local']})[/dim]"
        else:
            local_str = "[dim]n/a[/dim]"

        table.add_row(agent.display_name, global_str, local_str)

    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Install: hafiz agent install <name> [--local] [--path PATH] [--file FILE][/dim]")
    console.print("[dim]Custom:  hafiz agent install --path <dir> [--file <name>][/dim]")
