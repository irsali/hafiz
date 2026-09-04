"""hafiz agent — install/uninstall/list agent configurations."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from hafiz.core.agents import (
    _EXTRACTOR_CONTRACT_VERSION,
    AGENTS,
    current_skills_version,
    install_file,
    install_hooks,
    installed_skills_version,
    is_hafiz_managed,
    load_skills_content,
    resolve_target,
    uninstall_file,
    uninstall_hooks,
)
from hafiz.core.communications import DEFAULT_RETENTION_DAYS

console = Console()


def _resolve_hooks_target(name: str | None, *, local: bool) -> tuple[Path, object]:
    """Locate the settings file whose hooks we manage for ``name``."""
    if not name:
        raise ValueError(
            "Specify which agent's hooks to manage, e.g. `hafiz agent install claude-code --hooks`."
        )
    agent = AGENTS.get(name)
    if agent is None:
        raise ValueError(f"Unknown agent: {name}. Known agents: {', '.join(sorted(AGENTS))}")
    if not agent.supports_hooks:
        raise ValueError(
            f"{agent.display_name} has no hook surface hafiz can manage yet — "
            "only claude-code does. Capture transcripts manually with "
            "`hafiz import <agent>`."
        )
    scope = "local" if local else "global"
    raw = agent.hooks.settings.get(scope)
    if raw is None:
        available = ", ".join(sorted(agent.hooks.settings))
        raise ValueError(
            f"{agent.display_name} does not support {scope} hooks (available: {available})."
        )
    target = Path(raw).expanduser()
    if local and not target.is_absolute():
        target = Path.cwd() / target
    return target, agent


def run_agent_install_hooks(
    name: str | None = None,
    *,
    local: bool = False,
) -> None:
    """Install automatic transcript-capture hooks for an agent."""
    try:
        target, agent = _resolve_hooks_target(name, local=local)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    console.print(f"Installing hafiz capture hooks for [bold]{agent.display_name}[/bold]...")
    try:
        result = install_hooks(target, agent.hooks)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    icons = {"created": "[green]+[/green]", "updated": "[yellow]~[/yellow]"}
    console.print(f"  {icons.get(result['action'], '?')} {result['path']}")
    console.print(f"  [dim]events: {', '.join(result['events'])}[/dim]")
    if result["replaced"]:
        console.print(f"  [dim]replaced {result['replaced']} previous hafiz hook entry(ies)[/dim]")

    console.print(
        f"\n[green]Done.[/green] {agent.display_name} transcripts will now be captured "
        "automatically before compaction and at session end.\n"
        "[dim]Transcripts are stored locally, scoped to the project they ran in, and "
        f"retained for {DEFAULT_RETENTION_DAYS} days.\n"
        "  Inspect:  hafiz status            (capture freshness)\n"
        "  Redact:   hafiz forget <session>\n"
        "  Eject:    hafiz export --include-transcripts\n"
        "  Remove:   hafiz agent uninstall "
        f"{agent.name} --hooks[/dim]"
    )


def run_agent_uninstall_hooks(
    name: str | None = None,
    *,
    local: bool = False,
) -> None:
    """Remove hafiz capture hooks for an agent."""
    try:
        target, agent = _resolve_hooks_target(name, local=local)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    try:
        result = uninstall_hooks(target)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1)

    if result["action"] == "not_found":
        console.print(f"[yellow]No settings file at {result['path']} — nothing to do.[/yellow]")
        return
    if result["action"] == "skipped":
        console.print(f"[yellow]No hafiz hooks found in {result['path']}.[/yellow]")
        return

    console.print(f"[green]Removed[/green] {result['removed']} hafiz hook entry(ies)")
    console.print(f"  [dim]{result['path']}[/dim]")
    console.print(
        "\n[yellow]Automatic transcript capture is now off.[/yellow] "
        "[dim]Already-captured transcripts are untouched; "
        "remove them with `hafiz forget`.[/dim]"
    )


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

    # Detect version drift: an agent config with an older SKILLS_VERSION
    # than what we're about to install is running the previous contract
    # (e.g. v1 extractor vocabulary). Flag it so users know the refresh
    # is carrying a hard break — not a no-op cosmetic update.
    current_v = current_skills_version()
    prev_v = installed_skills_version(target)

    content = load_skills_content()
    wrapper = agent.wrapper if agent else None
    status = install_file(target, content, wrapper=wrapper)

    icons = {
        "created": "[green]+[/green]",
        "updated": "[yellow]~[/yellow]",
        "appended": "[cyan]»[/cyan]",
    }
    console.print(f"  {icons.get(status, '?')} {target}  [dim]({status})[/dim]")

    if prev_v is not None and prev_v < current_v:
        console.print(f"\n[yellow]Upgraded skills.md from v{prev_v} to v{current_v}.[/yellow]")
        # The extractor vocabulary break is specific to crossing v2 — it is
        # not a property of "some version went up". Printing it on every
        # bump told users to retrain extractors after purely additive
        # releases, which is how a real warning gets tuned out.
        if prev_v < _EXTRACTOR_CONTRACT_VERSION <= current_v:
            console.print(
                "  [yellow]This crosses a breaking contract change:[/yellow] "
                "extract import rejects v1 payloads (entity_type / "
                "relation_type vocabulary). See the 'Agent Extraction' section "
                "of the new skills.md and retrain any in-flight extractors."
            )

    if status == "appended":
        console.print(
            f"\n[green]Done.[/green] Hafiz block appended to your existing {target.name}; "
            "your instructions are preserved."
        )
    else:
        console.print(
            f"\n[green]Done.[/green] {display_name} is configured to use hafiz "
            f"(skills v{current_v})."
        )


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
            "\n[yellow]No hafiz-managed region found in this file — skipping.[/yellow]"
            "\nUse --force to delete the file anyway."
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
    console.print(
        "[dim]Install: hafiz agent install <name> [--local] [--path PATH] [--file FILE][/dim]"
    )
    console.print("[dim]Custom:  hafiz agent install --path <dir> [--file <name>][/dim]")
