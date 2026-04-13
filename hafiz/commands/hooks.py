"""hafiz hooks — install git hooks for automatic indexing."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from rich.console import Console

console = Console()

HOOK_TEMPLATE = """\
#!/usr/bin/env bash
# Hafiz post-commit hook — re-indexes changed files after each commit.
# Installed by: hafiz hooks install

set -e

REPO_DIR="$(git rev-parse --show-toplevel)"
{project_flag}

# Run in background so commit isn't blocked
nohup hafiz ingest --git-hook{project_opt} > /dev/null 2>&1 &
"""


def run_hooks_install(repo_path: str, *, project: str | None = None) -> None:
    """Install the post-commit hook into a git repository."""
    repo = Path(repo_path).resolve()
    git_dir = repo / ".git"

    if not git_dir.is_dir():
        console.print(f"[red]Not a git repository:[/red] {repo}")
        raise SystemExit(1)

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    hook_path = hooks_dir / "post-commit"

    if project:
        project_flag = f'PROJECT="{project}"'
        project_opt = f" -p {project}"
    else:
        project_flag = "# No project specified"
        project_opt = ""

    hook_content = HOOK_TEMPLATE.format(
        project_flag=project_flag,
        project_opt=project_opt,
    )

    # Check for existing hook
    if hook_path.exists():
        existing = hook_path.read_text()
        if "hafiz" in existing.lower():
            console.print("[yellow]Hafiz post-commit hook already installed.[/yellow]")
            return
        # Append to existing hook
        console.print("[yellow]Existing post-commit hook found — appending Hafiz hook.[/yellow]")
        with open(hook_path, "a") as f:
            f.write("\n\n# --- Hafiz post-commit hook ---\n")
            f.write(f"{project_flag}\n")
            f.write(f"nohup hafiz ingest --git-hook{project_opt} > /dev/null 2>&1 &\n")
    else:
        hook_path.write_text(hook_content)

    # Make executable
    hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    console.print(f"[green]Installed post-commit hook:[/green] {hook_path}")
