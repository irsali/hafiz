"""hafiz hooks — install git hooks for automatic indexing."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from rich.console import Console

console = Console()

POST_COMMIT_TEMPLATE = """\
#!/usr/bin/env bash
# Hafiz post-commit hook — re-indexes changed files after each commit.
# Installed by: hafiz hooks install

set -e

{project_flag}

# Run in background so commit isn't blocked
nohup hafiz ingest --git-hook{project_opt} > /dev/null 2>&1 &
"""

POST_MERGE_TEMPLATE = """\
#!/usr/bin/env bash
# Hafiz post-merge hook — re-indexes changed files after pull/merge.
# Installed by: hafiz hooks install

set -e

{project_flag}

# Run in background so merge isn't blocked
nohup hafiz ingest --git-hook{project_opt} > /dev/null 2>&1 &
"""

POST_REWRITE_TEMPLATE = """\
#!/usr/bin/env bash
# Hafiz post-rewrite hook — reconciles the commits table after a rebase
# or amend (Phase 5b of structural-grounding). The hook runs a regular
# re-ingest in the background, which picks up the new HEAD and marks
# orphaned commits as rewritten.
# Installed by: hafiz hooks install

set -e

{project_flag}

# Run in background so the rewrite operation isn't blocked
nohup hafiz ingest --git-hook{project_opt} > /dev/null 2>&1 &
"""


def _install_hook(
    hooks_dir: Path,
    hook_name: str,
    content: str,
    project_flag: str,
    project_opt: str,
) -> str:
    """Install a single git hook. Returns status: 'created', 'updated', 'exists'."""
    hook_path = hooks_dir / hook_name

    if hook_path.exists():
        existing = hook_path.read_text(encoding="utf-8")
        if "hafiz" in existing.lower():
            return "exists"
        # Append to existing hook
        with open(hook_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n# --- Hafiz {hook_name} hook ---\n")
            f.write(f"{project_flag}\n")
            f.write(f"nohup hafiz ingest --git-hook{project_opt} > /dev/null 2>&1 &\n")
        if os.name != "nt":
            hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return "appended"
    else:
        hook_path.write_text(content, encoding="utf-8")
        if os.name != "nt":
            hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return "created"


def run_hooks_install(repo_path: str, *, project: str | None = None) -> None:
    """Install post-commit and post-merge hooks into a git repository."""
    repo = Path(repo_path).resolve()
    git_dir = repo / ".git"

    if not git_dir.is_dir():
        console.print(f"[red]Not a git repository:[/red] {repo}")
        raise SystemExit(1)

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    if project:
        project_flag = f'PROJECT="{project}"'
        project_opt = f" -p {project}"
    else:
        project_flag = "# No project specified"
        project_opt = ""

    hooks = [
        ("post-commit", POST_COMMIT_TEMPLATE),
        ("post-merge", POST_MERGE_TEMPLATE),
        ("post-rewrite", POST_REWRITE_TEMPLATE),
    ]

    for hook_name, template in hooks:
        content = template.format(project_flag=project_flag, project_opt=project_opt)
        status = _install_hook(hooks_dir, hook_name, content, project_flag, project_opt)

        if status == "exists":
            console.print(f"[yellow]Hafiz {hook_name} hook already installed.[/yellow]")
        elif status == "appended":
            console.print(
                f"[yellow]Existing {hook_name} hook found — appended Hafiz hook.[/yellow]"
            )
        else:
            console.print(
                f"[green]Installed {hook_name} hook:[/green] {hooks_dir / hook_name}"
            )
