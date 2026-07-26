"""hafiz hooks — install git hooks for automatic indexing.

The hook body must name **both** the repo and the project. Omitting the project
used to be allowed, and the generated hook was:

    nohup hafiz ingest --git-hook > /dev/null 2>&1 &

No repo path, no project. ``ingest --git-hook`` then walked ``.`` and tagged
everything ``project=NULL``, which is worse than it sounds: ``files`` is unique
on ``(project, path)``, so the hook built a *second, untagged copy* of the repo
alongside the properly-tagged one. Measured on a real four-repo deployment —
1,951 untagged files whose paths mapped one-to-one onto the hooked repos (Admin
Portal: 699 tagged / 699 untagged). ``project=NULL`` also disables diff-driven
ingest and vanished-file tombstoning, so every commit re-walked the whole tree
and nothing was ever tombstoned, while the project-tagged rows stayed frozen
30-64 commits back.
"""

from __future__ import annotations

import os
import shlex
import stat
from pathlib import Path

from rich.console import Console

console = Console()

HOOK_HEADER = """\
#!/usr/bin/env bash
# Hafiz {hook_name} hook — {purpose}
# Installed by: hafiz hooks install
"""

# Both the repo and the project are pinned into the hook. The repo path because
# a hook's cwd is not guaranteed to be the work tree (worktrees, bare-adjacent
# setups, `git -C`); the project because an untagged ingest writes a duplicate
# shadow index instead of updating the real one.
HOOK_BODY = """
set -e

HAFIZ_REPO={repo}
HAFIZ_PROJECT={project}

# Run in background so the git operation isn't blocked.
nohup hafiz ingest "$HAFIZ_REPO" --git-hook --project "$HAFIZ_PROJECT" \\
  > /dev/null 2>&1 &
"""

HOOK_PURPOSE = {
    "post-commit": "re-indexes changed files after each commit.",
    "post-merge": "re-indexes changed files after pull/merge.",
    "post-rewrite": (
        "reconciles the commits table after a rebase or amend. The re-ingest "
        "picks up the new HEAD and marks orphaned commits as rewritten."
    ),
}


def _hook_text(hook_name: str, *, repo: Path, project: str) -> str:
    """Render a full hook script for ``hook_name``."""
    return HOOK_HEADER.format(
        hook_name=hook_name, purpose=HOOK_PURPOSE[hook_name]
    ) + HOOK_BODY.format(repo=shlex.quote(str(repo)), project=shlex.quote(project))


def _install_hook(hooks_dir: Path, hook_name: str, *, repo: Path, project: str) -> str:
    """Install a single git hook. Returns 'created', 'appended', or 'exists'."""
    hook_path = hooks_dir / hook_name
    content = _hook_text(hook_name, repo=repo, project=project)

    if hook_path.exists():
        existing = hook_path.read_text(encoding="utf-8")
        if "hafiz" in existing.lower():
            return "exists"
        # Append our block to a foreign hook rather than clobbering it.
        with open(hook_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n# --- Hafiz {hook_name} hook ---")
            f.write(HOOK_BODY.format(repo=shlex.quote(str(repo)), project=shlex.quote(project)))
        status = "appended"
    else:
        hook_path.write_text(content, encoding="utf-8")
        status = "created"

    if os.name != "nt":
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return status


def _conflicting_root(project: str, repo: Path) -> str | None:
    """Return the indexed root of ``project`` if it isn't this repo.

    Guards the case that silently corrupts an index: pointing a repo's hooks at
    a project name whose files live somewhere else entirely, so every commit
    here re-indexes (and tombstones against) an unrelated tree. Returns None
    when the project is new, when its root contains this repo, or when the
    check can't run — a DB outage must not block installing a hook.
    """
    try:
        import asyncio

        from hafiz.core.database import close_engine
        from hafiz.core.store import indexed_root_per_project

        async def _lookup() -> dict:
            try:
                return await indexed_root_per_project()
            finally:
                await close_engine()

        roots = asyncio.run(_lookup())
    except Exception as e:  # noqa: BLE001 — advisory check; never block on it
        console.print(f"[dim]Could not verify project scope against the index ({e}).[/dim]")
        return None

    root = roots.get(project)
    if not root:
        return None
    root_path = Path(root)
    if root_path == repo or root_path in repo.parents or repo in root_path.parents:
        return None
    return root


def run_hooks_install(
    repo_path: str,
    *,
    project: str | None = None,
    force: bool = False,
) -> None:
    """Install post-commit / post-merge / post-rewrite hooks into a repository.

    ``project`` defaults to the repo directory name. An untagged hook is never
    generated: it would write a duplicate untagged index rather than updating
    the project's own rows.
    """
    repo = Path(repo_path).resolve()
    git_dir = repo / ".git"

    if not git_dir.is_dir():
        console.print(f"[red]Not a git repository:[/red] {repo}")
        raise SystemExit(1)

    resolved_project = (project or repo.name).strip()
    if not resolved_project:
        console.print(f"[red]Could not derive a project name from:[/red] {repo}")
        raise SystemExit(1)

    conflict = _conflicting_root(resolved_project, repo)
    if conflict and not force:
        console.print(
            f"[red]Refusing to install:[/red] project "
            f"[bold]{resolved_project}[/bold] is already indexed under\n"
            f"  {conflict}\n"
            f"but you are installing hooks into\n"
            f"  {repo}\n\n"
            f"Committing here would re-index — and tombstone against — an "
            f"unrelated tree.\n"
            f"[dim]Pick a different name with --project, or pass --force if the "
            f"project really did move.[/dim]"
        )
        raise SystemExit(2)
    if conflict:
        console.print(
            f"[yellow]--force:[/yellow] repointing project "
            f"[bold]{resolved_project}[/bold] from {conflict} to {repo}."
        )

    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)

    for hook_name in ("post-commit", "post-merge", "post-rewrite"):
        status = _install_hook(hooks_dir, hook_name, repo=repo, project=resolved_project)

        if status == "exists":
            console.print(f"[yellow]Hafiz {hook_name} hook already installed.[/yellow]")
        elif status == "appended":
            console.print(
                f"[yellow]Existing {hook_name} hook found — appended Hafiz hook.[/yellow]"
            )
        else:
            console.print(f"[green]Installed {hook_name} hook:[/green] {hooks_dir / hook_name}")

    console.print(
        f"\n  Indexing [bold]{repo}[/bold] as project [bold]{resolved_project}[/bold]."
        f"\n  [dim]Check freshness any time with: hafiz status[/dim]"
    )
