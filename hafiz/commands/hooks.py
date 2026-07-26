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

Generating a correct hook is only half the fix. Re-installing over an existing
hafiz hook used to short-circuit on "already installed" — it discarded
``--project``, printed a summary claiming the new project, and exited 0. So
every repo that got a broken hook once could never be corrected through the
CLI; the operator had to know to delete the file by hand. **A hafiz-generated
hook is a managed artifact: re-installing converges it to the requested
config** and says what changed, keeping the previous file as ``.hafiz-bak``.
"""

from __future__ import annotations

import os
import re
import shlex
import stat
from dataclasses import dataclass
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

HOOK_NAMES = ("post-commit", "post-merge", "post-rewrite")

BACKUP_SUFFIX = ".hafiz-bak"


def _header_mark(hook_name: str) -> str:
    """Marks a file hafiz generated *in full* — including older versions."""
    return f"# Hafiz {hook_name} hook"


def _block_mark(hook_name: str) -> str:
    """Marks a hafiz block appended to somebody else's hook."""
    return f"# --- Hafiz {hook_name} hook ---"


def _hook_text(hook_name: str, *, repo: Path, project: str) -> str:
    """Render a full hook script for ``hook_name``."""
    return HOOK_HEADER.format(
        hook_name=hook_name, purpose=HOOK_PURPOSE[hook_name]
    ) + HOOK_BODY.format(repo=shlex.quote(str(repo)), project=shlex.quote(project))


def _block_text(hook_name: str, *, repo: Path, project: str) -> str:
    """Render just the hafiz block, for appending to a foreign hook."""
    return f"\n\n{_block_mark(hook_name)}" + HOOK_BODY.format(
        repo=shlex.quote(str(repo)), project=shlex.quote(project)
    )


def _hook_project(text: str) -> str | None:
    """Recover the project a hook is pinned to, across hook generations.

    Used only to describe an update in human terms, so an unparseable hook
    degrades to "unknown" rather than blocking the rewrite.
    """
    value = r"(\"[^\"]*\"|'[^']*'|\S+)"
    for pattern in (rf"HAFIZ_PROJECT={value}", rf"--project\s+{value}"):
        m = re.search(pattern, text)
        if m:
            found = m.group(1)
            if "$" in found:  # the variable reference, not the value
                continue
            try:
                return " ".join(shlex.split(found)) or None
            except ValueError:  # unbalanced quotes in a hand-edited hook
                return found.strip("\"'") or None
    return None


@dataclass
class _Plan:
    """What re-installing would do to one hook file, decided before any write."""

    hook_name: str
    action: str  # create | update | append | unchanged | unrecognized
    desired: str = ""  # full file content to write (empty for unrecognized)
    old_project: str | None = None
    overwrites: bool = False  # an existing file is being replaced → keep a backup


def _classify(hooks_dir: Path, hook_name: str, *, repo: Path, project: str) -> _Plan:
    """Decide the action for one hook without touching the filesystem.

    Whole-file hooks we generated are rewritten; a block we appended to
    somebody else's hook is replaced in place; a foreign hook that already runs
    hafiz without either of our markers is left alone unless forced.
    """
    path = hooks_dir / hook_name
    full = _hook_text(hook_name, repo=repo, project=project)
    if not path.exists():
        return _Plan(hook_name, "create", full)

    existing = path.read_text(encoding="utf-8")
    found_project = _hook_project(existing)

    if _block_mark(hook_name) in existing:
        head = existing[: existing.index(_block_mark(hook_name))].rstrip("\n")
        desired = head + _block_text(hook_name, repo=repo, project=project)
    elif _header_mark(hook_name) in existing:
        desired = full
    elif re.search(r"\bhafiz\b", existing, re.IGNORECASE):
        # Mentions hafiz but carries neither marker — a hand-written hook we
        # can't safely rewrite. Appending would fire a second ingest per commit,
        # which is exactly the silent waste this command is meant to prevent.
        return _Plan(hook_name, "unrecognized", old_project=found_project)
    else:
        return _Plan(
            hook_name,
            "append",
            existing.rstrip("\n") + _block_text(hook_name, repo=repo, project=project),
        )

    if desired == existing:
        return _Plan(hook_name, "unchanged", desired, found_project)
    return _Plan(hook_name, "update", desired, found_project, overwrites=True)


def _apply(plan: _Plan, hooks_dir: Path, *, repo: Path, project: str) -> None:
    """Execute one plan. Any overwrite leaves the previous file recoverable."""
    path = hooks_dir / plan.hook_name

    if plan.action == "unchanged":
        return

    if plan.action == "unrecognized":  # --force: append rather than rewrite
        (hooks_dir / (plan.hook_name + BACKUP_SUFFIX)).write_text(
            path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        with open(path, "a", encoding="utf-8") as f:
            f.write(_block_text(plan.hook_name, repo=repo, project=project))
    else:
        if plan.overwrites:
            (hooks_dir / (plan.hook_name + BACKUP_SUFFIX)).write_text(
                path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        path.write_text(plan.desired, encoding="utf-8")

    if os.name != "nt":
        path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


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
    the project's own rows. Re-running converges an existing hafiz hook onto
    the requested repo/project instead of skipping it.
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

    # Decide everything first: a refusal must not leave two of three hooks
    # rewritten.
    plans = [_classify(hooks_dir, name, repo=repo, project=resolved_project) for name in HOOK_NAMES]

    unrecognized = [p for p in plans if p.action == "unrecognized"]
    if unrecognized and not force:
        names = ", ".join(p.hook_name for p in unrecognized)
        console.print(
            f"[red]Refusing to install:[/red] {names} already run hafiz, but were "
            f"not generated by\n  hafiz hooks install\n\n"
            f"Appending would ingest this repo twice on every commit.\n"
            f"[dim]Remove the hafiz lines from those hooks and re-run, or pass "
            f"--force to append anyway.[/dim]"
        )
        raise SystemExit(2)

    for plan in plans:
        _apply(plan, hooks_dir, repo=repo, project=resolved_project)
        path = hooks_dir / plan.hook_name

        if plan.action == "unchanged":
            console.print(f"[dim]{plan.hook_name} already current.[/dim]")
        elif plan.action == "update":
            moved = (
                f" [dim](project: {plan.old_project or 'none'} → {resolved_project})[/dim]"
                if plan.old_project != resolved_project
                else ""
            )
            console.print(
                f"[green]Updated {plan.hook_name} hook:[/green] {path}{moved}\n"
                f"  [dim]previous version kept at {path.name}{BACKUP_SUFFIX}[/dim]"
            )
        elif plan.action in ("append", "unrecognized"):
            console.print(
                f"[yellow]Existing {plan.hook_name} hook found — appended Hafiz hook.[/yellow]"
            )
        else:
            console.print(f"[green]Installed {plan.hook_name} hook:[/green] {path}")

    console.print(
        f"\n  Indexing [bold]{repo}[/bold] as project [bold]{resolved_project}[/bold]."
        f"\n  [dim]Check freshness any time with: hafiz status[/dim]"
    )
