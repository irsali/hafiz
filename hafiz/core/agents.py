"""Agent configuration registry and file operations.

Supports installing hafiz skills into any AI coding agent's config directory.
Known agents (Claude Code, Cursor, GitHub Copilot) have sensible defaults;
unknown agents can specify --path/--file directly.
"""

from __future__ import annotations

import importlib.resources
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Marker to identify files installed by hafiz (checked in first 200 chars)
MARKER = "Installed by hafiz"

# Default filename when no agent name and no --file provided
DEFAULT_FILENAME = "skills.md"


# ── Wrappers ──────────────────────────────────────────────────────────────


def prepend_cursor_frontmatter(content: str) -> str:
    """Wrap content with Cursor .mdc frontmatter."""
    frontmatter = (
        "---\n"
        "description: Hafiz workspace intelligence\n"
        "alwaysApply: true\n"
        "---\n\n"
    )
    return frontmatter + content


# ── Registry ──────────────────────────────────────────────────────────────


@dataclass
class AgentDefaults:
    """Default paths for a known agent."""

    name: str
    display_name: str
    instructions: dict[str, str]  # {"global": "~/.claude/CLAUDE.md", "local": "CLAUDE.md"}
    wrapper: Callable[[str], str] | None = None

    @property
    def supports_global(self) -> bool:
        return "global" in self.instructions

    @property
    def supports_local(self) -> bool:
        return "local" in self.instructions


AGENTS: dict[str, AgentDefaults] = {
    "claude-code": AgentDefaults(
        name="claude-code",
        display_name="Claude Code",
        instructions={"global": "~/.claude/CLAUDE.md", "local": "CLAUDE.md"},
    ),
    "cursor": AgentDefaults(
        name="cursor",
        display_name="Cursor",
        instructions={"local": ".cursor/rules/hafiz.mdc"},
        wrapper=prepend_cursor_frontmatter,
    ),
    "github-copilot": AgentDefaults(
        name="github-copilot",
        display_name="GitHub Copilot",
        instructions={"local": ".github/copilot-instructions.md"},
    ),
}


# ── File operations ───────────────────────────────────────────────────────


def load_skills_content() -> str:
    """Load skills.md from package data."""
    ref = importlib.resources.files("hafiz.data.agents").joinpath("skills.md")
    return ref.read_text(encoding="utf-8")


def is_hafiz_managed(path: Path) -> bool:
    """Check if a file was installed by hafiz (marker in first 200 chars)."""
    if not path.exists():
        return False
    try:
        content = path.read_text(encoding="utf-8")
        return MARKER in content[:200]
    except (OSError, UnicodeDecodeError):
        return False


def resolve_target(
    name: str | None,
    *,
    local: bool,
    path_override: str | None,
    file_override: str | None,
) -> tuple[Path, AgentDefaults | None]:
    """Resolve the target file path from agent name, flags, and overrides.

    Returns (resolved_path, agent_defaults_or_None).
    Raises ValueError if target cannot be determined.
    """
    agent = AGENTS.get(name) if name else None

    if agent:
        scope = "local" if local else "global"
        default_path = agent.instructions.get(scope)
        if default_path is None:
            raise ValueError(
                f"{agent.display_name} does not support {'local' if local else 'global'} "
                f"installation. Use {'--local' if not local else 'without --local'}."
            )

        # Split default into directory and filename
        default_full = Path(default_path)
        default_dir = str(default_full.parent)
        default_file = default_full.name

        # Apply overrides
        final_dir = path_override if path_override else default_dir
        final_file = file_override if file_override else default_file

        target = Path(final_dir) / final_file

    else:
        # Unknown agent — require at least --path
        if not path_override:
            raise ValueError(
                "Unknown agent. Provide --path (and optionally --file) to specify "
                "where to install.\n"
                f"Known agents: {', '.join(sorted(AGENTS.keys()))}"
            )
        final_file = file_override if file_override else DEFAULT_FILENAME
        target = Path(path_override) / final_file

    # Expand ~ and resolve for local installs
    target = target.expanduser()
    if local and not target.is_absolute():
        target = Path.cwd() / target

    return target, agent


def install_file(target: Path, content: str) -> str:
    """Write content to target path. Returns status: 'created', 'updated', 'skipped'."""
    target = target.resolve()

    if target.exists():
        if is_hafiz_managed(target):
            target.write_text(content, encoding="utf-8")
            return "updated"
        else:
            return "skipped"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return "created"


def uninstall_file(target: Path, *, force: bool = False) -> str:
    """Remove a hafiz-managed file. Returns status: 'removed', 'skipped', 'not_found'."""
    target = target.resolve()

    if not target.exists():
        return "not_found"

    if force or is_hafiz_managed(target):
        target.unlink()
        # Clean up empty parent dirs (but don't remove cwd or home)
        try:
            target.parent.rmdir()
        except OSError:
            pass
        return "removed"

    return "skipped"
