"""Agent configuration registry and file operations.

Supports installing hafiz skills into any AI coding agent's config directory.
Known agents (Claude Code, Cursor, GitHub Copilot) have sensible defaults;
unknown agents can specify --path/--file directly.
"""

from __future__ import annotations

import importlib.resources
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Markers bracketing hafiz-managed content inside an instruction file.
# The region between them is owned by hafiz; everything outside belongs to the user.
START_MARKER = "<!-- Installed by hafiz — workspace intelligence layer -->"
END_MARKER = (
    "<!-- /Installed by hafiz — do not edit above this block; "
    "re-run `hafiz agent install` to update -->"
)

# Default filename when no agent name and no --file provided
DEFAULT_FILENAME = "skills.md"


# ── Wrappers ──────────────────────────────────────────────────────────────


def prepend_cursor_frontmatter(content: str) -> str:
    """Wrap content with Cursor .mdc frontmatter."""
    frontmatter = "---\ndescription: Hafiz workspace intelligence\nalwaysApply: true\n---\n\n"
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


def find_hafiz_region(text: str) -> tuple[int, int] | None:
    """Locate the hafiz-managed region in *text*.

    Returns (start, end) character indices such that ``text[start:end]`` covers
    the region from the line containing START_MARKER through the line
    containing END_MARKER (including its trailing newline if present).
    Returns None if the paired region is not found.
    """
    start_idx = text.find(START_MARKER)
    if start_idx == -1:
        return None
    line_start = text.rfind("\n", 0, start_idx) + 1  # 0 when no prior newline

    end_idx = text.find(END_MARKER, start_idx + len(START_MARKER))
    if end_idx == -1:
        return None
    trailing_nl = text.find("\n", end_idx + len(END_MARKER))
    line_end = len(text) if trailing_nl == -1 else trailing_nl + 1

    return line_start, line_end


def is_hafiz_managed(path: Path) -> bool:
    """Check whether *path* contains a paired hafiz-managed region."""
    if not path.exists():
        return False
    try:
        return find_hafiz_region(path.read_text(encoding="utf-8")) is not None
    except (OSError, UnicodeDecodeError):
        return False


# Matches ``<!-- SKILLS_VERSION: N -->`` anywhere in a skills region.
# Emitted by the shipped skills.md so agent installers can tell an
# out-of-date splice apart from a current one.
_SKILLS_VERSION_RE = re.compile(r"<!--\s*SKILLS_VERSION:\s*(\d+)\s*-->")


def current_skills_version() -> int:
    """Version of the skills.md shipping with this package. Source of
    truth for the agent contract."""
    marker = _SKILLS_VERSION_RE.search(load_skills_content())
    return int(marker.group(1)) if marker else 1


def installed_skills_version(target: Path) -> int | None:
    """Version of the skills.md currently spliced into ``target``, or
    None if the file doesn't exist or has no hafiz-managed region."""
    if not target.exists():
        return None
    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    region = find_hafiz_region(content)
    if region is None:
        return None
    start, end = region
    marker = _SKILLS_VERSION_RE.search(content[start:end])
    return int(marker.group(1)) if marker else 1


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


def install_file(
    target: Path,
    content: str,
    *,
    wrapper: Callable[[str], str] | None = None,
) -> str:
    """Install hafiz skills into *target*, preserving any user-owned content.

    - Fresh file: write ``wrapper(content)`` if a wrapper is provided, else *content*.
    - File with a paired hafiz region: splice *content* in place, leaving
      everything outside the markers untouched.
    - File without markers: append *content* after the user's existing content.

    Returns one of: ``'created'``, ``'updated'``, ``'appended'``.
    """
    target = target.resolve()

    if not target.exists():
        file_content = wrapper(content) if wrapper else content
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file_content, encoding="utf-8")
        return "created"

    existing = target.read_text(encoding="utf-8")
    region = find_hafiz_region(existing)

    if region is not None:
        start, end = region
        new_text = existing[:start] + content + existing[end:]
        target.write_text(new_text, encoding="utf-8")
        return "updated"

    # User-owned file — append the hafiz block after their content with a blank line.
    prefix = existing.rstrip("\n") + "\n\n"
    target.write_text(prefix + content, encoding="utf-8")
    return "appended"


def uninstall_file(target: Path, *, force: bool = False) -> str:
    """Remove the hafiz-managed region from *target*, preserving user content.

    - Paired region present: splice it out. Delete the file only if nothing
      non-whitespace remains.
    - No region and ``force=False``: skipped.
    - No region and ``force=True``: delete the whole file.

    Returns one of: ``'removed'``, ``'skipped'``, ``'not_found'``.
    """
    target = target.resolve()

    if not target.exists():
        return "not_found"

    existing = target.read_text(encoding="utf-8")
    region = find_hafiz_region(existing)

    if region is not None:
        start, end = region
        remaining = existing[:start] + existing[end:]
        if remaining.strip():
            target.write_text(remaining.rstrip("\n") + "\n", encoding="utf-8")
        else:
            target.unlink()
            try:
                target.parent.rmdir()
            except OSError:
                pass
        return "removed"

    if force:
        target.unlink()
        try:
            target.parent.rmdir()
        except OSError:
            pass
        return "removed"

    return "skipped"
