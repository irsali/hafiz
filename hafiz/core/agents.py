"""Agent configuration registry and file operations.

Supports installing hafiz skills into any AI coding agent's config directory.
Known agents (Claude Code, Cursor, GitHub Copilot) have sensible defaults;
unknown agents can specify --path/--file directly.
"""

from __future__ import annotations

import importlib.resources
import json
import os
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

# The skills.md version at which the agent-extraction contract broke
# (entity_type / relation_type vocabulary). Only a bump *crossing* this
# boundary warrants the retrain-your-extractors warning; later bumps are
# additive and must not cry wolf.
_EXTRACTOR_CONTRACT_VERSION = 2


# ── Wrappers ──────────────────────────────────────────────────────────────


def prepend_cursor_frontmatter(content: str) -> str:
    """Wrap content with Cursor .mdc frontmatter."""
    frontmatter = "---\ndescription: Hafiz workspace intelligence\nalwaysApply: true\n---\n\n"
    return frontmatter + content


# ── Registry ──────────────────────────────────────────────────────────────


# Sentinel embedded in every hook command hafiz writes into an agent's
# settings file. Hook config is JSON, not Markdown, so the paired-marker
# splicing used for instruction files doesn't apply — hafiz-owned entries
# are instead identified structurally, by this string appearing in the
# command. Everything without it belongs to the user and is never touched.
HOOK_SENTINEL = "# hafiz-managed"


@dataclass
class AgentHooks:
    """How to wire automatic transcript capture into an agent harness.

    ``events`` are the harness's own hook-event names. For capture we want
    the moments *before context is destroyed*: compaction discards turns
    mid-session, and session end is the last chance at the rest. Both fire
    the same idempotent command, so double-firing costs one no-op query.
    """

    settings: dict[str, str]  # {"global": "~/.claude/settings.json", …}
    events: tuple[str, ...]
    command: str
    timeout: int = 30


# `--from-hook` reads the harness payload on stdin and always exits 0, so
# the installed command needs no shell JSON handling and no `jq`. The
# `timeout` guard and `|| true` cover the one failure hafiz itself cannot:
# the process hanging (timeout exits 124).
_CAPTURE_COMMAND = (
    f"timeout 25s hafiz import claude-code --from-hook >/dev/null 2>&1 || true  {HOOK_SENTINEL}"
)


@dataclass
class AgentDefaults:
    """Default paths for a known agent."""

    name: str
    display_name: str
    instructions: dict[str, str]  # {"global": "~/.claude/CLAUDE.md", "local": "CLAUDE.md"}
    wrapper: Callable[[str], str] | None = None
    hooks: AgentHooks | None = None

    @property
    def supports_global(self) -> bool:
        return "global" in self.instructions

    @property
    def supports_local(self) -> bool:
        return "local" in self.instructions

    @property
    def supports_hooks(self) -> bool:
        return self.hooks is not None


AGENTS: dict[str, AgentDefaults] = {
    "claude-code": AgentDefaults(
        name="claude-code",
        display_name="Claude Code",
        instructions={"global": "~/.claude/CLAUDE.md", "local": "CLAUDE.md"},
        hooks=AgentHooks(
            settings={
                "global": "~/.claude/settings.json",
                "local": ".claude/settings.json",
            },
            events=("PreCompact", "SessionEnd"),
            command=_CAPTURE_COMMAND,
        ),
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


# ── Hook operations ───────────────────────────────────────────────────────


def _strip_hafiz_hooks(events: dict) -> int:
    """Remove every hafiz-owned entry from a harness ``hooks`` mapping,
    in place. Returns how many entries were removed.

    Walks the harness's nested shape (``event -> [group] -> group["hooks"]
    -> [entry]``) and drops only entries whose command carries
    :data:`HOOK_SENTINEL`. Groups emptied by that are removed too, and so
    are events left with no groups — a stale ``"SessionEnd": []`` is
    noise in a file the user reads.
    """
    removed = 0
    for event in list(events):
        groups = events.get(event)
        if not isinstance(groups, list):
            continue
        surviving_groups = []
        for group in groups:
            if not isinstance(group, dict):
                surviving_groups.append(group)
                continue
            entries = group.get("hooks")
            if not isinstance(entries, list):
                surviving_groups.append(group)
                continue
            kept = [
                e
                for e in entries
                if not (isinstance(e, dict) and HOOK_SENTINEL in str(e.get("command", "")))
            ]
            removed += len(entries) - len(kept)
            if not kept:
                # Whole group was ours — drop it rather than leave an
                # empty {"hooks": []} behind.
                continue
            group["hooks"] = kept
            surviving_groups.append(group)
        if surviving_groups:
            events[event] = surviving_groups
        else:
            del events[event]
    return removed


def _load_settings(target: Path) -> dict:
    """Read an agent settings file, tolerating absent/corrupt content.

    A settings file we cannot parse is *not* overwritten — the caller
    raises instead. Silently replacing a user's unparseable config would
    destroy hand-written configuration to install a convenience.
    """
    if not target.exists():
        return {}
    text = target.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{target} is not valid JSON ({exc}); refusing to overwrite it") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{target} does not contain a JSON object; refusing to overwrite it")
    return data


def _write_settings(target: Path, data: dict) -> None:
    """Write settings atomically, keeping a one-shot backup.

    This is a file the user owns and edits by hand, so: back it up once
    (never clobbering an existing backup, which would erase the last
    known-good copy on a second run), then swap the new content in via
    ``os.replace`` so a crash mid-write can't truncate it.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = target.with_suffix(target.suffix + ".hafiz-backup")
        if not backup.exists():
            backup.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")

    tmp = target.with_suffix(target.suffix + ".hafiz-tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def install_hooks(target: Path, hooks: AgentHooks) -> dict:
    """Install hafiz capture hooks into an agent's settings file.

    Idempotent: existing hafiz-owned entries are stripped first, so
    re-running updates the command in place rather than stacking copies.
    User-authored hooks — and every other key in the file — are preserved.

    Returns ``{"action", "path", "events", "replaced"}``.
    """
    data = _load_settings(target)
    existed = target.exists()

    events = data.setdefault("hooks", {})
    if not isinstance(events, dict):
        raise ValueError(f"{target} has a 'hooks' key that is not an object; refusing to edit it")

    replaced = _strip_hafiz_hooks(events)

    for event in hooks.events:
        groups = events.setdefault(event, [])
        if not isinstance(groups, list):
            raise ValueError(f"{target} has a non-list hooks.{event}; refusing to edit it")
        groups.append(
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": hooks.command,
                        "timeout": hooks.timeout,
                    }
                ]
            }
        )

    _write_settings(target, data)
    return {
        "action": "updated" if existed else "created",
        "path": str(target),
        "events": list(hooks.events),
        "replaced": replaced,
    }


def uninstall_hooks(target: Path) -> dict:
    """Remove hafiz capture hooks from an agent's settings file.

    Leaves the file (and every user-authored hook in it) otherwise
    untouched. Returns ``{"action", "path", "removed"}`` where action is
    ``'removed'``, ``'skipped'`` (nothing of ours present) or
    ``'not_found'``.
    """
    if not target.exists():
        return {"action": "not_found", "path": str(target), "removed": 0}

    data = _load_settings(target)
    events = data.get("hooks")
    if not isinstance(events, dict):
        return {"action": "skipped", "path": str(target), "removed": 0}

    removed = _strip_hafiz_hooks(events)
    if not removed:
        return {"action": "skipped", "path": str(target), "removed": 0}

    if not events:
        del data["hooks"]
    _write_settings(target, data)
    return {"action": "removed", "path": str(target), "removed": removed}


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
