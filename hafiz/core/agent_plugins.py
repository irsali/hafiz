"""Plugin-style agent integrations — `hafiz agent install hermes`.

Most agents take hafiz as an *instruction file* (see :mod:`hafiz.core.agents`).
Hermes is different: its memory is a Python plugin that Hermes loads from its
own home directory, selected by one key in its YAML config. Installing it
means writing two files and flipping that key — without disturbing anything
else in a config file the user owns and edits by hand.

PyYAML round-trips lose comments, so the config is edited **line-wise**: only
the single ``memory.provider`` line is touched. Everything else, comments and
key order included, stays byte-for-byte identical. If the file is shaped in a
way the line editor cannot handle safely (e.g. ``memory: {}`` flow style) we
change nothing and hand the user the exact lines to add.

Everything hafiz writes is recorded in ``.hafiz-managed`` inside the plugin
directory, which is how uninstall knows what is ours to remove.
"""

from __future__ import annotations

import importlib.resources
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hafiz

# Plugin files: (name in package data, name written into the Hermes plugin dir).
# The provider ships as ``provider.py`` so it is a normal, lintable, importable
# module in this repo, but Hermes loads a plugin from ``<dir>/__init__.py``.
HERMES_FILES: tuple[tuple[str, str], ...] = (
    ("provider.py", "__init__.py"),
    ("plugin.yaml", "plugin.yaml"),
)
HERMES_PACKAGE = "hafiz.data.hermes"
HERMES_PLUGIN_NAME = "hafiz"
MARKER_NAME = ".hafiz-managed"
BACKUP_SUFFIX = ".hafiz-backup"

_EMPTY_PROVIDER_TOKENS = {"", "''", '""', "null", "~"}


@dataclass(frozen=True)
class AgentPlugin:
    """A plugin-style integration: where it lives and what it is called."""

    name: str
    display_name: str
    home_env: str
    default_home: str
    restart_hint: str

    def home(self, path_override: str | None = None) -> Path:
        """Hermes home: ``--path``, else ``$HERMES_HOME``, else ``~/.hermes``."""
        raw = path_override or os.environ.get(self.home_env) or self.default_home
        return Path(raw).expanduser()


PLUGINS: dict[str, AgentPlugin] = {
    "hermes": AgentPlugin(
        name="hermes",
        display_name="Hermes",
        home_env="HERMES_HOME",
        default_home="~/.hermes",
        restart_hint="Restart Hermes to load it.",
    ),
}


# ── config.yaml line editing ──────────────────────────────────────────────

_TOP_KEY_RE = re.compile(r"^([A-Za-z_][\w-]*)\s*:(.*)$")
_PROVIDER_RE = re.compile(r"^(\s+)provider\s*:[ \t]*(.*?)[ \t]*$")


def _comment_suffix(value_part: str, scalar: str) -> str:
    """The ``# comment`` after ``scalar``, with the space YAML needs before ``#``.

    For an empty value (``provider:   # note``) the regex has already eaten the
    spaces, and pasting the comment straight after a new value would make it
    part of the scalar (``hafiz# note``).
    """
    comment = value_part[len(scalar) :]
    return f" {comment}" if comment and not comment[0].isspace() else comment


def _strip_comment(value: str) -> str:
    """Drop a trailing ``# comment`` from a YAML scalar (quote-aware enough
    for the simple scalars ``provider`` holds)."""
    quote = ""
    for i, ch in enumerate(value):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or value[i - 1] in " \t"):
            return value[:i].rstrip()
    return value.rstrip()


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
        return token[1:-1]
    return token


@dataclass
class _MemoryBlock:
    start: int  # index of the ``memory:`` line
    end: int  # one past the last line of the block
    provider_idx: int | None  # index of the ``provider:`` line, if present
    child_indent: str


def _find_memory_block(lines: list[str]) -> _MemoryBlock | str | None:
    """Locate the top-level ``memory:`` block.

    Returns the block, ``None`` when there is no ``memory`` key, or the string
    ``"unsupported"`` when the key exists in a form we will not edit safely.
    """
    start = None
    for i, line in enumerate(lines):
        m = _TOP_KEY_RE.match(line.rstrip("\n"))
        if m and m.group(1) == "memory":
            if _strip_comment(m.group(2).strip()):
                return "unsupported"  # inline value / flow mapping
            start = i
            break
    if start is None:
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped and not stripped.startswith("#") and not lines[j][0].isspace():
            end = j
            break

    child_indent = "  "
    provider_idx = None
    seen_child = False
    for j in range(start + 1, end):
        line = lines[j].rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not seen_child:
            child_indent = line[: len(line) - len(line.lstrip())]
            seen_child = True
        m = _PROVIDER_RE.match(line)
        if m and m.group(1) == child_indent:
            provider_idx = j
    return _MemoryBlock(start, end, provider_idx, child_indent)


def read_memory_provider(text: str) -> tuple[str, str | None]:
    """Current ``memory.provider`` of a Hermes config.

    Returns ``(status, raw_token)``: status is ``"set"`` (token is the raw YAML
    scalar, possibly ``"''"``), ``"absent"`` (no provider key) or
    ``"unsupported"`` (shape we will not edit).
    """
    lines = text.splitlines(keepends=True)
    block = _find_memory_block(lines)
    if block == "unsupported":
        return "unsupported", None
    if not isinstance(block, _MemoryBlock) or block.provider_idx is None:
        return "absent", None
    m = _PROVIDER_RE.match(lines[block.provider_idx].rstrip("\n"))
    return "set", _strip_comment(m.group(2)) if m else ""


def provider_name(raw: str | None) -> str:
    """The provider as a plain string; empty for absent/blank/null."""
    if raw is None:
        return ""
    token = _unquote(raw.strip())
    return "" if raw.strip() in _EMPTY_PROVIDER_TOKENS or token == "null" else token


def write_memory_provider(text: str, raw_value: str) -> tuple[str, dict[str, Any]] | None:
    """Return *text* with ``memory.provider`` set to the raw YAML scalar.

    Gives ``(new_text, undo)`` where ``undo`` records how to put things back
    (``previous_raw`` / ``created_block``), or ``None`` if the file's shape is
    unsupported. Only one line changes (or is added); comments, key order and
    every other key are untouched.
    """
    lines = text.splitlines(keepends=True)
    block = _find_memory_block(lines)
    if block == "unsupported":
        return None

    if block is None:
        sep = "" if not lines or lines[-1].endswith("\n") else "\n"
        new = text + sep + f"memory:\n  provider: {raw_value}\n"
        return new, {"previous_raw": None, "created_block": True}

    assert isinstance(block, _MemoryBlock)
    if block.provider_idx is None:
        lines.insert(block.start + 1, f"{block.child_indent}provider: {raw_value}\n")
        return "".join(lines), {"previous_raw": None, "created_block": False}

    old = lines[block.provider_idx]
    m = _PROVIDER_RE.match(old.rstrip("\n"))
    assert m is not None
    previous_raw = _strip_comment(m.group(2))
    value_part = m.group(2)
    comment = _comment_suffix(value_part, previous_raw)  # keeps any "  # comment"
    nl = "\n" if old.endswith("\n") else ""
    lines[block.provider_idx] = f"{m.group(1)}provider: {raw_value}{comment}{nl}"
    return "".join(lines), {"previous_raw": previous_raw, "created_block": False}


def restore_memory_provider(text: str, undo: dict[str, Any]) -> str:
    """Reverse :func:`write_memory_provider` using its recorded ``undo``."""
    lines = text.splitlines(keepends=True)
    block = _find_memory_block(lines)
    if not isinstance(block, _MemoryBlock) or block.provider_idx is None:
        return text
    previous = undo.get("previous_raw")
    if previous is not None:
        old = lines[block.provider_idx]
        m = _PROVIDER_RE.match(old.rstrip("\n"))
        assert m is not None
        value_part = m.group(2)
        comment = _comment_suffix(value_part, _strip_comment(value_part))
        nl = "\n" if old.endswith("\n") else ""
        lines[block.provider_idx] = f"{m.group(1)}provider: {previous}{comment}{nl}"
        return "".join(lines)
    del lines[block.provider_idx]
    if undo.get("created_block"):
        refreshed = _find_memory_block(lines)
        if isinstance(refreshed, _MemoryBlock):
            body = [ln for ln in lines[refreshed.start + 1 : refreshed.end] if ln.strip()]
            if not body:
                del lines[refreshed.start : refreshed.end]
    return "".join(lines)


# ── file helpers ──────────────────────────────────────────────────────────


def _read_config(path: Path) -> tuple[str, str]:
    """``(text with LF endings, the file's own newline)`` — so a CRLF config is
    written back CRLF instead of having every line ending rewritten."""
    raw = path.read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n"), "\r\n" if "\r\n" in raw else "\n"


def _atomic_write(target: Path, text: str, newline: str = "\n") -> None:
    tmp = target.with_suffix(target.suffix + ".hafiz-tmp")
    tmp.write_bytes(text.replace("\n", newline).encode("utf-8"))
    os.replace(tmp, target)


def _backup_once(target: Path) -> None:
    """One-shot backup; never clobbers an earlier (last known-good) copy."""
    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)


def _read_marker(plugin_dir: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((plugin_dir / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_plugin_file(name: str) -> str:
    """Read one shipped plugin file from package data."""
    return importlib.resources.files(HERMES_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def manual_config_lines(value: str = HERMES_PLUGIN_NAME) -> str:
    return f"memory:\n  provider: {value}"


# ── install / uninstall ───────────────────────────────────────────────────


def install_hermes(home: Path, *, explicit_home: bool = False) -> dict[str, Any]:
    """Install the Hermes memory provider under *home*.

    Idempotent. Returns ``{"plugin_dir", "files", "plugin_action", "config"}``
    where ``config`` is ``{"path", "action", ...}`` with action one of
    ``set`` · ``already`` · ``conflict`` · ``manual`` · ``no_config``.
    Raises ``ValueError`` when *home* does not exist and was not asked for
    explicitly (Hermes is evidently not installed there).
    """
    if not home.is_dir():
        if not explicit_home:
            raise ValueError(
                f"Hermes home not found at {home}. Run Hermes once so it creates it, "
                "set $HERMES_HOME, or pass --path <hermes-home>."
            )
        home.mkdir(parents=True, exist_ok=True)

    plugin_dir = home / "plugins" / HERMES_PLUGIN_NAME
    marker = _read_marker(plugin_dir)
    existed = plugin_dir.exists()
    plugin_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for src_name, dest_name in HERMES_FILES:
        dest = plugin_dir / dest_name
        content = load_plugin_file(src_name)
        if dest.exists() and marker is None and dest.read_text(encoding="utf-8") != content:
            # A hand-installed plugin hafiz did not write: keep it recoverable.
            _backup_once(dest)
        if not dest.exists() or dest.read_text(encoding="utf-8") != content:
            dest.write_text(content, encoding="utf-8")
        written.append(str(dest))

    config = _configure_hermes(home / "config.yaml", marker)
    new_marker = {"hafiz_version": hafiz.__version__, "config": config.pop("undo", None)}
    if marker is not None and new_marker["config"] is None:
        # Re-run after a first install that set the config: keep the undo record.
        new_marker["config"] = marker.get("config")
    (plugin_dir / MARKER_NAME).write_text(json.dumps(new_marker, indent=2) + "\n", encoding="utf-8")

    return {
        "plugin_dir": str(plugin_dir),
        "files": written,
        "plugin_action": "updated" if existed else "created",
        "config": config,
    }


def _configure_hermes(config_path: Path, marker: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(config_path)}
    if not config_path.exists():
        # Hermes writes its own config on first run; creating a partial one
        # here could shadow its defaults, so hand over the lines instead.
        return {**result, "action": "no_config", "lines": manual_config_lines()}

    text, newline = _read_config(config_path)
    status, raw = read_memory_provider(text)
    if status == "unsupported":
        return {**result, "action": "manual", "lines": manual_config_lines()}

    current = provider_name(raw)
    if current == HERMES_PLUGIN_NAME:
        return {**result, "action": "already"}
    if current:
        # Another memory provider is active. Hermes runs one at a time, so
        # switching is the user's call — never ours.
        return {**result, "action": "conflict", "current": current}

    edited = write_memory_provider(text, HERMES_PLUGIN_NAME)
    if edited is None:
        return {**result, "action": "manual", "lines": manual_config_lines()}
    new_text, undo = edited
    _backup_once(config_path)
    _atomic_write(config_path, new_text, newline)
    return {**result, "action": "set", "undo": undo}


def uninstall_hermes(home: Path) -> dict[str, Any]:
    """Remove what :func:`install_hermes` wrote, and nothing else.

    Returns ``{"action", "plugin_dir", "removed", "config"}``; action is
    ``removed`` or ``not_found``. A hand-installed plugin hafiz never wrote
    (no marker) is left alone — action ``skipped``.
    """
    plugin_dir = home / "plugins" / HERMES_PLUGIN_NAME
    if not plugin_dir.exists():
        return {"action": "not_found", "plugin_dir": str(plugin_dir), "removed": [], "config": None}
    marker = _read_marker(plugin_dir)
    if marker is None:
        return {"action": "skipped", "plugin_dir": str(plugin_dir), "removed": [], "config": None}

    removed: list[str] = []
    for _src, dest_name in HERMES_FILES:
        dest = plugin_dir / dest_name
        backup = dest.with_name(dest.name + BACKUP_SUFFIX)
        if dest.exists():
            dest.unlink()
            removed.append(str(dest))
        if backup.exists():  # put back the plugin we displaced at install time
            backup.rename(dest)
    (plugin_dir / MARKER_NAME).unlink(missing_ok=True)
    shutil.rmtree(plugin_dir / "__pycache__", ignore_errors=True)
    try:
        plugin_dir.rmdir()  # only succeeds when nothing of the user's is left
    except OSError:
        pass

    config = _unconfigure_hermes(home / "config.yaml", marker)
    return {
        "action": "removed",
        "plugin_dir": str(plugin_dir),
        "removed": removed,
        "config": config,
    }


def _unconfigure_hermes(config_path: Path, marker: dict[str, Any]) -> dict[str, Any]:
    undo = marker.get("config")
    if not config_path.exists():
        return {"path": str(config_path), "action": "none"}
    text, newline = _read_config(config_path)
    status, raw = read_memory_provider(text)
    still_ours = status == "set" and provider_name(raw) == HERMES_PLUGIN_NAME
    if not still_ours:
        return {"path": str(config_path), "action": "none"}
    if not undo:
        # hafiz did not set it (the user did, or it predates the install):
        # not ours to unset. Say so rather than leave a silent dangling name.
        return {"path": str(config_path), "action": "left", "current": HERMES_PLUGIN_NAME}
    _backup_once(config_path)
    _atomic_write(config_path, restore_memory_provider(text, undo), newline)
    return {"path": str(config_path), "action": "restored"}
