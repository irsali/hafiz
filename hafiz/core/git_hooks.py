"""Git integration for Hafiz — commit-aware ingestion and observation storage.

Uses subprocess for git operations (no gitpython dependency).
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from hafiz.core.chunker import LANGUAGE_MAP, chunk_file, should_ignore
from hafiz.core.config import get_settings
from hafiz.core.embeddings import embed_texts
from hafiz.core.observations import store_observation
from hafiz.core.store import delete_chunks_for_file, store_chunks

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 64


def _run_git(args: list[str], cwd: str | Path) -> str:
    """Run a git command and return stdout. Returns empty string on failure."""
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.debug("git %s failed: %s", " ".join(args), result.stderr.strip())
            return ""
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("git command failed: git %s", " ".join(args))
        return ""


def get_changed_files_from_commit(repo_path: str | Path) -> list[str]:
    """Get list of files changed in the most recent commit.

    Falls back to staged files if HEAD~1 doesn't exist (initial commit).
    Returns paths relative to repo root.
    """
    repo = Path(repo_path).resolve()

    # Try HEAD~1..HEAD first (normal commits)
    output = _run_git(["diff", "--name-only", "HEAD~1", "HEAD"], cwd=repo)
    if output:
        return [f for f in output.splitlines() if f.strip()]

    # Fallback: staged files (pre-commit or initial commit)
    output = _run_git(["diff", "--cached", "--name-only"], cwd=repo)
    if output:
        return [f for f in output.splitlines() if f.strip()]

    return []


def _get_commit_info(repo_path: Path) -> dict:
    """Get metadata about the latest commit."""
    commit_hash = _run_git(["rev-parse", "HEAD"], cwd=repo_path)
    author = _run_git(["log", "-1", "--format=%an <%ae>"], cwd=repo_path)
    message = _run_git(["log", "-1", "--format=%s"], cwd=repo_path)
    timestamp = _run_git(["log", "-1", "--format=%aI"], cwd=repo_path)

    return {
        "hash": commit_hash or "unknown",
        "author": author or "unknown",
        "message": message or "",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
    }


async def store_commit_observation(
    repo_path: str | Path,
    project: str | None = None,
) -> None:
    """Store the latest commit metadata as an observation."""
    repo = Path(repo_path).resolve()
    info = _get_commit_info(repo)

    changed = get_changed_files_from_commit(repo)
    files_summary = f" ({len(changed)} files)" if changed else ""

    content = (
        f"Commit {info['hash'][:8]}: {info['message']}{files_summary} "
        f"by {info['author']}"
    )

    await store_observation(
        content,
        obs_type="fact",
        source="git",
        project=project,
        tags=["commit", "git"],
        confidence=1.0,
        metadata={
            "commit_hash": info["hash"],
            "author": info["author"],
            "message": info["message"],
            "timestamp": info["timestamp"],
            "changed_files": changed,
        },
    )
    logger.info("Stored commit observation: %s", info["hash"][:8])


async def run_git_hook_ingest(
    repo_path: str | Path,
    project: str | None = None,
) -> tuple[int, int]:
    """Process only files changed in the latest commit.

    Returns (files_processed, chunks_stored).
    """
    repo = Path(repo_path).resolve()
    settings = get_settings()
    ignore_patterns = settings.workspace.ignore

    changed_files = get_changed_files_from_commit(repo)
    if not changed_files:
        logger.info("No changed files found in latest commit")
        return 0, 0

    total_chunks_stored = 0
    files_processed = 0

    for rel_path in changed_files:
        full_path = repo / rel_path
        path_obj = Path(rel_path)

        # Skip ignored files and unsupported extensions
        if should_ignore(path_obj, ignore_patterns):
            continue
        if path_obj.suffix.lower() not in LANGUAGE_MAP:
            continue

        # Delete old chunks for this file
        await delete_chunks_for_file(rel_path)

        # If file was deleted in the commit, just remove chunks
        if not full_path.exists():
            files_processed += 1
            continue

        # Chunk, embed, store
        chunks = chunk_file(full_path, relative_to=repo)
        if not chunks:
            files_processed += 1
            continue

        all_embeddings: list[list[float]] = []
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            texts = [c.content for c in batch]
            embeddings = await embed_texts(texts)
            all_embeddings.extend(embeddings)

        stored = await store_chunks(chunks, all_embeddings, project=project)
        total_chunks_stored += stored
        files_processed += 1
        logger.info("Git hook re-indexed %s: %d chunks", rel_path, stored)

    # Store commit observation
    await store_commit_observation(repo, project=project)

    return files_processed, total_chunks_stored
