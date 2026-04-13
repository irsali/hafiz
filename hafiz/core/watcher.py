"""File watcher for real-time re-indexing using watchdog.

Monitors a directory for file changes, debounces rapid saves,
and re-indexes only changed files (chunks + embeddings, no graph extraction).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from hafiz.core.chunker import LANGUAGE_MAP, chunk_file, should_ignore
from hafiz.core.config import get_settings
from hafiz.core.database import close_engine
from hafiz.core.embeddings import embed_texts
from hafiz.core.store import delete_chunks_for_file, store_chunks

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 0.5
EMBED_BATCH_SIZE = 64


class _DebouncedHandler(FileSystemEventHandler):
    """Watchdog handler that debounces rapid changes per file path."""

    def __init__(
        self,
        root: Path,
        project: str | None,
        ignore_patterns: list[str],
        on_reindex: callable | None = None,
        on_delete: callable | None = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.project = project
        self.ignore_patterns = ignore_patterns
        self.on_reindex = on_reindex
        self.on_delete = on_delete
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()

    def _is_relevant(self, path: Path) -> bool:
        """Check if a file should be processed."""
        if not path.is_file() and not path.suffix:
            return False
        if path.suffix.lower() not in LANGUAGE_MAP:
            return False
        if should_ignore(path, self.ignore_patterns):
            return False
        return True

    def _schedule(self, file_path: str, action: str) -> None:
        """Schedule a debounced action for a file path."""
        with self._lock:
            existing = self._timers.get(file_path)
            if existing is not None:
                existing.cancel()

            def _fire():
                with self._lock:
                    self._timers.pop(file_path, None)
                future = asyncio.run_coroutine_threadsafe(
                    self._handle(file_path, action), self._loop
                )
                try:
                    future.result(timeout=120)
                except Exception:
                    logger.exception("Error processing %s (%s)", file_path, action)

            timer = threading.Timer(DEBOUNCE_SECONDS, _fire)
            self._timers[file_path] = timer
            timer.start()

    async def _handle(self, file_path: str, action: str) -> None:
        """Handle a file change or deletion."""
        path = Path(file_path)
        rel_path = str(path.relative_to(self.root))

        if action == "delete":
            deleted = await delete_chunks_for_file(rel_path)
            logger.info("Deleted %d chunks for removed file: %s", deleted, rel_path)
            if self.on_delete:
                self.on_delete(rel_path, deleted)
            return

        # Re-index: delete old chunks, chunk, embed, store
        if not path.exists():
            return

        await delete_chunks_for_file(rel_path)
        chunks = chunk_file(path, relative_to=self.root)
        if not chunks:
            logger.info("No chunks produced for %s", rel_path)
            return

        # Embed in batches
        all_embeddings: list[list[float]] = []
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i : i + EMBED_BATCH_SIZE]
            texts = [c.content for c in batch]
            embeddings = await embed_texts(texts)
            all_embeddings.extend(embeddings)

        stored = await store_chunks(chunks, all_embeddings, project=self.project)
        logger.info("Re-indexed %s: %d chunks stored", rel_path, stored)
        if self.on_reindex:
            self.on_reindex(rel_path, stored)

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._is_relevant(path):
            self._schedule(event.src_path, "modify")

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._is_relevant(path):
            self._schedule(event.src_path, "create")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        # For deletes, check extension directly (file no longer exists)
        path = Path(event.src_path)
        if path.suffix.lower() in LANGUAGE_MAP:
            self._schedule(event.src_path, "delete")

    def shutdown(self) -> None:
        """Cancel all pending timers and stop the async loop."""
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)


def start_watcher(
    root: Path,
    *,
    project: str | None = None,
    on_reindex: callable | None = None,
    on_delete: callable | None = None,
) -> tuple[Observer, _DebouncedHandler]:
    """Start the file watcher on a directory.

    Returns (observer, handler) so the caller can stop them.
    """
    settings = get_settings()
    ignore_patterns = settings.workspace.ignore

    handler = _DebouncedHandler(
        root=root,
        project=project,
        ignore_patterns=ignore_patterns,
        on_reindex=on_reindex,
        on_delete=on_delete,
    )

    observer = Observer()
    observer.schedule(handler, str(root), recursive=True)
    observer.start()
    logger.info("Watching %s for changes (project=%s)", root, project)

    return observer, handler
