"""
Watcher: Auto-reindex files when they change on disk.

Features:
- Debounced: batches rapid file changes into single re-index operations.
- Handles deletions: removes documents from index when files are deleted.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .index import DocumentIndex
    from .splitter import TextSplitter

logger = logging.getLogger(__name__)

_DEFAULT_DEBOUNCE_SECONDS = 0.5


def _file_doc_id(file_path: Path) -> str:
    """Deterministic document ID from file path."""
    return hashlib.sha256(str(file_path.resolve()).encode()).hexdigest()


def watch_directory(
    path: str,
    index: DocumentIndex,
    extensions: Optional[List[str]] = None,
    splitter: Optional[TextSplitter] = None,
    debounce: float = _DEFAULT_DEBOUNCE_SECONDS,
) -> None:
    """Watch a directory and auto-reindex changed files.

    Requires: ``pip install 'ravenrag[watch]'``

    Args:
        path: Directory to watch.
        index: DocumentIndex to add documents to.
        extensions: File extensions to watch (e.g. [".txt", ".md"]).
        splitter: Optional TextSplitter for chunking.
        debounce: Seconds to wait before processing batched changes.
    """
    try:
        from watchfiles import Change, watch
    except ImportError:
        raise ImportError(
            "watchfiles is required for watch mode. Install with: pip install 'ravenrag[watch]'"
        ) from None

    from .loaders import load_text

    dir_path = Path(path).resolve()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    if extensions is None:
        extensions = [".txt", ".md", ".py"]

    logger.info("Watching %s for changes (extensions: %s)", path, extensions)

    # Track indexed file → doc IDs for deletion
    _file_doc_ids: Dict[str, List[str]] = {}

    for changes in watch(str(dir_path)):
        # Debounce: collect changes in a short window
        pending_upsert: Set[str] = set()
        pending_delete: Set[str] = set()

        for change_type, file_path_str in changes:
            fp = Path(file_path_str)
            if fp.suffix not in extensions:
                continue
            if change_type in (Change.added, Change.modified):
                pending_upsert.add(file_path_str)
                pending_delete.discard(file_path_str)
            elif change_type == Change.deleted:
                pending_delete.add(file_path_str)
                pending_upsert.discard(file_path_str)

        if debounce > 0:
            time.sleep(debounce)

        # Process deletions
        for file_path_str in pending_delete:
            doc_ids = _file_doc_ids.pop(file_path_str, None)
            if doc_ids:
                for doc_id in doc_ids:
                    try:
                        index.delete(doc_id)
                    except Exception:
                        logger.warning("Failed to delete doc %s", doc_id, exc_info=True)
                logger.info("Deleted %d chunks for %s", len(doc_ids), Path(file_path_str).name)
            else:
                logger.info("File deleted: %s (not tracked)", Path(file_path_str).name)

        # Process upserts
        for file_path_str in pending_upsert:
            fp = Path(file_path_str)
            try:
                doc = load_text(str(fp))
                docs = [doc]
                if splitter:
                    docs = splitter.split_documents(docs)
                index.add(docs)
                _file_doc_ids[file_path_str] = [d.id for d in docs]
                logger.info("Indexed: %s (%d chunks)", fp.name, len(docs))
            except Exception:
                logger.warning("Failed to index %s", fp, exc_info=True)
