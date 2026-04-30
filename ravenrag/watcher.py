"""
Watcher: Auto-reindex files when they change on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .index import DocumentIndex
    from .splitter import TextSplitter

logger = logging.getLogger(__name__)


def watch_directory(
    path: str,
    index: DocumentIndex,
    extensions: Optional[List[str]] = None,
    splitter: Optional[TextSplitter] = None,
) -> None:
    """Watch a directory and auto-reindex changed files.

    Requires: ``pip install 'ravenrag[watch]'``

    Args:
        path: Directory to watch.
        index: DocumentIndex to add documents to.
        extensions: File extensions to watch (e.g. [".txt", ".md"]).
        splitter: Optional TextSplitter for chunking.
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

    for changes in watch(str(dir_path)):
        for change_type, file_path_str in changes:
            fp = Path(file_path_str)

            if fp.suffix not in extensions:
                continue

            if change_type in (Change.added, Change.modified):
                try:
                    doc = load_text(str(fp))
                    docs = [doc]
                    if splitter:
                        docs = splitter.split_documents(docs)
                    index.add(docs)
                    logger.info("Indexed: %s (%d chunks)", fp.name, len(docs))
                except Exception:
                    logger.warning("Failed to index %s", fp, exc_info=True)

            elif change_type == Change.deleted:
                logger.info("File deleted: %s (manual re-index may be needed)", fp.name)
