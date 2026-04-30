"""
Document loaders: Load files from disk into Document objects.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .index import Document

logger = logging.getLogger(__name__)


def load_text(path: str, metadata: Optional[Dict] = None) -> Document:
    """Load a single text file as a Document."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    text = file_path.read_text(encoding="utf-8")
    doc_metadata = {"source": str(file_path), "filename": file_path.name}
    if metadata:
        doc_metadata.update(metadata)

    return Document(text=text, metadata=doc_metadata)


def load_directory(
    path: str,
    glob: str = "**/*.txt",
    metadata: Optional[Dict] = None,
) -> List[Document]:
    """Load all matching files from a directory as Documents.

    Supports: .txt, .md, .py, .json, .csv, .html, .xml, .yaml, .yml, .toml
    """
    dir_path = Path(path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    documents: List[Document] = []
    for file_path in sorted(dir_path.glob(glob)):
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError) as e:
            logger.warning("Skipping %s: %s", file_path, e)
            continue

        doc_metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "extension": file_path.suffix,
        }
        if metadata:
            doc_metadata.update(metadata)

        documents.append(Document(text=text, metadata=doc_metadata))

    return documents
