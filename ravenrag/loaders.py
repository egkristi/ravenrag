"""
Document loaders: Load files from disk into Document objects.

Supports a plugin system for custom file type handlers::

    from ravenrag.loaders import register_loader

    @register_loader(".pdf")
    def load_pdf(path, metadata=None):
        ...  # return Document
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .index import Document

logger = logging.getLogger(__name__)

# Plugin registry: extension -> loader function
_loader_registry: Dict[str, Callable] = {}


def register_loader(extension: str):
    """Register a custom loader for a file extension.

    The decorated function receives (path: str, metadata: dict | None)
    and must return a Document.

    Example::

        @register_loader(".pdf")
        def load_pdf(path, metadata=None):
            text = extract_text_from_pdf(path)
            return Document(text=text, metadata={"source": path, **(metadata or {})})
    """

    def decorator(func: Callable) -> Callable:
        ext = extension if extension.startswith(".") else f".{extension}"
        _loader_registry[ext] = func
        logger.debug("Registered loader for %s: %s", ext, func.__name__)
        return func

    return decorator


def get_registered_extensions() -> List[str]:
    """Return list of file extensions with registered loaders."""
    return list(_loader_registry.keys())


def load_text(path: str, metadata: Optional[Dict] = None) -> Document:
    """Load a single text file as a Document."""
    file_path = Path(path).resolve()
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
    Custom file types can be handled via ``register_loader()``.
    """
    dir_path = Path(path).resolve()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    documents: List[Document] = []
    for file_path in sorted(dir_path.glob(glob)):
        if not file_path.is_file():
            continue

        # Path traversal protection: ensure file is within target directory
        try:
            file_path.resolve().relative_to(dir_path)
        except ValueError:
            logger.warning("Skipping %s: outside target directory", file_path)
            continue

        # Check for registered plugin loader
        if file_path.suffix in _loader_registry:
            try:
                doc = _loader_registry[file_path.suffix](str(file_path), metadata)
                documents.append(doc)
                continue
            except Exception as e:
                logger.warning("Plugin loader failed for %s: %s", file_path, e)
                continue

        # Default: read as text
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
