"""
Export/Import: Backup and restore RavenRAG collections as JSONL.

Usage::

    raven export > backup.jsonl
    raven import backup.jsonl --collection restored
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .index import DocumentIndex

logger = logging.getLogger(__name__)


def export_index(index: DocumentIndex, output: TextIO) -> int:
    """Export all documents from an index as JSONL.

    Each line is a JSON object with keys: id, text, metadata.

    Args:
        index: The DocumentIndex to export.
        output: A writable text stream (e.g. sys.stdout or open file).

    Returns:
        Number of documents exported.
    """
    all_docs = index.store.get_all()
    ids = all_docs.get("ids", [])
    texts = all_docs.get("documents", [])
    metadatas = all_docs.get("metadatas") or [{}] * len(ids)

    count = 0
    for doc_id, text, metadata in zip(ids, texts, metadatas):
        record = {"id": doc_id, "text": text, "metadata": metadata or {}}
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1

    logger.info("Exported %d documents", count)
    return count


def import_index(index: DocumentIndex, input_path: str) -> int:
    """Import documents from a JSONL file into an index.

    Each line must be a JSON object with at least a ``text`` field.
    Optional: ``id``, ``metadata``.

    Args:
        index: The DocumentIndex to import into.
        input_path: Path to the JSONL file.

    Returns:
        Number of documents imported.
    """
    from .index import Document

    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Import file not found: {input_path}")

    docs = []
    for line_num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            logger.warning("Skipping invalid JSON on line %d: %s", line_num, e)
            continue

        text = record.get("text", "")
        if not text:
            logger.warning("Skipping line %d: missing 'text' field", line_num)
            continue

        docs.append(
            Document(
                text=text,
                metadata=record.get("metadata", {}),
                doc_id=record.get("id"),
            )
        )

    if docs:
        index.add(docs)
    logger.info("Imported %d documents from %s", len(docs), input_path)
    return len(docs)
