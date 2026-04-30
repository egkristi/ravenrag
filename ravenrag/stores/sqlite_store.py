"""
SqliteVecStore: SQLite-backed vector store with no external dependencies.

Uses pure-Python brute-force cosine similarity over numpy arrays.
Data stored in SQLite for persistence and portability.

No extra packages required beyond numpy (already a core dependency).
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..index import Document


def _cosine_distances(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine distance (1 - cosine_sim) between query and each row."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    matrix_norm = matrix / norms
    similarities = matrix_norm @ query_norm
    return 1.0 - similarities


class SqliteVecStore:
    """SQLite-backed vector store with brute-force cosine search.

    Args:
        persist_dir: Directory for the SQLite database file.
        collection_name: Table name prefix.
    """

    def __init__(
        self,
        persist_dir: str = "./ravenrag_db",
        collection_name: str = "documents",
    ):
        os.makedirs(persist_dir, exist_ok=True)
        db_path = Path(persist_dir) / f"{collection_name}.sqlite"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding BLOB NOT NULL
            )"""
        )
        self._conn.commit()

    def upsert(self, documents: List["Document"], embeddings: List[List[float]]) -> None:
        """Add or update documents with their embeddings."""
        rows = []
        for doc, emb in zip(documents, embeddings):
            emb_blob = np.array(emb, dtype=np.float32).tobytes()
            meta_json = json.dumps(doc.metadata or {}, ensure_ascii=False)
            rows.append((doc.id, doc.text, meta_json, emb_blob))
        self._conn.executemany(
            "INSERT OR REPLACE INTO documents (id, text, metadata, embedding) VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Find the most similar documents using cosine distance."""
        rows = self._conn.execute("SELECT id, text, metadata, embedding FROM documents").fetchall()
        if not rows:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        dim = len(query_embedding)

        ids = []
        texts = []
        metas = []
        embeddings = []

        for row_id, text, meta_json, emb_blob in rows:
            meta = json.loads(meta_json)
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            ids.append(row_id)
            texts.append(text)
            metas.append(meta)
            embeddings.append(np.frombuffer(emb_blob, dtype=np.float32)[:dim])

        if not ids:
            return []

        matrix = np.stack(embeddings)
        distances = _cosine_distances(query_vec, matrix)

        top_indices = np.argsort(distances)[:top_k]
        results = []
        for idx in top_indices:
            results.append(
                {
                    "id": ids[idx],
                    "text": texts[idx],
                    "metadata": metas[idx],
                    "distance": float(distances[idx]),
                }
            )
        return results

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._conn.commit()

    def count(self) -> int:
        """Return total document count."""
        return self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    def get_all(self) -> Dict:
        """Retrieve all documents and their metadata."""
        rows = self._conn.execute("SELECT id, text, metadata FROM documents").fetchall()
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [json.loads(r[2]) for r in rows],
        }

    def get_by_ids(self, ids: List[str]) -> Dict:
        """Retrieve specific documents by their IDs."""
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, text, metadata FROM documents WHERE id IN ({placeholders})",  # noqa: S608
            ids,
        ).fetchall()
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1] for r in rows],
            "metadatas": [json.loads(r[2]) for r in rows],
        }

    def clear(self) -> None:
        """Delete all documents."""
        self._conn.execute("DELETE FROM documents")
        self._conn.commit()
