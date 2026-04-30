"""
FaissStore: In-memory FAISS-backed vector store.

Requires: ``pip install faiss-cpu`` (or ``faiss-gpu``).

Fast, lightweight, no server needed. Data persisted to disk as
numpy arrays + JSON metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from ..index import Document


class FaissStore:
    """FAISS-backed vector store with JSON metadata persistence.

    Args:
        persist_dir: Directory to save index and metadata.
        collection_name: Logical collection name (used for file naming).
        dimension: Embedding dimension. Auto-detected on first upsert if None.
    """

    def __init__(
        self,
        persist_dir: str = "./ravenrag_db",
        collection_name: str = "documents",
        dimension: Optional[int] = None,
    ):
        try:
            import faiss  # noqa: F401
        except ImportError:
            raise ImportError("faiss-cpu is required for FaissStore. Install with: pip install faiss-cpu") from None

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._dimension = dimension

        os.makedirs(persist_dir, exist_ok=True)
        self._index_path = Path(persist_dir) / f"{collection_name}.faiss"
        self._meta_path = Path(persist_dir) / f"{collection_name}.meta.json"

        # In-memory state
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._metadatas: List[Dict] = []
        self._faiss_index = None

        self._load()

    def _load(self) -> None:
        """Load persisted index and metadata from disk."""
        import faiss

        if self._meta_path.exists():
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._ids = data.get("ids", [])
            self._texts = data.get("texts", [])
            self._metadatas = data.get("metadatas", [])

        if self._index_path.exists() and self._ids:
            self._faiss_index = faiss.read_index(str(self._index_path))
            self._dimension = self._faiss_index.d
        elif self._dimension:
            self._faiss_index = faiss.IndexFlatL2(self._dimension)

    def _save(self) -> None:
        """Persist index and metadata to disk."""
        import faiss

        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, str(self._index_path))

        data = {
            "ids": self._ids,
            "texts": self._texts,
            "metadatas": self._metadatas,
        }
        self._meta_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def upsert(self, documents: List["Document"], embeddings: List[List[float]]) -> None:
        """Add or update documents with their embeddings."""
        import faiss

        if not documents:
            return

        dim = len(embeddings[0])
        if self._faiss_index is None:
            self._dimension = dim
            self._faiss_index = faiss.IndexFlatL2(dim)

        for doc, emb in zip(documents, embeddings):
            if doc.id in self._ids:
                # Update: remove old, add new
                idx = self._ids.index(doc.id)
                self._ids.pop(idx)
                self._texts.pop(idx)
                self._metadatas.pop(idx)
                # Rebuild index (FAISS doesn't support single-row removal for IndexFlatL2)
                self._rebuild_faiss_without(idx)

            self._ids.append(doc.id)
            self._texts.append(doc.text)
            self._metadatas.append(doc.metadata or {})
            vec = np.array([emb], dtype=np.float32)
            self._faiss_index.add(vec)

        self._save()

    def _rebuild_faiss_without(self, remove_idx: int) -> None:
        """Rebuild FAISS index after removing a row."""
        import faiss

        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return

        n = self._faiss_index.ntotal
        all_vecs = np.empty((n, self._dimension), dtype=np.float32)
        for i in range(n):
            all_vecs[i] = self._faiss_index.reconstruct(i)

        keep = np.delete(all_vecs, remove_idx, axis=0)
        self._faiss_index = faiss.IndexFlatL2(self._dimension)
        if len(keep) > 0:
            self._faiss_index.add(keep)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Find the most similar documents."""
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []

        vec = np.array([query_embedding], dtype=np.float32)
        # Fetch extra if filtering
        fetch_k = min(self._faiss_index.ntotal, top_k * 4 if where else top_k)
        distances, indices = self._faiss_index.search(vec, fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            meta = self._metadatas[idx]
            if where and not all(meta.get(k) == v for k, v in where.items()):
                continue
            results.append(
                {
                    "id": self._ids[idx],
                    "text": self._texts[idx],
                    "metadata": meta,
                    "distance": float(dist),
                }
            )
            if len(results) >= top_k:
                break

        return results

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        if doc_id not in self._ids:
            return
        idx = self._ids.index(doc_id)
        self._ids.pop(idx)
        self._texts.pop(idx)
        self._metadatas.pop(idx)
        self._rebuild_faiss_without(idx)
        self._save()

    def count(self) -> int:
        """Return total document count."""
        return len(self._ids)

    def get_all(self) -> Dict:
        """Retrieve all documents and their metadata."""
        return {
            "ids": list(self._ids),
            "documents": list(self._texts),
            "metadatas": list(self._metadatas),
        }

    def get_by_ids(self, ids: List[str]) -> Dict:
        """Retrieve specific documents by their IDs."""
        result_ids = []
        result_docs = []
        result_metas = []
        for doc_id in ids:
            if doc_id in self._ids:
                idx = self._ids.index(doc_id)
                result_ids.append(doc_id)
                result_docs.append(self._texts[idx])
                result_metas.append(self._metadatas[idx])
        return {"ids": result_ids, "documents": result_docs, "metadatas": result_metas}

    def clear(self) -> None:
        """Delete all documents."""
        import faiss

        self._ids.clear()
        self._texts.clear()
        self._metadatas.clear()
        if self._dimension:
            self._faiss_index = faiss.IndexFlatL2(self._dimension)
        else:
            self._faiss_index = None
        self._save()
