"""
VectorStore: ChromaDB-backed persistent vector storage.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, List, Optional

import chromadb

if TYPE_CHECKING:
    from .index import Document

_PLACEHOLDER_METADATA = {"_ravenrag_placeholder": True}


def _strip_placeholder(metadata: Optional[Dict]) -> Dict:
    """Remove placeholder metadata used for ChromaDB compatibility."""
    if metadata == _PLACEHOLDER_METADATA:
        return {}
    return metadata or {}


class VectorStore:
    """Wraps ChromaDB for document storage and similarity search."""

    def __init__(self, persist_dir: str = "./ravenrag_db", collection_name: str = "documents"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, documents: List["Document"], embeddings: List[List[float]]) -> None:
        """Add or update documents with their embeddings."""
        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata if doc.metadata else None for doc in documents]
        # ChromaDB rejects empty dicts; pass None to omit metadata
        has_metadata = any(m is not None for m in metadatas)

        kwargs: Dict = {
            "ids": ids,
            "embeddings": embeddings,
            "documents": texts,
        }
        if has_metadata:
            kwargs["metadatas"] = [m if m else _PLACEHOLDER_METADATA for m in metadatas]

        self.collection.upsert(**kwargs)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Find the most similar documents, optionally filtered by metadata."""
        total = self.collection.count()
        if total == 0:
            return []

        kwargs: Dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, total),
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        # Flatten results
        output = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": _strip_placeholder(results["metadatas"][0][i]),
                    "distance": results["distances"][0][i],
                }
            )
        return output

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        self.collection.delete(ids=[doc_id])

    def count(self) -> int:
        """Return total document count."""
        return self.collection.count()

    def get_all(self) -> Dict:
        """Retrieve all documents and their metadata."""
        result = self.collection.get(include=["documents", "metadatas"])
        if result.get("metadatas"):
            result["metadatas"] = [_strip_placeholder(m) for m in result["metadatas"]]
        return result

    def clear(self) -> None:
        """Delete all documents in the collection."""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
