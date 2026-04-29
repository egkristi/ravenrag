"""
VectorStore: ChromaDB-backed persistent vector storage.
"""

import os
from typing import List, Dict
import chromadb
from chromadb.config import Settings


class VectorStore:
    """Wraps ChromaDB for document storage and similarity search."""

    def __init__(self, persist_dir: str = "./ravenrag_db", collection_name: str = "documents"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert(self, documents: List, embeddings: List[List[float]]) -> None:
        """Add or update documents with their embeddings."""
        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Find the most similar documents."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        # Flatten results
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        self.collection.delete(ids=[doc_id])

    def count(self) -> int:
        """Return total document count."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all documents in the collection."""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
