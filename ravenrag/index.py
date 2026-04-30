"""
DocumentIndex: Core indexing and retrieval logic.
"""

import hashlib
from typing import Dict, List, Optional

from .embed import Embedder
from .store import VectorStore


class Document:
    """A simple document container."""

    def __init__(self, text: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None):
        self.text = text
        self.metadata = metadata or {}
        self.id = doc_id or hashlib.sha256(text.encode()).hexdigest()

    def __repr__(self):
        return f"Document(id={self.id[:8]}..., text={self.text[:50]}...)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Document):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)


class DocumentIndex:
    """
    Main interface for indexing documents and querying them.
    """

    def __init__(
        self,
        persist_dir: str = "./ravenrag_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        self.embedder = Embedder(model_name=embedding_model)
        self.store = VectorStore(persist_dir=persist_dir)
        self.batch_size = batch_size

    def add(self, documents: List[Document]) -> None:
        """Index a list of documents (batched for large sets)."""
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            texts = [doc.text for doc in batch]
            embeddings = self.embedder.encode_batched(texts, batch_size=self.batch_size)
            self.store.upsert(batch, embeddings)

    def query(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict]:
        """Query the index. Optionally filter by metadata with `where`."""
        query_embedding = self.embedder.encode([query])[0]
        return self.store.search(query_embedding, top_k=top_k, where=where)

    def delete(self, doc_id: str) -> None:
        """Remove a document by ID."""
        self.store.delete(doc_id)

    def count(self) -> int:
        """Return the number of indexed documents."""
        return self.store.count()

    def clear(self) -> None:
        """Delete all documents."""
        self.store.clear()
