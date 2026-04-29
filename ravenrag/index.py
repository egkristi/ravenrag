"""
DocumentIndex: Core indexing and retrieval logic.
"""

import hashlib
from typing import List, Dict, Optional
from .store import VectorStore
from .embed import Embedder


class Document:
    """A simple document container."""
    def __init__(self, text: str, metadata: Optional[Dict] = None, doc_id: Optional[str] = None):
        self.text = text
        self.metadata = metadata or {}
        self.id = doc_id or hashlib.md5(text.encode()).hexdigest()

    def __repr__(self):
        return f"Document(id={self.id[:8]}..., text={self.text[:50]}...)"


class DocumentIndex:
    """
    Main interface for indexing documents and querying them.
    """
    def __init__(self, persist_dir: str = "./ravenrag_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = Embedder(model_name=embedding_model)
        self.store = VectorStore(persist_dir=persist_dir)

    def add(self, documents: List[Document]) -> None:
        """Index a list of documents."""
        texts = [doc.text for doc in documents]
        embeddings = self.embedder.encode(texts)
        self.store.upsert(documents, embeddings)

    def query(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query the index and return top-k matching documents with scores."""
        query_embedding = self.embedder.encode([query])[0]
        return self.store.search(query_embedding, top_k=top_k)

    def delete(self, doc_id: str) -> None:
        """Remove a document by ID."""
        self.store.delete(doc_id)

    def count(self) -> int:
        """Return the number of indexed documents."""
        return self.store.count()

    def clear(self) -> None:
        """Delete all documents."""
        self.store.clear()
