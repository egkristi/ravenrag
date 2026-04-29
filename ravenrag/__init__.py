"""
RavenRAG — Lightweight RAG for local documents.

Inspired by LlamaIndex but focused on:
- Local-first (no cloud dependencies)
- Minimal footprint
- Easy embedding + retrieval
- Persistent vector storage with ChromaDB
"""

__version__ = "0.1.0"

from .index import DocumentIndex
from .store import VectorStore
from .embed import Embedder

__all__ = ["DocumentIndex", "VectorStore", "Embedder"]
