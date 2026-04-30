"""
RavenRAG — Lightweight RAG for local documents.

Inspired by LlamaIndex but focused on:
- Local-first (no cloud dependencies)
- Minimal footprint
- Easy embedding + retrieval
- Persistent vector storage with ChromaDB
"""

__version__ = "0.2.0"

from .embed import Embedder
from .index import Document, DocumentIndex
from .loaders import load_directory, load_text
from .splitter import TextSplitter
from .store import VectorStore

__all__ = [
    "Document",
    "DocumentIndex",
    "VectorStore",
    "Embedder",
    "TextSplitter",
    "load_text",
    "load_directory",
]
