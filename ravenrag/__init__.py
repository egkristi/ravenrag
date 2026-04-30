"""
RavenRAG — Lightweight RAG for local documents.

Local-first, composable retrieval-augmented generation:
- Persistent vector storage with ChromaDB
- Multiple embedding backends (sentence-transformers, Ollama)
- Text and token-aware chunking
- Hybrid search (vector + BM25)
- Cross-encoder reranking
- LLM context formatting
- CLI and file watching
"""

__version__ = "0.3.0"

from .context import ContextFormatter
from .embed import Embedder, EmbeddingBackend, OllamaBackend
from .index import Document, DocumentIndex, QueryResult
from .loaders import load_directory, load_text
from .splitter import TextSplitter, TokenSplitter
from .store import VectorStore

__all__ = [
    "ContextFormatter",
    "Document",
    "DocumentIndex",
    "Embedder",
    "EmbeddingBackend",
    "OllamaBackend",
    "QueryResult",
    "TextSplitter",
    "TokenSplitter",
    "VectorStore",
    "load_directory",
    "load_text",
]
