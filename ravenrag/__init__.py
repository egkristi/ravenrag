"""
RavenRAG — Lightweight RAG for local documents.

Local-first, composable retrieval-augmented generation:
- Persistent vector storage with ChromaDB
- Multiple embedding backends (sentence-transformers, Ollama, OpenAI-compatible, vLLM)
- Text and token-aware chunking
- Hybrid search (vector + BM25)
- Cross-encoder reranking
- LLM context formatting with citations
- CLI, file watching, and HTTP API server
- Config file support (ravenrag.toml)
- Plugin loader system for custom file types
"""

__version__ = "0.4.0"

from .config import RavenConfig, load_config
from .context import ContextFormatter
from .embed import Embedder, EmbeddingBackend, OllamaBackend, OpenAIBackend, VLLMBackend
from .hybrid import HybridSearcher
from .index import Document, DocumentIndex, QueryResult
from .loaders import get_registered_extensions, load_directory, load_text, register_loader
from .rerank import Reranker
from .splitter import TextSplitter, TokenSplitter
from .store import VectorStore

__all__ = [
    "ContextFormatter",
    "Document",
    "DocumentIndex",
    "Embedder",
    "EmbeddingBackend",
    "HybridSearcher",
    "OllamaBackend",
    "OpenAIBackend",
    "QueryResult",
    "RavenConfig",
    "Reranker",
    "TextSplitter",
    "TokenSplitter",
    "VectorStore",
    "get_registered_extensions",
    "VLLMBackend",
    "load_config",
    "load_directory",
    "load_text",
    "register_loader",
]
