"""
RavenRAG — Lightweight RAG for local documents.

Local-first, composable retrieval-augmented generation:
- Persistent vector storage with ChromaDB (swappable via VectorStoreBackend)
- Multiple embedding backends (sentence-transformers, Ollama, OpenAI-compatible, vLLM)
- Text, token-aware, and semantic chunking
- Hybrid search (vector + BM25)
- Cross-encoder reranking
- Parent-child document retrieval
- LLM context formatting with citations
- Retrieval quality evaluation (MRR, NDCG, Recall)
- CLI, file watching, HTTP API server, and MCP server
- Config file support (ravenrag.toml) with env var overrides
- Plugin loader system for custom file types
- Incremental re-indexing with fingerprints
- Export/import (JSONL backup/restore)
"""

__version__ = "0.5.0"

from .config import RavenConfig, load_config
from .context import ContextFormatter
from .embed import Embedder, EmbeddingBackend, OllamaBackend, OpenAIBackend, VLLMBackend
from .eval import EvalResult, evaluate
from .hybrid import HybridSearcher
from .index import Document, DocumentIndex, QueryResult
from .loaders import get_registered_extensions, load_directory, load_text, register_loader
from .rerank import Reranker
from .splitter import SemanticSplitter, TextSplitter, TokenSplitter
from .store import VectorStore, VectorStoreBackend

__all__ = [
    "ContextFormatter",
    "Document",
    "DocumentIndex",
    "Embedder",
    "EmbeddingBackend",
    "EvalResult",
    "HybridSearcher",
    "OllamaBackend",
    "OpenAIBackend",
    "QueryResult",
    "RavenConfig",
    "Reranker",
    "SemanticSplitter",
    "TextSplitter",
    "TokenSplitter",
    "VLLMBackend",
    "VectorStore",
    "VectorStoreBackend",
    "evaluate",
    "get_registered_extensions",
    "load_config",
    "load_directory",
    "load_text",
    "register_loader",
]
