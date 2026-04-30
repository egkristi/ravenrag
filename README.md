# 🐦‍⬛ RavenRAG

[![CI](https://github.com/egkristi/ravenrag/actions/workflows/ci.yml/badge.svg)](https://github.com/egkristi/ravenrag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/ravenrag)](https://pypi.org/project/ravenrag/)

> *"Index your docs in one command, query them from anywhere."*

A lightweight, local-first **RAG (Retrieval-Augmented Generation)** library built for minimal dependencies and maximum simplicity.

No cloud required. No API keys. Just local embeddings, persistent vector storage, and clean retrieval — with a CLI, hybrid search, reranking, and LLM-ready context formatting.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏠 **Local-first** | Runs entirely on your machine — no external APIs |
| 🪶 **Lightweight** | Minimal core dependencies |
| 💾 **Persistent** | Vector store survives restarts |
| 🔍 **Semantic search** | Built on sentence-transformers embeddings |
| 🧠 **Semantic splitting** | Split at meaning boundaries, not character counts |
| 🧩 **Composable** | Mix and match index, store, embedder, reranker |
| ✂️ **Chunking** | Character, token-aware, and semantic splitting |
| 📂 **File loaders** | Load .txt, .md, .py and more — with plugin system |
| 🏷️ **Metadata filtering** | Filter search results by metadata |
| 🔀 **Hybrid search** | Vector + BM25 with reciprocal rank fusion |
| 🎯 **Reranking** | Cross-encoder reranking for precision |
| 🪆 **Parent-child** | Search chunks, return full parent documents |
| 💬 **Context formatting** | LLM-ready prompt generation with citations |
| 📌 **Citations** | Full provenance: source file + chunk reference |
| 📊 **Retrieval eval** | Built-in MRR, NDCG, Recall@k metrics |
| 🖥️ **CLI** | `raven index`, `raven query`, `raven serve`, `raven mcp`, `raven doctor` |
| 🌐 **API server** | Built-in HTTP server with auth & CORS |
| 📡 **MCP server** | Model Context Protocol for AI assistants |
| 🔌 **Pluggable backends** | sentence-transformers, Ollama, vLLM, OpenAI-compatible, or your own |
| 👁️ **Watch mode** | Auto-reindex on file changes (with debounce & delete) |
| ⚙️ **Config file** | `ravenrag.toml` or `pyproject.toml` + env vars |
| 🔄 **Incremental indexing** | Skip unchanged files via fingerprints |
| 💾 **Export/import** | JSONL backup and restore |
| 🧩 **Plugin loaders** | `@register_loader(".pdf")` for custom file types |

---

## 🚀 Quick Start

```bash
pip install ravenrag
```

```python
from ravenrag import DocumentIndex, Document

index = DocumentIndex(persist_dir="./my_docs")

docs = [
    Document("RAG stands for Retrieval-Augmented Generation."),
    Document("ChromaDB is a vector database for embeddings."),
    Document("Sentence-transformers provide local text embeddings."),
]
index.add(docs)

results = index.query("What is RAG?", top_k=2)
for r in results:
    print(f"{r.distance:.4f} | {r.text}")
```

---

## 🖥️ CLI

```bash
# Index a directory (incremental — skips unchanged files)
raven index ./docs --glob "**/*.md" --chunk-size 512

# Query (with optional hybrid search and reranking)
raven query "What is retrieval-augmented generation?"
raven query "auth flow" --hybrid --rerank -k 10

# Get a formatted LLM prompt
raven prompt "Explain RAG" -k 3

# Start the API server (with optional auth)
raven serve --port 8484

# Watch for changes and auto-reindex
raven watch ./docs --extensions ".md,.txt"

# Export/import (backup & restore)
raven export -o backup.jsonl
raven import backup.jsonl

# Diagnostics
raven doctor

# MCP server (for Claude, Copilot, Cursor)
raven mcp

# Show stats
raven info

# Debug mode
raven query "test" --verbose
```

---

## 🌐 API Server

Start a local HTTP server that any LLM app can query:

```bash
raven serve
# 🐦‍⬛ RavenRAG server running on http://127.0.0.1:8484

# With API key authentication
RAVENRAG_API_KEY=my-secret raven serve
```

```bash
# Search
curl -X POST http://localhost:8484/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-secret" \
  -d '{"query": "What is RAG?", "top_k": 3}'

# Get LLM-ready prompt
curl -X POST http://localhost:8484/prompt \
  -d '{"query": "Explain embeddings"}'

# Index new documents via API
curl -X POST http://localhost:8484/index \
  -d '{"documents": [{"text": "New doc", "metadata": {"source": "api"}}]}'

# Health check
curl http://localhost:8484/health
```

---

## ⚙️ Config File

Create a `ravenrag.toml` in your project root (or add `[tool.ravenrag]` to `pyproject.toml`):

```toml
[index]
persist_dir = "./my_db"
model = "all-MiniLM-L6-v2"
collection = "docs"
chunk_size = 512
chunk_overlap = 64

[search]
top_k = 5
rerank = true
hybrid = true
alpha = 0.6

[server]
host = "127.0.0.1"
port = 8484

[watch]
extensions = [".md", ".txt", ".py"]
```

CLI flags override config values. Config is auto-discovered by walking up from cwd.

### Environment Variables

| Variable | Overrides |
|----------|-----------|
| `RAVENRAG_DB` | `index.persist_dir` |
| `RAVENRAG_COLLECTION` | `index.collection` |
| `RAVENRAG_MODEL` | `index.model` |
| `RAVENRAG_TOP_K` | `search.top_k` |
| `RAVENRAG_HOST` | `server.host` |
| `RAVENRAG_PORT` | `server.port` |
| `RAVENRAG_API_KEY` | `server.api_key` |

---

## ✂️ Chunking

### Character-based

```python
from ravenrag import TextSplitter, Document

splitter = TextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_documents([Document("Very long text..." * 100)])
```

### Token-aware

```python
from ravenrag import TokenSplitter

splitter = TokenSplitter(chunk_size=256, chunk_overlap=32, model_name="all-MiniLM-L6-v2")
chunks = splitter.split_documents(docs)
```

### Semantic splitting

Split at natural meaning boundaries using embedding cosine similarity:

```python
from ravenrag import SemanticSplitter, Embedder

embedder = Embedder("all-MiniLM-L6-v2")
splitter = SemanticSplitter(embedder, threshold=0.5)
chunks = splitter.split_documents(docs)
```

---

## 📂 Loading Files

```python
from ravenrag import load_text, load_directory, DocumentIndex, TextSplitter

doc = load_text("notes.md")
docs = load_directory("./my_docs", glob="**/*.md")

splitter = TextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_documents(docs)

index = DocumentIndex(persist_dir="./my_index")
index.add(chunks)
```

---

## 💬 Context Formatting for LLMs

```python
# One-liner: query → formatted prompt
prompt = index.query_for_prompt("What is RAG?", top_k=3)
# Send `prompt` directly to your LLM

# Custom template
prompt = index.query_for_prompt(
    "Explain embeddings",
    template="Context:\n{context}\n\nAnswer the question: {query}",
)
```

---

## 🔀 Hybrid Search

Combines vector similarity with BM25 keyword matching via Reciprocal Rank Fusion:

```bash
pip install 'ravenrag[hybrid]'
```

```python
results = index.hybrid_query("retrieval augmented generation", alpha=0.5)
# alpha=1.0 → pure vector, alpha=0.0 → pure BM25
```

---

## 🎯 Reranking

Use a cross-encoder to rerank initial results for higher precision:

```python
results = index.query("What is RAG?", top_k=5, rerank=True)
# Fetches 4x results, reranks with cross-encoder, returns top 5
```

---

## 🏷️ Metadata Filtering

```python
results = index.query("machine learning", where={"source": "papers"})
```

---

## 🔌 Embedding Backends

### sentence-transformers (default)

```python
from ravenrag import DocumentIndex, Embedder

index = DocumentIndex(persist_dir="./db", embedding_backend=Embedder("all-MiniLM-L6-v2"))
```

### Ollama (local)

```python
from ravenrag import DocumentIndex, OllamaBackend

backend = OllamaBackend(model_name="nomic-embed-text")
index = DocumentIndex(persist_dir="./db", embedding_backend=backend)
```

### vLLM

```bash
vllm serve BAAI/bge-base-en-v1.5 --task embed
```

```python
from ravenrag import DocumentIndex, VLLMBackend

backend = VLLMBackend("BAAI/bge-base-en-v1.5")  # localhost:8000 by default
index = DocumentIndex(persist_dir="./db", embedding_backend=backend)
```

### OpenAI-compatible (LM Studio, LocalAI, TGI, Xinference, OpenAI, …)

```python
from ravenrag import DocumentIndex, OpenAIBackend

# LM Studio
backend = OpenAIBackend(model_name="nomic-embed-text-v1.5", base_url="http://localhost:1234/v1")

# OpenAI cloud
backend = OpenAIBackend(
    model_name="text-embedding-3-small",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)

index = DocumentIndex(persist_dir="./db", embedding_backend=backend)
```

### Your own backend

```python
from ravenrag import EmbeddingBackend

class MyBackend:
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...  # Your implementation

    def encode_batched(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        ...  # Your implementation

index = DocumentIndex(embedding_backend=MyBackend())
```

---

## � Citations & Provenance

Every result includes a `.citation` property for full traceability:

```python
results = index.query("auth flow", top_k=3)
for r in results:
    print(f"{r.citation}: {r.text[:80]}")
    # docs/auth.md#chunk3: The authentication flow starts with...
```

Citations are automatically included in LLM prompts via `query_for_prompt()`.

---

## 🧩 Plugin Loaders

Register custom loaders for any file type:

```python
from ravenrag import register_loader, Document

@register_loader(".pdf")
def load_pdf(path, metadata=None):
    text = my_pdf_extract(path)
    return Document(text=text, metadata={"source": path, **(metadata or {})})

# Now load_directory() automatically handles .pdf files
docs = load_directory("./docs", glob="**/*.*")
```

---

## 📦 Architecture

```
ravenrag/
├── index.py        → DocumentIndex, Document, QueryResult, query_parent
├── store.py        → VectorStore, VectorStoreBackend protocol
├── embed.py        → EmbeddingBackend, Embedder, Ollama/OpenAI/vLLM backends
├── splitter.py     → TextSplitter, TokenSplitter, SemanticSplitter
├── loaders.py      → load_text, load_directory, register_loader
├── rerank.py       → Reranker (cross-encoder)
├── hybrid.py       → HybridSearcher (BM25 + vector fusion)
├── context.py      → ContextFormatter (LLM prompt builder)
├── config.py       → RavenConfig, load_config (TOML + env vars)
├── server.py       → HTTP API server (auth, CORS, size limits)
├── mcp_server.py   → MCP stdio server (for AI assistants)
├── watcher.py      → watch_directory (debounce, delete support)
├── fingerprint.py  → FingerprintStore (incremental re-indexing)
├── eval.py         → evaluate (MRR, NDCG, Recall)
├── export.py       → export_index, import_index (JSONL)
└── cli.py          → CLI (raven command)
```

---

## 🛠️ Installation

```bash
# Core
pip install ravenrag

# With hybrid search
pip install 'ravenrag[hybrid]'

# With file watching
pip install 'ravenrag[watch]'

# With token-aware splitting
pip install 'ravenrag[tokens]'

# Everything
pip install 'ravenrag[all]'
```

### From source (uv)

```bash
git clone https://github.com/egkristi/ravenrag.git
cd ravenrag
uv sync --dev
```

### From source (pip)

```bash
pip install -e ".[dev]"
```

---

## 🧪 Tests

```bash
# Fast unit tests (mocked, no model download)
uv run pytest tests/ -m "not integration"

# Full integration tests (downloads model on first run)
uv run pytest tests/ -m "integration"

# All tests with coverage
uv run pytest tests/ -v --cov=ravenrag
```

---

## 🗺️ Roadmap

- [x] Document chunking (character + token-aware + semantic)
- [x] File loaders (with plugin system)
- [x] Metadata filtering
- [x] Hybrid search (BM25 + vector)
- [x] Cross-encoder reranking
- [x] LLM context formatting with citations
- [x] CLI tool
- [x] Watch mode (debounce, delete support)
- [x] Pluggable embedding backends (Ollama, vLLM, OpenAI-compatible, custom)
- [x] Named collections
- [x] Config file support (ravenrag.toml + env vars)
- [x] Built-in API server (auth, CORS)
- [x] Citation & provenance tracking
- [x] Plugin loader system
- [x] MCP server for AI assistants
- [x] Retrieval evaluation (MRR, NDCG, Recall)
- [x] Parent-child document retrieval
- [x] Incremental re-indexing (fingerprints)
- [x] Export/import (JSONL)
- [x] VectorStoreBackend protocol
- [ ] Async support
- [ ] Knowledge graph retrieval
- [ ] PDF / DOCX loaders (built-in)
- [ ] Streaming query results
- [ ] OpenAPI schema for server

---

## 📝 License

Dual-licensed under **AGPLv3** and a **commercial license**.

- Open source use: [AGPLv3](LICENSES/AGPLv3.txt)
- Commercial use: [Commercial License](LICENSES/COMMERCIAL.txt)
- Plain-English breakdown: [LICENSING.md](LICENSING.md)

---

> *RavenRAG: Remember what matters.*
