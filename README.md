# 🐦‍⬛ RavenRAG

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
| 🧩 **Composable** | Mix and match index, store, embedder, reranker |
| ✂️ **Chunking** | Character-based and token-aware text splitting |
| 📂 **File loaders** | Load .txt, .md, .py and more — with plugin system |
| 🏷️ **Metadata filtering** | Filter search results by metadata |
| 🔀 **Hybrid search** | Vector + BM25 with reciprocal rank fusion |
| 🎯 **Reranking** | Cross-encoder reranking for precision |
| 💬 **Context formatting** | LLM-ready prompt generation with citations |
| 📌 **Citations** | Full provenance: source file + chunk reference |
| 🖥️ **CLI** | `raven index`, `raven query`, `raven serve`, `raven watch` |
| 🌐 **API server** | Built-in HTTP server (`raven serve`) — RAG sidecar |
| 🔌 **Pluggable backends** | sentence-transformers, Ollama, or your own |
| 👁️ **Watch mode** | Auto-reindex on file changes |
| ⚙️ **Config file** | `ravenrag.toml` or `pyproject.toml [tool.ravenrag]` |
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
# Index a directory
raven index ./docs --glob "**/*.md" --chunk-size 512

# Query (with optional hybrid search and reranking)
raven query "What is retrieval-augmented generation?"
raven query "auth flow" --hybrid --rerank -k 10

# Get a formatted LLM prompt
raven prompt "Explain RAG" -k 3

# Start the API server
raven serve --port 8484

# Watch for changes and auto-reindex
raven watch ./docs --extensions ".md,.txt"

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
```

```bash
# Search
curl -X POST http://localhost:8484/query \
  -H "Content-Type: application/json" \
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

## 🔌 Custom Embedding Backends

### Ollama (local)

```python
from ravenrag import DocumentIndex, OllamaBackend

backend = OllamaBackend(model_name="nomic-embed-text")
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
├── index.py    → DocumentIndex, Document, QueryResult
├── store.py    → VectorStore (ChromaDB wrapper)
├── embed.py    → EmbeddingBackend protocol, Embedder, OllamaBackend
├── splitter.py → TextSplitter, TokenSplitter
├── loaders.py  → load_text, load_directory, register_loader
├── rerank.py   → Reranker (cross-encoder)
├── hybrid.py   → HybridSearcher (BM25 + vector fusion)
├── context.py  → ContextFormatter (LLM prompt builder)
├── config.py   → RavenConfig, load_config (TOML support)
├── server.py   → HTTP API server (stdlib, zero-dep)
├── watcher.py  → watch_directory (auto-reindex)
└── cli.py      → CLI (raven command)
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

- [x] Document chunking (character + token-aware)
- [x] File loaders (with plugin system)
- [x] Metadata filtering
- [x] Hybrid search (BM25 + vector)
- [x] Cross-encoder reranking
- [x] LLM context formatting with citations
- [x] CLI tool
- [x] Watch mode
- [x] Pluggable embedding backends (Ollama, custom)
- [x] Named collections
- [x] Config file support (ravenrag.toml)
- [x] Built-in API server (raven serve)
- [x] Citation & provenance tracking
- [x] Plugin loader system
- [ ] Async support
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
