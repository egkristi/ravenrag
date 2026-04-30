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
| 📂 **File loaders** | Load .txt, .md, .py and more from disk |
| 🏷️ **Metadata filtering** | Filter search results by metadata |
| 🔀 **Hybrid search** | Vector + BM25 with reciprocal rank fusion |
| 🎯 **Reranking** | Cross-encoder reranking for precision |
| 💬 **Context formatting** | LLM-ready prompt generation with sources |
| 🖥️ **CLI** | `raven index`, `raven query`, `raven watch` |
| 🔌 **Pluggable backends** | sentence-transformers, Ollama, or your own |
| 👁️ **Watch mode** | Auto-reindex on file changes |

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

# Query
raven query "What is retrieval-augmented generation?"

# Get a formatted LLM prompt
raven prompt "Explain RAG" -k 3

# Watch for changes and auto-reindex
raven watch ./docs --extensions ".md,.txt"

# Show stats
raven info
```

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

## 📦 Architecture

```
ravenrag/
├── index.py    → DocumentIndex, Document, QueryResult
├── store.py    → VectorStore (ChromaDB wrapper)
├── embed.py    → EmbeddingBackend protocol, Embedder, OllamaBackend
├── splitter.py → TextSplitter, TokenSplitter
├── loaders.py  → load_text, load_directory
├── rerank.py   → Reranker (cross-encoder)
├── hybrid.py   → HybridSearcher (BM25 + vector fusion)
├── context.py  → ContextFormatter (LLM prompt builder)
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
- [x] File loaders
- [x] Metadata filtering
- [x] Hybrid search (BM25 + vector)
- [x] Cross-encoder reranking
- [x] LLM context formatting
- [x] CLI tool
- [x] Watch mode
- [x] Pluggable embedding backends (Ollama, custom)
- [x] Named collections
- [ ] Async support
- [ ] PDF / DOCX loaders
- [ ] Streaming query results

---

## 📝 License

MIT — see [LICENSE](LICENSE)

---

> *RavenRAG: Remember what matters.*
