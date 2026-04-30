# 🐦‍⬛ RavenRAG

> *"Memory over noise."*

A lightweight, local-first **RAG (Retrieval-Augmented Generation)** library inspired by [LlamaIndex](https://www.llamaindex.ai/) but built for minimal dependencies and maximum simplicity.

No cloud required. No API keys. Just local embeddings, persistent vector storage, and clean retrieval.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🏠 **Local-first** | Runs entirely on your machine — no external APIs |
| 🪶 **Lightweight** | ~3 core dependencies (ChromaDB, sentence-transformers, numpy) |
| 💾 **Persistent** | Vector store survives restarts |
| 🔍 **Semantic search** | Built on sentence-transformers embeddings |
| 🧩 **Composable** | Mix and match index, store, and embedder |
| ✂️ **Chunking** | Built-in text splitter with overlapping chunks |
| 📂 **File loaders** | Load .txt, .md, .py and more from disk |
| 🏷️ **Metadata filtering** | Filter search results by metadata |

---

## 🚀 Quick Start

```bash
pip install ravenrag
```

```python
from ravenrag import DocumentIndex, Document

# Create an index
index = DocumentIndex(persist_dir="./my_docs")

# Add documents
docs = [
    Document("RAG stands for Retrieval-Augmented Generation."),
    Document("ChromaDB is a vector database for embeddings."),
    Document("Sentence-transformers provide local text embeddings."),
]
index.add(docs)

# Query
results = index.query("What is RAG?", top_k=2)
for r in results:
    print(f"Score: {r['distance']:.4f} | {r['text']}")
```

---

## ✂️ Chunking Long Documents

```python
from ravenrag import DocumentIndex, Document, TextSplitter

splitter = TextSplitter(chunk_size=512, chunk_overlap=64)

docs = [Document("Very long document text..." * 100)]
chunked = splitter.split_documents(docs)

index = DocumentIndex(persist_dir="./chunked_db")
index.add(chunked)
```

---

## 📂 Loading Files from Disk

```python
from ravenrag import load_text, load_directory, DocumentIndex, TextSplitter

# Single file
doc = load_text("notes.md")

# Entire directory (recursive)
docs = load_directory("./my_docs", glob="**/*.md")

# Combine with chunking
splitter = TextSplitter(chunk_size=512, chunk_overlap=64)
chunked = splitter.split_documents(docs)

index = DocumentIndex(persist_dir="./my_index")
index.add(chunked)
```

---

## 🏷️ Metadata Filtering

```python
results = index.query(
    "machine learning",
    top_k=5,
    where={"source": "research_papers"}
)
```

---

## 📦 Architecture

```
ravenrag/
├── index.py    → DocumentIndex + Document (high-level API)
├── store.py    → VectorStore (ChromaDB wrapper)
├── embed.py    → Embedder (sentence-transformers wrapper)
├── splitter.py → TextSplitter (chunking)
└── loaders.py  → File loaders (text, directory)
```

Each component can be used independently:

```python
from ravenrag.embed import Embedder
from ravenrag.store import VectorStore

embedder = Embedder(model_name="all-MiniLM-L6-v2")
store = VectorStore(persist_dir="./db")

embeddings = embedder.encode(["Hello world"])
store.upsert([doc], embeddings)
```

---

## 🛠️ Installation

### From source (uv)

```bash
git clone https://github.com/egkristi/ravenrag.git
cd ravenrag
uv sync --dev
```

### From source (pip)

```bash
pip install -e ".[dev]"

---

## 🧪 Tests

```bash
# Fast unit tests (mocked, no model download)
uv run pytest tests/ -m "not integration"

# Full integration tests (downloads model on first run)
uv run pytest tests/ -m "integration"

# All tests
uv run pytest tests/ -v
```

---

## 🗺️ Roadmap

- [x] Document chunking strategies
- [x] File loaders
- [x] Metadata filtering
- [ ] Async support
- [ ] Streaming query results
- [ ] Multiple embedding backends (ONNX, Ollama)
- [ ] Hybrid search (BM25 + vector)
- [ ] PDF / DOCX loaders

---

## 📝 License

MIT — see [LICENSE](LICENSE)

---

> *RavenRAG: Remember what matters.*
