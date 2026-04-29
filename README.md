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

Output:
```
Score: 0.2341 | RAG stands for Retrieval-Augmented Generation.
Score: 0.4567 | ChromaDB is a vector database for embeddings.
```

---

## 📦 Architecture

```
ravenrag/
├── index.py   → DocumentIndex (high-level API)
├── store.py   → VectorStore (ChromaDB wrapper)
└── embed.py   → Embedder (sentence-transformers wrapper)
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

### From source

```bash
git clone https://github.com/egkristi/ravenrag.git
cd ravenrag
pip install -e ".[dev]"
```

---

## 🧪 Tests

```bash
pytest tests/
```

---

## 🗺️ Roadmap

- [ ] Async support
- [ ] Streaming query results
- [ ] Multiple embedding backends (ONNX, Ollama)
- [ ] Document chunking strategies
- [ ] Hybrid search (BM25 + vector)

---

## 📝 License

MIT — see [LICENSE](LICENSE)

---

> *RavenRAG: Remember what matters.*
