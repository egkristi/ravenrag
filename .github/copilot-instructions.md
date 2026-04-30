# RavenRAG — AI Agent Instructions

> Authoritative reference for AI coding agents working on the RavenRAG project.

## 1. Project Identity

| Field | Value |
|-------|-------|
| Name | `ravenrag` |
| Version | `0.7.0` |
| License | AGPL-3.0-or-later |
| Python | `>=3.11` |
| Build | Hatchling (`pyproject.toml`) |
| Package manager | `uv` |
| CLI entry point | `raven` → `ravenrag.cli:app` (Typer) |
| Repo | `github.com/egkristi/ravenrag` |

## 2. Development Commands

```bash
# Install (dev mode with all extras)
uv sync --all-extras

# Lint
uv run ruff check ravenrag/ tests/

# Format check
uv run ruff format --check ravenrag/ tests/

# Auto-fix lint + format
uv run ruff check --fix ravenrag/ tests/ && uv run ruff format ravenrag/ tests/

# Unit tests (fast, mocked, no model download)
uv run pytest tests/ -v --tb=short -m "not integration"

# Unit tests with coverage (CI threshold: 75%)
uv run pytest tests/ -v --tb=short -m "not integration" --cov=ravenrag --cov-report=term-missing --cov-fail-under=75

# Integration tests (downloads model on first run)
uv run pytest tests/ -v --tb=short -m "integration"

# Single test file
uv run pytest tests/test_graph.py -v --tb=short

# Full CI pipeline locally
uv run ruff check ravenrag/ tests/ && uv run ruff format --check ravenrag/ tests/ && uv run pytest tests/ -v --tb=short -m "not integration" --cov=ravenrag --cov-report=term-missing --cov-fail-under=75
```

## 3. CI Pipeline

**File:** `.github/workflows/ci.yml`
**Runner:** `ubuntu-latest` with `astral-sh/setup-uv@v6`
**Matrix:** Python `[3.11, 3.12]`
**Steps:**
1. `uv sync --all-extras`
2. `ruff check` (lint)
3. `ruff format --check` (format)
4. `pytest -m "not integration" --cov-fail-under=75` (unit tests)
5. `pytest -m "integration"` (integration tests)

**After every commit+push:** check CI status with:
```bash
PAGER=cat GH_PAGER=cat GH_FORCE_TTY=0 NO_COLOR=1 gh run list -L 1
```

## 4. Architecture

```
ravenrag/
├── __init__.py          # Public API exports (37 symbols)
├── config.py            # RavenConfig dataclass, TOML loading, env overrides
├── index.py             # DocumentIndex (core), MultiCollectionRouter, QueryResult, Document
├── store.py             # VectorStoreBackend (Protocol), VectorStore (ChromaDB)
├── embed.py             # EmbeddingBackend (Protocol), Embedder, Ollama/OpenAI/vLLM backends
├── splitter.py          # TextSplitter, TokenSplitter, SemanticSplitter
├── hybrid.py            # HybridSearcher (RRF: vector + BM25)
├── graph.py             # KnowledgeGraph, GraphRetriever (RRF: graph + vector)
├── rerank.py            # Reranker (cross-encoder)
├── context.py           # ContextFormatter (LLM prompt builder)
├── loaders.py           # load_text, load_directory, register_loader (plugin system)
├── pipeline.py          # Pipeline (composable loader→splitter→index)
├── cache.py             # EmbeddingCache (thread-safe LRU)
├── eval.py              # evaluate() → EvalResult (MRR, NDCG, Recall@k)
├── export.py            # export_index, import_index (JSONL)
├── fingerprint.py       # FingerprintStore (SHA-256 incremental re-indexing)
├── timing.py            # @timed decorator, get_timings(), reset_timings()
├── watcher.py           # watch_directory (debounced, watchfiles-backed)
├── cli.py               # Typer CLI (11 commands)
├── server.py            # HTTP API (stdlib HTTPServer, OpenAPI 3.0.3)
├── mcp_server.py        # MCP server (JSON-RPC over stdio)
└── stores/
    ├── __init__.py      # Re-exports FaissStore, SqliteVecStore
    ├── faiss_store.py   # FAISS IndexFlatL2 + JSON metadata
    └── sqlite_store.py  # SQLite + numpy brute-force cosine
```

### Core Data Flow

```
Documents → TextSplitter/TokenSplitter/SemanticSplitter → Chunks
Chunks → Embedder.encode_batched() → Embeddings
(Chunks, Embeddings) → VectorStore.upsert()
Query → Embedder.encode() → VectorStore.search() → QueryResult[]
```

### Extension Protocols

```python
# Any class implementing these 2 methods works as an embedding backend
class EmbeddingBackend(Protocol):
    def encode(self, texts: List[str]) -> List[List[float]]: ...
    def encode_batched(self, texts: List[str], batch_size: int = 64) -> List[List[float]]: ...

# Any class implementing these 7 methods works as a vector store
class VectorStoreBackend(Protocol):
    def upsert(self, documents, embeddings) -> None: ...
    def search(self, query_embedding, top_k=5, where=None) -> List[Dict]: ...
    def delete(self, doc_id: str) -> None: ...
    def count(self) -> int: ...
    def get_all(self) -> Dict: ...
    def get_by_ids(self, ids: List[str]) -> Dict: ...
    def clear(self) -> None: ...
```

## 5. Code Conventions

### Import Ordering (enforced by ruff `I`)
```python
from __future__ import annotations       # 1. Always first

import json                               # 2. Stdlib
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np                        # 3. Third-party

from .index import Document, QueryResult  # 4. Local (relative imports)

if TYPE_CHECKING:                         # 5. Type-checking-only imports
    from .embed import EmbeddingBackend
```

### Naming
- Classes: `PascalCase` (`DocumentIndex`, `QueryResult`)
- Functions/methods: `snake_case` (`load_directory`, `encode_batched`)
- Private: `_prefix` (`_get_model`, `_loader_registry`)
- Constants: `_UPPER_SNAKE` (`_MAX_BODY_BYTES`, `_PLACEHOLDER_METADATA`)
- Async methods: `a` prefix (`aadd`, `aquery`, `ahybrid_query`)

### Docstrings — Google style
```python
def search(self, query: str, top_k: int = 5) -> List[QueryResult]:
    """Search using graph traversal fused with vector similarity.

    Args:
        query: Search query text.
        top_k: Number of results to return.

    Returns:
        List of QueryResult sorted by fused score.
    """
```

### Type Annotations
- `from __future__ import annotations` in every file
- `List[str]` style (typing module), not `list[str]`
- `TYPE_CHECKING` guard for circular import avoidance
- `@runtime_checkable` on Protocol classes

### Error Handling
- `ValueError` for input validation (empty query, invalid params)
- `ImportError` for missing optional deps — include install command in message:
  ```python
  raise ImportError("rank-bm25 is required. Install with: pip install 'ravenrag[hybrid]'") from None
  ```
- Always use `from None` or `from e` for exception chaining
- Pipeline errors: configurable via `on_error` param (`"skip"`, `"raise"`, callable)
- Server errors: catch-all → `logger.exception()` → generic 500 JSON (never leak internals)

### Logging
- Module-level: `logger = logging.getLogger(__name__)`
- Debug: config loading, timing, registration
- Warning: unknown config keys, file skips, fallbacks
- Exception: server errors (with traceback via `exc_info=True`)

### Lazy Loading Pattern
Used for optional dependencies and heavy models:
```python
def _get_model(self):
    if self._model is None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name)
    return self._model
```

### Ruff Config
```toml
line-length = 120
target-version = "py310"
select = ["E", "F", "W", "I"]
ignore = ["E203"]
```

## 6. Test Patterns

### Structure
- Framework: `pytest` with `pytest-cov`
- Markers: `@pytest.mark.integration` for tests that download models
- Class-based grouping: `class TestKnowledgeGraph:` (no unittest.TestCase inheritance)
- Coverage threshold: 75% (`--cov-fail-under=75`)

### Mocking Pattern
```python
from unittest.mock import MagicMock

class TestHybridSearcher:
    def _make_store_and_embedder(self):
        store = MagicMock()
        store.get_all.return_value = {
            "ids": ["d1", "d2"],
            "documents": ["text1", "text2"],
            "metadatas": [{"key": "val"}, {}],
        }
        store.search.return_value = [
            {"id": "d1", "text": "text1", "metadata": {}, "distance": 0.1},
        ]
        embedder = MagicMock()
        embedder.encode.return_value = [[0.1, 0.2, 0.3]]
        return store, embedder
```

### Server Test Pattern
Real HTTP tests with threaded server on port 0:
```python
class TestServerHTTP:
    def setup_method(self):
        self.server = create_server(mock_idx, host="127.0.0.1", port=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def teardown_method(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
```

### CLI Test Pattern
Uses `typer.testing.CliRunner` with ANSI stripping:
```python
import re
_ANSI_RE = re.compile(r"\x1b\[[^m]*m")
def _clean(text: str) -> str:
    return _ANSI_RE.sub("", text)

result = runner.invoke(app, ["query", "--help"])
assert "--hybrid" in _clean(result.output)
```

### Test File Naming
One test file per module: `test_{module}.py`
Test method naming: `test_<behavior_under_test>`

### All Test Files (25 files, 276+ tests)
```
tests/conftest.py, test_async.py, test_cache.py, test_cli.py,
test_config.py, test_config_validation.py, test_context.py,
test_embed.py, test_eval.py, test_export.py, test_faiss_store.py,
test_fingerprint.py, test_graph.py, test_hybrid.py, test_index.py,
test_loaders.py, test_loaders_builtin.py, test_mcp.py,
test_multi_collection.py, test_pipeline.py, test_rerank.py,
test_server.py, test_splitter.py, test_sqlite_store.py,
test_store.py, test_timing.py, test_watcher.py
```

## 7. Dependencies Between Modules

```
config.py         → standalone (stdlib only)
timing.py         → standalone (stdlib only)
cache.py          → standalone (stdlib only)
fingerprint.py    → standalone (stdlib only)
embed.py          → standalone (urllib.request for HTTP backends)
store.py          → chromadb; index.Document (TYPE_CHECKING)
index.py          → embed, store; cache/rerank/hybrid/context/graph (all lazy)
hybrid.py         → index.QueryResult; rank_bm25 (lazy)
graph.py          → index/store/embed (TYPE_CHECKING only)
rerank.py         → sentence_transformers (lazy)
context.py        → index (TYPE_CHECKING)
splitter.py       → index.Document; transformers (lazy)
loaders.py        → index.Document; pymupdf4llm/docx/bs4 (all lazy)
pipeline.py       → index (TYPE_CHECKING)
export.py         → index (TYPE_CHECKING)
eval.py           → index (TYPE_CHECKING)
watcher.py        → loaders, watchfiles (lazy)
server.py         → index, timing
mcp_server.py     → index, __init__
cli.py            → typer, config (lazy), all modules (lazy per-command)
stores/*          → faiss/numpy (lazy), index.Document (TYPE_CHECKING)
```

**Circular import avoidance:** All cross-module type imports use `TYPE_CHECKING` guards. The `_get_cache_class()` function in index.py breaks the index↔cache circular reference.

## 8. Adding New Features — Checklist

1. **Create module:** `ravenrag/<feature>.py` with `from __future__ import annotations`
2. **Use `TYPE_CHECKING`** for imports from index.py, store.py, embed.py
3. **Use lazy imports** for optional dependencies
4. **Add public classes/functions to** `ravenrag/__init__.py` (both import and `__all__`)
5. **Write tests:** `tests/test_<feature>.py` using MagicMock for store/embedder
6. **Run locally:**
   ```bash
   uv run ruff check ravenrag/ tests/ && uv run ruff format --check ravenrag/ tests/
   uv run pytest tests/ -v --tb=short -m "not integration" --cov=ravenrag --cov-fail-under=75
   ```
7. **Update README.md** if user-facing
8. **Commit and push:**
   ```bash
   git add -A && git commit -m "<type>(<scope>): <description>" && git push
   ```
9. **Monitor CI** after push

### Commit Message Format
```
<type>(<scope>): <short description>

<body — what and why>
```
Types: `feat`, `fix`, `refactor`, `test`, `docs`, `ci`, `chore`

## 9. Key Design Decisions

- **No framework for HTTP server** — stdlib `http.server` to avoid deps
- **No framework for MCP** — raw JSON-RPC over stdio
- **Protocol over ABC** — `@runtime_checkable` Protocol for duck typing
- **Lazy everything** — models, optional deps, and cross-module imports loaded on first use
- **RRF for fusion** — both HybridSearcher and GraphRetriever use reciprocal rank fusion with `k=60`
- **SHA-256 for IDs** — Document IDs default to SHA-256 of text
- **Thread-safe cache** — `threading.Lock` + `OrderedDict` LRU
- **No `requests` library** — HTTP via `urllib.request` throughout
- **ChromaDB as default store** — but fully swappable via `VectorStoreBackend` protocol

## 10. Optional Dependency Groups

| Extra | Packages | Used by |
|-------|----------|---------|
| `hybrid` | `rank-bm25` | `HybridSearcher` |
| `watch` | `watchfiles` | `watch_directory()` |
| `tokens` | `transformers` | `TokenSplitter` |
| `faiss` | `faiss-cpu` | `FaissStore` |
| `loaders` | `pymupdf4llm`, `python-docx`, `beautifulsoup4` | PDF/DOCX/HTML loaders |
| `all` | All of above | |
| `dev` | `pytest`, `pytest-cov`, `ruff` + hybrid+watch+tokens | |

## 11. HTTP API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check: `{"status": "ok"}` |
| GET | `/stats` | Document count + collection name |
| GET | `/collections` | List collection names |
| GET | `/metrics` | Timing + cache performance stats |
| GET | `/openapi.json` | Full OpenAPI 3.0.3 specification |
| POST | `/query` | Search (vector, hybrid, rerank) |
| POST | `/prompt` | LLM-ready formatted prompt |
| POST | `/index` | Index new documents |

Security: Bearer token auth, 10 MB body limit, top_k capped at 1000, CORS configurable.

## 12. CLI Commands

| Command | Description |
|---------|-------------|
| `raven index <path>` | Index documents (incremental via fingerprints) |
| `raven query <text>` | Search with `--hybrid`, `--rerank`, `--alpha` |
| `raven prompt <text>` | Get formatted LLM prompt |
| `raven serve` | Start HTTP API server |
| `raven watch <path>` | Watch & auto-reindex |
| `raven info` | Show index stats |
| `raven export` | Export JSONL |
| `raven import <file>` | Import JSONL |
| `raven doctor` | Diagnose setup |
| `raven mcp` | Start MCP server (stdio) |
| `raven benchmark` | Benchmark performance |

Shared flags: `--db`, `--collection`, `--model`, `--verbose`/`-v`
