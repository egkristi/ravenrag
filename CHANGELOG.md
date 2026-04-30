# Changelog

All notable changes to RavenRAG will be documented in this file.

## [0.5.0] — 2026-04-30

### Added
- **Semantic splitting**: `SemanticSplitter` splits documents at meaning boundaries using embedding similarity.
- **MCP server**: `raven mcp` exposes RavenRAG as a Model Context Protocol server for AI assistants (Claude, Copilot, Cursor).
- **Retrieval evaluation**: `evaluate()` computes MRR, NDCG, and Recall@k against ground truth.
- **Parent-child retrieval**: `index.query_parent()` searches chunks but returns full parent documents.
- **Incremental re-indexing**: `raven index` skips unchanged files using fingerprint tracking.
- **Export/import**: `raven export` / `raven import` for JSONL backup and restore.
- **`raven doctor`**: Diagnostic command to check setup health.
- **VectorStoreBackend protocol**: Abstraction to swap ChromaDB for other backends.
- **Server auth**: Optional API key (Bearer token) for HTTP server.
- **Server CORS**: Configurable `Access-Control-Allow-Origin`.
- **Request size limit**: Server rejects payloads over 10 MB.
- **top_k clamping**: Server clamps `top_k` to 1–1000.
- **HTTP timeouts**: All embedding backends now use `timeout=30` (configurable).
- **Environment variable config**: `RAVENRAG_DB`, `RAVENRAG_MODEL`, `RAVENRAG_API_KEY`, etc.
- **Config schema validation**: Warns on unknown keys (typo protection).
- **Path traversal protection**: `load_directory` validates files stay within target directory.
- **Watcher debounce**: Batches rapid file changes before re-indexing.
- **Watcher deletion support**: Removes documents from index when files are deleted.
- **`py.typed` marker**: PEP 561 typed package support.

### Changed
- Version bumped to 0.5.0.
- Ruff target-version set to `py310` (was `py39`).
- Stats endpoint no longer exposes internal `persist_dir`.

### Fixed
- Hybrid search `bm25_metadatas` unused variable removed.
- Config `_build_config` now validates keys and warns on typos.

## [0.4.0] — 2026-04-30

### Added
- Config file support (`ravenrag.toml` / `pyproject.toml [tool.ravenrag]`).
- Built-in HTTP API server (`raven serve`).
- Citation & provenance tracking (`QueryResult.citation`).
- Plugin loader system (`@register_loader` decorator).
- OpenAI-compatible and vLLM embedding backends.
- CLI flags: `--hybrid`, `--rerank`, `--verbose`.
- Cross-encoder reranking, hybrid search, context formatting, watch mode.

## [0.3.0] — 2026-04-29

### Added
- Initial release with core RAG functionality.
- sentence-transformers and Ollama embedding backends.
- ChromaDB vector store, text/token splitting, file loaders.
- CLI tool (`raven index`, `raven query`, `raven info`, `raven watch`, `raven prompt`).
