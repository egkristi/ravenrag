"""
Server: Lightweight HTTP API server for RavenRAG.

Exposes search/index/stats endpoints. Requires no extra dependencies
beyond the Python standard library — uses a built-in HTTP server.

For production workloads, see the ``uvicorn`` integration option.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class _RavenHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RavenRAG API."""

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format, *args)

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        body = self.rfile.read(length)
        return json.loads(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/health":
            self._send_json({"status": "ok"})
        elif path == "/stats":
            self._handle_stats()
        elif path == "/collections":
            self._handle_collections()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        try:
            body = self._read_json()
        except (json.JSONDecodeError, ValueError) as e:
            self._send_json({"error": f"Invalid JSON: {e}"}, 400)
            return

        if path == "/query":
            self._handle_query(body)
        elif path == "/prompt":
            self._handle_prompt(body)
        elif path == "/index":
            self._handle_index(body)
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_stats(self) -> None:
        index = self.server.raven_index  # type: ignore[attr-defined]
        self._send_json(
            {
                "documents": index.count(),
                "persist_dir": str(
                    getattr(index.store, "client")._identifier if hasattr(index.store, "client") else "unknown"
                ),
                "collection": index.store.collection.name,
            }
        )

    def _handle_collections(self) -> None:
        index = self.server.raven_index  # type: ignore[attr-defined]
        collections = index.store.client.list_collections()
        self._send_json(
            {
                "collections": [c.name for c in collections],
            }
        )

    def _handle_query(self, body: Dict) -> None:
        index = self.server.raven_index  # type: ignore[attr-defined]
        query = body.get("query", "")
        if not query:
            self._send_json({"error": "Missing 'query' field"}, 400)
            return

        top_k = body.get("top_k", 5)
        where = body.get("where")
        rerank = body.get("rerank", False)
        hybrid = body.get("hybrid", False)

        try:
            if hybrid:
                alpha = body.get("alpha", 0.5)
                results = index.hybrid_query(query, top_k=top_k, where=where, alpha=alpha)
            else:
                results = index.query(query, top_k=top_k, where=where, rerank=rerank)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)
            return

        self._send_json(
            {
                "query": query,
                "results": [
                    {
                        "id": r.id,
                        "text": r.text,
                        "metadata": r.metadata,
                        "distance": r.distance,
                        "rerank_score": r.rerank_score,
                    }
                    for r in results
                ],
            }
        )

    def _handle_prompt(self, body: Dict) -> None:
        index = self.server.raven_index  # type: ignore[attr-defined]
        query = body.get("query", "")
        if not query:
            self._send_json({"error": "Missing 'query' field"}, 400)
            return

        top_k = body.get("top_k", 5)
        where = body.get("where")
        template = body.get("template")

        try:
            prompt = index.query_for_prompt(query, top_k=top_k, where=where, template=template)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)
            return

        self._send_json({"query": query, "prompt": prompt})

    def _handle_index(self, body: Dict) -> None:
        index = self.server.raven_index  # type: ignore[attr-defined]
        documents = body.get("documents", [])
        if not documents:
            self._send_json({"error": "Missing 'documents' field (list of {text, metadata?})"}, 400)
            return

        from .index import Document

        docs = []
        for d in documents:
            if isinstance(d, str):
                docs.append(Document(text=d))
            elif isinstance(d, dict):
                docs.append(
                    Document(
                        text=d.get("text", ""),
                        metadata=d.get("metadata"),
                        doc_id=d.get("id"),
                    )
                )
            else:
                self._send_json({"error": f"Invalid document format: {type(d)}"}, 400)
                return

        try:
            index.add(docs)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)
            return

        self._send_json({"indexed": len(docs), "total": index.count()})


def create_server(
    index: Any,
    host: str = "127.0.0.1",
    port: int = 8484,
) -> HTTPServer:
    """Create an HTTP server bound to the given index.

    Args:
        index: A DocumentIndex instance.
        host: Bind address.
        port: Bind port.

    Returns:
        An HTTPServer ready to serve_forever().
    """
    server = HTTPServer((host, port), _RavenHandler)
    server.raven_index = index  # type: ignore[attr-defined]
    return server


def serve(
    persist_dir: str = "./ravenrag_db",
    collection_name: str = "documents",
    embedding_model: str = "all-MiniLM-L6-v2",
    host: str = "127.0.0.1",
    port: int = 8484,
) -> None:
    """Start the RavenRAG API server.

    Endpoints:
        GET  /health      → {"status": "ok"}
        GET  /stats       → {"documents": N, ...}
        GET  /collections → {"collections": [...]}
        POST /query       → {"query": "...", "top_k": 5, "where": {}, "rerank": false, "hybrid": false}
        POST /prompt      → {"query": "...", "top_k": 5, "template": "..."}
        POST /index       → {"documents": [{"text": "...", "metadata": {...}}]}
    """
    from .index import DocumentIndex

    index = DocumentIndex(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    server = create_server(index, host=host, port=port)

    logger.info("RavenRAG server starting on http://%s:%d", host, port)
    print(f"🐦‍⬛ RavenRAG server running on http://{host}:{port}")
    print(f"   Database: {persist_dir} (collection: {collection_name})")
    print(f"   Documents: {index.count()}")
    print("   Endpoints: /health /stats /query /prompt /index /collections")
    print("   Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
