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

_MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB
_MAX_TOP_K = 1000

_OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "RavenRAG API",
        "description": "Lightweight, local-first RAG retrieval API.",
        "version": "0.7.0",
    },
    "paths": {
        "/health": {
            "get": {
                "summary": "Health check",
                "operationId": "healthCheck",
                "responses": {
                    "200": {
                        "description": "Server is healthy",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"status": {"type": "string", "example": "ok"}},
                                }
                            }
                        },
                    }
                },
            }
        },
        "/stats": {
            "get": {
                "summary": "Index statistics",
                "operationId": "getStats",
                "responses": {
                    "200": {
                        "description": "Document count and collection info",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "documents": {"type": "integer"},
                                        "collection": {"type": "string"},
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/collections": {
            "get": {
                "summary": "List collections",
                "operationId": "listCollections",
                "responses": {
                    "200": {
                        "description": "Available collections",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "collections": {"type": "array", "items": {"type": "string"}},
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
        "/metrics": {
            "get": {
                "summary": "Timing and cache metrics",
                "operationId": "getMetrics",
                "responses": {
                    "200": {
                        "description": "Performance metrics",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
        "/query": {
            "post": {
                "summary": "Search documents",
                "operationId": "queryDocuments",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["query"],
                                "properties": {
                                    "query": {"type": "string", "description": "Search query text"},
                                    "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 1000},
                                    "where": {
                                        "type": "object",
                                        "description": "Metadata filter",
                                        "additionalProperties": True,
                                    },
                                    "rerank": {"type": "boolean", "default": False},
                                    "hybrid": {"type": "boolean", "default": False},
                                    "alpha": {
                                        "type": "number",
                                        "default": 0.5,
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                        "description": "Hybrid alpha: 1.0=vector, 0.0=BM25",
                                    },
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Search results",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"},
                                        "results": {
                                            "type": "array",
                                            "items": {"$ref": "#/components/schemas/QueryResult"},
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "400": {"description": "Missing or invalid query"},
                },
            }
        },
        "/prompt": {
            "post": {
                "summary": "Get LLM-ready prompt",
                "operationId": "queryForPrompt",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["query"],
                                "properties": {
                                    "query": {"type": "string"},
                                    "top_k": {"type": "integer", "default": 5},
                                    "where": {"type": "object", "additionalProperties": True},
                                    "template": {"type": "string", "description": "Custom prompt template"},
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Formatted prompt",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string"},
                                        "prompt": {"type": "string"},
                                    },
                                }
                            }
                        },
                    },
                    "400": {"description": "Missing query"},
                },
            }
        },
        "/index": {
            "post": {
                "summary": "Index new documents",
                "operationId": "indexDocuments",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["documents"],
                                "properties": {
                                    "documents": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "required": ["text"],
                                            "properties": {
                                                "text": {"type": "string"},
                                                "metadata": {"type": "object", "additionalProperties": True},
                                                "id": {"type": "string"},
                                            },
                                        },
                                    }
                                },
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Indexing result",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "indexed": {"type": "integer"},
                                        "total": {"type": "integer"},
                                    },
                                }
                            }
                        },
                    },
                    "400": {"description": "Missing or invalid documents"},
                },
            }
        },
        "/openapi.json": {
            "get": {
                "summary": "OpenAPI schema",
                "operationId": "getOpenApiSchema",
                "responses": {
                    "200": {
                        "description": "This OpenAPI specification",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
    },
    "components": {
        "schemas": {
            "QueryResult": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "metadata": {"type": "object", "additionalProperties": True},
                    "distance": {"type": "number"},
                    "rerank_score": {"type": "number", "nullable": True},
                },
            }
        },
        "securitySchemes": {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
            }
        },
    },
    "security": [{"BearerAuth": []}],
}


class _RavenHandler(BaseHTTPRequestHandler):
    """HTTP request handler for RavenRAG API."""

    def log_message(self, format: str, *args: Any) -> None:
        logger.info(format, *args)

    def _check_auth(self) -> bool:
        """Validate API key if one is configured on the server."""
        api_key: str | None = getattr(self.server, "raven_api_key", None)
        if not api_key:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {api_key}":
            return True
        self._send_json({"error": "Unauthorized"}, 401)
        return False

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        cors = getattr(self.server, "raven_cors_origin", None)
        if cors:
            self.send_header("Access-Control-Allow-Origin", cors)
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict | None:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        if length > _MAX_BODY_BYTES:
            self._send_json({"error": "Payload too large"}, 413)
            return None
        body = self.rfile.read(length)
        return json.loads(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Handle CORS preflight."""
        self.send_response(204)
        cors = getattr(self.server, "raven_cors_origin", None)
        if cors:
            self.send_header("Access-Control-Allow-Origin", cors)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if not self._check_auth():
            return
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/health":
            self._send_json({"status": "ok"})
        elif path == "/stats":
            self._handle_stats()
        elif path == "/collections":
            self._handle_collections()
        elif path == "/metrics":
            self._handle_metrics()
        elif path == "/openapi.json":
            self._send_json(_OPENAPI_SPEC)
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if not self._check_auth():
            return
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        try:
            body = self._read_json()
        except (json.JSONDecodeError, ValueError) as e:
            self._send_json({"error": f"Invalid JSON: {e}"}, 400)
            return
        if body is None:
            return  # _read_json already sent 413

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

    def _handle_metrics(self) -> None:
        """Return timing and cache metrics (Prometheus-style text or JSON)."""
        index = self.server.raven_index  # type: ignore[attr-defined]
        from .timing import get_timings

        metrics: Dict = {
            "documents": index.count(),
            "timings": get_timings(),
        }
        if hasattr(index, "_embedding_cache"):
            cache = index._embedding_cache
            metrics["cache"] = {
                "size": cache.size,
                "hits": cache.hits,
                "misses": cache.misses,
            }
        self._send_json(metrics)

    def _handle_query(self, body: Dict) -> None:
        index = self.server.raven_index  # type: ignore[attr-defined]
        query = body.get("query", "")
        if not query:
            self._send_json({"error": "Missing 'query' field"}, 400)
            return

        top_k = min(max(int(body.get("top_k", 5)), 1), _MAX_TOP_K)
        where = body.get("where")
        rerank = body.get("rerank", False)
        hybrid = body.get("hybrid", False)

        try:
            if hybrid:
                alpha = body.get("alpha", 0.5)
                results = index.hybrid_query(query, top_k=top_k, where=where, alpha=alpha)
            else:
                results = index.query(query, top_k=top_k, where=where, rerank=rerank)
        except Exception:
            logger.exception("Query failed")
            self._send_json({"error": "Internal server error"}, 500)
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
        except Exception:
            logger.exception("Prompt generation failed")
            self._send_json({"error": "Internal server error"}, 500)
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
        except Exception:
            logger.exception("Indexing failed")
            self._send_json({"error": "Internal server error"}, 500)
            return

        self._send_json({"indexed": len(docs), "total": index.count()})


def create_server(
    index: Any,
    host: str = "127.0.0.1",
    port: int = 8484,
    api_key: str | None = None,
    cors_origin: str | None = None,
) -> HTTPServer:
    """Create an HTTP server bound to the given index.

    Args:
        index: A DocumentIndex instance.
        host: Bind address.
        port: Bind port.
        api_key: Optional API key for Bearer auth.
        cors_origin: Optional CORS origin (e.g. ``"*"``).

    Returns:
        An HTTPServer ready to serve_forever().

    .. warning::

        This server does **not** support TLS. When using API key
        authentication, run behind a TLS-terminating reverse proxy
        (nginx, Caddy, etc.) to protect credentials in transit.
    """
    server = HTTPServer((host, port), _RavenHandler)
    server.raven_index = index  # type: ignore[attr-defined]
    server.raven_api_key = api_key  # type: ignore[attr-defined]
    server.raven_cors_origin = cors_origin  # type: ignore[attr-defined]
    return server


def serve(
    persist_dir: str = "./ravenrag_db",
    collection_name: str = "documents",
    embedding_model: str = "all-MiniLM-L6-v2",
    host: str = "127.0.0.1",
    port: int = 8484,
    api_key: str | None = None,
    cors_origin: str | None = None,
) -> None:
    """Start the RavenRAG API server.

    Endpoints:
        GET  /health       → {"status": "ok"}
        GET  /stats        → {"documents": N, ...}
        GET  /collections  → {"collections": [...]}
        GET  /metrics      → timing & cache stats
        GET  /openapi.json → OpenAPI 3.0 schema
        POST /query        → {"query": "...", "top_k": 5, "where": {}, "rerank": false, "hybrid": false}
        POST /prompt       → {"query": "...", "top_k": 5, "template": "..."}
        POST /index        → {"documents": [{"text": "...", "metadata": {...}}]}
    """
    from .index import DocumentIndex

    index = DocumentIndex(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    server = create_server(index, host=host, port=port, api_key=api_key, cors_origin=cors_origin)

    logger.info("RavenRAG server starting on http://%s:%d", host, port)
    print(f"🐦‍⬛ RavenRAG server running on http://{host}:{port}")
    print(f"   Database: {persist_dir} (collection: {collection_name})")
    print(f"   Documents: {index.count()}")
    if api_key:
        print("   Auth: API key required (Bearer token)")
    print("   Endpoints: /health /stats /query /prompt /index /collections /openapi.json")
    print("   Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
