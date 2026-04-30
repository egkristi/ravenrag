"""Tests for the HTTP API server."""

import json
import threading
import urllib.error
import urllib.request
from unittest.mock import MagicMock

from ravenrag.index import QueryResult
from ravenrag.server import create_server


class TestServerEndpoints:
    def _make_mock_index(self, count=0, query_results=None):
        idx = MagicMock()
        idx.count.return_value = count
        idx.store.collection.name = "test"
        idx.store.client.list_collections.return_value = [MagicMock(name="test")]
        if query_results is not None:
            idx.query.return_value = query_results
            idx.hybrid_query.return_value = query_results
            idx.query_for_prompt.return_value = "formatted prompt"
        else:
            idx.query.return_value = []
            idx.hybrid_query.return_value = []
        return idx

    def test_create_server(self):
        idx = self._make_mock_index()
        server = create_server(idx, host="127.0.0.1", port=0)
        assert server.raven_index is idx
        server.server_close()

    def test_create_server_with_api_key(self):
        idx = self._make_mock_index()
        server = create_server(idx, port=0, api_key="test-key")
        assert server.raven_api_key == "test-key"
        server.server_close()

    def test_create_server_with_cors(self):
        idx = self._make_mock_index()
        server = create_server(idx, port=0, cors_origin="*")
        assert server.raven_cors_origin == "*"
        server.server_close()


class TestServerHTTP:
    """Real HTTP request tests against a running server."""

    def setup_method(self):
        self.idx = MagicMock()
        self.idx.count.return_value = 5
        self.idx.store.collection.name = "test"
        coll_mock = MagicMock()
        coll_mock.name = "test"
        self.idx.store.client.list_collections.return_value = [coll_mock]
        self.idx.query.return_value = [
            QueryResult(id="d1", text="result text", metadata={"source": "a.md"}, distance=0.1),
        ]
        self.idx.hybrid_query.return_value = [
            QueryResult(id="d1", text="result text", metadata={"source": "a.md"}, distance=0.1),
        ]
        self.idx.query_for_prompt.return_value = "formatted prompt"
        self.idx.add.return_value = None

        self.server = create_server(self.idx, host="127.0.0.1", port=0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def teardown_method(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _get(self, path):
        req = urllib.request.Request(f"{self.base_url}{path}")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read()), resp.status
        except urllib.error.HTTPError as e:
            return json.loads(e.read()), e.code

    def _post(self, path, data, headers=None):
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read()), resp.status
        except urllib.error.HTTPError as e:
            return json.loads(e.read()), e.code

    def test_health(self):
        data, status = self._get("/health")
        assert status == 200
        assert data["status"] == "ok"

    def test_stats(self):
        data, status = self._get("/stats")
        assert status == 200
        assert data["documents"] == 5
        assert data["collection"] == "test"

    def test_collections(self):
        data, status = self._get("/collections")
        assert status == 200
        assert "collections" in data

    def test_query(self):
        data, status = self._post("/query", {"query": "test", "top_k": 3})
        assert status == 200
        assert data["query"] == "test"
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == "d1"

    def test_query_missing_field(self):
        data, status = self._post("/query", {})
        assert status == 400
        assert "error" in data

    def test_prompt(self):
        data, status = self._post("/prompt", {"query": "explain"})
        assert status == 200
        assert "prompt" in data

    def test_prompt_missing_field(self):
        data, status = self._post("/prompt", {})
        assert status == 400

    def test_index_documents(self):
        data, status = self._post(
            "/index",
            {"documents": [{"text": "new doc", "metadata": {"source": "api"}}]},
        )
        assert status == 200
        assert data["indexed"] == 1

    def test_index_empty(self):
        data, status = self._post("/index", {"documents": []})
        assert status == 400

    def test_get_404(self):
        data, status = self._get("/nonexistent")
        assert status == 404

    def test_post_404(self):
        data, status = self._post("/nonexistent", {})
        assert status == 404

    def test_server_error_is_generic(self):
        """Server should return generic error, not leak internals."""
        self.idx.query.side_effect = RuntimeError("secret internal detail")
        data, status = self._post("/query", {"query": "test"})
        assert status == 500
        assert "Internal server error" in data["error"]
        assert "secret" not in data["error"]


class TestServerAuth:
    """Test API key authentication."""

    def setup_method(self):
        self.idx = MagicMock()
        self.idx.count.return_value = 0
        self.idx.store.collection.name = "test"

        self.server = create_server(self.idx, host="127.0.0.1", port=0, api_key="test-secret")
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def teardown_method(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_no_auth_returns_401(self):
        req = urllib.request.Request(f"{self.base_url}/health")
        try:
            urllib.request.urlopen(req)
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 401

    def test_wrong_key_returns_401(self):
        req = urllib.request.Request(
            f"{self.base_url}/health",
            headers={"Authorization": "Bearer wrong-key"},
        )
        try:
            urllib.request.urlopen(req)
            assert False, "Should have raised"
        except urllib.error.HTTPError as e:
            assert e.code == 401

    def test_correct_key_succeeds(self):
        req = urllib.request.Request(
            f"{self.base_url}/health",
            headers={"Authorization": "Bearer test-secret"},
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            assert data["status"] == "ok"


class TestServerOpenAPI:
    """Test OpenAPI schema endpoint."""

    def setup_method(self):
        self.idx = MagicMock()
        self.idx.count.return_value = 0
        self.idx.store.collection.name = "test"

        self.server = create_server(self.idx, host="127.0.0.1", port=0)
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def teardown_method(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _get(self, path):
        req = urllib.request.Request(f"{self.base_url}{path}")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read()), resp.status
        except urllib.error.HTTPError as e:
            return json.loads(e.read()), e.code

    def test_openapi_returns_200(self):
        data, status = self._get("/openapi.json")
        assert status == 200

    def test_openapi_has_required_fields(self):
        data, _ = self._get("/openapi.json")
        assert data["openapi"] == "3.0.3"
        assert "info" in data
        assert "paths" in data

    def test_openapi_info(self):
        data, _ = self._get("/openapi.json")
        assert data["info"]["title"] == "RavenRAG API"
        assert "version" in data["info"]

    def test_openapi_lists_all_endpoints(self):
        data, _ = self._get("/openapi.json")
        paths = data["paths"]
        assert "/health" in paths
        assert "/stats" in paths
        assert "/collections" in paths
        assert "/metrics" in paths
        assert "/query" in paths
        assert "/prompt" in paths
        assert "/index" in paths
        assert "/openapi.json" in paths

    def test_openapi_query_schema(self):
        data, _ = self._get("/openapi.json")
        query_schema = data["paths"]["/query"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert "query" in query_schema["properties"]
        assert "top_k" in query_schema["properties"]
        assert "hybrid" in query_schema["properties"]

    def test_openapi_components(self):
        data, _ = self._get("/openapi.json")
        assert "QueryResult" in data["components"]["schemas"]
        assert "BearerAuth" in data["components"]["securitySchemes"]
