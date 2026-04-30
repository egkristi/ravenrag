"""Tests for Embedder and backends."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ravenrag.embed import Embedder, EmbeddingBackend, OllamaBackend, OpenAIBackend, VLLMBackend


class TestEmbedder:
    def test_encode_calls_model(self):
        embedder = Embedder(model_name="all-MiniLM-L6-v2")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        embedder._model = mock_model

        result = embedder.encode(["Hello", "World"])

        mock_model.encode.assert_called_once_with(["Hello", "World"], show_progress_bar=False)
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_encode_batched(self):
        embedder = Embedder()
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([[0.1, 0.2]]),
            np.array([[0.3, 0.4]]),
        ]
        embedder._model = mock_model

        result = embedder.encode_batched(["a", "b"], batch_size=1)

        assert len(result) == 2
        assert mock_model.encode.call_count == 2

    def test_lazy_model_loading(self):
        embedder = Embedder()
        assert embedder._model is None

    def test_implements_protocol(self):
        embedder = Embedder()
        assert isinstance(embedder, EmbeddingBackend)

    @pytest.mark.integration
    def test_real_encode(self):
        embedder = Embedder()
        embeddings = embedder.encode(["Hello world", "Another sentence"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0
        assert isinstance(embeddings[0], list)
        assert isinstance(embeddings[0][0], float)


class TestOllamaBackend:
    def test_encode_calls_api(self):
        backend = OllamaBackend(model_name="nomic-embed-text", base_url="http://localhost:11434")
        response_data = json.dumps({"embeddings": [[0.1, 0.2, 0.3]]}).encode()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = response_data
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_resp

            result = backend.encode(["test text"])

        assert result == [[0.1, 0.2, 0.3]]

    def test_encode_batched(self):
        backend = OllamaBackend()
        response1 = json.dumps({"embeddings": [[0.1, 0.2]]}).encode()
        response2 = json.dumps({"embeddings": [[0.3, 0.4]]}).encode()

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp1 = MagicMock()
            mock_resp1.read.return_value = response1
            mock_resp1.__enter__ = lambda s: s
            mock_resp1.__exit__ = MagicMock(return_value=False)

            mock_resp2 = MagicMock()
            mock_resp2.read.return_value = response2
            mock_resp2.__enter__ = lambda s: s
            mock_resp2.__exit__ = MagicMock(return_value=False)

            mock_urlopen.side_effect = [mock_resp1, mock_resp2]
            result = backend.encode_batched(["a", "b"], batch_size=1)

        assert len(result) == 2

    def test_connection_error(self):
        import urllib.error

        backend = OllamaBackend(base_url="http://localhost:99999")

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                backend.encode(["test"])

    def test_implements_protocol(self):
        backend = OllamaBackend()
        assert isinstance(backend, EmbeddingBackend)


class TestOpenAIBackend:
    def _mock_response(self, embeddings):
        """Build a mock urlopen response for /v1/embeddings."""
        data = [{"embedding": emb, "index": i} for i, emb in enumerate(embeddings)]
        body = json.dumps({"object": "list", "data": data}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_encode_calls_api(self):
        backend = OpenAIBackend(model_name="text-embedding-3-small", base_url="http://localhost:8000/v1")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response([[0.1, 0.2, 0.3]])
            result = backend.encode(["test text"])

        assert result == [[0.1, 0.2, 0.3]]
        # Verify it hit /v1/embeddings
        called_req = mock_urlopen.call_args[0][0]
        assert called_req.full_url == "http://localhost:8000/v1/embeddings"

    def test_encode_preserves_order(self):
        """Results are sorted by index even if API returns them out of order."""
        backend = OpenAIBackend(model_name="m", base_url="http://localhost:8000/v1")
        # Return out-of-order
        data = [
            {"embedding": [0.3], "index": 1},
            {"embedding": [0.1], "index": 0},
        ]
        body = json.dumps({"object": "list", "data": data}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = backend.encode(["a", "b"])

        assert result == [[0.1], [0.3]]

    def test_api_key_sent_as_bearer(self):
        backend = OpenAIBackend(model_name="m", base_url="http://localhost:8000/v1", api_key="sk-test123")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response([[0.1]])
            backend.encode(["test"])

        called_req = mock_urlopen.call_args[0][0]
        assert called_req.get_header("Authorization") == "Bearer sk-test123"

    def test_no_api_key_no_header(self):
        backend = OpenAIBackend(model_name="m", base_url="http://localhost:8000/v1")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = self._mock_response([[0.1]])
            backend.encode(["test"])

        called_req = mock_urlopen.call_args[0][0]
        assert called_req.get_header("Authorization") is None

    def test_connection_error(self):
        import urllib.error

        backend = OpenAIBackend(model_name="m", base_url="http://localhost:99999/v1")

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                backend.encode(["test"])

    def test_http_error(self):
        import urllib.error

        backend = OpenAIBackend(model_name="m", base_url="http://localhost:8000/v1")

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(None, 401, "Unauthorized", {}, None),
        ):
            with pytest.raises(ConnectionError, match="HTTP 401"):
                backend.encode(["test"])

    def test_encode_batched(self):
        backend = OpenAIBackend(model_name="m", base_url="http://localhost:8000/v1")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = [
                self._mock_response([[0.1, 0.2]]),
                self._mock_response([[0.3, 0.4]]),
            ]
            result = backend.encode_batched(["a", "b"], batch_size=1)

        assert len(result) == 2
        assert mock_urlopen.call_count == 2

    def test_implements_protocol(self):
        backend = OpenAIBackend(model_name="m")
        assert isinstance(backend, EmbeddingBackend)


class TestVLLMBackend:
    def test_inherits_openai_backend(self):
        backend = VLLMBackend()
        assert isinstance(backend, OpenAIBackend)

    def test_default_model(self):
        backend = VLLMBackend()
        assert backend.model_name == "BAAI/bge-base-en-v1.5"
        assert backend.base_url == "http://localhost:8000/v1"

    def test_custom_model(self):
        backend = VLLMBackend(model_name="intfloat/e5-large-v2", base_url="http://gpu-server:8000/v1")
        assert backend.model_name == "intfloat/e5-large-v2"

    def test_implements_protocol(self):
        backend = VLLMBackend()
        assert isinstance(backend, EmbeddingBackend)
