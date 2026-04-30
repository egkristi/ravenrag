"""Tests for Embedder and backends."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ravenrag.embed import Embedder, EmbeddingBackend, OllamaBackend


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
