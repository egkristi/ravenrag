"""Tests for Embedder."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from ravenrag.embed import Embedder


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

    @pytest.mark.integration
    def test_real_encode(self):
        embedder = Embedder()
        embeddings = embedder.encode(["Hello world", "Another sentence"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) > 0
        assert isinstance(embeddings[0], list)
        assert isinstance(embeddings[0][0], float)
