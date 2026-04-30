"""Tests for the embedding and query cache."""

from ravenrag.cache import EmbeddingCache


class TestEmbeddingCache:
    def test_put_and_get(self):
        cache = EmbeddingCache(maxsize=10)
        cache.put("hello", [1.0, 2.0, 3.0])
        assert cache.get("hello") == [1.0, 2.0, 3.0]

    def test_miss_returns_none(self):
        cache = EmbeddingCache(maxsize=10)
        assert cache.get("missing") is None

    def test_lru_eviction(self):
        cache = EmbeddingCache(maxsize=2)
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        cache.put("c", [3.0])  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == [2.0]
        assert cache.get("c") == [3.0]

    def test_hit_miss_counters(self):
        cache = EmbeddingCache(maxsize=10)
        cache.put("x", [1.0])
        cache.get("x")  # hit
        cache.get("y")  # miss
        assert cache.hits == 1
        assert cache.misses == 1

    def test_get_or_compute(self):
        cache = EmbeddingCache(maxsize=10)
        cache.put("cached", [1.0, 2.0])

        computed = []

        def mock_encode(texts):
            computed.extend(texts)
            return [[float(i)] for i, _ in enumerate(texts)]

        result = cache.get_or_compute(["cached", "new"], mock_encode)
        assert result[0] == [1.0, 2.0]  # from cache
        assert result[1] == [0.0]  # computed
        assert computed == ["new"]  # only "new" was sent to compute

    def test_disabled_cache(self):
        cache = EmbeddingCache(maxsize=0)
        cache.put("x", [1.0])
        assert cache.get("x") is None
        assert cache.size == 0

    def test_clear(self):
        cache = EmbeddingCache(maxsize=10)
        cache.put("a", [1.0])
        cache.put("b", [2.0])
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_size(self):
        cache = EmbeddingCache(maxsize=10)
        assert cache.size == 0
        cache.put("a", [1.0])
        assert cache.size == 1
