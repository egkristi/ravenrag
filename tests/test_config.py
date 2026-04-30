"""Tests for config loading."""

from ravenrag.config import RavenConfig, _build_config, load_config


class TestRavenConfig:
    def test_defaults(self):
        cfg = RavenConfig()
        assert cfg.index.persist_dir == "./ravenrag_db"
        assert cfg.index.collection == "documents"
        assert cfg.index.model == "all-MiniLM-L6-v2"
        assert cfg.search.top_k == 5
        assert cfg.search.rerank is False
        assert cfg.search.hybrid is False
        assert cfg.server.host == "127.0.0.1"
        assert cfg.server.port == 8484

    def test_build_config_from_dict(self):
        data = {
            "index": {"persist_dir": "/tmp/test_db", "collection": "my_col", "model": "custom-model"},
            "search": {"top_k": 10, "rerank": True, "hybrid": True, "alpha": 0.7},
            "server": {"host": "0.0.0.0", "port": 9000},
            "watch": {"extensions": [".md", ".rst"]},
        }
        cfg = _build_config(data)
        assert cfg.index.persist_dir == "/tmp/test_db"
        assert cfg.index.collection == "my_col"
        assert cfg.search.top_k == 10
        assert cfg.search.rerank is True
        assert cfg.search.alpha == 0.7
        assert cfg.server.port == 9000
        assert cfg.watch_extensions == [".md", ".rst"]


class TestLoadConfig:
    def test_no_config_returns_defaults(self, tmp_path):
        cfg = load_config(search_dir=str(tmp_path))
        assert isinstance(cfg, RavenConfig)
        assert cfg.index.persist_dir == "./ravenrag_db"

    def test_load_ravenrag_toml(self, tmp_path):
        toml_content = """
[index]
persist_dir = "./custom_db"
collection = "notes"
chunk_size = 256

[search]
top_k = 3
rerank = true
"""
        (tmp_path / "ravenrag.toml").write_text(toml_content)
        cfg = load_config(search_dir=str(tmp_path))
        assert cfg.index.persist_dir == "./custom_db"
        assert cfg.index.collection == "notes"
        assert cfg.index.chunk_size == 256
        assert cfg.search.top_k == 3
        assert cfg.search.rerank is True

    def test_load_pyproject_toml(self, tmp_path):
        toml_content = """
[tool.ravenrag.index]
persist_dir = "./pyproject_db"

[tool.ravenrag.search]
hybrid = true
alpha = 0.8
"""
        (tmp_path / "pyproject.toml").write_text(toml_content)
        cfg = load_config(search_dir=str(tmp_path))
        assert cfg.index.persist_dir == "./pyproject_db"
        assert cfg.search.hybrid is True
        assert cfg.search.alpha == 0.8

    def test_ravenrag_toml_takes_precedence(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[tool.ravenrag.index]\npersist_dir = "./pyproject_db"\n')
        (tmp_path / "ravenrag.toml").write_text('[index]\npersist_dir = "./raven_db"\n')
        cfg = load_config(search_dir=str(tmp_path))
        assert cfg.index.persist_dir == "./raven_db"
