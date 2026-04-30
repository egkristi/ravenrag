"""
Config: Load project settings from ravenrag.toml or pyproject.toml.

Precedence (highest wins): environment variables > config file > defaults.

Environment variables:
    RAVENRAG_DB           → index.persist_dir
    RAVENRAG_COLLECTION   → index.collection
    RAVENRAG_MODEL        → index.model
    RAVENRAG_TOP_K        → search.top_k
    RAVENRAG_HOST         → server.host
    RAVENRAG_PORT         → server.port
    RAVENRAG_API_KEY      → server.api_key
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Known config keys — auto-derived from dataclass fields
def _derive_known_keys() -> Dict[str, set]:
    return {
        "index": {f.name for f in fields(IndexConfig)},
        "search": {f.name for f in fields(SearchConfig)},
        "server": {f.name for f in fields(ServerConfig)},
        "watch": {"extensions"},
    }


@dataclass
class IndexConfig:
    persist_dir: str = "./ravenrag_db"
    collection: str = "documents"
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    chunk_size: int = 512
    chunk_overlap: int = 64
    glob: str = "**/*.txt"


@dataclass
class SearchConfig:
    top_k: int = 5
    rerank: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid: bool = False
    alpha: float = 0.5


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8484
    api_key: str = ""
    cors_origin: str = ""

    def __post_init__(self) -> None:
        if not (0 <= self.port <= 65535):
            raise ValueError(f"port must be 0-65535, got {self.port}")


@dataclass
class RavenConfig:
    index: IndexConfig = field(default_factory=IndexConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    watch_extensions: List[str] = field(default_factory=lambda: [".txt", ".md", ".py"])


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_toml(path: Path) -> Dict:
    """Parse a TOML file using tomllib (3.11+) or tomli fallback."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            raise ImportError(
                "tomli is required for TOML parsing on Python < 3.11. Install with: pip install tomli"
            ) from None

    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_config(search_dir: Optional[str] = None) -> RavenConfig:
    """Load config from ravenrag.toml or [tool.ravenrag] in pyproject.toml.

    Searches in order:
    1. ravenrag.toml in search_dir (or cwd)
    2. pyproject.toml [tool.ravenrag] in search_dir (or cwd)
    3. Walk up parent directories

    Returns default config if no file found.
    """
    start = Path(search_dir).resolve() if search_dir else Path.cwd()

    # Search upward for config files
    for directory in [start, *start.parents]:
        raven_toml = directory / "ravenrag.toml"
        if raven_toml.is_file():
            logger.debug("Loading config from %s", raven_toml)
            data = _parse_toml(raven_toml)
            return _build_config(data)

        pyproject = directory / "pyproject.toml"
        if pyproject.is_file():
            data = _parse_toml(pyproject)
            if "tool" in data and "ravenrag" in data["tool"]:
                logger.debug("Loading config from %s [tool.ravenrag]", pyproject)
                return _build_config(data["tool"]["ravenrag"])

    return RavenConfig()


def _build_config(data: Dict) -> RavenConfig:
    """Build a RavenConfig from parsed TOML data."""
    config = RavenConfig()
    known_keys = _derive_known_keys()

    for section_name in ("index", "search", "server"):
        if section_name not in data:
            continue
        section_obj = getattr(config, section_name)
        known = known_keys.get(section_name, set())
        for key, value in data[section_name].items():
            key = key.replace("-", "_")
            if key not in known:
                logger.warning("Unknown config key [%s].%s — ignoring (typo?)", section_name, key)
                continue
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)

    if "watch" in data:
        known_watch = known_keys.get("watch", set())
        for key in data["watch"]:
            if key.replace("-", "_") not in known_watch:
                logger.warning("Unknown config key [watch].%s — ignoring (typo?)", key)
        if "extensions" in data["watch"]:
            config.watch_extensions = data["watch"]["extensions"]

    if "watch_extensions" in data:
        config.watch_extensions = data["watch_extensions"]

    # Environment variable overrides (highest precedence)
    _apply_env_overrides(config)

    return config


def _apply_env_overrides(config: RavenConfig) -> None:
    """Apply RAVENRAG_* environment variables over config values."""
    env_map = {
        "RAVENRAG_DB": ("index", "persist_dir", str),
        "RAVENRAG_COLLECTION": ("index", "collection", str),
        "RAVENRAG_MODEL": ("index", "model", str),
        "RAVENRAG_TOP_K": ("search", "top_k", int),
        "RAVENRAG_HOST": ("server", "host", str),
        "RAVENRAG_PORT": ("server", "port", int),
        "RAVENRAG_API_KEY": ("server", "api_key", str),
    }
    for env_var, (section, attr, cast) in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                setattr(getattr(config, section), attr, cast(value))
                logger.debug("Config override from %s", env_var)
            except (ValueError, TypeError):
                logger.warning("Invalid value for %s=%r — ignoring", env_var, value)
