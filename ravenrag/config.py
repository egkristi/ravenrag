"""
Config: Load project settings from ravenrag.toml or pyproject.toml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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
    """Parse a TOML file, supporting Python 3.9+ (tomllib) and fallback."""
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            # Minimal TOML parser for simple key=value configs
            return _parse_toml_minimal(path)

    return tomllib.loads(path.read_text(encoding="utf-8"))


def _parse_toml_minimal(path: Path) -> Dict:
    """Minimal TOML parser for simple flat configs (fallback for Python 3.9-3.10)."""
    result: Dict = {}
    current_section = result
    section_path: List[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].strip()
            section_path = section_name.split(".")
            current_section = result
            for part in section_path:
                current_section = current_section.setdefault(part, {})
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Parse value types
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.startswith("[") and value.endswith("]"):
                # Simple list parsing
                inner = value[1:-1].strip()
                if inner:
                    value = [v.strip().strip("\"'") for v in inner.split(",")]
                else:
                    value = []
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            current_section[key] = value

    return result


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

    if "index" in data:
        for key, value in data["index"].items():
            key = key.replace("-", "_")
            if hasattr(config.index, key):
                setattr(config.index, key, value)

    if "search" in data:
        for key, value in data["search"].items():
            key = key.replace("-", "_")
            if hasattr(config.search, key):
                setattr(config.search, key, value)

    if "server" in data:
        for key, value in data["server"].items():
            key = key.replace("-", "_")
            if hasattr(config.server, key):
                setattr(config.server, key, value)

    if "watch_extensions" in data:
        config.watch_extensions = data["watch_extensions"]
    elif "watch" in data and "extensions" in data["watch"]:
        config.watch_extensions = data["watch"]["extensions"]

    return config
