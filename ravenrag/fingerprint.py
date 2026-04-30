"""
Fingerprint: Track file hashes for incremental re-indexing.

Stores SHA-256 hashes of indexed files so that ``raven index`` can
skip unchanged files on subsequent runs and remove deleted ones.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class FingerprintStore:
    """Persist file hashes alongside the vector database.

    The fingerprint file (``_fingerprints.json``) lives in ``persist_dir``
    and maps absolute file paths to their SHA-256 content hashes.
    """

    def __init__(self, persist_dir: str) -> None:
        self._path = Path(persist_dir) / "_fingerprints.json"
        self._hashes: Dict[str, str] = {}
        self._load()

    # -- persistence --------------------------------------------------

    def _load(self) -> None:
        if self._path.is_file():
            try:
                self._hashes = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load fingerprints: %s", e)
                self._hashes = {}

    def save(self) -> None:
        """Write fingerprints to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._hashes, indent=2), encoding="utf-8")

    # -- hashing ------------------------------------------------------

    @staticmethod
    def hash_file(path: Path) -> str:
        """Return the SHA-256 hex digest of a file's contents."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    # -- diff logic ---------------------------------------------------

    def diff(
        self,
        file_paths: List[Path],
    ) -> Tuple[List[Path], List[Path], List[str]]:
        """Compare current files against stored fingerprints.

        Args:
            file_paths: All files that *should* be indexed.

        Returns:
            Tuple of (new_or_changed, unchanged, deleted_keys).
            - ``new_or_changed``: files to (re-)index.
            - ``unchanged``: files whose hash matches — can skip.
            - ``deleted_keys``: absolute-path keys in the store that
              are no longer present on disk (should be removed from index).
        """
        current_keys = {str(p.resolve()) for p in file_paths}

        new_or_changed: List[Path] = []
        unchanged: List[Path] = []

        for p in file_paths:
            key = str(p.resolve())
            file_hash = self.hash_file(p)
            if self._hashes.get(key) == file_hash:
                unchanged.append(p)
            else:
                new_or_changed.append(p)

        deleted_keys = [k for k in self._hashes if k not in current_keys]

        return new_or_changed, unchanged, deleted_keys

    def update(self, path: Path) -> None:
        """Store the current hash of a file."""
        key = str(path.resolve())
        self._hashes[key] = self.hash_file(path)

    def remove(self, key: str) -> None:
        """Remove a fingerprint entry."""
        self._hashes.pop(key, None)
