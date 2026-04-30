"""Tests for FingerprintStore."""

import json

from ravenrag.fingerprint import FingerprintStore


class TestFingerprintStore:
    def test_hash_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        h = FingerprintStore.hash_file(f)
        assert len(h) == 64  # SHA-256 hex digest
        # Same content = same hash
        f2 = tmp_path / "test2.txt"
        f2.write_text("hello world", encoding="utf-8")
        assert FingerprintStore.hash_file(f2) == h

    def test_hash_file_different_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello", encoding="utf-8")
        f2 = tmp_path / "b.txt"
        f2.write_text("world", encoding="utf-8")
        assert FingerprintStore.hash_file(f1) != FingerprintStore.hash_file(f2)

    def test_diff_all_new(self, tmp_path):
        db = tmp_path / "db"
        db.mkdir()
        store = FingerprintStore(str(db))

        f = tmp_path / "new.txt"
        f.write_text("content", encoding="utf-8")

        new, unchanged, deleted = store.diff([f])
        assert len(new) == 1
        assert len(unchanged) == 0
        assert len(deleted) == 0

    def test_diff_unchanged(self, tmp_path):
        db = tmp_path / "db"
        db.mkdir()
        store = FingerprintStore(str(db))

        f = tmp_path / "file.txt"
        f.write_text("content", encoding="utf-8")
        store.update(f)
        store.save()

        # Reload
        store2 = FingerprintStore(str(db))
        new, unchanged, deleted = store2.diff([f])
        assert len(new) == 0
        assert len(unchanged) == 1
        assert len(deleted) == 0

    def test_diff_changed(self, tmp_path):
        db = tmp_path / "db"
        db.mkdir()
        store = FingerprintStore(str(db))

        f = tmp_path / "file.txt"
        f.write_text("original", encoding="utf-8")
        store.update(f)
        store.save()

        # Modify file
        f.write_text("modified", encoding="utf-8")
        store2 = FingerprintStore(str(db))
        new, unchanged, deleted = store2.diff([f])
        assert len(new) == 1
        assert len(unchanged) == 0

    def test_diff_deleted(self, tmp_path):
        db = tmp_path / "db"
        db.mkdir()
        store = FingerprintStore(str(db))

        f = tmp_path / "file.txt"
        f.write_text("content", encoding="utf-8")
        store.update(f)
        store.save()

        # Diff with empty file list
        store2 = FingerprintStore(str(db))
        new, unchanged, deleted = store2.diff([])
        assert len(deleted) == 1

    def test_save_and_load(self, tmp_path):
        db = tmp_path / "db"
        db.mkdir()
        store = FingerprintStore(str(db))

        f = tmp_path / "file.txt"
        f.write_text("content", encoding="utf-8")
        store.update(f)
        store.save()

        fp_file = db / "_fingerprints.json"
        assert fp_file.exists()
        data = json.loads(fp_file.read_text(encoding="utf-8"))
        assert str(f.resolve()) in data

    def test_remove(self, tmp_path):
        db = tmp_path / "db"
        db.mkdir()
        store = FingerprintStore(str(db))

        f = tmp_path / "file.txt"
        f.write_text("content", encoding="utf-8")
        store.update(f)
        store.remove(str(f.resolve()))
        store.save()

        store2 = FingerprintStore(str(db))
        new, unchanged, deleted = store2.diff([f])
        assert len(new) == 1  # Appears as new since fingerprint was removed

    def test_corrupt_fingerprint_file(self, tmp_path):
        db = tmp_path / "db"
        db.mkdir()
        (db / "_fingerprints.json").write_text("not json", encoding="utf-8")
        store = FingerprintStore(str(db))
        # Should not raise, just start with empty hashes
        assert store._hashes == {}
