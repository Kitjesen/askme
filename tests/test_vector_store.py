"""Tests for VectorStore — works without sentence-transformers via mocking."""

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from askme.memory.vector_store import VectorStore


# -- Helpers ------------------------------------------------------------------

def _patch_available(val):
    """Patch _check_st_available to return a fixed value."""
    return patch("askme.memory.vector_store._check_st_available", return_value=val)


def _make_store(tmp_path=None, available=True):
    """Create a VectorStore with mocked encoder."""
    store_path = tmp_path / "store.json" if tmp_path else None
    with _patch_available(available):
        store = VectorStore(store_path=store_path)
    # Mock the encoder so we never import sentence_transformers
    store._encode = MagicMock(side_effect=_mock_encode)
    # Patch available check on the instance's method calls too
    store._check = available
    return store


_DIM = 8


def _mock_encode(texts):
    """Return deterministic fake embeddings based on text hash."""
    result = []
    for t in texts:
        np.random.seed(hash(t) % (2**31))
        vec = np.random.randn(_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        result.append(vec)
    return np.array(result)


# -- Tests: basic operations -------------------------------------------------

class TestAvailability:
    def test_available_when_st_installed(self):
        with _patch_available(True):
            store = VectorStore()
            assert store.available is True

    def test_unavailable_when_st_missing(self):
        with _patch_available(False):
            store = VectorStore()
            assert store.available is False

    def test_add_noop_when_unavailable(self):
        with _patch_available(False):
            store = VectorStore()
            store.add("hello")
            assert store.size == 0

    def test_search_empty_when_unavailable(self):
        with _patch_available(False):
            store = VectorStore()
            assert store.search("hello") == []


class TestAddAndSearch:
    def test_add_increases_size(self, tmp_path):
        with _patch_available(True):
            store = _make_store(tmp_path)
            assert store.size == 0
            store.add("first entry")
            assert store.size == 1
            store.add("second entry")
            assert store.size == 2

    def test_add_empty_text_ignored(self, tmp_path):
        with _patch_available(True):
            store = _make_store(tmp_path)
            store.add("")
            store.add("   ")
            assert store.size == 0

    def test_search_returns_results(self, tmp_path):
        with _patch_available(True):
            store = _make_store(tmp_path)
            store.add("warehouse A temperature alert")
            store.add("warehouse B normal status")
            store.add("warehouse A temperature alert again")

            results = store.search("temperature alert", top_k=2)
            assert len(results) == 2
            assert "text" in results[0]
            assert "score" in results[0]
            assert "metadata" in results[0]

    def test_search_empty_store_returns_empty(self, tmp_path):
        with _patch_available(True):
            store = _make_store(tmp_path)
            assert store.search("anything") == []

    def test_search_empty_query_returns_empty(self, tmp_path):
        with _patch_available(True):
            store = _make_store(tmp_path)
            store.add("some text")
            assert store.search("") == []
            assert store.search("   ") == []

    def test_search_top_k_limits_results(self, tmp_path):
        with _patch_available(True):
            store = _make_store(tmp_path)
            for i in range(10):
                store.add(f"entry number {i}")
            results = store.search("entry", top_k=3)
            assert len(results) == 3

    def test_metadata_preserved(self, tmp_path):
        with _patch_available(True):
            store = _make_store(tmp_path)
            store.add("test entry", {"source": "test", "id": 42})
            results = store.search("test entry", top_k=1)
            assert results[0]["metadata"]["source"] == "test"
            assert results[0]["metadata"]["id"] == 42


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        with _patch_available(True):
            store = _make_store(tmp_path)
            store.add("persistent entry", {"tag": "a"})
            store.add("another entry", {"tag": "b"})
            store.save()

            # Verify file exists
            store_file = tmp_path / "store.json"
            assert store_file.exists()

            # Load into new store
            store2 = VectorStore(store_path=store_file)
            assert store2.size == 2

    def test_load_nonexistent_file(self, tmp_path):
        store = VectorStore(store_path=tmp_path / "missing.json")
        assert store.size == 0

    def test_save_creates_directories(self, tmp_path):
        with _patch_available(True):
            deep_path = tmp_path / "a" / "b" / "c" / "store.json"
            store = _make_store()
            store._store_path = deep_path
            store.add("test")
            store.save()
            assert deep_path.exists()


class TestThreadSafety:
    def test_concurrent_adds(self, tmp_path):
        with _patch_available(True):
            store = _make_store(tmp_path)
            errors = []

            def add_entries(start):
                try:
                    for i in range(20):
                        store.add(f"thread entry {start + i}")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=add_entries, args=(i * 20,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors
            assert store.size == 80
