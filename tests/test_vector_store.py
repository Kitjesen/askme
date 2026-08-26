"""Tests for VectorStore — works without fastembed via mocking."""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from askme.memory.vector_store import (
    VectorStore,
    _fastembed_model_status,
    _top_score_indices,
)

# -- Helpers ------------------------------------------------------------------

def _patch_available(val):
    """Patch _check_fastembed_available to return a fixed value."""
    return patch("askme.memory.vector_store._check_fastembed_available", return_value=val)


def _make_store(tmp_path=None, available=True):
    """Create a VectorStore with mocked encoder."""
    store_path = tmp_path / "store.json" if tmp_path else None
    with _patch_available(available):
        store = VectorStore(store_path=store_path)
    # Mock the encoder so we never import fastembed
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
class TestPackaging:
    def test_memory_extra_installs_local_vector_backend(self):
        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        memory_extra = pyproject.split("memory = [", 1)[1].split("]", 1)[0]

        assert '"fastembed>=0.4.0"' in memory_extra


class TestAvailability:
    def test_dependency_installed_without_cached_model_is_not_runtime_ready(
        self,
        tmp_path,
    ):
        status = _fastembed_model_status(
            cache_dir=tmp_path,
            dependency_installed=True,
        )
        with (
            _patch_available(True),
            patch(
                "askme.memory.vector_store._fastembed_model_status",
                return_value=status,
            ),
        ):
            store = VectorStore()
            assert store.dependency_installed is True
            assert store.available is False
            assert store.model_status["reason"] == "model_artifacts_missing"
            assert store.model_status["network_checked"] is False

    def test_complete_local_model_artifacts_are_runtime_ready(self, tmp_path):
        snapshot = (
            tmp_path
            / "models--qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
            / "snapshots"
            / "local-revision"
        )
        snapshot.mkdir(parents=True)
        refs = snapshot.parents[1] / "refs"
        refs.mkdir()
        (refs / "main").write_text("local-revision", encoding="utf-8")
        for filename in (
            "config.json",
            "model_optimized.onnx",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            (snapshot / filename).write_bytes(b"local-artifact")

        status = _fastembed_model_status(
            cache_dir=tmp_path,
            dependency_installed=True,
        )

        assert status["ready"] is True
        assert status["cached"] is True
        assert status["model_path"] == str(snapshot)
        assert status["missing_artifacts"] == []
        assert status["network_checked"] is False

    def test_unavailable_when_fastembed_missing(self):
        with _patch_available(False):
            store = VectorStore()
            assert store.dependency_installed is False
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

    def test_embedding_model_is_pinned_to_cpu_provider(self):
        constructor = MagicMock(return_value=object())
        fake_module = SimpleNamespace(TextEmbedding=constructor)

        with (
            patch.dict(sys.modules, {"fastembed": fake_module}),
            patch("askme.memory.vector_store._MODEL_CACHE", {}),
            _patch_available(True),
        ):
            VectorStore()._get_model()

        constructor.assert_called_once_with(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            providers=["CPUExecutionProvider"],
            cuda=False,
            local_files_only=True,
        )


class TestTopScoreIndices:
    def test_returns_scores_descending_without_full_sorting(self):
        scores = np.array([0.2, 0.9, 0.1, 0.7, 0.4], dtype=np.float32)

        assert _top_score_indices(scores, 3).tolist() == [1, 3, 4]

    def test_non_positive_top_k_returns_empty(self):
        scores = np.array([0.2, 0.9, 0.1], dtype=np.float32)

        assert _top_score_indices(scores, 0).tolist() == []
        assert _top_score_indices(scores, -2).tolist() == []

    def test_top_k_larger_than_scores_returns_all_sorted(self):
        scores = np.array([0.2, 0.9, 0.1], dtype=np.float32)

        assert _top_score_indices(scores, 10).tolist() == [1, 0, 2]

    def test_empty_scores_returns_empty(self):
        scores = np.array([], dtype=np.float32)
        assert _top_score_indices(scores, 3).tolist() == []

    def test_single_element_returns_its_index(self):
        scores = np.array([0.5], dtype=np.float32)
        assert _top_score_indices(scores, 3).tolist() == [0]

    def test_top_k_one_with_multiple_scores(self):
        scores = np.array([0.2, 0.9, 0.1, 0.7], dtype=np.float32)
        assert _top_score_indices(scores, 1).tolist() == [1]

    def test_identical_scores_returns_correct_count(self):
        scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = _top_score_indices(scores, 2)
        assert len(result) == 2
        # argpartition is not stable for tied scores — only verify count and bounds
        assert all(0 <= idx < 3 for idx in result.tolist())


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
