"""Large-scale memory regression tests with offline deterministic fakes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from askme.memory.bridge import MemoryBridge
from askme.memory.vector_store import VectorStore


def _patch_vector_available():
    return patch("askme.memory.vector_store._check_st_available", return_value=True)


def _one_hot_encode(texts: list[str]) -> np.ndarray:
    """Encode item-N text so matching query-N deterministically ranks item-N first."""
    vectors = []
    for text in texts:
        idx = 0
        for part in text.replace("-", " ").split():
            if part.isdigit():
                idx = int(part) % 16
                break
        vec = np.zeros(16, dtype=np.float32)
        vec[idx] = 1.0
        vectors.append(vec)
    return np.array(vectors)


def test_vector_store_large_corpus_search_shape_and_top_k(tmp_path):
    with _patch_vector_available():
        store = VectorStore(store_path=tmp_path / "vectors.json")
        store._encode = MagicMock(side_effect=_one_hot_encode)

        for idx in range(1024):
            store.add(
                f"memory item-{idx} zone-{idx % 16}",
                {"idx": idx, "zone": idx % 16},
            )

        results = store.search("query item-42", top_k=7)

    assert store.size == 1024
    assert len(results) == 7
    assert all(set(result) == {"text", "score", "metadata"} for result in results)
    assert all(isinstance(result["text"], str) for result in results)
    assert all(isinstance(result["score"], float) for result in results)
    assert all(isinstance(result["metadata"], dict) for result in results)
    assert all(result["metadata"]["zone"] == 10 for result in results)
    assert store._encode.call_count == 1025


def _make_bridge_with_mem0(search_results):
    config = {
        "memory": {
            "enabled": True,
            "backend": "mem0",
            "embed_model": "test-model",
            "retrieve_timeout": 2.0,
        },
        "app": {"data_dir": "data"},
        "brain": {
            "api_key": "test-key",
            "base_url": "http://test.invalid",
            "model": "test-model",
        },
    }
    bridge = MemoryBridge(config=config)
    bridge._mem0 = MagicMock()
    bridge._mem0.search = MagicMock(return_value=search_results)
    return bridge


@pytest.mark.asyncio
async def test_memory_bridge_large_mem0_result_cap():
    search_results = {
        "results": [
            {"memory": f"large memory result {idx}"}
            for idx in range(250)
        ]
    }
    bridge = _make_bridge_with_mem0(search_results)

    result = await bridge.retrieve("large memory query")

    lines = result.splitlines()
    assert lines == [f"- large memory result {idx}" for idx in range(5)]
    assert "large memory result 5" not in result
    bridge._mem0.search.assert_called_once_with("large memory query", user_id="robot")
