"""Tests for MemoryBridge — Mem0 primary + VectorStore fallback."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from askme.memory.bridge import MemoryBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_config(enabled=True):
    """Patch get_config to return a minimal config dict."""
    cfg = {
        "memory": {"enabled": enabled, "embed_model": "test-model", "retrieve_timeout": 2.0},
        "app": {"data_dir": "data"},
        "brain": {"api_key": "test-key", "base_url": "http://test", "model": "test-model"},
    }
    return patch("askme.memory.bridge.get_config", return_value=cfg)


def _patch_vector_store():
    """Patch VectorStore to a mock that reports unavailable."""
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.available = False
    mock_instance.size = 0
    mock_cls.return_value = mock_instance
    return patch("askme.memory.bridge.VectorStore", mock_cls), mock_instance


def _make_bridge(enabled=True):
    """Create a MemoryBridge with mocked dependencies."""
    vs_patch, vs_mock = _patch_vector_store()
    with _patch_config(enabled=enabled), vs_patch:
        bridge = MemoryBridge()
    bridge._store = vs_mock
    return bridge, vs_mock


def _make_mem0_mock(search_results=None):
    """Create a mock Mem0 Memory instance."""
    mock = MagicMock()
    mock.search = MagicMock(return_value=search_results or {"results": []})
    mock.add = MagicMock(return_value={"results": []})
    return mock


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_disabled_bridge(self):
        bridge, _ = _make_bridge(enabled=False)
        assert bridge._enabled is False
        assert bridge.available is False

    def test_enabled_no_mem0_no_vectorstore(self):
        bridge, vs = _make_bridge(enabled=True)
        vs.available = False
        # Mem0 not initialised yet, VectorStore unavailable
        assert bridge.available is False

    def test_available_when_mem0_ready(self):
        bridge, _ = _make_bridge(enabled=True)
        bridge._mem0 = _make_mem0_mock()
        assert bridge.available is True

    def test_available_when_vectorstore_ready(self):
        bridge, vs = _make_bridge(enabled=True)
        vs.available = True
        # No Mem0, but VectorStore works
        assert bridge.available is True


# ---------------------------------------------------------------------------
# Tests: _ensure_mem0
# ---------------------------------------------------------------------------

class TestEnsureMem0:
    def test_returns_true_when_already_initialised(self):
        bridge, _ = _make_bridge()
        bridge._mem0 = _make_mem0_mock()
        assert bridge._ensure_mem0() is True

    def test_returns_false_when_disabled(self):
        bridge, _ = _make_bridge(enabled=False)
        assert bridge._ensure_mem0() is False

    def test_returns_false_after_failure(self):
        bridge, _ = _make_bridge()
        bridge._mem0_failed = True
        assert bridge._ensure_mem0() is False

    def test_initialises_mem0_on_first_call(self):
        bridge, _ = _make_bridge()
        mock_mem0 = _make_mem0_mock()
        mock_memory_cls = MagicMock()
        mock_memory_cls.from_config = MagicMock(return_value=mock_mem0)

        with patch.dict("sys.modules", {"mem0": MagicMock(Memory=mock_memory_cls)}):
            result = bridge._ensure_mem0()

        assert result is True
        assert bridge._mem0 is mock_mem0

    def test_sets_failed_flag_on_import_error(self):
        bridge, _ = _make_bridge()

        with patch("builtins.__import__", side_effect=ImportError("no mem0")):
            result = bridge._ensure_mem0()

        assert result is False
        assert bridge._mem0_failed is True

    def test_does_not_retry_after_failure(self):
        bridge, _ = _make_bridge()
        bridge._mem0_failed = True
        # Should not attempt import
        assert bridge._ensure_mem0() is False


# ---------------------------------------------------------------------------
# Tests: retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self):
        bridge, _ = _make_bridge(enabled=False)
        result = await bridge.retrieve("test query")
        assert result == ""

    @pytest.mark.asyncio
    async def test_mem0_retrieve_success(self):
        bridge, _ = _make_bridge()
        mock_mem0 = _make_mem0_mock(search_results={
            "results": [
                {"memory": "仓库A温度异常"},
                {"memory": "仓库B正常"},
            ]
        })
        bridge._mem0 = mock_mem0

        result = await bridge.retrieve("仓库情况")
        assert "仓库A温度异常" in result
        assert "仓库B正常" in result
        mock_mem0.search.assert_called_once_with("仓库情况", user_id="robot")

    @pytest.mark.asyncio
    async def test_mem0_retrieve_empty_results(self):
        bridge, _ = _make_bridge()
        bridge._mem0 = _make_mem0_mock(search_results={"results": []})

        result = await bridge.retrieve("nothing")
        assert result == ""

    @pytest.mark.asyncio
    async def test_mem0_retrieve_no_results_key(self):
        bridge, _ = _make_bridge()
        bridge._mem0 = _make_mem0_mock(search_results={})

        result = await bridge.retrieve("nothing")
        assert result == ""

    @pytest.mark.asyncio
    async def test_mem0_retrieve_filters_empty_memories(self):
        bridge, _ = _make_bridge()
        bridge._mem0 = _make_mem0_mock(search_results={
            "results": [
                {"memory": "有内容"},
                {"memory": ""},
                {"memory": "也有内容"},
            ]
        })

        result = await bridge.retrieve("test")
        assert "有内容" in result
        assert "也有内容" in result
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_fallback_to_vectorstore(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[
            {"text": "fallback result", "score": 0.8, "metadata": {}},
        ])

        result = await bridge.retrieve("test")
        assert "fallback result" in result
        vs.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_fallback_filters_low_score(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[
            {"text": "low score", "score": 0.2, "metadata": {}},
        ])

        result = await bridge.retrieve("test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_when_both_unavailable(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = False

        result = await bridge.retrieve("test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_mem0_retrieve_exception_returns_empty(self):
        bridge, _ = _make_bridge()
        mock_mem0 = MagicMock()
        mock_mem0.search = MagicMock(side_effect=RuntimeError("network error"))
        bridge._mem0 = mock_mem0

        result = await bridge.retrieve("test")
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: save
# ---------------------------------------------------------------------------

class TestSave:
    @pytest.mark.asyncio
    async def test_save_disabled_noop(self):
        bridge, _ = _make_bridge(enabled=False)
        await bridge.save("user", "assistant")  # should not raise

    @pytest.mark.asyncio
    async def test_mem0_save(self):
        bridge, _ = _make_bridge()
        mock_mem0 = _make_mem0_mock()
        bridge._mem0 = mock_mem0

        await bridge.save("你好", "你好，有什么任务？")
        mock_mem0.add.assert_called_once()
        call_args = mock_mem0.add.call_args
        assert "你好" in call_args[0][0]
        assert call_args[1]["user_id"] == "robot"

    @pytest.mark.asyncio
    async def test_mem0_save_truncates_reply(self):
        bridge, _ = _make_bridge()
        mock_mem0 = _make_mem0_mock()
        bridge._mem0 = mock_mem0

        long_reply = "x" * 500
        await bridge.save("q", long_reply)
        call_text = mock_mem0.add.call_args[0][0]
        # Reply should be truncated to 200 chars
        assert len(call_text) < 250  # user line + truncated reply

    @pytest.mark.asyncio
    async def test_mem0_save_exception_swallowed(self):
        bridge, _ = _make_bridge()
        mock_mem0 = MagicMock()
        mock_mem0.add = MagicMock(side_effect=RuntimeError("save error"))
        bridge._mem0 = mock_mem0

        await bridge.save("user", "assistant")  # should not raise

    @pytest.mark.asyncio
    async def test_fallback_save_to_vectorstore(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True

        await bridge.save("user", "assistant")
        vs.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_noop_when_both_unavailable(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = False

        await bridge.save("user", "assistant")  # should not raise
        vs.add.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    @pytest.mark.asyncio
    async def test_warmup_disabled_noop(self):
        bridge, _ = _make_bridge(enabled=False)
        await bridge.warmup()  # should not raise

    @pytest.mark.asyncio
    async def test_warmup_with_mem0(self):
        bridge, _ = _make_bridge()
        mock_mem0 = _make_mem0_mock()

        # Make _ensure_mem0 succeed and set mem0
        def side_effect():
            bridge._mem0 = mock_mem0
            return True

        with patch.object(bridge, "_ensure_mem0", side_effect=side_effect):
            await bridge.warmup()

    @pytest.mark.asyncio
    async def test_warmup_falls_back_to_vectorstore(self):
        bridge, vs = _make_bridge()
        vs.available = True
        bridge._mem0_failed = True

        with patch.object(bridge, "_ensure_mem0", return_value=False):
            await bridge.warmup()
            vs.search.assert_called_once_with("warmup", 1)


# ---------------------------------------------------------------------------
# Tests: properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_vector_store_property(self):
        bridge, vs = _make_bridge()
        assert bridge.vector_store is vs

    def test_import_existing_data_noop_when_disabled(self):
        bridge, _ = _make_bridge(enabled=False)
        assert bridge.import_existing_data() == 0

    def test_import_existing_data_noop_when_unavailable(self):
        bridge, vs = _make_bridge()
        vs.available = False
        assert bridge.import_existing_data() == 0
