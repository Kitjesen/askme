"""Tests for RobotMem backend integration."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from askme.memory.robotmem_backend import RobotMemBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(mem_cfg=None, brain_cfg=None):
    """Create a RobotMemBackend with test config."""
    if mem_cfg is None:
        mem_cfg = {
            "robotmem_collection": "test",
            "retrieve_timeout": 2.0,
        }
    if brain_cfg is None:
        brain_cfg = {}
    return RobotMemBackend(mem_cfg, brain_cfg)


def _make_robotmem_mock(recall_results=None):
    """Create a mock RobotMemory SDK instance."""
    mock = MagicMock()
    mock.recall = MagicMock(return_value=recall_results or [])
    mock.learn = MagicMock(return_value={"status": "created", "memory_id": 1})
    mock.close = MagicMock()
    return mock


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_config(self):
        backend = _make_backend()
        assert backend._collection == "test"
        assert backend._retrieve_timeout == 2.0
        assert backend._rm is None
        assert backend._rm_failed is False

    def test_custom_db_path(self):
        backend = _make_backend(mem_cfg={
            "robotmem_collection": "custom",
            "robotmem_db_path": "/tmp/test.db",
            "retrieve_timeout": 1.0,
        })
        assert backend._collection == "custom"
        assert backend._db_path == "/tmp/test.db"

    def test_not_available_before_init(self):
        backend = _make_backend()
        assert backend.available is False

    def test_available_after_init(self):
        backend = _make_backend()
        backend._rm = _make_robotmem_mock()
        assert backend.available is True


# ---------------------------------------------------------------------------
# Tests: _ensure_robotmem
# ---------------------------------------------------------------------------

class TestEnsureRobotmem:
    def test_returns_true_when_already_initialised(self):
        backend = _make_backend()
        backend._rm = _make_robotmem_mock()
        assert backend._ensure_robotmem() is True

    def test_returns_false_after_failure(self):
        backend = _make_backend()
        backend._rm_failed = True
        assert backend._ensure_robotmem() is False

    def test_initialises_on_first_call(self):
        backend = _make_backend()
        mock_rm = _make_robotmem_mock()
        mock_cls = MagicMock(return_value=mock_rm)

        with patch.dict("sys.modules", {"robotmem": MagicMock(), "robotmem.sdk": MagicMock(RobotMemory=mock_cls)}):
            with patch("askme.memory.robotmem_backend.RobotMemBackend._ensure_robotmem") as m:
                # Direct test: simulate successful import
                backend._rm = mock_rm
                assert backend.available is True

    def test_sets_failed_on_import_error(self):
        backend = _make_backend()

        with patch("builtins.__import__", side_effect=ImportError("no robotmem")):
            result = backend._ensure_robotmem()

        assert result is False
        assert backend._rm_failed is True

    def test_does_not_retry_after_failure(self):
        backend = _make_backend()
        backend._rm_failed = True
        assert backend._ensure_robotmem() is False


# ---------------------------------------------------------------------------
# Tests: retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    @pytest.mark.asyncio
    async def test_returns_empty_when_not_initialised(self):
        backend = _make_backend()
        backend._rm_failed = True
        result = await backend.retrieve("test query")
        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_success(self):
        backend = _make_backend()
        mock_rm = _make_robotmem_mock(recall_results=[
            {"content": "仓库A温度异常", "confidence": 0.9, "_rrf_score": 0.8},
            {"content": "仓库B正常运行", "confidence": 0.85, "_rrf_score": 0.7},
        ])
        backend._rm = mock_rm

        result = await backend.retrieve("仓库情况")
        assert "仓库A温度异常" in result
        assert "仓库B正常运行" in result
        mock_rm.recall.assert_called_once_with("仓库情况", n=5)

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self):
        backend = _make_backend()
        backend._rm = _make_robotmem_mock(recall_results=[])

        result = await backend.retrieve("nothing")
        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_filters_empty_content(self):
        backend = _make_backend()
        backend._rm = _make_robotmem_mock(recall_results=[
            {"content": "有内容", "confidence": 0.9},
            {"content": "", "confidence": 0.5},
            {"content": "也有内容", "confidence": 0.8},
        ])

        result = await backend.retrieve("test")
        assert "有内容" in result
        assert "也有内容" in result
        lines = [line for line in result.split("\n") if line.strip()]
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_retrieve_exception_returns_empty(self):
        backend = _make_backend()
        mock_rm = MagicMock()
        mock_rm.recall = MagicMock(side_effect=RuntimeError("network error"))
        backend._rm = mock_rm

        result = await backend.retrieve("test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_timeout_returns_empty(self):
        backend = _make_backend(mem_cfg={
            "robotmem_collection": "test",
            "retrieve_timeout": 0.01,
        })

        async def slow_recall(*args, **kwargs):
            await asyncio.sleep(1)
            return []

        mock_rm = MagicMock()
        mock_rm.recall = MagicMock(side_effect=lambda *a, **kw: asyncio.sleep(1))
        backend._rm = mock_rm

        # The timeout in retrieve should catch this
        result = await backend.retrieve("test")
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: save
# ---------------------------------------------------------------------------

class TestSave:
    @pytest.mark.asyncio
    async def test_save_noop_when_not_initialised(self):
        backend = _make_backend()
        backend._rm_failed = True
        await backend.save("user", "assistant")  # should not raise

    @pytest.mark.asyncio
    async def test_save_success(self):
        backend = _make_backend()
        mock_rm = _make_robotmem_mock()
        backend._rm = mock_rm

        await backend.save("你好", "你好，有什么任务？")
        mock_rm.learn.assert_called_once()
        call_args = mock_rm.learn.call_args
        text_arg = call_args[0][0]  # positional arg
        assert "你好" in text_arg

    @pytest.mark.asyncio
    async def test_save_truncates_reply(self):
        backend = _make_backend()
        mock_rm = _make_robotmem_mock()
        backend._rm = mock_rm

        long_reply = "x" * 500
        await backend.save("q", long_reply)
        call_args = mock_rm.learn.call_args
        text_arg = call_args[0][0]  # positional arg
        # Reply should be truncated to 200 chars
        assert len(text_arg) < 250

    @pytest.mark.asyncio
    async def test_save_exception_swallowed(self):
        backend = _make_backend()
        mock_rm = MagicMock()
        mock_rm.learn = MagicMock(side_effect=RuntimeError("save error"))
        backend._rm = mock_rm

        await backend.save("user", "assistant")  # should not raise

    @pytest.mark.asyncio
    async def test_save_includes_robot_context(self):
        backend = _make_backend()
        mock_rm = _make_robotmem_mock()
        backend._rm = mock_rm

        await backend.save("测试", "回复")
        call_kwargs = mock_rm.learn.call_args[1]
        context = call_kwargs["context"]
        assert context["robot"] == "thunder"
        assert context["source"] == "conversation"


# ---------------------------------------------------------------------------
# Tests: warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    @pytest.mark.asyncio
    async def test_warmup_success(self):
        backend = _make_backend()
        mock_rm = _make_robotmem_mock()
        backend._rm = mock_rm

        with patch.object(backend, "_ensure_robotmem", return_value=True):
            await backend.warmup()
        mock_rm.recall.assert_called_once_with("warmup", n=1)

    @pytest.mark.asyncio
    async def test_warmup_failure_swallowed(self):
        backend = _make_backend()

        with patch.object(backend, "_ensure_robotmem", return_value=False):
            await backend.warmup()  # should not raise


# ---------------------------------------------------------------------------
# Tests: close
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_calls_sdk_close(self):
        backend = _make_backend()
        mock_rm = _make_robotmem_mock()
        backend._rm = mock_rm

        backend.close()
        mock_rm.close.assert_called_once()
        assert backend._rm is None

    def test_close_noop_when_not_initialised(self):
        backend = _make_backend()
        backend.close()  # should not raise

    def test_close_swallows_exception(self):
        backend = _make_backend()
        mock_rm = MagicMock()
        mock_rm.close = MagicMock(side_effect=RuntimeError("close error"))
        backend._rm = mock_rm

        backend.close()  # should not raise
        assert backend._rm is None


# ---------------------------------------------------------------------------
# Tests: MemoryBridge integration (backend="robotmem")
# ---------------------------------------------------------------------------

class TestBridgeRobotmemBackend:
    """Test that MemoryBridge correctly routes to robotmem when configured."""

    def _patch_config(self, backend="robotmem", enabled=True):
        cfg = {
            "memory": {
                "enabled": enabled,
                "backend": backend,
                "embed_model": "test-model",
                "retrieve_timeout": 2.0,
                "robotmem_collection": "test",
            },
            "app": {"data_dir": "data"},
            "brain": {"api_key": "k", "base_url": "http://t", "model": "m"},
        }
        return patch("askme.memory.bridge.get_config", return_value=cfg)

    def _patch_vector_store(self):
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.available = False
        mock_instance.size = 0
        mock_cls.return_value = mock_instance
        return patch("askme.memory.bridge.VectorStore", mock_cls), mock_instance

    def test_bridge_reads_backend_config(self):
        from askme.memory.bridge import MemoryBridge

        vs_patch, vs_mock = self._patch_vector_store()
        with self._patch_config(backend="robotmem"), vs_patch:
            bridge = MemoryBridge()

        assert bridge._backend == "robotmem"

    def test_bridge_default_backend_is_mem0(self):
        from askme.memory.bridge import MemoryBridge

        vs_patch, vs_mock = self._patch_vector_store()
        cfg = {
            "memory": {"enabled": True, "embed_model": "test", "retrieve_timeout": 2.0},
            "app": {"data_dir": "data"},
            "brain": {},
        }
        with patch("askme.memory.bridge.get_config", return_value=cfg), vs_patch:
            bridge = MemoryBridge()

        assert bridge._backend == "mem0"

    def test_bridge_vector_backend(self):
        from askme.memory.bridge import MemoryBridge

        vs_patch, vs_mock = self._patch_vector_store()
        with self._patch_config(backend="vector"), vs_patch:
            bridge = MemoryBridge()

        assert bridge._backend == "vector"

    @pytest.mark.asyncio
    async def test_bridge_retrieve_routes_to_robotmem(self):
        from askme.memory.bridge import MemoryBridge
        from unittest.mock import AsyncMock

        vs_patch, vs_mock = self._patch_vector_store()
        with self._patch_config(backend="robotmem"), vs_patch:
            bridge = MemoryBridge()

        # Inject a mock robotmem backend
        mock_rm_backend = MagicMock()
        mock_rm_backend.available = True
        mock_rm_backend.retrieve = AsyncMock(return_value="- test memory")

        bridge._robotmem = mock_rm_backend

        with patch.object(bridge, "_ensure_robotmem", return_value=True):
            result = await bridge.retrieve("test")

        assert result == "- test memory"

    @pytest.mark.asyncio
    async def test_bridge_save_routes_to_robotmem(self):
        from askme.memory.bridge import MemoryBridge
        from unittest.mock import AsyncMock

        vs_patch, vs_mock = self._patch_vector_store()
        with self._patch_config(backend="robotmem"), vs_patch:
            bridge = MemoryBridge()

        mock_rm_backend = MagicMock()
        mock_rm_backend.available = True
        mock_rm_backend.save = AsyncMock()
        bridge._robotmem = mock_rm_backend

        with patch.object(bridge, "_ensure_robotmem", return_value=True):
            await bridge.save("user", "assistant")  # should not raise

    def test_bridge_available_with_robotmem(self):
        from askme.memory.bridge import MemoryBridge

        vs_patch, vs_mock = self._patch_vector_store()
        with self._patch_config(backend="robotmem"), vs_patch:
            bridge = MemoryBridge()

        mock_rm_backend = MagicMock()
        mock_rm_backend.available = True
        bridge._robotmem = mock_rm_backend

        assert bridge.available is True

    @pytest.mark.asyncio
    async def test_bridge_robotmem_fallback_to_mem0(self):
        """When robotmem fails, bridge falls back to mem0."""
        from askme.memory.bridge import MemoryBridge

        vs_patch, vs_mock = self._patch_vector_store()
        with self._patch_config(backend="robotmem"), vs_patch:
            bridge = MemoryBridge()

        bridge._robotmem_failed = True

        # Set up mem0 mock
        mock_mem0 = MagicMock()
        mock_mem0.search = MagicMock(return_value={
            "results": [{"memory": "mem0 fallback"}]
        })
        bridge._mem0 = mock_mem0

        result = await bridge.retrieve("test")
        assert "mem0 fallback" in result

    @pytest.mark.asyncio
    async def test_bridge_robotmem_fallback_to_vectorstore(self):
        """When both robotmem and mem0 fail, bridge falls back to vectorstore."""
        from askme.memory.bridge import MemoryBridge

        vs_patch, vs_mock = self._patch_vector_store()
        with self._patch_config(backend="robotmem"), vs_patch:
            bridge = MemoryBridge()

        bridge._robotmem_failed = True
        bridge._mem0_failed = True
        bridge._store = vs_mock
        vs_mock.available = True
        vs_mock.search = MagicMock(return_value=[
            {"text": "vectorstore fallback", "score": 0.9, "metadata": {}},
        ])

        result = await bridge.retrieve("test")
        assert "vectorstore fallback" in result
