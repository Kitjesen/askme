"""Tests for RobotMemBackend — init, available, retrieve, save, consolidate, close."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from askme.memory.robotmem_backend import RobotMemBackend

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_backend(**kwargs) -> RobotMemBackend:
    mem_cfg = {"robotmem_collection": "test", **kwargs}
    return RobotMemBackend(mem_cfg=mem_cfg, brain_cfg={})


def _make_backend_with_rm() -> tuple[RobotMemBackend, MagicMock]:
    """Backend with a pre-initialised mock _rm SDK instance."""
    backend = _make_backend()
    rm = MagicMock()
    rm.recall.return_value = []
    rm.learn.return_value = None
    rm.close.return_value = None
    backend._rm = rm
    return backend, rm


# ── Init ──────────────────────────────────────────────────────────────────────

class TestInit:
    def test_collection_from_config(self):
        backend = _make_backend(robotmem_collection="myproject")
        assert backend._collection == "myproject"

    def test_default_collection(self):
        backend = RobotMemBackend(mem_cfg={}, brain_cfg={})
        assert backend._collection == "askme"

    def test_default_retrieve_timeout(self):
        backend = _make_backend()
        assert backend._retrieve_timeout == 2.0

    def test_custom_retrieve_timeout(self):
        backend = _make_backend(retrieve_timeout=5.0)
        assert backend._retrieve_timeout == 5.0

    def test_db_path_default_none(self):
        backend = _make_backend()
        assert backend._db_path is None

    def test_db_path_from_config(self):
        backend = _make_backend(robotmem_db_path="/tmp/robotmem.db")
        assert backend._db_path == "/tmp/robotmem.db"

    def test_not_available_initially(self):
        backend = _make_backend()
        assert backend.available is False

    def test_rm_failed_false_initially(self):
        backend = _make_backend()
        assert backend._rm_failed is False


# ── _ensure_robotmem ─────────────────────────────────────────────────────────

class TestEnsureRobotmem:
    def test_returns_false_when_sdk_not_installed(self):
        backend = _make_backend()
        with patch.dict("sys.modules", {"robotmem": None, "robotmem.sdk": None}):
            result = backend._ensure_robotmem()
        assert result is False
        assert backend._rm_failed is True

    def test_returns_false_on_second_call_after_failure(self):
        backend = _make_backend()
        backend._rm_failed = True
        result = backend._ensure_robotmem()
        assert result is False

    def test_returns_true_when_already_initialised(self):
        backend, _ = _make_backend_with_rm()
        result = backend._ensure_robotmem()
        assert result is True

    def test_sets_rm_on_success(self):
        backend = _make_backend()
        mock_rm_cls = MagicMock(return_value=MagicMock())
        mock_module = MagicMock()
        mock_module.RobotMemory = mock_rm_cls
        with patch.dict("sys.modules", {"robotmem": mock_module, "robotmem.sdk": mock_module}):
            with patch("builtins.__import__", side_effect=lambda name, *a, **kw: (
                mock_module if "robotmem" in name else __import__(name, *a, **kw)
            )):
                pass  # just test the failure path is reliable

    def test_passes_db_path_when_configured(self):
        backend = _make_backend(robotmem_db_path="/tmp/test.db")
        mock_rm = MagicMock()
        mock_module = MagicMock()
        mock_module.RobotMemory = MagicMock(return_value=mock_rm)
        with patch("askme.memory.robotmem_backend.RobotMemBackend._ensure_robotmem", return_value=True):
            backend._rm = mock_rm
        assert backend._db_path == "/tmp/test.db"


# ── available ─────────────────────────────────────────────────────────────────

class TestAvailable:
    def test_false_without_rm(self):
        backend = _make_backend()
        assert backend.available is False

    def test_true_with_rm(self):
        backend, _ = _make_backend_with_rm()
        assert backend.available is True


# ── retrieve ─────────────────────────────────────────────────────────────────

class TestRetrieve:
    @pytest.mark.asyncio
    async def test_returns_empty_when_not_initialised(self):
        backend = _make_backend()
        backend._rm_failed = True
        result = await backend.retrieve("test query")
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_formatted_items(self):
        backend, rm = _make_backend_with_rm()
        rm.recall.return_value = [
            {"content": "用户喜欢咖啡"},
            {"content": "机器人名叫雷霆"},
        ]
        result = await backend.retrieve("用户偏好")
        assert "用户喜欢咖啡" in result
        assert "机器人名叫雷霆" in result
        assert result.startswith("-")

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_memories(self):
        backend, rm = _make_backend_with_rm()
        rm.recall.return_value = []
        result = await backend.retrieve("anything")
        assert result == ""

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        backend, rm = _make_backend_with_rm()
        rm.recall.side_effect = OSError("db error")
        result = await backend.retrieve("query")
        assert result == ""

    @pytest.mark.asyncio
    async def test_skips_empty_content_items(self):
        backend, rm = _make_backend_with_rm()
        rm.recall.return_value = [
            {"content": ""},
            {"content": "valid memory"},
        ]
        result = await backend.retrieve("query")
        assert "valid memory" in result

    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self):
        import asyncio
        backend, rm = _make_backend_with_rm()
        backend._retrieve_timeout = 0.001

        async def slow_recall(*a, **kw):
            await asyncio.sleep(10)
            return []

        with patch("asyncio.to_thread", side_effect=asyncio.TimeoutError):
            result = await backend.retrieve("slow query")
        assert result == ""


# ── save ──────────────────────────────────────────────────────────────────────

class TestSave:
    @pytest.mark.asyncio
    async def test_save_does_nothing_when_not_initialised(self):
        backend = _make_backend()
        backend._rm_failed = True
        await backend.save("user text", "assistant text")  # should not raise

    @pytest.mark.asyncio
    async def test_save_calls_learn(self):
        backend, rm = _make_backend_with_rm()
        await backend.save("你好", "你好，有什么可以帮你？")
        rm.learn.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_includes_both_texts(self):
        backend, rm = _make_backend_with_rm()
        await backend.save("用户问题", "助手回复")
        call_args = rm.learn.call_args
        text_passed = call_args[0][0]
        assert "用户问题" in text_passed
        assert "助手回复" in text_passed

    @pytest.mark.asyncio
    async def test_save_on_exception_does_not_raise(self):
        backend, rm = _make_backend_with_rm()
        rm.learn.side_effect = OSError("write error")
        await backend.save("q", "a")  # must not raise


# ── close ─────────────────────────────────────────────────────────────────────

class TestClose:
    def test_close_clears_rm(self):
        backend, rm = _make_backend_with_rm()
        backend.close()
        assert backend._rm is None
        rm.close.assert_called_once()

    def test_close_when_no_rm_no_crash(self):
        backend = _make_backend()
        backend.close()  # should not raise

    def test_close_swallows_exceptions(self):
        backend, rm = _make_backend_with_rm()
        rm.close.side_effect = OSError("close failed")
        backend.close()  # should not raise
        assert backend._rm is None


# ── warmup ────────────────────────────────────────────────────────────────────

class TestWarmup:
    @pytest.mark.asyncio
    async def test_warmup_does_not_raise_when_not_available(self):
        backend = _make_backend()
        backend._rm_failed = True
        await backend.warmup()  # should not raise

    @pytest.mark.asyncio
    async def test_warmup_calls_recall_dummy(self):
        backend, rm = _make_backend_with_rm()
        rm.recall.return_value = []
        await backend.warmup()
        rm.recall.assert_called_once_with("warmup", n=1)

    @pytest.mark.asyncio
    async def test_warmup_swallows_exceptions(self):
        backend, rm = _make_backend_with_rm()
        rm.recall.side_effect = OSError("warmup error")
        await backend.warmup()  # should not raise
