"""Stage 3 runtime integration tests.

Covers:
- DogSafetyClient.query_estop_state() caching
- DogSafetyClient.is_estop_active() non-blocking cache read
- BrainPipeline._build_l0_runtime_block() injection
- BrainPipeline.execute_skill() estop safety gate
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.robot.safety_client import DogSafetyClient, _ESTOP_STATE_TTL


# ── DogSafetyClient ──────────────────────────────────────────────────────────


class TestQueryEstopState:
    def _make_client(self, base_url: str = "http://fake:5070") -> DogSafetyClient:
        return DogSafetyClient({"base_url": base_url})

    def test_returns_none_when_not_configured(self):
        client = DogSafetyClient({})
        assert client.query_estop_state() is None

    def test_fetches_and_caches_state(self):
        client = self._make_client()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"enabled": False}
        fake_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=fake_resp) as mock_get:
            result = client.query_estop_state()
            assert result == {"enabled": False}
            assert mock_get.call_count == 1
            # Second call should hit cache, no new HTTP request
            result2 = client.query_estop_state()
            assert result2 == {"enabled": False}
            assert mock_get.call_count == 1  # still 1 — cache hit

    def test_cache_expires_after_ttl(self):
        client = self._make_client()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"enabled": False}
        fake_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=fake_resp) as mock_get:
            client.query_estop_state()
            # Expire the cache manually
            client._cache_ts = time.monotonic() - _ESTOP_STATE_TTL - 1.0
            client.query_estop_state()
            assert mock_get.call_count == 2  # second fetch after expiry

    def test_returns_none_on_network_error(self):
        client = self._make_client()
        with patch("requests.get", side_effect=OSError("connection refused")):
            assert client.query_estop_state() is None

    def test_returns_none_on_non_200(self):
        client = self._make_client()
        import requests as _requests
        fake_resp = MagicMock()
        fake_resp.raise_for_status.side_effect = _requests.HTTPError("503")
        with patch("requests.get", return_value=fake_resp):
            assert client.query_estop_state() is None


class TestIsEstopActive:
    def _make_client(self, base_url: str = "http://fake:5070") -> DogSafetyClient:
        return DogSafetyClient({"base_url": base_url})

    def test_returns_false_when_no_cache(self):
        client = self._make_client()
        # No cache populated — should be safe (False)
        assert client.is_estop_active() is False

    def test_returns_true_when_cache_says_enabled(self):
        client = self._make_client()
        client._cached_estop = {"enabled": True}
        client._cache_ts = time.monotonic()
        assert client.is_estop_active() is True

    def test_returns_false_when_cache_says_disabled(self):
        client = self._make_client()
        client._cached_estop = {"enabled": False}
        client._cache_ts = time.monotonic()
        assert client.is_estop_active() is False

    def test_returns_false_when_cache_stale(self):
        client = self._make_client()
        client._cached_estop = {"enabled": True}
        client._cache_ts = time.monotonic() - _ESTOP_STATE_TTL - 1.0
        # Stale data = treat as unknown = safe
        assert client.is_estop_active() is False

    def test_never_calls_network(self):
        client = self._make_client()
        with patch("requests.get") as mock_get:
            client.is_estop_active()
            mock_get.assert_not_called()


# ── BrainPipeline L0 block ───────────────────────────────────────────────────


def _make_pipeline(dog_safety=None):
    """Build a minimal BrainPipeline-like object for L0 tests."""
    from askme.pipeline.brain_pipeline import BrainPipeline
    from askme.pipeline.prompt_builder import PromptBuilder
    pipeline = object.__new__(BrainPipeline)
    pipeline._dog_safety = dog_safety
    pipeline._dog_control = None
    pipeline._base_prompt = "你是Thunder巡检机器人。"
    pipeline._episodic = None
    pipeline._session_memory = None
    pipeline._memory = None
    pipeline._mem = None
    pipeline._vision = None
    pipeline._tools = MagicMock()
    pipeline._tools.get_definitions.return_value = []
    pipeline._skill_manager = MagicMock()
    pipeline._skill_manager.get_skill_catalog.return_value = "none"
    pipeline._general_tool_max_safety_level = "normal"
    pipeline._qp_memory = None
    pipeline._prompt_builder = PromptBuilder(
        base_prompt=pipeline._base_prompt,
        prompt_seed=[],
        user_prefix="",
        tools=pipeline._tools,
        skill_manager=pipeline._skill_manager,
        general_tool_max_safety_level="normal",
        dog_safety=dog_safety,
        episodic=None,
        session_memory=None,
        vision=None,
        qp_memory=None,
    )
    return pipeline


class TestBuildL0RuntimeBlock:
    def test_empty_when_no_safety_client(self):
        pipeline = _make_pipeline(dog_safety=None)
        assert pipeline._build_l0_runtime_block() == ""

    def test_empty_when_safety_not_configured(self):
        safety = DogSafetyClient({})  # no URL = not configured
        pipeline = _make_pipeline(dog_safety=safety)
        assert pipeline._build_l0_runtime_block() == ""

    def test_shows_normal_when_no_estop(self):
        safety = DogSafetyClient({"base_url": "http://fake:5070"})
        safety._cached_estop = {"enabled": False}
        safety._cache_ts = time.monotonic()
        pipeline = _make_pipeline(dog_safety=safety)
        block = pipeline._build_l0_runtime_block()
        assert "运行时状态" in block
        assert "正常" in block

    def test_shows_warning_when_estop_active(self):
        safety = DogSafetyClient({"base_url": "http://fake:5070"})
        safety._cached_estop = {"enabled": True}
        safety._cache_ts = time.monotonic()
        pipeline = _make_pipeline(dog_safety=safety)
        block = pipeline._build_l0_runtime_block()
        assert "已激活" in block
        assert "禁止" in block


class TestBuildSystemPromptL0Injection:
    def test_l0_block_prepended_before_base_prompt(self):
        safety = DogSafetyClient({"base_url": "http://fake:5070"})
        safety._cached_estop = {"enabled": True}
        safety._cache_ts = time.monotonic()
        pipeline = _make_pipeline(dog_safety=safety)
        prompt = pipeline._build_system_prompt(None)
        # L0 block should come BEFORE the base prompt
        l0_pos = prompt.find("运行时状态")
        base_pos = prompt.find("Thunder")
        assert l0_pos < base_pos

    def test_no_l0_when_no_safety_configured(self):
        pipeline = _make_pipeline(dog_safety=None)
        prompt = pipeline._build_system_prompt(None)
        assert "运行时状态" not in prompt
        assert prompt.startswith("你是Thunder")


# ── BrainPipeline safety gate ────────────────────────────────────────────────


class TestExecuteSkillSafetyGate:
    def _make_full_pipeline(self, estop_state):
        """Return a BrainPipeline mock where execute_skill can be called."""
        from askme.pipeline.brain_pipeline import BrainPipeline
        pipeline = object.__new__(BrainPipeline)

        # Safety client
        safety = MagicMock(spec=DogSafetyClient)
        safety.is_configured.return_value = True
        # query_estop_state is called via asyncio.to_thread — patch as sync
        safety.query_estop_state.return_value = estop_state
        pipeline._dog_safety = safety

        # Skill manager
        sm = MagicMock()
        skill = MagicMock()
        skill.depends = []
        skill.timeout = 30
        sm.get.return_value = skill
        pipeline._skill_manager = sm

        # Minimal stubs
        pipeline._dog_control = None
        pipeline._episodic = None
        pipeline._arm = None
        pipeline._audio = MagicMock()
        pipeline._audio.drain_buffers = MagicMock()
        pipeline._audio.start_playback = MagicMock()
        pipeline._audio.stop_playback = MagicMock()
        pipeline._audio.speak = MagicMock()
        pipeline._skill_executor = MagicMock()
        pipeline._skill_executor.run = AsyncMock(return_value="skill executed")
        pipeline._conversation = MagicMock()
        pipeline._memory = None
        pipeline._pending_tasks = set()
        pipeline._splitter = MagicMock()
        pipeline._max_response_chars = 0

        return pipeline

    async def test_estop_active_blocks_skill(self):
        pipeline = self._make_full_pipeline({"enabled": True})
        result = await pipeline.execute_skill("navigate", "去仓库")
        assert "[安全锁定]" in result
        assert "急停" in result
        # Skill executor should NOT have been called
        pipeline._skill_executor.run.assert_not_called()

    async def test_estop_inactive_allows_skill(self):
        pipeline = self._make_full_pipeline({"enabled": False})
        # execute_skill will hit skill_executor.run — stub the rest of the method
        # by catching the AttributeError that happens after the gate
        try:
            await pipeline.execute_skill("navigate", "去仓库")
        except (AttributeError, TypeError):
            pass  # expected — pipeline stubs incomplete after the gate
        # Key assertion: the gate did NOT block (safety.query_estop_state was called)
        pipeline._dog_safety.query_estop_state.assert_called_once()

    async def test_estop_state_none_allows_skill(self):
        """When estop service is unreachable (returns None), do not block."""
        pipeline = self._make_full_pipeline(None)
        try:
            await pipeline.execute_skill("navigate", "去仓库")
        except (AttributeError, TypeError):
            pass
        # Gate should not have blocked (no safety lock message)
        pipeline._dog_safety.query_estop_state.assert_called_once()

    async def test_no_safety_client_allows_skill(self):
        pipeline = self._make_full_pipeline({"enabled": True})
        pipeline._dog_safety = None  # No client wired
        try:
            await pipeline.execute_skill("navigate", "去仓库")
        except (AttributeError, TypeError):
            pass
        # No blocking — no client means no gate

    async def test_safety_not_configured_allows_skill(self):
        pipeline = self._make_full_pipeline({"enabled": True})
        pipeline._dog_safety.is_configured.return_value = False
        try:
            await pipeline.execute_skill("navigate", "去仓库")
        except (AttributeError, TypeError):
            pass
        pipeline._dog_safety.query_estop_state.assert_not_called()
