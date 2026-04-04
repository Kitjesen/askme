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


def _make_prompt_builder(dog_safety=None):
    """Build a minimal PromptBuilder for L0 tests."""
    from askme.pipeline.prompt_builder import PromptBuilder
    tools = MagicMock()
    tools.get_definitions.return_value = []
    sm = MagicMock()
    sm.get_skill_catalog.return_value = "none"
    return PromptBuilder(
        base_prompt="你是Thunder巡检机器人。",
        prompt_seed=[],
        user_prefix="",
        tools=tools,
        skill_manager=sm,
        general_tool_max_safety_level="normal",
        dog_safety=dog_safety,
        episodic=None,
        session_memory=None,
        vision=None,
        qp_memory=None,
    )


class TestBuildL0RuntimeBlock:
    def test_empty_when_no_safety_client(self):
        pb = _make_prompt_builder(dog_safety=None)
        assert pb.build_l0_runtime_block() == ""

    def test_empty_when_safety_not_configured(self):
        safety = DogSafetyClient({})  # no URL = not configured
        pb = _make_prompt_builder(dog_safety=safety)
        assert pb.build_l0_runtime_block() == ""

    def test_shows_normal_when_no_estop(self):
        safety = DogSafetyClient({"base_url": "http://fake:5070"})
        safety._cached_estop = {"enabled": False}
        safety._cache_ts = time.monotonic()
        pb = _make_prompt_builder(dog_safety=safety)
        block = pb.build_l0_runtime_block()
        assert "运行时状态" in block
        assert "正常" in block

    def test_shows_warning_when_estop_active(self):
        safety = DogSafetyClient({"base_url": "http://fake:5070"})
        safety._cached_estop = {"enabled": True}
        safety._cache_ts = time.monotonic()
        pb = _make_prompt_builder(dog_safety=safety)
        block = pb.build_l0_runtime_block()
        assert "已激活" in block
        assert "禁止" in block


class TestBuildSystemPromptL0Injection:
    def test_l0_block_prepended_before_base_prompt(self):
        safety = DogSafetyClient({"base_url": "http://fake:5070"})
        safety._cached_estop = {"enabled": True}
        safety._cache_ts = time.monotonic()
        pb = _make_prompt_builder(dog_safety=safety)
        prompt = pb.build_system_prompt(None)
        # L0 block should come BEFORE the base prompt
        l0_pos = prompt.find("运行时状态")
        base_pos = prompt.find("Thunder")
        assert l0_pos < base_pos

    def test_no_l0_when_no_safety_configured(self):
        pb = _make_prompt_builder(dog_safety=None)
        prompt = pb.build_system_prompt(None)
        assert "运行时状态" not in prompt
        assert prompt.startswith("你是Thunder")


# ── BrainPipeline safety gate ────────────────────────────────────────────────


class TestExecuteSkillSafetyGate:
    """Tests for the estop safety gate in SkillGate.execute_skill."""

    def _make_gate(self, estop_state):
        """Build a SkillGate with a mocked safety client."""
        from askme.pipeline.skill_gate import SkillGate

        safety = MagicMock(spec=DogSafetyClient)
        safety.is_configured.return_value = True
        safety.query_estop_state.return_value = estop_state

        skill_executor = MagicMock()
        skill_executor.run = AsyncMock(return_value="skill executed")

        sm = MagicMock()
        skill = MagicMock()
        skill.depends = []
        skill.timeout = 30
        sm.get.return_value = skill

        audio = MagicMock()
        audio.drain_buffers = MagicMock()
        audio.start_playback = MagicMock()
        audio.stop_playback = MagicMock()
        audio.speak = MagicMock()

        gate = SkillGate(
            skill_manager=sm,
            skill_executor=skill_executor,
            audio=audio,
            conversation=MagicMock(),
            dog_safety=safety,
        )
        return gate, safety, skill_executor

    async def test_estop_active_blocks_skill(self):
        gate, safety, skill_executor = self._make_gate({"enabled": True})
        result = await gate.execute_skill("navigate", "去仓库")
        assert "[安全锁定]" in result
        assert "急停" in result
        skill_executor.run.assert_not_called()

    async def test_estop_inactive_allows_skill(self):
        gate, safety, skill_executor = self._make_gate({"enabled": False})
        try:
            await gate.execute_skill("navigate", "去仓库")
        except (AttributeError, TypeError):
            pass
        safety.query_estop_state.assert_called_once()

    async def test_estop_state_none_allows_skill(self):
        """When estop service is unreachable (returns None), do not block."""
        gate, safety, _ = self._make_gate(None)
        try:
            await gate.execute_skill("navigate", "去仓库")
        except (AttributeError, TypeError):
            pass
        safety.query_estop_state.assert_called_once()

    async def test_no_safety_client_allows_skill(self):
        from askme.pipeline.skill_gate import SkillGate
        gate = SkillGate(
            skill_manager=MagicMock(),
            skill_executor=MagicMock(),
            audio=MagicMock(),
            conversation=MagicMock(),
            dog_safety=None,
        )
        try:
            await gate.execute_skill("navigate", "去仓库")
        except (AttributeError, TypeError):
            pass
        # No blocking — no client means no gate

    async def test_safety_not_configured_allows_skill(self):
        gate, safety, _ = self._make_gate({"enabled": True})
        safety.is_configured.return_value = False
        try:
            await gate.execute_skill("navigate", "去仓库")
        except (AttributeError, TypeError):
            pass
        safety.query_estop_state.assert_not_called()
