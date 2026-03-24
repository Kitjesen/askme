"""Tests for StrategyGenerator — LLM-based action suggestions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from askme.memory.strategy import StrategyGenerator, Suggestion


def _make_generator(llm_response=""):
    """Create a StrategyGenerator with a mocked LLM."""
    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=llm_response)
    gen = StrategyGenerator(mock_llm)
    return gen, mock_llm


class TestSuggest:
    @pytest.mark.asyncio
    async def test_parses_formatted_response(self):
        response = (
            "增加仓库A巡检频率|温度异常频率升高，需要密切监控|0.8\n"
            "通知管理员|连续3天温度异常|0.6\n"
            "检查空调系统|可能是制冷设备故障|0.7"
        )
        gen, llm = _make_generator(response)
        suggestions = await gen.suggest(trends="仓库A温度异常频率升高3倍")

        assert len(suggestions) == 3
        assert suggestions[0].action == "增加仓库A巡检频率"
        assert suggestions[0].reason == "温度异常频率升高，需要密切监控"
        assert suggestions[0].confidence == 0.8
        assert suggestions[2].confidence == 0.7

    @pytest.mark.asyncio
    async def test_empty_context_returns_empty(self):
        gen, llm = _make_generator()
        suggestions = await gen.suggest()
        assert suggestions == []
        llm.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self):
        gen, llm = _make_generator()
        llm.chat = AsyncMock(side_effect=RuntimeError("LLM down"))
        suggestions = await gen.suggest(trends="something")
        assert suggestions == []

    @pytest.mark.asyncio
    async def test_caps_at_three_suggestions(self):
        response = "\n".join(
            f"action{i}|reason{i}|0.{i}" for i in range(1, 7)
        )
        gen, _ = _make_generator(response)
        suggestions = await gen.suggest(world_state="busy")
        assert len(suggestions) == 3

    @pytest.mark.asyncio
    async def test_confidence_clamped(self):
        gen, _ = _make_generator("action|reason|1.5\naction2|reason2|-0.3")
        suggestions = await gen.suggest(trends="test")
        assert suggestions[0].confidence == 1.0
        assert suggestions[1].confidence == 0.0

    @pytest.mark.asyncio
    async def test_invalid_confidence_defaults(self):
        gen, _ = _make_generator("action|reason|not_a_number")
        suggestions = await gen.suggest(trends="test")
        assert len(suggestions) == 1
        assert suggestions[0].confidence == 0.5

    @pytest.mark.asyncio
    async def test_skips_malformed_lines(self):
        response = "good action|good reason|0.9\nno pipe here\n||\nanother|reason|0.5"
        gen, _ = _make_generator(response)
        suggestions = await gen.suggest(associations="test")
        assert len(suggestions) == 2
        assert suggestions[0].action == "good action"
        assert suggestions[1].action == "another"


class TestParse:
    def test_empty_string(self):
        assert StrategyGenerator._parse("") == []

    def test_no_pipes(self):
        assert StrategyGenerator._parse("just plain text") == []

    def test_too_few_parts(self):
        assert StrategyGenerator._parse("action|reason") == []

    def test_whitespace_handling(self):
        result = StrategyGenerator._parse("  action  |  reason  |  0.7  ")
        assert len(result) == 1
        assert result[0].action == "action"
        assert result[0].reason == "reason"
        assert result[0].confidence == 0.7
