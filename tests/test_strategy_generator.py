"""Tests for StrategyGenerator — _parse method and suggest with mocked LLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from askme.memory.strategy import StrategyGenerator, Suggestion

# ── Suggestion dataclass ──────────────────────────────────────────────────────

class TestSuggestion:
    def test_fields(self):
        s = Suggestion(action="巡逻A区", reason="异常较多", confidence=0.8)
        assert s.action == "巡逻A区"
        assert s.reason == "异常较多"
        assert s.confidence == 0.8


# ── StrategyGenerator._parse ──────────────────────────────────────────────────

class TestParse:
    def test_single_valid_line(self):
        result = StrategyGenerator._parse("巡逻A区|异常较多|0.8")
        assert len(result) == 1
        assert result[0].action == "巡逻A区"
        assert result[0].reason == "异常较多"
        assert result[0].confidence == 0.8

    def test_multiple_lines(self):
        text = "巡逻A区|异常|0.9\n检查设备|传感器故障|0.7"
        result = StrategyGenerator._parse(text)
        assert len(result) == 2

    def test_caps_at_3(self):
        text = "\n".join([f"动作{i}|原因{i}|0.5" for i in range(10)])
        result = StrategyGenerator._parse(text)
        assert len(result) == 3

    def test_invalid_line_skipped(self):
        text = "no pipe here\n巡逻|原因|0.5"
        result = StrategyGenerator._parse(text)
        assert len(result) == 1

    def test_empty_action_skipped(self):
        text = "|some reason|0.5"
        result = StrategyGenerator._parse(text)
        assert len(result) == 0

    def test_confidence_clamped_to_01(self):
        text = "行动|原因|2.5"
        result = StrategyGenerator._parse(text)
        assert result[0].confidence <= 1.0

    def test_negative_confidence_clamped(self):
        text = "行动|原因|-0.5"
        result = StrategyGenerator._parse(text)
        assert result[0].confidence >= 0.0

    def test_invalid_confidence_defaults_to_05(self):
        text = "行动|原因|not_a_float"
        result = StrategyGenerator._parse(text)
        assert result[0].confidence == 0.5

    def test_empty_text_returns_empty(self):
        assert StrategyGenerator._parse("") == []

    def test_too_few_parts_skipped(self):
        text = "action|reason_only"
        result = StrategyGenerator._parse(text)
        assert len(result) == 0

    def test_whitespace_stripped(self):
        text = "  巡逻  |  原因  |  0.7  "
        result = StrategyGenerator._parse(text)
        assert result[0].action == "巡逻"
        assert result[0].reason == "原因"
        assert abs(result[0].confidence - 0.7) < 0.01


# ── StrategyGenerator.suggest ─────────────────────────────────────────────────

class TestSuggest:
    def _make_gen(self, llm_response: str) -> StrategyGenerator:
        llm = MagicMock()
        llm.chat = AsyncMock(return_value=llm_response)
        return StrategyGenerator(llm)

    async def test_no_context_returns_empty(self):
        gen = self._make_gen("anything")
        result = await gen.suggest()
        assert result == []

    async def test_with_trends_calls_llm(self):
        gen = self._make_gen("巡逻A|异常多|0.8")
        result = await gen.suggest(trends="温度升高")
        assert len(result) == 1
        gen._llm.chat.assert_called_once()

    async def test_llm_failure_returns_empty(self):
        llm = MagicMock()
        llm.chat = AsyncMock(side_effect=RuntimeError("LLM error"))
        gen = StrategyGenerator(llm)
        result = await gen.suggest(trends="some trend")
        assert result == []

    async def test_all_context_fields_included_in_prompt(self):
        gen = self._make_gen("行动|原因|0.5")
        await gen.suggest(
            trends="trend info",
            associations="assoc info",
            world_state="world info",
            procedures="proc info",
        )
        prompt_arg = gen._llm.chat.call_args[0][0]
        all_text = " ".join(m["content"] for m in prompt_arg)
        assert "trend info" in all_text
        assert "assoc info" in all_text
        assert "world info" in all_text
        assert "proc info" in all_text

    async def test_returns_parsed_suggestions(self):
        gen = self._make_gen("巡逻|传感器|0.9\n检查|异常|0.7")
        result = await gen.suggest(world_state="异常状态")
        assert len(result) == 2
        assert result[0].action == "巡逻"
