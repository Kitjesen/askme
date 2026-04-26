"""Tests for AddressDetector — rule-based speech address classification."""

from __future__ import annotations

import time

from askme.voice.address_detector import AddressDetector


def _make_detector(enabled: bool = True, name_window: float = 30.0) -> AddressDetector:
    return AddressDetector(config={"enabled": enabled, "name_window": name_window})


class TestDisabled:
    def test_returns_true_when_disabled(self):
        det = _make_detector(enabled=False)
        assert det.is_addressed("吃饭了吗") is True

    def test_returns_true_for_casual_when_disabled(self):
        det = _make_detector(enabled=False)
        assert det.is_addressed("哈哈哈好玩") is True


class TestEmptyText:
    def test_empty_string_returns_false_when_enabled(self):
        det = _make_detector()
        assert det.is_addressed("") is False

    def test_whitespace_only_returns_false(self):
        det = _make_detector()
        assert det.is_addressed("   ") is False


class TestRobotName:
    def test_exact_name_thunder(self):
        det = _make_detector()
        assert det.is_addressed("thunder 你好") is True

    def test_chinese_name_雷霆(self):
        det = _make_detector()
        assert det.is_addressed("雷霆过来") is True

    def test_chinese_name_机器人(self):
        det = _make_detector()
        assert det.is_addressed("机器人检查一下") is True

    def test_name_activates_window(self):
        det = _make_detector(name_window=30.0)
        det.is_addressed("雷霆你好")
        # After name activation, next utterance should be addressed
        assert det.is_addressed("去仓库那边看看") is True  # would be addressed anyway by verb
        # Check window is actually set
        assert det._name_activated_until > time.monotonic()

    def test_name_window_expires(self):
        det = _make_detector(name_window=0.001)
        det.is_addressed("雷霆你好")
        time.sleep(0.01)  # let the tiny window expire
        # After expiry, casual chat should return False
        assert det.is_addressed("哈哈好吃") is False


class TestAddressPronoun:
    def test_你_addressed(self):
        det = _make_detector()
        assert det.is_addressed("你去哪里") is True

    def test_您_addressed(self):
        det = _make_detector()
        assert det.is_addressed("您好请问") is True


class TestSingleCharCommands:
    def test_停_addressed(self):
        det = _make_detector()
        assert det.is_addressed("停") is True

    def test_站_addressed(self):
        det = _make_detector()
        assert det.is_addressed("站") is True

    def test_two_char_坐下_via_command_verb(self):
        # "坐下" matches _COMMAND_VERBS, not _SINGLE_CHAR_COMMANDS
        det = _make_detector()
        assert det.is_addressed("坐下") is True


class TestCommandVerbs:
    def test_检查_addressed(self):
        det = _make_detector()
        assert det.is_addressed("检查一下温度") is True

    def test_导航_addressed(self):
        det = _make_detector()
        assert det.is_addressed("导航到3号仓库") is True

    def test_帮我_addressed(self):
        det = _make_detector()
        assert det.is_addressed("帮我拿一下") is True

    def test_巡检_addressed(self):
        det = _make_detector()
        assert det.is_addressed("开始巡检") is True


class TestQuestionPatterns:
    def test_几点_addressed(self):
        det = _make_detector()
        assert det.is_addressed("现在几点了") is True

    def test_温度_question_addressed(self):
        det = _make_detector()
        assert det.is_addressed("现在什么温度") is True

    def test_状态_addressed(self):
        det = _make_detector()
        assert det.is_addressed("什么状态") is True


class TestCasualChat:
    def test_吃饭_casual_not_addressed(self):
        det = _make_detector()
        assert det.is_addressed("去吃饭吗") is False

    def test_下班_casual(self):
        det = _make_detector()
        assert det.is_addressed("下班了走走") is False

    def test_哈哈_casual(self):
        det = _make_detector()
        assert det.is_addressed("哈哈太好玩了") is False

    def test_回家_casual(self):
        det = _make_detector()
        assert det.is_addressed("晚上回家吗") is False

    def test_外卖_casual(self):
        det = _make_detector()
        assert det.is_addressed("点外卖吧") is False


class TestShortText:
    def test_short_ambiguous_text_defaults_addressed(self):
        det = _make_detector()
        # 3 chars, no casual signals, no command verbs → default addressed (short-text rule)
        assert det.is_addressed("嗯好吧") is True

    def test_short_casual_returns_false_via_casual_signal(self):
        det = _make_detector()
        # "走吧" is a casual signal
        assert det.is_addressed("走吧") is False


class TestDefaultFallback:
    def test_unknown_long_text_defaults_addressed(self):
        det = _make_detector()
        # No robot name, pronoun, command, question, or casual signal
        # Long enough that short-text rule doesn't apply
        result = det.is_addressed("这个东西放在那边位置可以吗")
        assert result is True  # default safe

    def test_returns_bool(self):
        det = _make_detector()
        result = det.is_addressed("hello")
        assert isinstance(result, bool)
