"""Detect whether speech is addressed to the robot or is casual bystander chat.

Pure rule-based, 0ms latency. No LLM dependency.
Falls back to "addressed" (safe default — better to respond than to miss a command).

Supports "name activation" — when the robot's name is detected, a 30-second
window opens where ALL subsequent speech is treated as addressed. This gives
operators a natural "呼名+对话" interaction pattern without needing explicit
wake words for every sentence.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# Robot names / identifiers — if present, definitely talking to us
_ROBOT_NAMES = frozenset([
    "thunder", "雷霆", "机器人", "小雷", "机器狗", "巡检",
])

# Direct address pronouns — "你" directed at the robot
_ADDRESS_PRONOUNS = frozenset(["你", "您"])

# Command verbs — if present, likely a task command for the robot
_COMMAND_VERBS = frozenset([
    "检查", "巡检", "导航", "前往", "带我去", "报告", "汇报",
    "停下", "站起", "坐下", "起来",
    "抓取", "扫描", "拍照", "记录", "测量",
    "查一下", "看一下", "帮我", "告诉我", "回答",
    "几点", "什么时间", "温度", "湿度", "状态",
    "音量", "语速", "静音", "够了", "别说",
    "停止", "取消", "继续", "确认", "执行",
    "导航到", "走到", "去仓库", "去厂房",
])

# Single-char commands that ARE robot commands (only exact match)
_SINGLE_CHAR_COMMANDS = frozenset(["停", "站", "坐", "起", "退"])

# Casual chat signals — if these dominate and no command verbs, probably not for us
_CASUAL_SIGNALS = frozenset([
    "吃饭", "吃了吗", "吃什么", "下班", "走走", "走吧", "哈哈", "嘿嘿", "呵呵",
    "电影", "游戏", "好玩", "好吃", "无聊", "累死", "中午", "晚上",
    "老婆", "老公", "孩子", "回家", "周末", "放假", "休息",
    "微信", "手机", "抖音", "快手", "外卖", "咖啡", "奶茶",
    "早上好", "晚安", "拜拜", "再见", "明天见",
])

# Question patterns that imply addressing someone nearby (could be robot)
_QUESTION_TO_ENTITY = re.compile(r"(几点|多少度|什么温度|什么时间|什么状态|怎么样了)")


class AddressDetector:
    """Determine if user speech is addressed to the robot.

    Returns True (addressed) or False (bystander chat).
    Default-safe: returns True when uncertain.

    Config keys (under ``voice.address_detection``)::

        enabled: bool  - Enable/disable (default False)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        # Name activation window: after hearing robot name, treat all speech
        # as addressed for this many seconds (natural "呼名+对话" pattern)
        self._name_window: float = float(cfg.get("name_window", 30.0))
        self._name_activated_until: float = 0.0  # monotonic deadline

    def is_addressed(self, text: str) -> bool:
        """Check if *text* is addressed to the robot.

        Returns True if addressed (should respond), False if bystander chat.
        When disabled or uncertain, returns True (safe default).
        """
        if not self.enabled:
            return True

        text_lower = text.lower().strip()
        if not text_lower:
            return False

        # Rule 0: Name activation window — if robot name was recently heard,
        # treat everything as addressed (natural "Thunder, 检查温度" then
        # follow-up "再看看那边" without repeating the name)
        if time.monotonic() < self._name_activated_until:
            logger.debug("[Address] YES: name window active (%.0fs left)",
                         self._name_activated_until - time.monotonic())
            return True

        # Rule 1: Robot name mentioned → definitely addressed + activate window
        for name in _ROBOT_NAMES:
            if name in text_lower:
                self._name_activated_until = time.monotonic() + self._name_window
                logger.info("[Address] YES: robot name '%s' → window open %.0fs",
                            name, self._name_window)
                return True

        # Rule 2: Direct address pronoun "你/您" → likely addressed
        for p in _ADDRESS_PRONOUNS:
            if p in text_lower:
                logger.debug("[Address] YES: pronoun '%s' found", p)
                return True

        # Rule 3: Single-char exact command (停/站/坐) → addressed
        if len(text_lower) <= 2 and text_lower in _SINGLE_CHAR_COMMANDS:
            logger.debug("[Address] YES: single-char command '%s'", text_lower)
            return True

        # Rule 4: Command verb present → likely a task command
        for verb in _COMMAND_VERBS:
            if verb in text_lower:
                logger.debug("[Address] YES: command verb '%s' found", verb)
                return True

        # Rule 5: Question pattern implying addressing an entity
        if _QUESTION_TO_ENTITY.search(text_lower):
            logger.debug("[Address] YES: question-to-entity pattern found")
            return True

        # Rule 6: Casual chat signals → likely not for us
        casual_count = sum(1 for s in _CASUAL_SIGNALS if s in text_lower)
        if casual_count > 0:
            logger.info("[Address] NO: casual signals (%d) found in '%s'", casual_count, text[:30])
            return False

        # Rule 7: Very short text (<=4 chars) with no signals → ambiguous, default addressed
        if len(text_lower) <= 4:
            logger.debug("[Address] YES: short text, default addressed")
            return True

        # Rule 8: Longer text with no command signals → likely casual
        if len(text_lower) > 6:
            logger.info("[Address] NO: long text with no command signals: '%s'", text[:30])
            return False

        # Default: addressed (safe — better to respond than miss a command)
        logger.debug("[Address] YES: default (uncertain)")
        return True
