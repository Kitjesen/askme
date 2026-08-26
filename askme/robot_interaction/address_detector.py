"""Detect whether speech is addressed to the robot or is casual bystander chat.

Pure rule-based, 0ms latency. No LLM dependency.
The uncertainty policy is configurable so public deployments can prefer silence
while supervised deployments preserve the legacy respond-on-uncertain behavior.

Supports "name activation" — when the robot's name is detected, a 30-second
window opens where ALL subsequent speech is treated as addressed. This gives
operators a natural "呼名+对话" interaction pattern without needing explicit
wake words for every sentence.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Default robot names / identifiers — deployments can replace these via config.
_DEFAULT_ROBOT_NAMES = frozenset([
    "thunder", "雷霆", "机器人", "小雷", "机器狗", "巡检",
])

# Direct address pronouns — "你" directed at the robot
_ADDRESS_PRONOUNS = frozenset(["你", "您"])

# Single-char commands that ARE robot commands (only exact match)
_SINGLE_CHAR_COMMANDS = frozenset(["停", "站", "坐", "起", "退"])

# Casual chat signals — if present, probably not for us
_CASUAL_SIGNALS = frozenset([
    "吃饭", "吃了吗", "吃什么", "下班", "走走", "走吧", "哈哈", "嘿嘿", "呵呵",
    "电影", "游戏", "好玩", "好吃", "无聊", "累死", "中午", "晚上",
    "老婆", "老公", "孩子", "回家", "周末", "放假", "休息",
    "微信", "手机", "抖音", "快手", "外卖", "咖啡", "奶茶",
    "早上好", "晚安", "拜拜", "再见", "明天见",
])

class AddressDetector:
    """Determine if user speech is addressed to the robot.

    Returns True (addressed) or False (bystander chat).
    Default-safe: returns True when uncertain.

    Config keys (under ``voice.address_detection``)::

        enabled: bool  - Enable/disable (default False)
        names: list[str]  - Robot names that activate direct-address mode
        uncertain_policy: "addressed" | "ignore"
        allow_pronoun_address: bool
        name_window_allows_ambiguous: bool
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.enabled: bool = bool(cfg.get("enabled", False))
        configured_names = cfg.get("names")
        if configured_names is None:
            self._robot_names = _DEFAULT_ROBOT_NAMES
        else:
            if isinstance(configured_names, str):
                configured_names = [configured_names]
            self._robot_names = frozenset(
                str(name).strip().lower()
                for name in configured_names
                if str(name).strip()
            )
        configured_aliases = cfg.get("aliases", {})
        if isinstance(configured_aliases, dict):
            self._name_aliases = tuple(
                sorted(
                    (
                        (str(alias).strip(), str(canonical).strip())
                        for alias, canonical in configured_aliases.items()
                        if str(alias).strip() and str(canonical).strip()
                    ),
                    key=lambda item: len(item[0]),
                    reverse=True,
                )
            )
        else:
            self._name_aliases = ()
        # Name activation window: after hearing robot name, treat all speech
        # as addressed for this many seconds (natural "呼名+对话" pattern)
        self._name_window: float = float(cfg.get("name_window", 30.0))
        self._name_activated_until: float = 0.0  # monotonic deadline
        policy = str(cfg.get("uncertain_policy", "addressed")).strip().lower()
        self._uncertain_is_addressed = policy not in {"ignore", "silent", "reject"}
        self._allow_pronoun_address = bool(cfg.get("allow_pronoun_address", True))
        self._name_window_allows_ambiguous = bool(
            cfg.get("name_window_allows_ambiguous", True)
        )

    def normalize_text(self, text: str) -> str:
        """Replace ASR variants of the robot name with its canonical name."""
        normalized = text
        for alias, canonical in self._name_aliases:
            normalized = normalized.replace(alias, canonical)
        return normalized

    def is_addressed(self, text: str) -> bool:
        """Check if *text* is addressed to the robot.

        Returns True if addressed (should respond), False if bystander chat.
        When disabled or uncertain, returns True (safe default).
        """
        if not self.enabled:
            return True

        text_lower = self.normalize_text(text).lower().strip()
        if not text_lower:
            return False

        now = time.monotonic()
        name_window_active = now < self._name_activated_until

        # Legacy deployments can let a recent explicit name authorize ambiguous
        # follow-ups. Public deployments disable this and rely on the turn gate.
        if name_window_active and self._name_window_allows_ambiguous:
            logger.debug("[Address] YES: name window active (%.0fs left)",
                         self._name_activated_until - now)
            return True

        # Rule 1: Robot name mentioned → definitely addressed + activate window
        for name in self._robot_names:
            if name in text_lower:
                if self._name_window > 0:
                    self._name_activated_until = now + self._name_window
                logger.info("[Address] YES: robot name '%s' → window open %.0fs",
                            name, self._name_window)
                return True

        # Rule 2: Direct address pronoun "你/您" → likely addressed
        for p in _ADDRESS_PRONOUNS:
            if p in text_lower and (
                self._allow_pronoun_address or name_window_active
            ):
                logger.debug("[Address] YES: pronoun '%s' found", p)
                return True

        # Rule 3: Single-char exact command (停/站/坐) → addressed
        if len(text_lower) <= 2 and text_lower in _SINGLE_CHAR_COMMANDS:
            logger.debug("[Address] YES: single-char command '%s'", text_lower)
            return True

        # Command and question shape alone are not evidence of direct address.
        # Bystander speech frequently contains phrases such as "帮我查一下" or
        # "现在几点"; public-mode admission must instead rely on explicit name,
        # pronoun, wake authorization, or the narrow local-safety commands above.

        # Rule 4: Casual chat signals → likely not for us
        casual_count = sum(1 for s in _CASUAL_SIGNALS if s in text_lower)
        if casual_count > 0:
            logger.info("[Address] NO: casual signals (%d) found in '%s'", casual_count, text[:30])
            return False

        # Rule 5: Very short text (<=4 chars) with no signals is ambiguous.
        if len(text_lower) <= 4:
            logger.debug(
                "[Address] %s: short text, uncertainty policy",
                "YES" if self._uncertain_is_addressed else "NO",
            )
            return self._uncertain_is_addressed

        logger.debug(
            "[Address] %s: default uncertainty policy",
            "YES" if self._uncertain_is_addressed else "NO",
        )
        return self._uncertain_is_addressed
