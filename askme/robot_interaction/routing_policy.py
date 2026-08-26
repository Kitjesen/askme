"""Configurable policy values for first-hop interaction routing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

DEFAULT_BUILTIN_COMMANDS: frozenset[str] = frozenset({
    "/quit",
    "/exit",
    "exit",
    "quit",
    "/clear",
    "/history",
    "/help",
    "/skills",
})

DEFAULT_ESTOP_KEYWORDS: frozenset[str] = frozenset({
    "紧急停止",
    "急停",
    "emergency stop",
    "estop",
    "e-stop",
})

DEFAULT_QUICK_REPLIES: Mapping[str, str] = MappingProxyType({
    "你好": "你好，有什么需要帮忙的？",
    "谢谢": "不客气。",
    "谢谢你": "不客气，随时叫我。",
    "再见": "再见，有事随时叫我。",
    "拜拜": "拜拜。",
    "在吗": "在的，有什么事？",
    "你在吗": "在的，说吧。",
    "嗯": "嗯，我在听。",
    "好的": "好的。",
    "\u60a8\u597d": "\u60a8\u597d\uff0c\u6709\u4ec0\u4e48\u53ef\u4ee5\u5e2e\u60a8\uff1f",
    "\u55e8": "\u55ef\uff0c\u6211\u5728\u542c\u3002",
    "\u4f60\u662f\u8c01": (
        "\u6211\u662f\u5c0f\u7b97\uff0c\u4e00\u53ea\u667a\u80fd\u670d\u52a1\u673a\u5668\u72d7\uff0c"
        "\u53ef\u4ee5\u8fdb\u884c\u5bf9\u8bdd\u3001\u5de1\u68c0\u3001\u5bfc\u822a\u548c\u73b0\u573a\u4fe1\u606f\u67e5\u8be2\u3002"
    ),
    "\u4ecb\u7ecd\u4e00\u4e0b\u81ea\u5df1": (
        "\u6211\u662f\u5c0f\u7b97\uff0c\u4e00\u53ea\u667a\u80fd\u670d\u52a1\u673a\u5668\u72d7\uff0c"
        "\u53ef\u4ee5\u8fdb\u884c\u5bf9\u8bdd\u3001\u5de1\u68c0\u3001\u5bfc\u822a\u548c\u73b0\u573a\u4fe1\u606f\u67e5\u8be2\u3002"
    ),
    "\u81ea\u6211\u4ecb\u7ecd\u4e00\u4e0b": (
        "\u6211\u662f\u5c0f\u7b97\uff0c\u4e00\u53ea\u667a\u80fd\u670d\u52a1\u673a\u5668\u72d7\uff0c"
        "\u53ef\u4ee5\u8fdb\u884c\u5bf9\u8bdd\u3001\u5de1\u68c0\u3001\u5bfc\u822a\u548c\u73b0\u573a\u4fe1\u606f\u67e5\u8be2\u3002"
    ),
})

DEFAULT_NEGATION_PREFIXES: tuple[str, ...] = (
    "不要",
    "不用",
    "不能",
    "不想",
    "不让",
    "没有",
    "别再",
    "不",
    "别",
    "没",
)

DEFAULT_QUESTION_SUFFIXES: tuple[str, ...] = ("吗", "么", "呢", "嘛")

DEFAULT_QUESTION_SAFE_SKILLS: frozenset[str] = frozenset({
    "get_time",
    "nav_query",
    "system_status",
    "recall_memory",
    "list_skills",
    "workspace_info",
    "list_directory",
    "patrol_report",
    "daily_summary",
    "lookup_place",
    "answer_wayfinding",
    "recommend_route",
    "offer_wayfinding_help",
})


def _normalize_ascii(text: str) -> str:
    return str(text).strip().lower()


@dataclass(frozen=True)
class RoutingPolicy:
    """Static knobs that control deterministic interaction routing.

    Keeping these values out of ``IntentRouter`` makes the routing engine easier
    to test and lets product/runtime code swap policy without changing callers.
    """

    builtin_commands: Iterable[str] = DEFAULT_BUILTIN_COMMANDS
    estop_keywords: Iterable[str] = DEFAULT_ESTOP_KEYWORDS
    quick_replies: Mapping[str, str] = field(default_factory=lambda: DEFAULT_QUICK_REPLIES)
    min_trigger_length: int = 2
    negation_prefixes: Iterable[str] = DEFAULT_NEGATION_PREFIXES
    question_suffixes: Iterable[str] = DEFAULT_QUESTION_SUFFIXES
    question_safe_skills: Iterable[str] = DEFAULT_QUESTION_SAFE_SKILLS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "builtin_commands",
            frozenset(_normalize_ascii(command) for command in self.builtin_commands),
        )
        object.__setattr__(
            self,
            "estop_keywords",
            frozenset(_normalize_ascii(keyword) for keyword in self.estop_keywords),
        )
        object.__setattr__(
            self,
            "quick_replies",
            MappingProxyType(dict(self.quick_replies)),
        )
        object.__setattr__(
            self,
            "negation_prefixes",
            tuple(sorted(self.negation_prefixes, key=len, reverse=True)),
        )
        object.__setattr__(self, "question_suffixes", tuple(self.question_suffixes))
        object.__setattr__(
            self,
            "question_safe_skills",
            frozenset(self.question_safe_skills),
        )


DEFAULT_ROUTING_POLICY = RoutingPolicy()
