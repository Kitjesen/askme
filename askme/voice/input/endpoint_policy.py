"""Conservative endpoint decisions for ordinary Chinese ASR partials.

The policy only decides whether an utterance *could* be committed. It never
stops capture itself, keeping shadow evaluation separate from audio control.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from enum import StrEnum


class EndpointMode(StrEnum):
    """How an endpoint recommendation is exposed to the caller."""

    OFF = "off"
    SHADOW = "shadow"
    ACTIVE = "active"


@dataclass(frozen=True)
class EndpointDecision:
    """Observable result of one partial-transcript observation."""

    would_commit: bool
    should_commit: bool
    reason: str
    required_silence_ms: float
    observed_silence_ms: float
    stable_partial_ms: float
    cancelled: bool = False


@dataclass(frozen=True)
class _EndpointClass:
    required_silence_ms: float
    ready_reason: str
    waiting_reason: str


class EndpointPolicy:
    """Track transcript stability and silence without controlling capture."""

    _SHORT_COMMANDS = frozenset(
        {
            "请让一下",
            "让一下",
            "停一下",
            "等一下",
            "过来",
            "回去",
            "开始",
            "停止",
            "取消",
            "确认",
        }
    )
    _TERMINAL_PUNCTUATION = "。！？!?；;"
    _FILLER_ONLY_RE = re.compile(r"^(?:嗯+|呃+|额+|啊+|这个|那个|就是|怎么说)[，,。！？!?、]*$")
    _HESITATION_RE = re.compile(
        r"^(?:嗯+|呃+|额+|这个|那个|就是|怎么说)[，,、]?|"
        r"(?:让我想想|我想想|想一下)$"
    )
    _UNFINISHED_ENDING_RE = re.compile(
        r"(?:但是|然后|因为|所以|如果|要是|虽然|而且|还有|或者|并且|不过|"
        r"可是|和|跟|与|再|先|当|等到|直到|为了|由于|关于|至于|"
        r"想|想要|需要|打算|准备|请|帮我|麻烦|去|到|把|给|对|从|向|往)$"
    )
    _COMPLETE_ENDING_RE = re.compile(
        r"(?:吗|呢|吧|呀|啊|了|好|谢谢|再见|多少|什么|谁|哪里|哪儿|在哪|"
        r"怎么样|为什么|怎么办|几点|位置|时间|天气|[一二三四五六七八九十\d]+楼)$"
    )
    _SUBJECT_PREDICATE_ENDING_RE = re.compile(r"(?:在|是|有).{1,12}$")

    def __init__(
        self,
        *,
        mode: EndpointMode | str = EndpointMode.SHADOW,
        short_command_silence_ms: float = 400.0,
        complete_silence_ms: float = 500.0,
        incomplete_silence_ms: float = 1000.0,
        hesitation_silence_ms: float = 1200.0,
        stable_partial_ms: float = 180.0,
        min_confidence: float = 0.55,
    ) -> None:
        self.mode = EndpointMode(str(mode).lower()) if not isinstance(mode, EndpointMode) else mode
        self.short_command_silence_ms = max(100.0, float(short_command_silence_ms))
        self.complete_silence_ms = max(100.0, float(complete_silence_ms))
        self.incomplete_silence_ms = max(self.complete_silence_ms, float(incomplete_silence_ms))
        self.hesitation_silence_ms = max(
            self.incomplete_silence_ms,
            float(hesitation_silence_ms),
        )
        self.required_stable_partial_ms = max(0.0, float(stable_partial_ms))
        self.min_confidence = min(1.0, max(0.0, float(min_confidence)))
        self.reset()

    def reset(self) -> None:
        self._last_text = ""
        self._stable_started_at: float | None = None
        self._silence_started_at: float | None = None
        self._candidate_pending = False

    def observe(
        self,
        *,
        partial_text: str,
        quiet: bool,
        confidence: float | None = None,
        now: float | None = None,
    ) -> EndpointDecision:
        current = time.monotonic() if now is None else float(now)
        text = self._normalize(partial_text)

        if self.mode is EndpointMode.OFF:
            self.reset()
            return self._decision(reason="mode_off")

        if text != self._last_text:
            self._last_text = text
            self._stable_started_at = current
            self._silence_started_at = None

        if not quiet:
            cancelled = self._candidate_pending
            self._silence_started_at = None
            self._stable_started_at = current
            self._candidate_pending = False
            return self._decision(reason="speech_resumed", cancelled=cancelled)

        if not text:
            self._silence_started_at = None
            self._candidate_pending = False
            return self._decision(reason="empty_partial")

        if self._FILLER_ONLY_RE.fullmatch(text):
            self._silence_started_at = None
            self._candidate_pending = False
            return self._decision(reason="non_lexical_partial")

        if self._is_low_confidence(confidence):
            self._silence_started_at = None
            self._candidate_pending = False
            return self._decision(reason="low_confidence")

        if self._silence_started_at is None:
            self._silence_started_at = current
        self._candidate_pending = True

        observed_silence_ms = (current - self._silence_started_at) * 1000.0
        stable_since = self._stable_started_at if self._stable_started_at is not None else current
        stable_ms = (current - stable_since) * 1000.0
        endpoint_class = self._classify(text)
        silence_ready = observed_silence_ms >= endpoint_class.required_silence_ms
        stable_ready = stable_ms >= self.required_stable_partial_ms
        would_commit = silence_ready and stable_ready
        if would_commit:
            reason = endpoint_class.ready_reason
        elif not stable_ready:
            reason = "partial_not_stable"
        else:
            reason = endpoint_class.waiting_reason
        return EndpointDecision(
            would_commit=would_commit,
            should_commit=would_commit and self.mode is EndpointMode.ACTIVE,
            reason=reason,
            required_silence_ms=endpoint_class.required_silence_ms,
            observed_silence_ms=observed_silence_ms,
            stable_partial_ms=stable_ms,
        )

    def _decision(self, *, reason: str, cancelled: bool = False) -> EndpointDecision:
        return EndpointDecision(
            would_commit=False,
            should_commit=False,
            reason=reason,
            required_silence_ms=0.0,
            observed_silence_ms=0.0,
            stable_partial_ms=0.0,
            cancelled=cancelled,
        )

    def _classify(self, text: str) -> _EndpointClass:
        lexical_text = text.rstrip(self._TERMINAL_PUNCTUATION)
        if lexical_text in self._SHORT_COMMANDS:
            return _EndpointClass(
                self.short_command_silence_ms,
                "short_command_ready",
                "short_command_waiting",
            )
        if self._HESITATION_RE.search(lexical_text):
            return _EndpointClass(
                self.hesitation_silence_ms,
                "hesitation_timeout",
                "hesitation_waiting",
            )
        if text.endswith(tuple(self._TERMINAL_PUNCTUATION)) or self._looks_complete(lexical_text):
            return _EndpointClass(
                self.complete_silence_ms,
                "complete_utterance_ready",
                "complete_utterance_waiting",
            )
        if self._UNFINISHED_ENDING_RE.search(lexical_text):
            return _EndpointClass(
                self.incomplete_silence_ms,
                "unfinished_clause_timeout",
                "unfinished_clause_waiting",
            )
        return _EndpointClass(
            self.incomplete_silence_ms,
            "ordinary_utterance_timeout",
            "ordinary_utterance_waiting",
        )

    @classmethod
    def _looks_complete(cls, text: str) -> bool:
        return bool(
            cls._COMPLETE_ENDING_RE.search(text) or cls._SUBJECT_PREDICATE_ENDING_RE.search(text)
        )

    def _is_low_confidence(self, confidence: float | None) -> bool:
        if confidence is None:
            return False
        value = float(confidence)
        return not math.isfinite(value) or value < self.min_confidence

    @staticmethod
    def _normalize(text: str) -> str:
        return "".join(str(text or "").strip().split())


__all__ = ["EndpointDecision", "EndpointMode", "EndpointPolicy"]
