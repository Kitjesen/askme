"""Retry a TTS synthesize call the way the official Xiaozhi server does.

Up to five attempts. An exception or an empty result consumes one attempt and
is logged with the original display text. Programmer errors (``TypeError`` and
``ValueError``) are not retried. After success, the log records how many
retries were used. After the last failure, an error is logged and a
:class:`TTSRetryResult` is returned. This helper does not sleep and does not
raise for an exhausted TTS failure.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

DEFAULT_TTS_MAX_ATTEMPTS = 5

T = TypeVar("T")


class TTSSynthesisFailure(Exception):
    """All TTS attempts failed or returned empty audio."""

    def __init__(
        self,
        original_text: str,
        attempts: int,
        last_error: BaseException | None = None,
    ) -> None:
        self.original_text = original_text
        self.attempts = attempts
        self.last_error = last_error
        detail = "空音频" if last_error is None else str(last_error)
        super().__init__(
            f"语音生成失败: {original_text}，请检查网络或服务是否正常"
            f"（已尝试 {attempts} 次，最后错误: {detail}）"
        )


@dataclass(frozen=True, slots=True)
class TTSRetryResult:
    """Outcome of :func:`synthesize_with_retry`.

    ``attempts`` is the number of calls made. ``retries`` is how many of those
    calls failed before a success, matching the official success log. A total
    failure reports ``retries == attempts``.
    """

    ok: bool
    attempts: int
    value: Any = None
    error: BaseException | None = None

    @property
    def retries(self) -> int:
        if self.ok:
            return max(0, self.attempts - 1)
        return self.attempts


def synthesize_with_retry(
    call: Callable[[], T],
    *,
    original_text: str,
    max_attempts: int = DEFAULT_TTS_MAX_ATTEMPTS,
    is_empty: Callable[[T], bool] | None = None,
    log: logging.Logger | None = None,
) -> TTSRetryResult:
    """Call ``call`` until it returns audio, or until ``max_attempts`` is used.

    ``is_empty`` decides which successful returns count as "no audio". The
    default treats ``None``, ``False``, and empty bytes/str/list/dict as empty.
    ``True`` and any other non-empty value are success.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    empty = _is_empty_result if is_empty is None else is_empty
    sink = logger if log is None else log
    last_error: BaseException | None = None
    last_value: T | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            value = call()
        except (TypeError, ValueError):
            raise
        except Exception as exc:
            last_error = exc
            last_value = None
            sink.warning(
                "语音生成失败%d次: %s，错误: %s",
                attempt,
                original_text,
                exc,
            )
        else:
            if empty(value):
                last_error = None
                last_value = value
                sink.warning(
                    "语音生成失败%d次: %s，错误: %s",
                    attempt,
                    original_text,
                    "空音频",
                )
            else:
                sink.info(
                    "语音生成成功: %s，重试%d次",
                    original_text,
                    attempt - 1,
                )
                return TTSRetryResult(ok=True, attempts=attempt, value=value, error=None)
    sink.error("语音生成失败: %s，请检查网络或服务是否正常", original_text)
    failure = TTSSynthesisFailure(original_text, max_attempts, last_error)
    return TTSRetryResult(
        ok=False,
        attempts=max_attempts,
        value=last_value,
        error=failure,
    )


def _is_empty_result(value: Any) -> bool:
    if value is None or value is False:
        return True
    if isinstance(value, (bytes, bytearray, str, list, tuple, dict)):
        return len(value) == 0
    return False
