"""LLM stream processor — think filter, sentence splitting, TTS piping, tool accumulation."""

from __future__ import annotations

import asyncio
import logging
import math
import time as _time
from contextvars import ContextVar
from dataclasses import replace
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from askme.pipeline.core.protocols import CancellationToken
from askme.pipeline.core.trace import get_tracer

if TYPE_CHECKING:
    from askme.llm.core.client import LLMClient
    from askme.llm.core.contracts import LLMCallContext
    from askme.pipeline.core.tool_executor import ToolExecutor
    from askme.ports import AudioFrontendPort
    from askme.tools.core.tool_registry import ToolRegistry
    from askme.voice.core.stream_splitter import StreamSplitter

logger = logging.getLogger(__name__)


class _ThinkFilter:
    """Strip ``<think>...</think>`` blocks from incremental streaming text.

    MiniMax-M2.5 (and other reasoning models) emit a ``<think>`` block before
    the actual answer.  This filter removes it in O(n) without buffering the
    entire response — safe for real-time TTS piping.
    """

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def feed(self, text: str) -> str:
        self._buf += text
        out: list[str] = []
        while True:
            if self._in_think:
                idx = self._buf.find("</think>")
                if idx < 0:
                    if len(self._buf) > 8:
                        self._buf = self._buf[-8:]
                    return "".join(out)
                self._buf = self._buf[idx + 8 :]
                self._in_think = False
            else:
                idx = self._buf.find("<think>")
                if idx < 0:
                    safe = max(0, len(self._buf) - 7)
                    out.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                    return "".join(out)
                out.append(self._buf[:idx])
                self._buf = self._buf[idx + 7 :]
                self._in_think = True

    def flush(self) -> str:
        if self._in_think:
            self._buf = ""
            return ""
        r = self._buf
        self._buf = ""
        return r

    def reset(self) -> None:
        self._in_think = False
        self._buf = ""


class StreamProcessor:
    """Handles LLM streaming: think filtering, sentence splitting, TTS piping, tool accumulation."""

    # Long-tail fuse, not per-turn filler: normal turns speak within ~0.9s (P95),
    # so a 1.5s delay only fires on the slowest ~5% of turns — exactly the turns
    # users perceive as "too slow". Kirmayr et al. (CHI 2026) shows intermediate
    # feedback during the wait significantly improves perceived speed and trust;
    # Levinson (2015): unmarked silence beyond ~1s reads as hesitation.
    # Fires AFTER the acknowledge chime (~0.3s), so the user hears: chime ->
    # (slow turn only) thinking tone -> first semantic audio.
    THINKING_DELAY = 1.5
    SLOW_NETWORK_DELAY = 8.0  # only alert on genuine network stalls
    TRUNCATION_HINT = "还有更多内容，说继续我就接着说。"

    def __init__(
        self,
        *,
        llm: LLMClient,
        audio: AudioFrontendPort | None,
        tools: ToolRegistry,
        tool_executor: ToolExecutor,
        splitter: StreamSplitter,
        general_tool_max_safety_level: str,
        max_response_chars: int,
        voice_tts_coalesce: bool = False,
        voice_model: str | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> None:
        self._llm = llm
        self._audio = audio
        self._tools = tools
        self._tool_executor = tool_executor
        self._splitter = splitter
        self._general_tool_max_safety_level = general_tool_max_safety_level
        self._max_response_chars = max_response_chars
        self._voice_tts_coalesce = voice_tts_coalesce
        self._voice_model = voice_model
        self._think_filter = _ThinkFilter()
        self._cancel_token = cancel_token
        self._tool_turn_cancel_token: ContextVar[CancellationToken | None] = ContextVar(
            f"askme_tool_turn_cancel_{id(self)}",
            default=None,
        )
        self._tool_llm_call_context: ContextVar[LLMCallContext | None] = ContextVar(
            f"askme_tool_llm_context_{id(self)}",
            default=None,
        )

    def set_audio(self, audio: AudioFrontendPort) -> None:
        self._audio = audio

    def _create_thinking_task(
        self,
        include_slow_network: bool = False,
        *,
        cancel_token: CancellationToken | None = None,
        semantic_payload_seen: asyncio.Event | None = None,
    ) -> tuple[asyncio.Task[None], asyncio.Task[None] | None]:
        def _feedback_blocked() -> bool:
            return bool(
                (cancel_token is not None and cancel_token.is_set())
                or (semantic_payload_seen is not None and semantic_payload_seen.is_set())
            )

        async def _play_after(delay: float) -> None:
            await asyncio.sleep(delay)
            if _feedback_blocked():
                return
            # Give cancellation or a semantic payload already queued on the
            # loop one final chance to linearize before the audio handoff.
            await asyncio.sleep(0)
            if _feedback_blocked():
                return

            def _play_if_payload_still_missing() -> None:
                audio = self._audio
                if audio is not None and (
                    semantic_payload_seen is None or not semantic_payload_seen.is_set()
                ):
                    # This dedicated feedback/chime interface must stay separate
                    # from semantic ``speak`` and its TTS-first-audio telemetry.
                    audio.play_thinking()

            atomic_runner = getattr(cancel_token, "try_run", None)
            if callable(atomic_runner):
                atomic_runner(_play_if_payload_still_missing)
            elif not _feedback_blocked():
                _play_if_payload_still_missing()

        feedback_delay = self.THINKING_DELAY
        audio = self._audio
        if audio is not None:
            configured_delay = audio.processing_feedback_delay_s
            if (
                isinstance(configured_delay, (int, float))
                and not isinstance(configured_delay, bool)
                and math.isfinite(configured_delay)
            ):
                feedback_delay = max(0.0, float(configured_delay))
        thinking_task = asyncio.create_task(_play_after(feedback_delay))

        slow_network_task: asyncio.Task[None] | None = None
        if include_slow_network:
            slow_network_task = asyncio.create_task(_play_after(self.SLOW_NETWORK_DELAY))

        return thinking_task, slow_network_task

    @staticmethod
    async def _cancel_and_wait(*tasks: asyncio.Task[None] | None) -> None:
        live_tasks = [task for task in tasks if task is not None]
        for task in live_tasks:
            task.cancel()
        if live_tasks:
            await asyncio.gather(*live_tasks, return_exceptions=True)

    @staticmethod
    async def _iter_until_cancelled(stream, cancel_token: CancellationToken | None):
        """Yield stream chunks while remaining responsive to turn cancellation.

        An ``async for`` cannot inspect a cooperative token while ``__anext__``
        is waiting for the provider's first byte.  Race each pending next chunk
        against a lightweight token watcher, and always reap both tasks before
        returning so a cancelled turn cannot retain an HTTP stream or timer.
        """
        if cancel_token is None:
            async for chunk in stream:
                yield chunk
            return

        async def _wait_for_cancel() -> None:
            while not cancel_token.is_set():
                await asyncio.sleep(0.01)

        iterator = aiter(stream)
        cancel_task = asyncio.create_task(_wait_for_cancel())
        next_task: asyncio.Future[Any] | None = None
        try:
            while not cancel_token.is_set():
                next_task = asyncio.ensure_future(anext(iterator))
                done, _ = await asyncio.wait(
                    (next_task, cancel_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_task in done or cancel_token.is_set():
                    next_task.cancel()
                    await asyncio.gather(next_task, return_exceptions=True)
                    next_task = None
                    break
                try:
                    chunk = next_task.result()
                except StopAsyncIteration:
                    next_task = None
                    break
                next_task = None
                yield chunk
        finally:
            if next_task is not None:
                next_task.cancel()
                await asyncio.gather(next_task, return_exceptions=True)
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)

    @staticmethod
    def _has_payload(chunk: Any) -> bool:
        """Return whether a stream chunk carries non-empty model output."""
        try:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                return False
            delta = choices[0].delta
            return bool(getattr(delta, "content", None) or getattr(delta, "tool_calls", None))
        except Exception:  # noqa: BLE001 - never break streaming on a timing probe
            return True

    async def consume_llm_stream(
        self,
        stream,
        source: str = "voice",
        turn_cancel_token: CancellationToken | None = None,
    ) -> tuple[str, dict[int, dict[str, str]]]:
        """Consume LLM stream: apply think filter, feed splitter -> TTS, enforce truncation.

        Returns (full_text, tool_calls_acc).
        """
        full_response = ""
        tool_calls_acc: dict[int, dict[str, str]] = {}
        spoke_any = False

        audio = self._audio
        is_voice = source == "voice" and audio is not None
        chars_spoken = 0
        truncated = False
        char_limit = self._max_response_chars if is_voice else 0
        coalesce_voice = is_voice and self._voice_tts_coalesce
        voice_chunks: list[str] = []
        suppress_voice_output = False
        active_cancel_token = turn_cancel_token or self._cancel_token

        def _queue_voice(text: str) -> None:
            nonlocal spoke_any
            if (
                not text
                or audio is None
                or suppress_voice_output
                or (active_cancel_token is not None and active_cancel_token.is_set())
            ):
                return
            if coalesce_voice:
                voice_chunks.append(text)
            else:
                audio.speak(text)
            spoke_any = True

        async for chunk in self._iter_until_cancelled(stream, active_cancel_token):
            if active_cancel_token is not None and active_cancel_token.is_set():
                break
            delta = chunk.choices[0].delta

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

                if is_voice:
                    suppress_voice_output = True
                    self._think_filter.reset()
                    self._splitter.reset()
                    if spoke_any and audio is not None:
                        audio.drain_buffers()
                    spoke_any = False
                    voice_chunks.clear()

            if delta.content:
                clean = self._think_filter.feed(delta.content)
                if clean:
                    full_response += clean
                    if is_voice and not truncated:
                        for sentence in self._splitter.feed(clean):
                            if char_limit and chars_spoken + len(sentence) > char_limit:
                                _queue_voice(sentence)
                                _queue_voice(self.TRUNCATION_HINT)
                                truncated = True
                                logger.info(
                                    "Voice truncation at %d chars (limit %d)",
                                    chars_spoken + len(sentence),
                                    char_limit,
                                )
                                break
                            _queue_voice(sentence)
                            chars_spoken += len(sentence)

        if active_cancel_token is not None and active_cancel_token.is_set():
            return full_response, tool_calls_acc

        think_tail = self._think_filter.flush()
        if think_tail:
            full_response += think_tail
            if is_voice and not truncated:
                for sentence in self._splitter.feed(think_tail):
                    if char_limit and chars_spoken + len(sentence) > char_limit:
                        _queue_voice(sentence)
                        truncated = True
                        break
                    _queue_voice(sentence)
                    chars_spoken += len(sentence)
        if is_voice and not truncated:
            remainder = self._splitter.flush()
            if remainder:
                _queue_voice(remainder)

        if coalesce_voice and voice_chunks and audio is not None:
            audio.speak("".join(voice_chunks).strip())

        return full_response, tool_calls_acc

    @staticmethod
    def _supported_tool_context_kwargs(
        callback: Any,
        *,
        turn_cancel_token: CancellationToken | None,
        llm_call_context: LLMCallContext | None,
    ) -> dict[str, Any]:
        """Pass turn policy context without breaking legacy test/runtime adapters."""

        try:
            parameters = signature(callback).parameters
            accepts_kwargs = any(
                parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
        except (TypeError, ValueError):
            return {}

        context = {
            "turn_cancel_token": turn_cancel_token,
            "llm_call_context": llm_call_context,
        }
        if accepts_kwargs:
            return context
        return {name: value for name, value in context.items() if name in parameters}

    async def stream_with_tools(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        model: str | None = None,
        source: str = "voice",
        conversation_session_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
        llm_call_context: LLMCallContext | None = None,
    ) -> str:
        """Stream LLM response, speak sentences immediately, handle tool calls."""
        active_cancel_token = turn_cancel_token or self._cancel_token
        if active_cancel_token is not None and active_cancel_token.is_set():
            return ""
        tool_definitions = self._tools.get_definitions(
            max_safety_level=self._general_tool_max_safety_level
        )
        tool_names = [td.get("function", {}).get("name") for td in tool_definitions]
        logger.info("LLM tools available (%d): %s", len(tool_definitions), tool_names)
        ttft_logged = False
        t_start = _time.perf_counter()
        self._splitter.reset()
        self._think_filter.reset()

        audio = self._audio
        is_voice = source == "voice" and audio is not None

        # Voice turns use a tighter token cap for lower latency.
        per_call_max_tokens: int | None = None
        if is_voice:
            try:
                from askme.config import get_config

                per_call_max_tokens = get_config().get("brain", {}).get("voice_max_tokens")
            except Exception:
                per_call_max_tokens = None

        thinking_task: asyncio.Task[None] | None = None
        slow_network_task: asyncio.Task[None] | None = None
        semantic_payload_seen = asyncio.Event()
        externally_armed = bool(audio is not None and audio.processing_feedback_armed is True)
        if is_voice and not externally_armed:
            thinking_task, slow_network_task = self._create_thinking_task(
                include_slow_network=True,
                cancel_token=active_cancel_token,
                semantic_payload_seen=semantic_payload_seen,
            )

        try:

            async def _ttft_stream():
                nonlocal ttft_logged, thinking_task, slow_network_task
                async for chunk in self._llm.chat_stream(
                    messages,
                    tools=tool_definitions,
                    tool_choice="auto",
                    model=model,
                    max_tokens=per_call_max_tokens,
                    cancel_token=active_cancel_token,
                    context=llm_call_context,
                ):
                    has_payload = self._has_payload(chunk)
                    if has_payload and not ttft_logged:
                        ttft_logged = True
                        elapsed = _time.perf_counter() - t_start
                        logger.info("TTFT: %.2fs", elapsed)
                        get_tracer().record_span("ttft", elapsed * 1000, model=model or "default")
                    # Cancel the intermediate-feedback timers only on the first
                    # chunk carrying real content (text or tool call).  Empty
                    # keep-alive deltas must not suppress the long-tail thinking
                    # tone — that would defeat the 1.5s fuse on slow turns.
                    if has_payload:
                        first_semantic_payload = not semantic_payload_seen.is_set()
                        semantic_payload_seen.set()
                        if first_semantic_payload and externally_armed and audio is not None:
                            try:
                                audio.cancel_processing_feedback()
                            except Exception as exc:
                                logger.debug(
                                    "processing feedback cancel failed at first payload: %s",
                                    exc,
                                )
                        if thinking_task is not None or slow_network_task is not None:
                            await self._cancel_and_wait(
                                thinking_task,
                                slow_network_task,
                            )
                            thinking_task = None
                            slow_network_task = None
                    yield chunk

            full_response, tool_calls_acc = await self.consume_llm_stream(
                _ttft_stream(),
                source=source,
                turn_cancel_token=active_cancel_token,
            )
        finally:
            await self._cancel_and_wait(thinking_task, slow_network_task)

        if active_cancel_token is not None and active_cancel_token.is_set():
            return full_response

        if tool_calls_acc:
            if audio is not None:
                audio.drain_buffers()
            context_token = self._tool_turn_cancel_token.set(active_cancel_token)
            llm_context_token = self._tool_llm_call_context.set(llm_call_context)
            try:
                tool_context_kwargs = self._supported_tool_context_kwargs(
                    self._tool_executor.execute_tools,
                    turn_cancel_token=active_cancel_token,
                    llm_call_context=llm_call_context,
                )
                full_response = await self._tool_executor.execute_tools(
                    tool_calls_acc,
                    system_prompt,
                    model=model,
                    source=source,
                    conversation_session_id=conversation_session_id,
                    **tool_context_kwargs,
                )
            finally:
                self._tool_llm_call_context.reset(llm_context_token)
                self._tool_turn_cancel_token.reset(context_token)

        return full_response

    async def stream_and_speak(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        source: str = "voice",
        turn_cancel_token: CancellationToken | None = None,
        llm_call_context: LLMCallContext | None = None,
    ) -> str:
        """Stream a follow-up LLM response and pipe to TTS."""
        active_cancel_token = (
            turn_cancel_token or self._tool_turn_cancel_token.get() or self._cancel_token
        )
        inherited_llm_context = self._tool_llm_call_context.get()
        active_llm_context: LLMCallContext | None
        if llm_call_context is None and inherited_llm_context is not None:
            active_llm_context = replace(
                inherited_llm_context,
                call_id=uuid4().hex,
                purpose="tool_followup",
            )
        else:
            active_llm_context = llm_call_context or inherited_llm_context
        self._splitter.reset()
        self._think_filter.reset()
        full_response, _ = await self.consume_llm_stream(
            self._llm.chat_stream(
                messages,
                model=model,
                cancel_token=active_cancel_token,
                context=active_llm_context,
            ),
            source=source,
            turn_cancel_token=active_cancel_token,
        )
        return full_response

    def reset(self) -> None:
        """Reset internal state for a new turn."""
        self._think_filter.reset()
        self._splitter.reset()
