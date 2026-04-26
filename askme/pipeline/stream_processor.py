"""LLM stream processor — think filter, sentence splitting, TTS piping, tool accumulation."""

from __future__ import annotations

import asyncio
import logging
import time as _time
from typing import TYPE_CHECKING, Any

from askme.pipeline.trace import get_tracer

if TYPE_CHECKING:
    from askme.llm.client import LLMClient
    from askme.pipeline.tool_executor import ToolExecutor
    from askme.tools.tool_registry import ToolRegistry
    from askme.voice.audio_agent import AudioAgent
    from askme.voice.stream_splitter import StreamSplitter

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
                self._buf = self._buf[idx + 8:]
                self._in_think = False
            else:
                idx = self._buf.find("<think>")
                if idx < 0:
                    safe = max(0, len(self._buf) - 7)
                    out.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                    return "".join(out)
                out.append(self._buf[:idx])
                self._buf = self._buf[idx + 7:]
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

    THINKING_DELAY = 1.2          # seconds before playing the "thinking" audio cue
    SLOW_NETWORK_DELAY = 5.0      # seconds before playing a second "slow network" cue
    TRUNCATION_HINT = "还有更多内容，说继续我就接着说。"

    def __init__(
        self,
        *,
        llm: LLMClient,
        audio: AudioAgent | None,
        tools: ToolRegistry,
        tool_executor: ToolExecutor,
        splitter: StreamSplitter,
        general_tool_max_safety_level: int,
        max_response_chars: int,
        voice_model: str | None = None,
        cancel_token: asyncio.Event | None = None,
    ) -> None:
        self._llm = llm
        self._audio = audio
        self._tools = tools
        self._tool_executor = tool_executor
        self._splitter = splitter
        self._general_tool_max_safety_level = general_tool_max_safety_level
        self._max_response_chars = max_response_chars
        self._voice_model = voice_model
        self._think_filter = _ThinkFilter()
        self._cancel_token = cancel_token

    def set_audio(self, audio: AudioAgent) -> None:
        self._audio = audio

    def _create_thinking_task(
        self, include_slow_network: bool = False,
    ) -> tuple[asyncio.Task[None], asyncio.Task[None] | None]:
        async def _thinking_indicator() -> None:
            await asyncio.sleep(self.THINKING_DELAY)
            self._audio.play_thinking()

        thinking_task = asyncio.create_task(_thinking_indicator())

        slow_network_task: asyncio.Task[None] | None = None
        if include_slow_network:
            async def _slow_network_indicator() -> None:
                await asyncio.sleep(self.SLOW_NETWORK_DELAY)
                self._audio.play_thinking()

            slow_network_task = asyncio.create_task(_slow_network_indicator())

        return thinking_task, slow_network_task

    async def consume_llm_stream(
        self,
        stream,
        source: str = "voice",
    ) -> tuple[str, dict[int, dict[str, str]]]:
        """Consume LLM stream: apply think filter, feed splitter -> TTS, enforce truncation.

        Returns (full_text, tool_calls_acc).
        """
        full_response = ""
        tool_calls_acc: dict[int, dict[str, str]] = {}
        spoke_any = False

        is_voice = source == "voice"
        chars_spoken = 0
        truncated = False
        char_limit = self._max_response_chars if is_voice else 0

        async for chunk in stream:
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

                if spoke_any:
                    self._audio.drain_buffers()
                    spoke_any = False

            if delta.content:
                clean = self._think_filter.feed(delta.content)
                if clean:
                    full_response += clean
                    if not truncated:
                        for sentence in self._splitter.feed(clean):
                            if char_limit and chars_spoken + len(sentence) > char_limit:
                                self._audio.speak(sentence)
                                self._audio.speak(self.TRUNCATION_HINT)
                                spoke_any = True
                                truncated = True
                                logger.info(
                                    "Voice truncation at %d chars (limit %d)",
                                    chars_spoken + len(sentence), char_limit,
                                )
                                break
                            self._audio.speak(sentence)
                            chars_spoken += len(sentence)
                            spoke_any = True

        think_tail = self._think_filter.flush()
        if think_tail:
            full_response += think_tail
            if not truncated:
                for sentence in self._splitter.feed(think_tail):
                    if char_limit and chars_spoken + len(sentence) > char_limit:
                        self._audio.speak(sentence)
                        spoke_any = True
                        truncated = True
                        break
                    self._audio.speak(sentence)
                    chars_spoken += len(sentence)
                    spoke_any = True
        if not truncated:
            remainder = self._splitter.flush()
            if remainder:
                self._audio.speak(remainder)
                spoke_any = True

        return full_response, tool_calls_acc

    async def stream_with_tools(
        self, messages: list[dict[str, Any]], system_prompt: str,
        model: str | None = None, source: str = "voice",
    ) -> str:
        """Stream LLM response, speak sentences immediately, handle tool calls."""
        tool_definitions = self._tools.get_definitions(
            max_safety_level=self._general_tool_max_safety_level
        )
        tool_names = [td.get("function", {}).get("name") for td in tool_definitions]
        logger.info("LLM tools available (%d): %s", len(tool_definitions), tool_names)
        ttft_logged = False
        t_start = _time.perf_counter()
        self._splitter.reset()
        self._think_filter.reset()

        is_voice = source == "voice"

        thinking_task: asyncio.Task[None] | None = None
        slow_network_task: asyncio.Task[None] | None = None
        if is_voice:
            thinking_task, slow_network_task = self._create_thinking_task(
                include_slow_network=True,
            )

        try:
            async def _ttft_stream():
                nonlocal ttft_logged, thinking_task, slow_network_task
                async for chunk in self._llm.chat_stream(
                    messages, tools=tool_definitions, tool_choice="auto", model=model,
                    cancel_token=self._cancel_token,
                ):
                    if not ttft_logged:
                        ttft_logged = True
                        elapsed = _time.perf_counter() - t_start
                        logger.info("TTFT: %.2fs", elapsed)
                        get_tracer().record_span("ttft", elapsed * 1000, model=model or "default")
                        if thinking_task is not None:
                            thinking_task.cancel()
                            thinking_task = None
                        if slow_network_task is not None:
                            slow_network_task.cancel()
                            slow_network_task = None
                    yield chunk

            full_response, tool_calls_acc = await self.consume_llm_stream(
                _ttft_stream(), source=source,
            )
        finally:
            if thinking_task is not None:
                thinking_task.cancel()
            if slow_network_task is not None:
                slow_network_task.cancel()

        if tool_calls_acc:
            self._audio.drain_buffers()
            full_response = await self._tool_executor.execute_tools(
                tool_calls_acc, system_prompt, model=model, source=source,
            )

        return full_response

    async def stream_and_speak(
        self, messages: list[dict[str, Any]], model: str | None = None,
        source: str = "voice",
    ) -> str:
        """Stream a follow-up LLM response and pipe to TTS."""
        self._splitter.reset()
        self._think_filter.reset()
        full_response, _ = await self.consume_llm_stream(
            self._llm.chat_stream(messages, model=model, cancel_token=self._cancel_token),
            source=source,
        )
        return full_response

    def reset(self) -> None:
        """Reset internal state for a new turn."""
        self._think_filter.reset()
        self._splitter.reset()
