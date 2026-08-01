"""Protocol contracts and shared types for BrainPipeline sub-components.

Public surface:
  - ``StreamProcessorProtocol``, ``SkillGateProtocol``, ``TurnExecutorProtocol``
    — structural (typing.Protocol) contracts; tests inject mocks directly.
  - ``TurnContext`` — immutable per-turn snapshot; cancel_token shared across
    all sub-components; set by handle_estop() for autonomous E-STOP.
  - ``PipelineHooks`` re-exported for convenience (defined in hooks.py).
  - ``ToolCallRecord`` re-exported for convenience.

Inspired by Claude Code's patterns:
  - Context objects propagated immutably through the pipeline.
  - AbortSignal equivalent (thread-safe cancel token).
  - Hook system (PipelineHooks) for lifecycle callbacks.
  - Structured tool results (ToolCallRecord).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from askme.llm.core.contracts import LLMCallContext

# Re-export hook types for one-stop import convenience
from askme.pipeline.core.hooks import PipelineHooks, ToolCallRecord  # noqa: F401


@runtime_checkable
class CancellationToken(Protocol):
    """Minimal cancellation signal shared across event-loop and audio threads."""

    def is_set(self) -> bool: ...

    def set(self) -> None: ...


@runtime_checkable
class StreamProcessorProtocol(Protocol):
    """Contract for the LLM streaming + think-filter + TTS piping component."""

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
        """Stream LLM response, speak sentences via TTS, handle tool calls."""
        ...

    async def stream_and_speak(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        source: str = "voice",
        turn_cancel_token: CancellationToken | None = None,
        llm_call_context: LLMCallContext | None = None,
    ) -> str:
        """Stream a follow-up LLM response and pipe to TTS."""
        ...

    async def consume_llm_stream(
        self,
        stream: Any,
        source: str = "voice",
        turn_cancel_token: CancellationToken | None = None,
    ) -> tuple[str, dict[int, dict[str, str]]]:
        """Consume raw LLM stream: think filter, TTS, truncation.

        Returns ``(full_text, tool_calls_acc)``.
        """
        ...

    def reset(self) -> None:
        """Reset internal streaming state for a new turn."""
        ...

    def set_audio(self, audio: Any) -> None:
        """Late-bind the AudioAgent (set post-build by VoiceModule/TextModule)."""
        ...


@dataclass(frozen=True, slots=True)
class SkillExecutionDisposition:
    """Canonical Turn settlement decision returned by an injected skill gate."""

    status: Literal["succeeded", "failed", "cancelled"]
    code: str

    def __post_init__(self) -> None:
        if self.status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError(f"invalid skill execution disposition status: {self.status}")
        if not str(self.code or "").strip():
            raise ValueError("skill execution disposition code must not be empty")


@runtime_checkable
class SkillResultClassifierProtocol(Protocol):
    """Optional extension for gates that classify their legacy string result."""

    def classify_execution_result(
        self,
        result: str,
        *,
        skill_name: str = "",
    ) -> SkillExecutionDisposition:
        """Classify the legacy result for canonical Turn settlement."""
        ...


@runtime_checkable
class SkillGateProtocol(Protocol):
    """Contract for the skill execution + safety gate component."""

    @property
    def last_spoken_text(self) -> str:
        """The most recent text spoken during skill execution."""
        ...

    async def execute_skill(
        self,
        skill_name: str,
        user_text: str,
        extra_context: str = "",
        source: str = "voice",
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_cancel_token: CancellationToken | None = None,
    ) -> str:
        """Execute a named skill and speak the result."""
        ...

    def extract_semantic_target(self, user_text: str) -> str:
        """Extract navigation/object target from vague user input."""
        ...

    def set_audio(self, audio: Any) -> None: ...

    def set_skill_manager(self, manager: Any) -> None: ...

    def set_skill_executor(self, executor: Any) -> None: ...

    def set_agent_shell(self, shell: Any) -> None: ...


@runtime_checkable
class TurnExecutorProtocol(Protocol):
    """Contract for the full turn orchestration component."""

    @property
    def last_spoken_text(self) -> str:
        """The most recent assistant reply (used by repeat_last skill)."""
        ...

    async def process(
        self,
        user_text: str,
        *,
        memory_task: asyncio.Task[str] | None = None,
        source: str = "voice",
        conversation_session_id: str | None = None,
        voice_turn_id: str | None = None,
        turn_epoch: int | None = None,
        turn_cancel_token: CancellationToken | None = None,
    ) -> str:
        """Run the full pipeline for *user_text*. Returns assistant reply."""
        ...

    def start_idle_reflection(self, idle_seconds: float = 300.0) -> asyncio.Task[None] | None:
        """Start a background dream-consolidation task."""
        ...

    def start_memory_prefetch(self, user_text: str) -> asyncio.Task[str]:
        """Start memory retrieval as a background task."""
        ...

    async def shutdown(self) -> None:
        """Cancel all in-flight background tasks."""
        ...

    def set_audio(self, audio: Any) -> None:
        """Late-bind AudioAgent."""
        ...


@dataclass(frozen=True)
class TurnContext:
    """Immutable snapshot of per-turn context.

    Passed through the pipeline so sub-components share state without coupling.
    ``cancel_token`` is a thread-safe signal set by cancellation or E-STOP.
    Each sub-component checks it independently — no manual coordination needed.

    Example::

        token = asyncio.Event()
        ctx = TurnContext(user_text="巡检A区", source="voice", cancel_token=token)
        # E-STOP from any thread:
        token.set()
        # StreamProcessor, SkillGate, TurnExecutor all stop on their own.
    """

    user_text: str
    source: str
    cancel_token: CancellationToken
    voice_model: str | None = None
    conversation_session_id: str | None = None
    voice_turn_id: str | None = None
    turn_epoch: int | None = None
