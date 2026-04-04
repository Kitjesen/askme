"""Protocol contracts for BrainPipeline sub-components.

Defines three structural interfaces (typing.Protocol) so that:
  - Tests can inject plain mock objects without touching private attributes.
  - BrainPipeline's constructor accepts any conforming implementation.
  - Sub-components can be swapped independently.

Also exports TurnContext — an immutable per-turn snapshot inspired by
Claude Code's context-passing pattern.  cancel_token is set by
BrainPipeline.handle_estop(); each sub-component checks it autonomously
so E-STOP coordination requires no manual per-component calls.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StreamProcessorProtocol(Protocol):
    """Contract for the LLM streaming + think-filter + TTS piping component."""

    async def stream_with_tools(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
        model: str | None = None,
        source: str = "voice",
    ) -> str:
        """Stream LLM response, speak sentences via TTS, handle tool calls."""
        ...

    async def stream_and_speak(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        source: str = "voice",
    ) -> str:
        """Stream a follow-up LLM response and pipe to TTS."""
        ...

    def reset(self) -> None:
        """Reset internal streaming state for a new turn."""
        ...

    def set_audio(self, audio: Any) -> None:
        """Late-bind the AudioAgent (set post-build by VoiceModule/TextModule)."""
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
    ) -> str:
        """Run the full pipeline for *user_text*. Returns assistant reply."""
        ...

    def start_idle_reflection(
        self, idle_seconds: float = 300.0
    ) -> asyncio.Task[None] | None:
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
    ``cancel_token`` is an asyncio.Event set by BrainPipeline.handle_estop().
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
    cancel_token: asyncio.Event
    voice_model: str | None = None
    memory_task: "asyncio.Task[str] | None" = field(default=None, compare=False)
