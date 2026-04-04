"""Pipeline lifecycle hooks — inspired by Claude Code's hook system.

Claude Code fires hooks (PreToolUse, PostToolUse, UserPromptSubmit, Stop) at
well-defined lifecycle points, letting users inject behaviour without modifying
core code.  We apply the same pattern to BrainPipeline:

  - ``pre_turn``   : fired before the LLM is called; can return False to skip
  - ``post_turn``  : fired after the assistant reply is ready
  - ``pre_tool``   : fired before each tool execution; can override arguments or
                     block the call by returning None for the result
  - ``post_tool``  : fired after each tool execution; can override the result
  - ``on_estop``   : fired synchronously when handle_estop() is called

Usage::

    from askme.pipeline.hooks import PipelineHooks

    hooks = PipelineHooks()

    @hooks.on_pre_turn
    async def log_turn(ctx):
        logging.info("Turn started: %s", ctx.user_text)

    @hooks.on_post_tool
    async def audit_tool(record):
        db.save(record)

    pipeline = BrainPipeline(..., hooks=hooks)

Hook callbacks are async (or sync for on_estop).  Errors in hooks are caught
and logged — they never propagate into the main pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ToolCallRecord — immutable record of a single tool execution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCallRecord:
    """Immutable snapshot of one tool call.

    Inspired by Claude Code's structured tool results: each call has a clear
    identity (``call_id``, ``tool_name``), inputs (``arguments``), output
    (``result``), and execution metadata (``elapsed_ms``, ``timed_out``).

    Passed to ``post_tool`` hooks for auditing, metrics, or result overriding.
    """

    call_id: str
    tool_name: str
    arguments: str
    result: str
    elapsed_ms: float
    timed_out: bool = False
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Hook callback types
# ---------------------------------------------------------------------------

# pre_turn(ctx) → bool | None  (False / None = proceed; True = skip this turn)
PreTurnHook = Callable[["TurnContext"], Awaitable[bool | None]]  # type: ignore[name-defined]

# post_turn(ctx, response) → None
PostTurnHook = Callable[["TurnContext", str], Awaitable[None]]  # type: ignore[name-defined]

# pre_tool(record_without_result) → str | None  (None = skip & use empty result)
PreToolHook = Callable[[ToolCallRecord], Awaitable[str | None]]

# post_tool(record) → str  (return value replaces the tool result in conversation)
PostToolHook = Callable[[ToolCallRecord], Awaitable[str]]

# on_estop() → None  (synchronous; fires immediately inside handle_estop)
EStopHook = Callable[[], None]


# ---------------------------------------------------------------------------
# PipelineHooks
# ---------------------------------------------------------------------------

@dataclass
class PipelineHooks:
    """Registry of lifecycle hooks for BrainPipeline.

    Each hook type accepts a list of callbacks.  Use the ``@hooks.on_*``
    decorators or append directly to the lists.

    All async hooks are run sequentially in registration order.
    Exceptions in hooks are caught and logged — they never crash the pipeline.

    Example::

        hooks = PipelineHooks()

        @hooks.on_pre_turn
        async def guard(ctx):
            if 'forbidden' in ctx.user_text:
                return True  # skip this turn

        @hooks.on_post_tool
        async def metrics(record):
            statsd.timing('tool.' + record.tool_name, record.elapsed_ms)
            return record.result  # unchanged
    """

    pre_turn: list[PreTurnHook] = field(default_factory=list)
    post_turn: list[PostTurnHook] = field(default_factory=list)
    pre_tool: list[PreToolHook] = field(default_factory=list)
    post_tool: list[PostToolHook] = field(default_factory=list)
    estop: list[EStopHook] = field(default_factory=list)

    # ── Decorator API (mirrors Claude Code's settings-based hook registration) ──

    def on_pre_turn(self, fn: PreTurnHook) -> PreTurnHook:
        """Decorator: register a pre-turn hook."""
        self.pre_turn.append(fn)
        return fn

    def on_post_turn(self, fn: PostTurnHook) -> PostTurnHook:
        """Decorator: register a post-turn hook."""
        self.post_turn.append(fn)
        return fn

    def on_pre_tool(self, fn: PreToolHook) -> PreToolHook:
        """Decorator: register a pre-tool hook."""
        self.pre_tool.append(fn)
        return fn

    def on_post_tool(self, fn: PostToolHook) -> PostToolHook:
        """Decorator: register a post-tool hook."""
        self.post_tool.append(fn)
        return fn

    def on_estop(self, fn: EStopHook) -> EStopHook:
        """Decorator: register a synchronous E-STOP hook."""
        self.estop.append(fn)
        return fn

    # ── Async firing helpers ───────────────────────────────────────────────

    async def fire_pre_turn(self, ctx: "TurnContext") -> bool:  # type: ignore[name-defined]
        """Fire all pre_turn hooks. Returns True if any hook says to skip."""
        for hook in self.pre_turn:
            try:
                result = await hook(ctx)
                if result is True:
                    logger.info("[hooks.pre_turn] Hook requested turn skip: %s", hook.__name__)
                    return True
            except Exception as exc:
                logger.warning("[hooks.pre_turn] Hook %s raised: %s", hook.__name__, exc)
        return False

    async def fire_post_turn(self, ctx: "TurnContext", response: str) -> None:  # type: ignore[name-defined]
        """Fire all post_turn hooks."""
        for hook in self.post_turn:
            try:
                await hook(ctx, response)
            except Exception as exc:
                logger.warning("[hooks.post_turn] Hook %s raised: %s", hook.__name__, exc)

    async def fire_pre_tool(self, record: ToolCallRecord) -> "_ProceedType | str":
        """Fire all pre_tool hooks.

        Returns the first non-None override result (a string), or ``_PROCEED``
        sentinel when no hook intercepted the call.
        """
        for hook in self.pre_tool:
            try:
                override = await hook(record)
                if override is not None:
                    return override
            except Exception as exc:
                logger.warning("[hooks.pre_tool] Hook %s raised: %s", hook.__name__, exc)
        return _PROCEED  # sentinel: no override

    async def fire_post_tool(self, record: ToolCallRecord) -> str:
        """Fire all post_tool hooks, each may transform the result."""
        result = record.result
        for hook in self.post_tool:
            try:
                result = await hook(dataclasses_replace(record, result=result))
            except Exception as exc:
                logger.warning("[hooks.post_tool] Hook %s raised: %s", hook.__name__, exc)
        return result

    def fire_estop(self) -> None:
        """Fire all E-STOP hooks synchronously (called from handle_estop)."""
        for hook in self.estop:
            try:
                hook()
            except Exception as exc:
                logger.warning("[hooks.estop] Hook %s raised: %s", hook.__name__, exc)


# Sentinel returned by fire_pre_tool when no hook overrides the result.
# Using a proper sentinel class so ``result is _PROCEED`` has a correct type.


class _ProceedType:
    """Singleton sentinel: fire_pre_tool found no hook override — proceed normally."""

    _instance: "_ProceedType | None" = None

    def __new__(cls) -> "_ProceedType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<_PROCEED>"


_PROCEED = _ProceedType()


def dataclasses_replace(obj: ToolCallRecord, **changes: object) -> ToolCallRecord:
    """Functional update for frozen dataclass (like dataclasses.replace)."""
    import dataclasses
    return dataclasses.replace(obj, **changes)
