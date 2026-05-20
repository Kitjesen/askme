"""Core turn-processing pipeline modules.

This package owns pure turn orchestration, prompt/stream processing, tracing,
and frame contracts. Product code should import these core primitives from
here or from their concrete owner modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AudioRawFrame": ("askme.pipeline.core.frames", "AudioRawFrame"),
    "BrainPipeline": ("askme.pipeline.core.brain_pipeline", "BrainPipeline"),
    "CancellationToken": ("askme.pipeline.core.frames", "CancellationToken"),
    "DataFrame": ("askme.pipeline.core.frames", "DataFrame"),
    "Frame": ("askme.pipeline.core.frames", "Frame"),
    "FramePipeline": ("askme.pipeline.core.frames", "FramePipeline"),
    "FrameProcessor": ("askme.pipeline.core.frames", "FrameProcessor"),
    "IntentFrame": ("askme.pipeline.core.frames", "IntentFrame"),
    "InterruptFrame": ("askme.pipeline.core.frames", "InterruptFrame"),
    "LLMFullResponseFrame": ("askme.pipeline.core.frames", "LLMFullResponseFrame"),
    "LLMStartFrame": ("askme.pipeline.core.frames", "LLMStartFrame"),
    "LLMTextFrame": ("askme.pipeline.core.frames", "LLMTextFrame"),
    "MetricsFrame": ("askme.pipeline.core.frames", "MetricsFrame"),
    "PipelineHooks": ("askme.pipeline.core.hooks", "PipelineHooks"),
    "PipelineTracer": ("askme.pipeline.core.trace", "PipelineTracer"),
    "PromptBuilder": ("askme.pipeline.core.prompt_builder", "PromptBuilder"),
    "Span": ("askme.pipeline.core.trace", "Span"),
    "StartInterruptFrame": ("askme.pipeline.core.frames", "StartInterruptFrame"),
    "StopInterruptFrame": ("askme.pipeline.core.frames", "StopInterruptFrame"),
    "StreamProcessor": ("askme.pipeline.core.stream_processor", "StreamProcessor"),
    "StreamProcessorProtocol": (
        "askme.pipeline.core.protocols",
        "StreamProcessorProtocol",
    ),
    "SystemFrame": ("askme.pipeline.core.frames", "SystemFrame"),
    "TTSAudioFrame": ("askme.pipeline.core.frames", "TTSAudioFrame"),
    "TTSSpeakFrame": ("askme.pipeline.core.frames", "TTSSpeakFrame"),
    "ToolCallRecord": ("askme.pipeline.core.hooks", "ToolCallRecord"),
    "ToolExecutor": ("askme.pipeline.core.tool_executor", "ToolExecutor"),
    "Trace": ("askme.pipeline.core.trace", "Trace"),
    "TranscriptionFrame": ("askme.pipeline.core.frames", "TranscriptionFrame"),
    "TurnContext": ("askme.pipeline.core.protocols", "TurnContext"),
    "TurnExecutor": ("askme.pipeline.core.turn_executor", "TurnExecutor"),
    "TurnExecutorProtocol": (
        "askme.pipeline.core.protocols",
        "TurnExecutorProtocol",
    ),
    "VADFrame": ("askme.pipeline.core.frames", "VADFrame"),
    "get_tracer": ("askme.pipeline.core.trace", "get_tracer"),
    "strip_think_blocks": ("askme.pipeline.core.utils", "strip_think_blocks"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
