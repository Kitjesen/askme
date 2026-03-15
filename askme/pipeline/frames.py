"""Frame-based pipeline abstraction inspired by Pipecat/LiveKit.

Provides a unified data flow model where all data moves through the pipeline
as typed Frames. SystemFrames (interrupts) bypass queues and propagate
immediately to all downstream processors.

This module coexists with the current architecture — processors can be
adopted incrementally without rewriting existing code.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame hierarchy
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """Base class for all pipeline data."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.monotonic)


class SystemFrame(Frame):
    """High-priority frames that bypass queues and propagate immediately."""

    pass


class DataFrame(Frame):
    """Normal-priority frames that flow through the processing queue."""

    pass


# -- System frames ----------------------------------------------------------

@dataclass
class InterruptFrame(SystemFrame):
    """Signal to cancel all downstream processing immediately.

    When pushed into a FramePipeline, this frame is delivered to every
    processor's ``handle_interrupt()`` method — bypassing the normal queue.
    """

    reason: str = "unknown"  # "barge_in", "estop", "stop_command", "timeout"


@dataclass
class StartInterruptFrame(SystemFrame):
    """User started speaking — begin interrupt sequence."""

    pass


@dataclass
class StopInterruptFrame(SystemFrame):
    """Interrupt sequence complete — resume normal processing."""

    pass


# -- Data frames ------------------------------------------------------------

@dataclass
class AudioRawFrame(DataFrame):
    """Raw audio samples from the microphone."""

    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sample_rate: int = 16000
    peak: int = 0


@dataclass
class VADFrame(DataFrame):
    """Voice activity detection result for a chunk of audio."""

    is_speech: bool = False
    peak: int = 0


@dataclass
class TranscriptionFrame(DataFrame):
    """ASR transcription result."""

    text: str = ""
    is_final: bool = True
    language: str = "zh"


@dataclass
class IntentFrame(DataFrame):
    """Intent routing result."""

    intent_type: str = "general"  # "general", "voice_trigger", "estop", "command"
    skill_name: str | None = None
    user_text: str = ""


@dataclass
class LLMStartFrame(DataFrame):
    """LLM generation has started (first token received)."""

    model: str = ""
    ttft_ms: float = 0.0


@dataclass
class LLMTextFrame(DataFrame):
    """Incremental LLM text output (streaming chunk)."""

    text: str = ""


@dataclass
class LLMFullResponseFrame(DataFrame):
    """Complete LLM response after streaming finishes."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TTSSpeakFrame(DataFrame):
    """Text to be synthesized and spoken."""

    text: str = ""


@dataclass
class TTSAudioFrame(DataFrame):
    """Synthesized audio ready for playback."""

    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    sample_rate: int = 24000


@dataclass
class MetricsFrame(SystemFrame):
    """Pipeline timing / observability data."""

    stage: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Processor and Pipeline
# ---------------------------------------------------------------------------

class FrameProcessor(ABC):
    """Base class for pipeline stage processors.

    Each processor consumes frames and yields zero or more output frames.
    Interrupt handling is separate and immediate.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name or self.__class__.__name__
        self._interrupted = False

    @abstractmethod
    async def process_frame(self, frame: Frame) -> list[Frame]:
        """Process a single frame and return output frames."""
        ...

    async def handle_interrupt(self, frame: InterruptFrame) -> None:
        """Handle an interrupt frame — cancel ongoing work."""
        self._interrupted = True
        logger.debug("[%s] interrupted: %s", self.name, frame.reason)

    def reset_interrupt(self) -> None:
        """Clear the interrupted flag for the next turn."""
        self._interrupted = False


class PassthroughProcessor(FrameProcessor):
    """A no-op processor that passes frames through unchanged."""

    async def process_frame(self, frame: Frame) -> list[Frame]:
        return [frame]


class FramePipeline:
    """Chain of FrameProcessors with dual-queue interrupt propagation.

    Normal DataFrames flow sequentially through each processor.
    SystemFrames (interrupts) are delivered to ALL processors immediately,
    bypassing the normal queue — just like Pipecat's design.
    """

    def __init__(self, processors: list[FrameProcessor] | None = None) -> None:
        self.processors: list[FrameProcessor] = processors or []
        self._metrics: list[MetricsFrame] = []

    def add(self, processor: FrameProcessor) -> "FramePipeline":
        """Add a processor to the end of the pipeline."""
        self.processors.append(processor)
        return self

    async def push_frame(self, frame: Frame) -> list[Frame]:
        """Push a frame through the pipeline.

        SystemFrames propagate to all processors immediately.
        DataFrames flow sequentially, each processor's output feeding the next.
        """
        if isinstance(frame, InterruptFrame):
            await self._propagate_interrupt(frame)
            return [frame]

        if isinstance(frame, SystemFrame):
            if isinstance(frame, MetricsFrame):
                self._metrics.append(frame)
            return [frame]

        # Normal data flow
        frames: list[Frame] = [frame]
        for processor in self.processors:
            if processor._interrupted:
                logger.debug(
                    "[Pipeline] Skipping %s — interrupted", processor.name
                )
                return []
            next_frames: list[Frame] = []
            for f in frames:
                t0 = time.monotonic()
                result = await processor.process_frame(f)
                dt = (time.monotonic() - t0) * 1000
                if dt > 1.0:
                    self._metrics.append(MetricsFrame(
                        stage=processor.name,
                        duration_ms=dt,
                        metadata={"input_type": type(f).__name__},
                    ))
                next_frames.extend(result)
            frames = next_frames
            if not frames:
                break

        return frames

    async def _propagate_interrupt(self, frame: InterruptFrame) -> None:
        """Deliver interrupt to all processors simultaneously."""
        tasks = [proc.handle_interrupt(frame) for proc in self.processors]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def reset(self) -> None:
        """Reset all processors for a new turn."""
        for proc in self.processors:
            proc.reset_interrupt()
        self._metrics.clear()

    def get_metrics(self) -> list[MetricsFrame]:
        """Return collected metrics since last reset."""
        return list(self._metrics)


# ---------------------------------------------------------------------------
# Cancellation token for async task coordination
# ---------------------------------------------------------------------------

class CancellationToken:
    """Lightweight cooperative cancellation for pipeline stages.

    Usage::

        token = CancellationToken()
        # In producer:
        while not token.is_cancelled:
            await produce_chunk()
        # From consumer:
        token.cancel()
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._event = asyncio.Event()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True
        self._event.set()

    async def wait(self, timeout: float | None = None) -> bool:
        """Wait until cancelled or timeout. Returns True if cancelled."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def reset(self) -> None:
        self._cancelled = False
        self._event.clear()
