"""Provider-neutral contracts for one bidirectional speech-to-speech session.

The realtime provider owns network protocol details only.  Robot intent,
permission, safety, hardware dispatch, and physical audio devices remain owned
by the existing AskMe layers.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from askme.voice.core.media_contracts import VoiceMediaFrame


class RealtimeVoiceEventType(StrEnum):
    """Stable events emitted by any realtime speech provider."""

    CONNECTION_READY = "connection_ready"
    SESSION_READY = "session_ready"
    INPUT_SPEECH_STARTED = "input_speech_started"
    INPUT_TRANSCRIPT_DELTA = "input_transcript_delta"
    INPUT_TRANSCRIPT_FINAL = "input_transcript_final"
    RESPONSE_STARTED = "response_started"
    OUTPUT_TEXT_DELTA = "output_text_delta"
    OUTPUT_AUDIO = "output_audio"
    RESPONSE_DONE = "response_done"
    USAGE = "usage"
    INTERRUPTED = "interrupted"
    SESSION_CLOSED = "session_closed"
    ERROR = "error"


@dataclass(frozen=True)
class RealtimeVoiceSessionContext:
    """Safe session inputs shared across provider implementations."""

    session_id: str
    dialog_id: str = ""
    bot_name: str = "小算"
    system_role: str = ""
    speaking_style: str = ""
    input_mode: str = "audio"
    input_sample_rate: int = 16_000
    output_sample_rate: int = 24_000
    output_format: str = "pcm_s16le"
    allow_tool_calls: bool = False
    allow_hardware_dispatch: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RealtimeVoiceEvent:
    """Normalized provider event with optional transcript or PCM payload."""

    event_type: RealtimeVoiceEventType
    session_id: str = ""
    generation: int = 0
    provider: str = ""
    provider_event_id: int | None = None
    transcript: str = ""
    text: str = ""
    is_final: bool = False
    audio: VoiceMediaFrame | None = None
    error: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        audio: dict[str, Any] | None = None
        if self.audio is not None:
            audio = {
                "sample_rate": self.audio.sample_rate,
                "channels": self.audio.channels,
                "bytes": len(self.audio.pcm),
                "duration_ms": self.audio.duration_ms,
            }
        return {
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "generation": self.generation,
            "provider": self.provider,
            "provider_event_id": self.provider_event_id,
            "transcript": self.transcript,
            "text": self.text,
            "is_final": self.is_final,
            "audio": audio,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class RealtimeDialogueSession(Protocol):
    """Blocking/thread-safe boundary used by the current local audio loop."""

    def start(self, context: RealtimeVoiceSessionContext) -> bool: ...

    def offer_audio(self, frame: VoiceMediaFrame) -> bool: ...

    def finish_input(self) -> bool: ...

    def interrupt(self, reason: str) -> None: ...

    def next_event(self, timeout: float | None = None) -> RealtimeVoiceEvent | None: ...

    def events(self) -> Iterator[RealtimeVoiceEvent]: ...

    def close(self, reason: str = "shutdown") -> None: ...

    def status_snapshot(self) -> dict[str, Any]: ...
