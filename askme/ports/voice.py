"""Ports for the voice input/output provider boundary."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AudioRouterPort(Protocol):
    """Coordinates input/output audio ownership and classifies audio errors."""

    @property
    def is_output_active(self) -> bool: ...

    def output_session(self) -> AbstractContextManager[None]: ...

    def wait_for_input_ready(self, timeout: float = 10.0) -> bool: ...

    def classify_error(self, exc: BaseException) -> Any: ...


@runtime_checkable
class AudioFrontendPort(Protocol):
    """Unified audio frontend used by runtime, pipeline, tools, and UI bridges."""

    awaiting_confirmation: bool
    tts: Any

    @property
    def state(self) -> Any: ...

    @property
    def is_busy(self) -> bool: ...

    @property
    def is_muted(self) -> bool: ...

    @property
    def is_input_open(self) -> bool: ...

    def start_input(self) -> None: ...

    def stop_input(self) -> None: ...

    def listen_loop(self) -> str | None: ...

    def speak(self, text: str) -> None: ...

    def start_playback(self) -> None: ...

    def stop_playback(self) -> None: ...

    def wait_speaking_done(self, timeout: float = 30.0) -> bool: ...

    async def speak_and_wait(self, text: str) -> None: ...

    async def speak_cached_and_wait(self, text: str, *, cache_key: str) -> bool: ...

    def drain_buffers(self) -> None: ...

    def stop_immediately(self) -> None: ...

    def set_volume(self, value: float) -> float: ...

    def adjust_volume(self, delta: float) -> float: ...

    def set_speed(self, value: float) -> float: ...

    def adjust_speed(self, delta: float) -> float: ...

    def mute(self) -> None: ...

    def unmute(self) -> None: ...

    def acknowledge(self) -> None: ...

    @property
    def processing_feedback_delay_s(self) -> float: ...

    @property
    def processing_feedback_armed(self) -> bool: ...

    def arm_processing_feedback(self, cancel_token: Any | None = None) -> bool: ...

    def cancel_processing_feedback(self) -> None: ...

    def play_thinking(self) -> None: ...

    def speak_error(self) -> None: ...

    def status_snapshot(self) -> dict[str, Any]: ...

    def shutdown(self) -> None: ...


@runtime_checkable
class PlaybackOwnerTokenPort(Protocol):
    """Opaque immutable authority for one physical playback interval."""

    voice_turn_id: str | None
    epoch: int


@runtime_checkable
class TurnOwnedPlaybackPort(Protocol):
    """Optional additive capability for turn-bound local playback."""

    def start_playback(
        self,
        *,
        voice_turn_id: str | None = None,
    ) -> PlaybackOwnerTokenPort | None: ...

    def stop_playback(self, token: PlaybackOwnerTokenPort | None = None) -> None: ...


@runtime_checkable
class RealtimeApprovalPort(Protocol):
    """One provider response admitted by the local turn/safety owner."""

    initial_text: str

    @property
    def completed(self) -> bool: ...

    def wait(self, timeout: float | None = None) -> str: ...


@runtime_checkable
class RealtimePreparedApprovalPort(RealtimeApprovalPort, Protocol):
    """Provider response validated locally but not yet released to playback."""

    generation: int


@runtime_checkable
class RealtimeTwoPhaseVoiceFrontendPort(Protocol):
    """Optional prepare/release capability used by production S2S frontends.

    Kept separate from :class:`RealtimeVoiceFrontendPort` so older test fakes
    and third-party frontends retain their legacy one-step surface.  Production
    frontends implement this protocol so the Conversation Ledger can durably
    begin the Turn before a single provider PCM frame is released.
    """

    def prepare_realtime_general_chat(
        self,
        local_text: str,
        *,
        expected_generation: int = 0,
    ) -> RealtimePreparedApprovalPort | None: ...

    def release_realtime_general_chat(
        self,
        approval: RealtimePreparedApprovalPort,
        *,
        expected_generation: int = 0,
        voice_turn_id: str | None = None,
    ) -> bool: ...


@runtime_checkable
class GenerationBoundRealtimePlaybackPort(Protocol):
    """Optional cross-thread finish/abort authority for provider PCM."""

    def finish_realtime_playback(self, *, expected_generation: int) -> bool: ...

    def abort_realtime_playback(
        self,
        reason: str,
        *,
        expected_generation: int = 0,
    ) -> bool: ...


@runtime_checkable
class RealtimeVoiceFrontendPort(Protocol):
    """Explicit optional S2S surface consumed by the voice turn loop.

    The frontend owns physical playback and provider-session mechanics.  The
    voice loop remains the only layer allowed to admit an ordinary-chat turn.
    """

    last_turn_realtime_generation: int
    last_turn_realtime_baseline_generation: int

    def realtime_general_chat_ready(self) -> bool: ...

    def realtime_capture_active(self) -> bool: ...

    def discard_realtime_turn(
        self,
        reason: str,
        *,
        expected_generation: int = 0,
        after_generation: int = 0,
    ) -> None: ...

    def abort_realtime_playback(self, reason: str) -> None: ...

    def realtime_playback_started(self) -> bool: ...


@runtime_checkable
class ASRProviderPort(Protocol):
    """Speech recognition provider status surface."""

    def status_snapshot(self) -> dict[str, Any]: ...


@runtime_checkable
class TTSProviderPort(Protocol):
    """Speech synthesis provider surface."""

    def speak(self, text: str) -> None: ...

    def start_playback(self) -> None: ...

    def stop_playback(self) -> None: ...

    def wait_done(self, timeout: float = 30.0) -> bool: ...

    def status_snapshot(self) -> dict[str, Any]: ...

    def shutdown(self) -> None: ...


@runtime_checkable
class VoiceTurnBridgePort(Protocol):
    """Runtime bridge contract used by the voice gateway service."""

    def status_snapshot(self) -> dict[str, Any]: ...

    def handle_voice_text(
        self,
        text: str,
        *,
        conversation_session_id: str | None = None,
        session_id: str | None = None,
        operator_id: str | None = None,
        robot_id: str | None = None,
        site_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...

    def handle_text_input(
        self,
        text: str,
        *,
        conversation_session_id: str | None = None,
        session_id: str | None = None,
        operator_id: str | None = None,
        robot_id: str | None = None,
        site_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None: ...


@runtime_checkable
class VoiceIOPort(Protocol):
    """Simple blocking voice I/O contract for edge tools such as MCP."""

    def listen_once(self) -> str | None: ...

    def speak_and_wait(self, text: str) -> None: ...

    def shutdown(self) -> None: ...
