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

    def play_thinking(self) -> None: ...

    def speak_error(self) -> None: ...

    def status_snapshot(self) -> dict[str, Any]: ...

    def shutdown(self) -> None: ...


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
