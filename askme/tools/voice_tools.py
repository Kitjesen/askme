"""
Voice control tools for askme.

These tools expose mute/unmute/stop-speaking operations to the LLM tool-call
path, enabling the assistant to control its own voice I/O programmatically
(e.g. "mute yourself during the demo", "stop talking when I say X").

The fast path for user-initiated voice commands (闭麦, 开麦, 静音) bypasses
these tools entirely — VoiceLoop handles them at zero latency before any LLM
call.  These tools exist so the LLM can also trigger the same operations when
it decides to on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .tool_registry import BaseTool, ToolRegistry

if TYPE_CHECKING:
    from ..voice.audio_agent import AudioAgent


class MuteMicTool(BaseTool):
    """Software-mute: stop responding to voice input until unmuted."""

    name = "mute_mic"
    description = '关闭麦克风监听——助手进入静默模式，不再响应语音指令（说"开麦"恢复）'
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def __init__(self, audio: AudioAgent) -> None:
        self._audio = audio

    def execute(self, **kwargs: Any) -> str:
        self._audio.mute()
        return '已关闭麦克风监听。说"开麦"来重新打开。'


class UnmuteMicTool(BaseTool):
    """Resume voice listening after a mute."""

    name = "unmute_mic"
    description = "重新开启麦克风监听——从静默模式恢复正常语音响应"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def __init__(self, audio: AudioAgent) -> None:
        self._audio = audio

    def execute(self, **kwargs: Any) -> str:
        self._audio.unmute()
        return "已重新开启麦克风监听。"


class StopSpeakingTool(BaseTool):
    """Immediately stop any ongoing TTS playback."""

    name = "stop_speaking"
    description = "立即停止当前语音播放，清空TTS队列"
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    safety_level = "normal"

    def __init__(self, audio: AudioAgent) -> None:
        self._audio = audio

    def execute(self, **kwargs: Any) -> str:
        self._audio.drain_buffers()
        return "已停止语音播放。"


_VOICE_TOOL_CLASSES = [MuteMicTool, UnmuteMicTool, StopSpeakingTool]


def register_voice_tools(registry: ToolRegistry, audio: AudioAgent) -> None:
    """Instantiate and register all voice control tools, injecting the AudioAgent."""
    for tool_cls in _VOICE_TOOL_CLASSES:
        registry.register(tool_cls(audio))  # type: ignore[call-arg]
