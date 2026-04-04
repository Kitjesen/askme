"""Tests for voice control tools — mute/unmute/stop_speaking and registration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from askme.tools.voice_tools import (
    MuteMicTool,
    StopSpeakingTool,
    UnmuteMicTool,
    _VOICE_TOOL_CLASSES,
    register_voice_tools,
)


# ── MuteMicTool ───────────────────────────────────────────────────────────────

class TestMuteMicTool:
    def setup_method(self):
        self.audio = MagicMock()
        self.tool = MuteMicTool(self.audio)

    def test_name(self):
        assert MuteMicTool.name == "mute_mic"

    def test_execute_calls_mute(self):
        self.tool.execute()
        self.audio.mute.assert_called_once()

    def test_execute_returns_confirmation(self):
        result = self.tool.execute()
        assert isinstance(result, str)
        assert len(result) > 0


# ── UnmuteMicTool ─────────────────────────────────────────────────────────────

class TestUnmuteMicTool:
    def setup_method(self):
        self.audio = MagicMock()
        self.tool = UnmuteMicTool(self.audio)

    def test_name(self):
        assert UnmuteMicTool.name == "unmute_mic"

    def test_execute_calls_unmute(self):
        self.tool.execute()
        self.audio.unmute.assert_called_once()

    def test_execute_returns_confirmation(self):
        result = self.tool.execute()
        assert isinstance(result, str)
        assert len(result) > 0


# ── StopSpeakingTool ──────────────────────────────────────────────────────────

class TestStopSpeakingTool:
    def setup_method(self):
        self.audio = MagicMock()
        self.tool = StopSpeakingTool(self.audio)

    def test_name(self):
        assert StopSpeakingTool.name == "stop_speaking"

    def test_execute_drains_buffers(self):
        self.tool.execute()
        self.audio.drain_buffers.assert_called_once()

    def test_execute_returns_confirmation(self):
        result = self.tool.execute()
        assert isinstance(result, str)


# ── register_voice_tools ──────────────────────────────────────────────────────

class TestRegisterVoiceTools:
    def test_registers_all_three_tools(self):
        registry = MagicMock()
        audio = MagicMock()
        register_voice_tools(registry, audio)
        assert registry.register.call_count == len(_VOICE_TOOL_CLASSES)

    def test_all_tools_injected_with_audio(self):
        registry = MagicMock()
        audio = MagicMock()
        register_voice_tools(registry, audio)
        for call in registry.register.call_args_list:
            tool = call[0][0]
            assert tool._audio is audio

    def test_tool_class_list_has_three_entries(self):
        assert len(_VOICE_TOOL_CLASSES) == 3

    def test_all_three_names_registered(self):
        registry = MagicMock()
        audio = MagicMock()
        register_voice_tools(registry, audio)
        names = {call[0][0].name for call in registry.register.call_args_list}
        assert "mute_mic" in names
        assert "unmute_mic" in names
        assert "stop_speaking" in names
