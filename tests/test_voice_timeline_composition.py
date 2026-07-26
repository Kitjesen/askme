from __future__ import annotations

from unittest.mock import MagicMock

from askme.providers.voice import build_audio_frontend
from askme.voice.core import TimelineQuery, VoiceTurnTimeline
from askme.voice.core.turn_trace import VoiceTurnTraceRecorder


def test_audio_frontend_uses_one_in_memory_timeline_for_its_trace_recorder() -> None:
    stack = build_audio_frontend(
        {"voice": {"tts": {"backend": "edge"}}},
        voice_mode=False,
        metrics=MagicMock(),
    )
    try:
        assert isinstance(stack.turn_timeline, VoiceTurnTimeline)
        recorder = getattr(stack.audio, "_turn_traces")
        assert isinstance(recorder, VoiceTurnTraceRecorder)
        assert recorder._timeline is stack.turn_timeline
        assert stack.turn_timeline.snapshot(TimelineQuery(limit=1)).events == ()
    finally:
        stack.audio.shutdown()
