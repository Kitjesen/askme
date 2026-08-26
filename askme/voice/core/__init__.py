"""Core voice contracts and pure helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "AskmeEdgeVoiceContract": ("askme.voice.core.generated_contracts", "AskmeEdgeVoiceContract"),
    "EndpointSpec": ("askme.voice.core.generated_contracts", "EndpointSpec"),
    "InterruptDecision": ("askme.voice.core.media_contracts", "InterruptDecision"),
    "InterruptRequest": ("askme.voice.core.media_contracts", "InterruptRequest"),
    "JsonlVoiceTimelineStore": (
        "askme.voice.core.turn_timeline",
        "JsonlVoiceTimelineStore",
    ),
    "MediaSession": ("askme.voice.core.media_contracts", "MediaSession"),
    "MemoryVoiceTimelineStore": (
        "askme.voice.core.turn_timeline",
        "MemoryVoiceTimelineStore",
    ),
    "PunctuationRestorer": ("askme.voice.core.punctuation", "PunctuationRestorer"),
    "StreamSplitter": ("askme.voice.core.stream_splitter", "StreamSplitter"),
    "TimelineClockError": ("askme.voice.core.turn_timeline", "TimelineClockError"),
    "TimelineConflict": ("askme.voice.core.turn_timeline", "TimelineConflict"),
    "TimelineQuery": ("askme.voice.core.turn_timeline", "TimelineQuery"),
    "TimelineRecordReceipt": (
        "askme.voice.core.turn_timeline",
        "TimelineRecordReceipt",
    ),
    "TimelineRecordStatus": (
        "askme.voice.core.turn_timeline",
        "TimelineRecordStatus",
    ),
    "TimelineStoreError": ("askme.voice.core.turn_timeline", "TimelineStoreError"),
    "TurnDetector": ("askme.voice.core.media_contracts", "TurnDetector"),
    "VoiceTimelineEventInput": (
        "askme.voice.core.turn_timeline",
        "VoiceTimelineEventInput",
    ),
    "VoiceTimelineRecord": ("askme.voice.core.turn_timeline", "VoiceTimelineRecord"),
    "VoiceTimelineScope": ("askme.voice.core.turn_timeline", "VoiceTimelineScope"),
    "VoiceTimelineSnapshot": (
        "askme.voice.core.turn_timeline",
        "VoiceTimelineSnapshot",
    ),
    "VoiceTimelineStage": ("askme.voice.core.turn_timeline", "VoiceTimelineStage"),
    "VoiceGateway": ("askme.voice.core.media_contracts", "VoiceGateway"),
    "VoiceMediaFrame": ("askme.voice.core.media_contracts", "VoiceMediaFrame"),
    "VoiceMediaStatus": ("askme.voice.core.media_contracts", "VoiceMediaStatus"),
    "VoiceTraceStage": ("askme.voice.core.turn_trace", "VoiceTraceStage"),
    "VoiceTurnEvent": ("askme.voice.core.media_contracts", "VoiceTurnEvent"),
    "VoiceTurnEventType": ("askme.voice.core.media_contracts", "VoiceTurnEventType"),
    "VoiceTurnTimeline": ("askme.voice.core.turn_timeline", "VoiceTurnTimeline"),
    "VoiceTurnTrace": ("askme.voice.core.turn_trace", "VoiceTurnTrace"),
    "VoiceTurnTraceRecorder": ("askme.voice.core.turn_trace", "VoiceTurnTraceRecorder"),
    "evaluate_voice_turn_slo": ("askme.voice.core.turn_trace", "evaluate_voice_turn_slo"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
