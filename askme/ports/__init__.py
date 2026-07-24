"""Application ports for hardware and provider boundaries."""

from __future__ import annotations

from askme.ports.arm_control import ArmControlPort
from askme.ports.led import LedBridgePort, LedControllerPort
from askme.ports.navigation import NavigationPort
from askme.ports.perception import (
    ChangeMonitorPort,
    InteractionPerceptionPort,
    SceneIntelligencePort,
    VisionPort,
)
from askme.ports.robot_control import RobotControlPort
from askme.ports.safety import SafetyPort
from askme.ports.speech_playback import (
    PlaybackTarget,
    SpeechAudioArtifactFile,
    SpeechActor,
    SpeechDelivery,
    SpeechPlaybackError,
    SpeechPlaybackJob,
    SpeechPlaybackPort,
    SpeechPlaybackRequest,
    SpeechPlaybackState,
    SpeechPlaybackTimestamps,
    SpeechPriority,
)
from askme.ports.spatial_memory import TemporalMemoryPort
from askme.ports.voice import (
    ASRProviderPort,
    AudioFrontendPort,
    AudioRouterPort,
    TTSProviderPort,
    VoiceIOPort,
    VoiceTurnBridgePort,
)

__all__ = [
    "ArmControlPort",
    "ASRProviderPort",
    "AudioFrontendPort",
    "AudioRouterPort",
    "ChangeMonitorPort",
    "InteractionPerceptionPort",
    "LedBridgePort",
    "LedControllerPort",
    "NavigationPort",
    "PlaybackTarget",
    "SpeechAudioArtifactFile",
    "RobotControlPort",
    "SafetyPort",
    "SceneIntelligencePort",
    "SpeechActor",
    "SpeechDelivery",
    "SpeechPlaybackError",
    "SpeechPlaybackJob",
    "SpeechPlaybackPort",
    "SpeechPlaybackRequest",
    "SpeechPlaybackState",
    "SpeechPlaybackTimestamps",
    "SpeechPriority",
    "TemporalMemoryPort",
    "TTSProviderPort",
    "VisionPort",
    "VoiceIOPort",
    "VoiceTurnBridgePort",
]
