"""Concrete voice provider adapters.

This is the only provider-level module that constructs the low-level voice
stack used by higher layers: audio router, ASR manager, TTS engine, and the
audio frontend facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from askme.ports import (
    ASRProviderPort,
    AudioFrontendPort,
    AudioRouterPort,
    SpeechPlaybackPort,
    TTSProviderPort,
    VoiceIOPort,
    VoiceTurnBridgePort,
)


@dataclass(frozen=True)
class VoiceProviderStack:
    """Provider-owned voice capabilities exposed through ports."""

    audio: AudioFrontendPort
    audio_router: AudioRouterPort
    asr: ASRProviderPort | None = None
    tts: TTSProviderPort | None = None


class EdgeVoiceIO:
    """Provider-owned blocking voice I/O used by edge tool surfaces."""

    def __init__(self, config: dict[str, Any]) -> None:
        from askme.voice.input.asr import ASREngine
        from askme.voice.input.vad import VADEngine
        from askme.voice.output.tts import TTSEngine

        voice_cfg = config.get("voice", config) if isinstance(config, dict) else {}
        self.asr = ASREngine(voice_cfg.get("asr", {}))
        self.vad = VADEngine(voice_cfg.get("vad", {}))
        self.tts = TTSEngine(voice_cfg.get("tts", {}))
        self._config = config

    def listen_once(self) -> str | None:
        """Record one utterance and return recognized text."""
        from askme.voice.input.mic_input import MicInput

        sample_rate: int = self.asr.sample_rate
        speech_active = False
        stream = self.asr.create_stream()
        mic = MicInput.from_config(self._config)

        with mic.open() as mic_ctx:
            for _ in range(300):
                samples = mic_ctx.read_chunk()
                samples_int16 = (samples * 32768).astype(np.int16)
                self.vad.accept_waveform(samples_int16)

                if self.vad.is_speech_detected():
                    speech_active = True
                    stream.accept_waveform(sample_rate, samples)
                    while self.asr.is_ready(stream):
                        self.asr.decode_stream(stream)
                elif speech_active:
                    speech_active = False
                    stream.accept_waveform(sample_rate, samples)
                    while self.asr.is_ready(stream):
                        self.asr.decode_stream(stream)

                if self.asr.is_endpoint(stream):
                    text = self.asr.get_result(stream).strip()
                    if text:
                        return text

        return None

    def speak_and_wait(self, text: str) -> None:
        self.tts.speak(text)
        self.tts.start_playback()
        self.tts.wait_done()
        self.tts.stop_playback()

    def shutdown(self) -> None:
        self.tts.shutdown()


def build_audio_frontend(
    config: dict[str, Any],
    *,
    voice_mode: bool = True,
    metrics: Any | None = None,
) -> VoiceProviderStack:
    """Build the concrete audio frontend stack behind voice ports."""
    from askme.voice.orchestration.audio_agent import AudioAgent
    from askme.voice.output.audio_router import AudioRouter

    audio_router = AudioRouter()
    audio = AudioAgent(
        config,
        voice_mode=voice_mode,
        metrics=metrics,
        audio_router=audio_router,
    )
    return VoiceProviderStack(
        audio=audio,
        audio_router=audio_router,
        asr=getattr(audio, "_asr_mgr", None),
        tts=getattr(audio, "tts", None),
    )


def build_tts_provider(config: dict[str, Any]) -> TTSProviderPort:
    """Build only the configured TTS provider."""
    from askme.voice.output.tts import TTSEngine

    cfg = config if isinstance(config, dict) else {}
    voice_cfg = cfg.get("voice", cfg) if isinstance(cfg.get("voice", cfg), dict) else {}
    tts_cfg = voice_cfg.get("tts", voice_cfg) if isinstance(voice_cfg, dict) else {}
    return TTSEngine(tts_cfg if isinstance(tts_cfg, dict) else {})


def build_voice_runtime_bridge(config: dict[str, Any]) -> VoiceTurnBridgePort:
    """Build the runtime voice-turn bridge behind the gateway contract."""
    from askme.providers.voice_runtime import VoiceRuntimeBridge

    return VoiceRuntimeBridge(config)


def build_edge_voice_io(config: dict[str, Any]) -> VoiceIOPort:
    """Build blocking voice I/O for edge tools behind a stable port."""
    return EdgeVoiceIO(config)


def resolve_voice_profile_id(profile_id: str) -> str:
    """Resolve product voice labels behind the voice provider boundary."""
    from askme.voice.output.voice_profiles import resolve_voice_profile_id as resolve

    return resolve(profile_id)


def build_speech_playback(
    config: dict[str, Any],
    *,
    audio: AudioFrontendPort,
) -> SpeechPlaybackPort:
    """Build the product playback coordinator for this edge robot only."""
    from askme.voice.playback.service import SpeechPlaybackService

    cfg = config if isinstance(config, dict) else {}
    voice_cfg = cfg.get("voice") if isinstance(cfg.get("voice"), dict) else {}
    playback_cfg = (
        voice_cfg.get("playback")
        if isinstance(voice_cfg.get("playback"), dict)
        else {}
    )
    ota_cfg = cfg.get("ota") if isinstance(cfg.get("ota"), dict) else {}
    ota_device = (
        ota_cfg.get("device")
        if isinstance(ota_cfg.get("device"), dict)
        else {}
    )
    field_cfg = (
        cfg.get("field_operations")
        if isinstance(cfg.get("field_operations"), dict)
        else {}
    )
    cloud_asr = (
        voice_cfg.get("cloud_asr")
        if isinstance(voice_cfg.get("cloud_asr"), dict)
        else {}
    )
    robot_id = str(
        playback_cfg.get("robot_id")
        or field_cfg.get("robot_id")
        or ota_device.get("robot_id")
        or ""
    ).strip()
    device_id = str(
        playback_cfg.get("device_id")
        or ota_device.get("robot_id")
        or cloud_asr.get("device_id")
        or robot_id
    ).strip()
    site_id = str(
        playback_cfg.get("site_id")
        or ota_device.get("site_id")
        or ""
    ).strip()
    tts_cfg = voice_cfg.get("tts")
    if not isinstance(tts_cfg, dict):
        tts_cfg = {}
    raw_profiles = tts_cfg.get("voice_profiles")
    allowed_profiles = set(raw_profiles) if isinstance(raw_profiles, dict) else set()
    return SpeechPlaybackService(
        audio=audio,
        robot_id=robot_id,
        device_id=device_id,
        site_id=site_id,
        max_queue_size=int(playback_cfg.get("max_queue_size", 32)),
        max_text_chars=int(playback_cfg.get("max_text_chars", 500)),
        artifact_dir=str(playback_cfg.get("artifact_dir") or "artifacts/voice/playback"),
        allowed_voice_profiles=allowed_profiles,
        ledger_path=str(playback_cfg.get("ledger_path") or "data/voice/playback_ledger.json"),
        max_history=int(playback_cfg.get("max_history", 500)),
    )
