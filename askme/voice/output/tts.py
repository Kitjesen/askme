"""TTS Engine - three backends: local sherpa-onnx, edge-tts, or MiniMax streaming."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from askme.interfaces.tts import TTSBackend
from askme.voice.output.phrase_cache import PhraseAudioCache
from askme.voice.output.voice_profiles import (
    VoiceProfile,
    build_voice_profiles,
    resolve_voice_profile_id,
)

try:
    import sounddevice as sd
except ModuleNotFoundError:
    class _SoundDeviceStub:
        InputStream = None
        class CallbackFlags: pass
        class default:
            device = (None, None)
        @staticmethod
        def play(*args: object, **kwargs: object) -> None: ...
        @staticmethod
        def stop() -> None: ...
        @staticmethod
        def wait() -> None: ...
        @staticmethod
        def query_devices(device: object = None, kind: object = None) -> object:
            return {}
        class OutputStream:
            def __init__(self, *args: object, **kwargs: object) -> None: ...
            def __enter__(self) -> _SoundDeviceStub.OutputStream: return self
            def __exit__(self, *args: object) -> None: ...
    sd = _SoundDeviceStub()  # type: ignore[assignment]

if TYPE_CHECKING:
    from askme.voice.output.audio_router import AudioRouter

logger = logging.getLogger(__name__)


class TTSEngine(TTSBackend):
    """Text-to-speech engine with three backends:

    - **local** (default): sherpa-onnx VITS/MeloTTS — ~0.5-1s latency, no network.
    - **edge**: Microsoft Edge TTS — ~3s latency, requires internet.
    - **minimax**: MiniMax T2A v2 — SSE streaming, ~1s TTFT, incremental playback.

    Config dict expected keys (under voice.tts)::

        backend: str          - "local", "edge", or "minimax" (default "local")
        # Local backend
        model_dir: str        - path to sherpa-onnx TTS model directory
        num_threads: int      - inference threads (default 4)
        speed: float          - speech speed (default 1.0)
        sid: int              - speaker ID (default 0)
        # Edge backend
        voice: str            - Edge TTS voice name (default "zh-CN-YunxiNeural")
        rate: str             - Speed adjustment (default "+0%")
        # MiniMax backend
        minimax_api_key: str  - MiniMax API key
        minimax_tts_url: str  - MiniMax TTS base URL
        minimax_tts_model: str - TTS model name (default "speech-2.8-hd")
        minimax_voice_id: str - Voice ID (default "male-qn-qingse")
        minimax_sample_rate: int - MiniMax output sample rate (default 24000)
        # Common
        sample_rate: int      - playback sample rate (default 24000)
        output_device: int|str - sounddevice output device
    """

    # MiniMax consecutive failure threshold before temporarily disabling it.
    # After _MINIMAX_FAIL_THRESHOLD consecutive failures, the backend is
    # bypassed for _MINIMAX_BACKOFF_SECONDS seconds to avoid per-call timeout.
    _MINIMAX_FAIL_THRESHOLD = 3
    _MINIMAX_BACKOFF_SECONDS = 300.0  # 5 minutes
    _MINIMAX_MIN_STREAM_SAMPLES = 2400
    _SOUND_CUE_SPECS: dict[str, tuple[tuple[float, float], ...]] = {
        "soft_chime": ((659.25, 0.08), (783.99, 0.1)),
        "welcome_chime": ((523.25, 0.07), (659.25, 0.07), (880.0, 0.11)),
        "notice_beep": ((740.0, 0.09),),
        "emergency_tone": ((880.0, 0.08), (0.0, 0.04), (880.0, 0.08), (0.0, 0.04), (988.0, 0.12)),
        "quiet_ping": ((587.33, 0.08),),
        "fault_tone": ((392.0, 0.08), (0.0, 0.035), (330.0, 0.12)),
        "confirm_chime": ((659.25, 0.07), (523.25, 0.09)),
        "brand_chime": ((523.25, 0.06), (659.25, 0.06), (783.99, 0.09)),
    }

    # Regex patterns for cleaning text before TTS
    _RE_EMOJI = re.compile(r'[\U00010000-\U0010ffff]')
    _RE_BOLD = re.compile(r'\*\*(.+?)\*\*')
    _RE_ITALIC = re.compile(r'\*(.+?)\*')
    _RE_CODE = re.compile(r'`(.+?)`')
    _RE_HEADER = re.compile(r'^#+\s*', flags=re.MULTILINE)
    _RE_LIST = re.compile(r'^[-*]\s+', flags=re.MULTILINE)
    _RE_IMG = re.compile(r'!\[.*?\]\(.*?\)')
    _RE_LINK = re.compile(r'\[(.+?)\]\(.*?\)')
    _INTERNAL_TEXT_MARKERS = (
        "[SILENT]",
        "DSML",
        "<TOOL_CALL",
        "TOOL_CALLS>",
    )

    def __init__(
        self,
        config: dict[str, Any],
        *,
        audio_router: AudioRouter | None = None,
        echo_reference: Any | None = None,
    ) -> None:
        self._backend: str = config.get("backend", "local")
        self._fallback_backend: str = str(
            config.get("fallback_backend", "edge")
        ).strip().lower()
        if self._fallback_backend not in {"local", "edge"}:
            logger.warning(
                "Unknown TTS fallback_backend=%s, using edge",
                self._fallback_backend,
            )
            self._fallback_backend = "edge"
        self._sample_rate: int = int(config.get("sample_rate", 24000))
        self._output_device: int | str | None = config.get("output_device")
        self._output_transport: str = str(config.get("output_transport", "auto")).lower()
        self._tts_text_coalesce_seconds: float = max(
            0.0, float(config.get("text_coalesce_seconds", 0.0))
        )
        self._tts_text_coalesce_max_chars: int = max(
            1, int(config.get("text_coalesce_max_chars", 160))
        )
        self._phrase_cache = PhraseAudioCache(
            config.get("phrase_cache_dir", "~/.cache/askme/voice_phrases"),
            enabled=bool(config.get("phrase_cache_enabled", True)),
        )
        self._phrase_prime_lock = threading.Lock()

        # Local backend config
        self._model_dir: str = config.get("model_dir", "models/tts/vits-melo-tts-zh_en")
        self._num_threads: int = int(config.get("num_threads", 4))
        self._speed: float = float(config.get("speed", 1.0))
        self._sid: int = int(config.get("sid", 0))

        # Edge backend config
        self._voice: str = config.get("voice", "zh-CN-YunxiNeural")
        self._rate: str = str(config.get("rate", "+0%"))

        # MiniMax backend config
        self._minimax_api_key: str = config.get("minimax_api_key", "")
        self._minimax_tts_url: str = config.get("minimax_tts_url", "https://api.minimax.chat/v1")
        self._minimax_tts_ws_url: str = config.get(
            "minimax_tts_ws_url", "wss://api.minimax.io/ws/v1/t2a_v2"
        )
        self._minimax_tts_transport: str = str(
            config.get("minimax_tts_transport", "sse")
        ).lower()
        self._minimax_tts_model: str = config.get("minimax_tts_model", "speech-2.8-hd")
        self._minimax_voice_id: str = config.get("minimax_voice_id", "male-qn-qingse")
        self._minimax_sample_rate: int = int(config.get("minimax_sample_rate", 24000))
        self._minimax_bitrate: int = int(config.get("minimax_bitrate", 128000))
        self._minimax_audio_format: str = str(config.get("minimax_audio_format", "pcm"))
        # Voice tuning: speed (0.5-2.0), vol (0-10), pitch (-12 to 12 semitones)
        self._minimax_speed: float = float(config.get("minimax_speed", 1.0))
        self._minimax_vol: float = min(10.0, max(0.0, float(config.get("minimax_vol", 1.0))))
        self._minimax_pitch: int = int(config.get("minimax_pitch", 0))
        # Emotion: "" (auto), happy, sad, angry, fearful, disgusted, surprised, calm
        self._minimax_emotion: str = config.get("minimax_emotion", "")
        self._voice_profile_cues_enabled = bool(config.get("voice_profile_cues_enabled", True))
        self._voice_profiles: dict[str, VoiceProfile] = build_voice_profiles(
            config,
            default_voice_id=self._minimax_voice_id,
        )
        state_path = config.get("voice_profile_state_path") or os.getenv(
            "ASKME_VOICE_PROFILE_STATE_PATH"
        )
        self._voice_profile_state_path: Path | None = Path(state_path) if state_path else None
        self._voice_profile_persistence_error: str | None = None
        self._active_voice_profile_id: str = str(
            config.get("voice_profile", "patrol_default")
        )
        persisted_profile_id = self._load_persisted_voice_profile_id()
        if persisted_profile_id:
            self._active_voice_profile_id = persisted_profile_id
        if self._active_voice_profile_id in self._voice_profiles:
            self._apply_voice_profile(self._voice_profiles[self._active_voice_profile_id])
        self._minimax_leading_silence_preserve_seconds: float = float(
            config.get("minimax_leading_silence_preserve_seconds", 0.16)
        )
        self._minimax_onset_threshold: float = float(
            config.get("minimax_onset_threshold", 0.0005)
        )
        self._output_tail_silence_seconds: float = max(
            0.0,
            min(2.0, float(config.get("output_tail_silence_seconds", 0.0))),
        )
        self._aplay_start_buffer_seconds: float = max(
            0.0,
            min(10.0, float(config.get("aplay_start_buffer_seconds", 0.0))),
        )
        self._aplay_wait_for_synthesis_complete: bool = bool(
            config.get("aplay_wait_for_synthesis_complete", False)
        )
        self._aplay_drain_timeout_seconds: float = max(
            1.0,
            min(120.0, float(config.get("aplay_drain_timeout_seconds", 30.0))),
        )

        # Consecutive failure tracking for MiniMax auto-disable
        self._minimax_fail_count: int = 0
        self._minimax_disabled_until: float = 0.0  # monotonic time

        # Volume multiplier applied to all PCM output (0.0–1.0)
        self._volume: float = float(config.get("volume", 1.0))
        if self._active_voice_profile_id in self._voice_profiles:
            self._apply_voice_profile(self._voice_profiles[self._active_voice_profile_id])

        # Queues and buffers
        self.tts_text_queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
        self.tts_buffer: deque[np.ndarray] = deque()
        self._buffer_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._generation = 0

        # Playback state — guarded by _playback_lock
        self._playback_lock = threading.Lock()
        self._is_playing = False
        self._playback_thread: threading.Thread | None = None
        # aplay subprocess (Linux only); non-None while a chunk is being played
        self._aplay_proc: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._aplay_lock = threading.Lock()  # guards _aplay_proc r/w across threads
        self._aplay_bin: str | None = shutil.which("aplay")
        self._playback_busy = threading.Event()
        self._usb_audio_proc: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._usb_audio_stream_proc: subprocess.Popen | None = None  # type: ignore[type-arg]
        self._usb_audio_stream_ready_at: float = 0.0
        self._usb_audio_lock = threading.Lock()
        self._usb_audio_session_lock = threading.Lock()
        self._usb_audio_binary: str | None = config.get("usb_audio_binary")
        self._usb_audio_source: str | None = config.get("usb_audio_source")
        self._usb_direct_persistent_stream: bool = bool(
            config.get("usb_direct_persistent_stream", False)
        )
        self._usb_direct_trust_persistent_warm_state: bool = bool(
            config.get("usb_direct_trust_persistent_warm_state", False)
        )
        self._usb_direct_stream_start_grace_seconds: float = float(
            config.get("usb_direct_stream_start_grace_seconds", 0.08)
        )
        self._usb_direct_preroll_seconds: float = float(
            config.get("usb_direct_preroll_seconds", 1.5)
        )
        self._usb_direct_stream_guard_seconds: float = float(
            config.get("usb_direct_stream_guard_seconds", 0.25)
        )
        self._usb_direct_speech_leadin_seconds: float = float(
            config.get("usb_direct_speech_leadin_seconds", self._usb_direct_preroll_seconds)
        )
        self._usb_direct_speech_warm_leadin_seconds: float = float(
            config.get("usb_direct_speech_warm_leadin_seconds", 0.12)
        )
        self._usb_direct_speech_gain: float = float(
            config.get("usb_direct_speech_gain", 1.0)
        )
        self._usb_direct_speech_wake_signal_seconds: float = float(
            config.get("usb_direct_speech_wake_signal_seconds", 0.0)
        )
        self._usb_direct_speech_wake_signal_gain: float = float(
            config.get("usb_direct_speech_wake_signal_gain", 0.08)
        )
        self._usb_direct_speech_wake_noise_gain: float = float(
            config.get("usb_direct_speech_wake_noise_gain", 260.0 / 32767.0)
        )
        self._usb_direct_speech_wake_signal_hz: float = float(
            config.get("usb_direct_speech_wake_signal_hz", 880.0)
        )
        self._usb_direct_speech_wake_gap_seconds: float = float(
            config.get("usb_direct_speech_wake_gap_seconds", 0.04)
        )
        self._usb_direct_speech_onset_cushion_seconds: float = float(
            config.get("usb_direct_speech_onset_cushion_seconds", 0.0)
        )
        self._usb_direct_speech_onset_cushion_gain: float = float(
            config.get("usb_direct_speech_onset_cushion_gain", 0.18)
        )
        self._usb_direct_speech_onset_gap_seconds: float = float(
            config.get("usb_direct_speech_onset_gap_seconds", 0.08)
        )
        self._usb_direct_background_prewarm: bool = bool(
            config.get("usb_direct_background_prewarm", False)
        )
        self._usb_direct_coalesce_timeout: float = float(
            config.get("usb_direct_coalesce_timeout", 8.0)
        )
        self._usb_direct_stream_drain_grace_seconds: float = float(
            config.get("usb_direct_stream_drain_grace_seconds", 0.35)
        )
        self._usb_audio_build_failed = False
        # Immediate stop flag: checked by _playback_loop to abort mid-chunk
        self._stop_requested = threading.Event()
        # After a barge-in, streamed LLM text may continue arriving briefly.
        # Drop it until the next explicit playback lifecycle starts.
        self._discard_text_until_restart = threading.Event()
        # Pre-roll warm state: skip 400ms pre-roll when DAC was recently active
        self._last_aplay_close: float = 0.0  # monotonic time of last aplay close
        _PREROLL_WARM_WINDOW = 5.0  # seconds — DAC stays warm after close
        self._preroll_warm_window = _PREROLL_WARM_WINDOW
        self._usb_direct_warming = threading.Event()

        # AudioRouter for device ownership coordination (optional)
        self._audio_router: AudioRouter | None = audio_router
        # Optional microphone-side software AEC reference.
        self._echo_reference = echo_reference

        # Local TTS engine (lazy init)
        self._local_tts: Any | None = None
        self._local_sample_rate: int = 0

        # Auto-detect backend
        if self._backend == "minimax" and not self._minimax_api_key:
            if self._fallback_backend == "local" and os.path.isdir(self._model_dir):
                logger.warning("MiniMax TTS: no API key configured, using local fallback")
                self._backend = "local"
            else:
                logger.warning("MiniMax TTS: no API key configured, falling back to edge-tts")
                self._backend = "edge"
        if self._backend == "local" and not os.path.isdir(self._model_dir):
            logger.warning("Local TTS model not found at %s, falling back to edge-tts", self._model_dir)
            self._backend = "edge"
        if self._output_transport not in {"auto", "aplay", "sounddevice", "usb_direct"}:
            logger.warning("Unknown TTS output_transport=%s, using auto", self._output_transport)
            self._output_transport = "auto"
        if self._minimax_tts_transport not in {"sse", "websocket", "ws"}:
            logger.warning(
                "Unknown minimax_tts_transport=%s, using sse",
                self._minimax_tts_transport,
            )
            self._minimax_tts_transport = "sse"

        if self._backend == "local":
            self._init_local_tts()

        logger.info("TTS backend: %s", self._backend)
        self._log_output_devices()

        # Start TTS worker thread
        self._worker_thread = threading.Thread(target=self._tts_loop, daemon=True)
        self._worker_thread.start()

    # ------------------------------------------------------------------
    # Output device discovery
    # ------------------------------------------------------------------

    def _log_output_devices(self) -> None:
        """Log available output devices so the user can pick the right one."""
        try:
            import sounddevice as _sd

            devices = _sd.query_devices()
            default_out = _sd.default.device[1]
            output_lines: list[str] = []
            for i, dev in enumerate(devices):
                if dev["max_output_channels"] > 0:
                    marker = " << SELECTED" if (
                        self._output_device == i
                        or (self._output_device is None and i == default_out)
                    ) else ""
                    output_lines.append(
                        f"  [{i}] {dev['name']} (ch={dev['max_output_channels']}){marker}"
                    )

            selected_label = (
                f"index {self._output_device}"
                if self._output_device is not None
                else f"system default [{default_out}]"
            )
            logger.info(
                "TTS output device: %s\nAvailable output devices:\n%s",
                selected_label,
                "\n".join(output_lines),
            )
        except Exception as exc:
            logger.debug("Could not enumerate output devices: %s", exc)

    # ------------------------------------------------------------------
    # Local TTS init
    # ------------------------------------------------------------------

    def _init_local_tts(self) -> None:
        """Initialize sherpa-onnx OfflineTts for local synthesis."""
        try:
            import sherpa_onnx

            model_dir = self._model_dir

            # Detect model file
            model_file = os.path.join(model_dir, "model.onnx")
            if not os.path.exists(model_file):
                # Try aishell3 naming
                for name in ("vits-aishell3.onnx", "vits-aishell3.int8.onnx"):
                    candidate = os.path.join(model_dir, name)
                    if os.path.exists(candidate):
                        model_file = candidate
                        break

            # Detect optional files
            lexicon = os.path.join(model_dir, "lexicon.txt")
            tokens = os.path.join(model_dir, "tokens.txt")
            dict_dir = os.path.join(model_dir, "dict")

            # Build rule FSTs list
            rule_fsts = []
            for name in ("date.fst", "number.fst", "phone.fst", "new_heteronym.fst"):
                path = os.path.join(model_dir, name)
                if os.path.exists(path):
                    rule_fsts.append(path)

            rule_fars = []
            for name in ("rule.far",):
                path = os.path.join(model_dir, name)
                if os.path.exists(path):
                    rule_fars.append(path)

            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=model_file,
                        lexicon=lexicon if os.path.exists(lexicon) else "",
                        tokens=tokens,
                        dict_dir=dict_dir if os.path.isdir(dict_dir) else "",
                    ),
                    num_threads=self._num_threads,
                    provider="cpu",
                ),
                rule_fsts=",".join(rule_fsts),
                rule_fars=",".join(rule_fars),
                max_num_sentences=1,
            )

            self._local_tts = sherpa_onnx.OfflineTts(tts_config)

            # Warmup and detect sample rate
            warmup_audio = self._local_tts.generate("测试", sid=self._sid, speed=self._speed)
            self._local_sample_rate = warmup_audio.sample_rate
            logger.info(
                "Local TTS initialized: model=%s, sample_rate=%d",
                os.path.basename(model_file), self._local_sample_rate,
            )

        except Exception as exc:
            logger.warning("Local TTS init failed: %s — falling back to edge-tts", exc)
            self._local_tts = None
            self._backend = "edge"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Strip emoji/markdown from *text* and queue it for TTS generation."""
        if not text:
            return
        if self._discard_text_until_restart.is_set():
            logger.info("TTS interrupted: dropping late streamed text")
            return
        clean = text
        clean = self._RE_EMOJI.sub('', clean)
        clean = self._RE_BOLD.sub(r'\1', clean)
        clean = self._RE_ITALIC.sub(r'\1', clean)
        clean = self._RE_CODE.sub(r'\1', clean)
        clean = self._RE_HEADER.sub('', clean)
        clean = self._RE_LIST.sub('', clean)
        clean = self._RE_IMG.sub('', clean)
        clean = self._RE_LINK.sub(r'\1', clean)
        clean = clean.strip()
        if clean and self._is_speakable_text(clean):
            logger.info("speak queued: %r", clean[:60])
            self.tts_text_queue.put((self._get_generation(), clean))

    def queue_cached_pcm(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        cache_key: str = "",
    ) -> bool:
        """Queue cached mono PCM through the normal cancellable speech path."""

        audio = np.asarray(samples, dtype=np.float32)
        if (
            audio.ndim != 1
            or len(audio) == 0
            or int(sample_rate) <= 0
            or not np.all(np.isfinite(audio))
        ):
            return False

        generation = self._get_generation()
        prepared = self._resample(audio, int(sample_rate), self._sample_rate)
        if len(prepared) == 0:
            return False
        tail_count = int(self._sample_rate * self._output_tail_silence_seconds)

        with self._generation_lock:
            if generation != self._generation:
                return False
            with self._buffer_lock:
                self.tts_buffer.append(prepared.astype(np.float32, copy=True))
                if tail_count > 0:
                    self.tts_buffer.append(np.zeros(tail_count, dtype=np.float32))
        logger.info(
            "TTS phrase cache queued: key=%s samples=%d",
            cache_key or "anonymous",
            len(prepared),
        )
        return True

    def queue_cached_phrase(self, text: str, *, cache_key: str) -> bool:
        """Queue a persisted phrase without invoking any TTS provider."""

        storage_key = self._phrase_cache_storage_key(text, cache_key)
        cached = self._phrase_cache.get(storage_key)
        if cached is None:
            logger.info("TTS phrase cache miss: key=%s", cache_key)
            return False
        return self.queue_cached_pcm(
            cached.samples,
            cached.sample_rate,
            cache_key=storage_key,
        )

    def prime_cached_phrase(self, text: str, *, cache_key: str) -> dict[str, Any]:
        """Synthesize one phrase off-line from playback and persist its PCM."""

        storage_key = self._phrase_cache_storage_key(text, cache_key)
        existing = self._phrase_cache.get(storage_key)
        if existing is not None:
            return {
                "cached": True,
                "created": False,
                "cache_key": storage_key,
                "samples": len(existing.samples),
                "sample_rate": existing.sample_rate,
            }

        with self._phrase_prime_lock:
            if self.is_active() or self.tts_text_queue.unfinished_tasks:
                return {
                    "cached": False,
                    "created": False,
                    "cache_key": storage_key,
                    "reason": "tts_busy",
                }
            with self._buffer_lock:
                if self.tts_buffer:
                    return {
                        "cached": False,
                        "created": False,
                        "cache_key": storage_key,
                        "reason": "audio_buffer_not_empty",
                    }

            generation = self._get_generation()
            self._generate_audio(text, generation)
            with self._buffer_lock:
                chunks = list(self.tts_buffer)
                self.tts_buffer.clear()
            if not chunks:
                return {
                    "cached": False,
                    "created": False,
                    "cache_key": storage_key,
                    "reason": "synthesis_empty",
                }
            audio = np.concatenate(chunks).astype(np.float32, copy=False)
            tail_count = int(self._sample_rate * self._output_tail_silence_seconds)
            if tail_count > 0 and len(audio) > tail_count:
                audio = audio[:-tail_count]
            created = self._phrase_cache.put(storage_key, audio, self._sample_rate)
            return {
                "cached": created,
                "created": created,
                "cache_key": storage_key,
                "samples": len(audio),
                "sample_rate": self._sample_rate,
                "reason": "" if created else "cache_write_failed",
            }

    def _phrase_cache_storage_key(self, text: str, cache_key: str) -> str:
        signature = {
            "cache_key": cache_key,
            "text": text,
            "backend": self._backend,
            "voice_profile": self._active_voice_profile_id,
            "voice": self._voice,
            "rate": self._rate,
            "minimax_voice": self._minimax_voice_id,
            "speed": self._minimax_speed,
            "pitch": self._minimax_pitch,
            "emotion": self._minimax_emotion,
            "sample_rate": self._sample_rate,
        }
        digest = hashlib.sha256(
            json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        return f"{cache_key}-{digest}"

    @staticmethod
    def _is_speakable_text(text: str) -> bool:
        """Return True for normal text and short numeric/CJK utterances."""
        upper = text.upper()
        if any(marker in upper for marker in TTSEngine._INTERNAL_TEXT_MARKERS):
            logger.error("TTS blocked internal protocol text")
            return False
        if len(text) > 1:
            return True
        char = text[0]
        return char.isdigit() or "\u4e00" <= char <= "\u9fff"

    def start_playback(self) -> None:
        """Start the sounddevice output stream in a background thread."""
        with self._playback_lock:
            if self._is_playing:
                return
            self._discard_text_until_restart.clear()
            self._is_playing = True
            self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self._playback_thread.start()

    def stop_playback(self) -> None:
        """Stop playback immediately."""
        with self._playback_lock:
            self._is_playing = False
            thread = self._playback_thread
            self._playback_thread = None
        self._kill_aplay()
        self._kill_usb_audio()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        # stop_immediately() can race with playback-loop shutdown. Do not let
        # an unconsumed stop request cancel the first audio of the next turn.
        self._stop_requested.clear()

    def is_active(self) -> bool:
        """Return True if audio is buffered or playback is in progress."""
        with self._playback_lock:
            playing = self._is_playing
        return playing or self._has_buffered_audio()

    def wait_done(self, timeout: float = 30.0) -> bool:
        """Block until all queued text has been synthesised and played.

        Args:
            timeout: Maximum seconds to wait for playback to finish after
                     synthesis is complete.  Prevents infinite blocking when
                     the audio device or TTS backend is unavailable.
        """
        deadline = time.monotonic() + timeout
        with self.tts_text_queue.all_tasks_done:
            while self.tts_text_queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "wait_done: timed out after %.1fs waiting for TTS synthesis",
                        timeout,
                    )
                    return False
                self.tts_text_queue.all_tasks_done.wait(timeout=min(0.05, remaining))
        while self._has_buffered_audio() or self._playback_busy.is_set():
            if time.monotonic() >= deadline:
                logger.warning("wait_done: timed out after %.1fs waiting for buffer drain", timeout)
                return False
            time.sleep(0.05)
        # Wait for the last chunk to finish playing.
        # aplay: proc.communicate() is synchronous, but _aplay_proc is cleared
        # only after communicate() returns, so poll it.
        while self._aplay_proc is not None:
            if time.monotonic() >= deadline:
                logger.warning("wait_done: timed out after %.1fs waiting for aplay", timeout)
                return False
            time.sleep(0.02)
        while self._usb_audio_proc is not None:
            if time.monotonic() >= deadline:
                logger.warning("wait_done: timed out after %.1fs waiting for MCP01 USB audio", timeout)
                return False
            time.sleep(0.02)
        return True
        # Fallback: wait for any sounddevice stream (non-aplay systems).
        try:
            sd.wait()
        except Exception:
            pass

    def play_feedback_audio(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Play a short non-TTS feedback sound through the active output path.

        Sunrise uses MCP01 USB direct when ALSA does not expose a usable card.
        ACK/thinking chimes should use that same path so the user hears prompt
        feedback during LLM/TTS latency gaps.
        """
        if not self._should_use_usb_direct():
            return False

        samples = np.asarray(audio, dtype=np.float32)
        if len(samples) == 0:
            return True
        if sample_rate != self._sample_rate:
            samples = self._resample(samples, sample_rate, self._sample_rate)
        if self._volume != 1.0:
            samples = samples * self._volume
            np.clip(samples, -1.0, 1.0, out=samples)

        self._playback_busy.set()
        try:
            return self._play_chunk_usb_direct_with_preroll(samples)
        finally:
            self._playback_busy.clear()

    def drain_buffers(self) -> None:
        """Clear all pending TTS text and audio buffers."""
        self._advance_generation()
        while not self.tts_text_queue.empty():
            try:
                self.tts_text_queue.get_nowait()
                self.tts_text_queue.task_done()
            except queue.Empty:
                break
        self._clear_audio_buffer()
        self._kill_aplay()
        self._kill_usb_audio()

    def stop_immediately(self) -> None:
        """Signal the playback loop to abort the current chunk immediately.

        Unlike drain_buffers() which clears pending queues, this also
        interrupts the chunk currently being written to aplay/sounddevice.
        The _playback_loop checks _stop_requested and exits the current
        chunk early.  The flag is auto-cleared when _playback_loop resumes.
        """
        self._discard_text_until_restart.set()
        self._stop_requested.set()
        self._kill_aplay()
        self._kill_usb_audio()

    def shutdown(self) -> None:
        """Signal the worker thread to exit and stop playback."""
        self.drain_buffers()
        self.tts_text_queue.put(None)
        self.stop_playback()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    @property
    def backend(self) -> str:
        """Return the active TTS backend name."""
        return self._backend

    def status_snapshot(self) -> dict[str, Any]:
        """Return non-secret TTS provider and playback status for health/UI."""
        with self._playback_lock:
            playing = self._is_playing
        with self._buffer_lock:
            buffered_chunks = len(self.tts_buffer)
            buffered_samples = int(sum(len(chunk) for chunk in self.tts_buffer))
        minimax_disabled_remaining_s = max(
            0.0, self._minimax_disabled_until - time.monotonic()
        )
        return {
            "backend": self._backend,
            "fallback_backend": self._fallback_backend,
            "output_transport": self._output_transport,
            "sample_rate": self._sample_rate,
            "is_playing": playing,
            "playback_busy": self._playback_busy.is_set(),
            "stop_requested": self._stop_requested.is_set(),
            "queued_text_items": self.tts_text_queue.qsize(),
            "buffered_chunks": buffered_chunks,
            "buffered_samples": buffered_samples,
            "minimax": {
                "configured": bool(self._minimax_api_key),
                "transport": self._minimax_tts_transport,
                "model": self._minimax_tts_model,
                "voice_id": self._minimax_voice_id,
                "active_profile": self._active_voice_profile_id,
                "active_profile_settings": self._active_voice_profile_settings(),
                "profile_persistence_status": self._voice_profile_persistence_status(),
                "profile_persistence_error": self._voice_profile_persistence_error,
                "profiles": [profile.to_dict() for profile in self._voice_profiles.values()],
                "sample_rate": self._minimax_sample_rate,
                "format": self._minimax_audio_format,
                "url": self._minimax_tts_url,
                "ws_url": self._minimax_tts_ws_url,
                "failure_count": self._minimax_fail_count,
                "disabled_remaining_s": round(minimax_disabled_remaining_s, 2),
            },
        }

    def voice_profiles_payload(self) -> dict[str, Any]:
        """Return selectable voice styles for the product UI."""

        return {
            "active_profile": self._active_voice_profile_id,
            "active_profile_settings": self._active_voice_profile_settings(),
            "persistence_status": self._voice_profile_persistence_status(),
            "persistence_error": self._voice_profile_persistence_error,
            "sound_cues_enabled": self._voice_profile_cues_enabled,
            "available_sound_cues": sorted(self._SOUND_CUE_SPECS),
            "profiles": [profile.to_dict() for profile in self._voice_profiles.values()],
        }

    def set_voice_profile_payload(self, body: dict[str, Any]) -> dict[str, Any]:
        """Apply a voice style and optionally play its sample sentence."""

        requested_profile_id = str(body.get("profile_id") or body.get("id") or "").strip()
        profile_id = resolve_voice_profile_id(requested_profile_id)
        if not profile_id:
            return {"updated": False, "reason": "missing_profile_id"}
        profile = self._voice_profiles.get(profile_id)
        if profile is None:
            return {
                "updated": False,
                "reason": "unknown_profile",
                "profile_id": requested_profile_id or profile_id,
                "resolved_profile_id": profile_id,
                "available": list(self._voice_profiles),
            }
        self._apply_voice_profile(profile)
        self._persist_voice_profile(profile)
        if body.get("speak_sample"):
            sound_cue = self.queue_sound_cue(profile.cue)
            self.speak(str(body.get("sample_text") or profile.sample_text))
            self.start_playback()
        else:
            sound_cue = {"queued": False, "cue": profile.cue, "reason": "speak_sample_disabled"}
        return {
            "updated": True,
            "active_profile": self._active_voice_profile_id,
            "requested_profile": requested_profile_id,
            "resolved_profile": profile_id,
            "profile": profile.to_dict(),
            "applied_settings": self._active_voice_profile_settings(),
            "sound_cue": sound_cue,
            "persistence_status": self._voice_profile_persistence_status(),
            "persistence_error": self._voice_profile_persistence_error,
        }

    def queue_sound_cue(self, cue: str) -> dict[str, Any]:
        """Queue a short local PCM cue before or between spoken responses."""

        cue_id = str(cue or "").strip()
        if not cue_id or cue_id == "none":
            return {"queued": False, "cue": cue_id or "none", "reason": "no_cue"}
        if not self._voice_profile_cues_enabled:
            return {"queued": False, "cue": cue_id, "reason": "cues_disabled"}
        samples = self._build_sound_cue(cue_id)
        if samples is None or len(samples) == 0:
            return {"queued": False, "cue": cue_id, "reason": "unknown_cue"}
        with self._buffer_lock:
            self.tts_buffer.append(samples)
        return {
            "queued": True,
            "cue": cue_id,
            "sample_rate": self._sample_rate,
            "samples": int(len(samples)),
            "duration_s": round(len(samples) / float(self._sample_rate), 3),
        }

    def _build_sound_cue(self, cue: str) -> np.ndarray | None:
        spec = self._SOUND_CUE_SPECS.get(cue)
        if spec is None:
            return None
        chunks: list[np.ndarray] = []
        for freq, duration_s in spec:
            count = max(1, int(self._sample_rate * duration_s))
            if freq <= 0:
                chunks.append(np.zeros(count, dtype=np.float32))
                continue
            t = np.linspace(0.0, duration_s, count, endpoint=False, dtype=np.float32)
            tone = np.sin(2.0 * np.pi * float(freq) * t).astype(np.float32)
            envelope = np.sin(np.linspace(0.0, np.pi, count, dtype=np.float32))
            chunks.append((tone * envelope * 0.22).astype(np.float32))
        gap = np.zeros(max(1, int(self._sample_rate * 0.025)), dtype=np.float32)
        interleaved: list[np.ndarray] = []
        for chunk in chunks:
            interleaved.append(chunk)
            interleaved.append(gap)
        if not interleaved:
            return None
        samples = np.concatenate(interleaved).astype(np.float32)
        np.clip(samples, -0.35, 0.35, out=samples)
        return samples

    def _apply_voice_profile(self, profile: VoiceProfile) -> None:
        self._active_voice_profile_id = profile.profile_id
        self._minimax_voice_id = profile.voice_id
        self.set_speed(profile.speed)
        self.set_volume(profile.volume)
        self._minimax_pitch = int(profile.pitch)
        self._minimax_emotion = profile.emotion

    def _active_voice_profile_settings(self) -> dict[str, Any]:
        profile = self._voice_profiles.get(self._active_voice_profile_id)
        if profile is None:
            return {
                "profile_id": self._active_voice_profile_id,
                "voice_id": self._minimax_voice_id,
                "speed": self.speed,
                "volume": self.volume,
                "pitch": self._minimax_pitch,
                "emotion": self._minimax_emotion,
                "known_profile": False,
            }
        return {
            "profile_id": profile.profile_id,
            "label": profile.label,
            "use_case": profile.use_case,
            "voice_id": profile.voice_id,
            "speed": profile.speed,
            "volume": profile.volume,
            "pitch": profile.pitch,
            "emotion": profile.emotion,
            "category": profile.category,
            "cue": profile.cue,
            "known_profile": True,
        }

    def _voice_profile_persistence_status(self) -> str:
        if self._voice_profile_state_path is None:
            return "session_only"
        if self._voice_profile_persistence_error:
            return "persistence_failed"
        return "persistent"

    def _load_persisted_voice_profile_id(self) -> str | None:
        path = self._voice_profile_state_path
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._voice_profile_persistence_error = f"load_failed:{exc.__class__.__name__}"
            return None
        profile_id = resolve_voice_profile_id(str(payload.get("active_profile") or ""))
        if profile_id in self._voice_profiles:
            self._voice_profile_persistence_error = None
            return profile_id
        self._voice_profile_persistence_error = "load_failed:unknown_profile"
        return None

    def _persist_voice_profile(self, profile: VoiceProfile) -> None:
        path = self._voice_profile_state_path
        if path is None:
            self._voice_profile_persistence_error = None
            return
        payload = {
            "active_profile": profile.profile_id,
            "label": profile.label,
            "voice_id": profile.voice_id,
            "speed": profile.speed,
            "volume": profile.volume,
            "pitch": profile.pitch,
            "emotion": profile.emotion,
            "updated_at": time.time(),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f".{path.name}.tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(path)
            self._voice_profile_persistence_error = None
        except OSError as exc:
            self._voice_profile_persistence_error = f"save_failed:{exc.__class__.__name__}"

    # ------------------------------------------------------------------
    # Runtime volume / speed control
    # ------------------------------------------------------------------

    _VOLUME_MIN = 0.05
    _VOLUME_MAX = 1.0
    _SPEED_MIN = 0.5
    _SPEED_MAX = 2.0

    def set_volume(self, value: float) -> float:
        """Set PCM output volume (0.05–1.0). Returns the new value."""
        self._volume = max(self._VOLUME_MIN, min(self._VOLUME_MAX, float(value)))
        return self._volume

    def adjust_volume(self, delta: float) -> float:
        """Adjust volume by delta (+/-). Returns the new value."""
        return self.set_volume(self._volume + delta)

    def set_speed(self, value: float) -> float:
        """Set speech speed across all backends (0.5–2.0). Returns new value."""
        speed = max(self._SPEED_MIN, min(self._SPEED_MAX, float(value)))
        self._speed = speed
        self._minimax_speed = speed
        # edge-tts rate is a percent string, e.g. "+20%" or "-30%"
        pct = round((speed - 1.0) * 100)
        self._rate = f"+{pct}%" if pct >= 0 else f"{pct}%"
        return speed

    def adjust_speed(self, delta: float) -> float:
        """Adjust speed by delta (+/-). Returns new value."""
        return self.set_speed(self._speed + delta)

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def speed(self) -> float:
        return self._speed

    # ------------------------------------------------------------------
    # Sounddevice callback
    # ------------------------------------------------------------------

    def play_audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for ``sd.OutputStream``."""
        if status:
            logger.debug("Playback status: %s", status)

        n = 0
        with self._buffer_lock:
            while n < frames and self.tts_buffer:
                remaining = frames - n
                current_chunk = self.tts_buffer[0]
                k = current_chunk.shape[0]

                if remaining <= k:
                    outdata[n:, 0] = current_chunk[:remaining]
                    if remaining == k:
                        self.tts_buffer.popleft()
                    else:
                        self.tts_buffer[0] = current_chunk[remaining:]
                    n = frames
                    break

                outdata[n : n + k, 0] = self.tts_buffer.popleft()
                n += k

        if n < frames:
            outdata[n:, 0] = 0
        if self._volume != 1.0:
            outdata[:, 0] *= self._volume
            np.clip(outdata[:, 0], -1.0, 1.0, out=outdata[:, 0])
        self._publish_echo_reference(outdata[:, 0])

    def _publish_echo_reference(self, samples: np.ndarray) -> None:
        """Publish actual mono PCM output to the microphone AEC, if enabled."""
        if self._echo_reference is None:
            return
        try:
            self._echo_reference.push_playback(samples, self._sample_rate)
        except Exception as exc:
            logger.debug("AEC playback reference publish failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal — worker thread
    # ------------------------------------------------------------------

    def _tts_loop(self) -> None:
        """Worker thread: consume text items and generate audio."""
        while True:
            item = self.tts_text_queue.get()
            if item is None:
                self.tts_text_queue.task_done()
                break
            generation, text = item
            acknowledgements = 1
            stop_after_item = False
            try:
                text, extra_acknowledgements, stop_after_item = self._coalesce_tts_text(
                    generation, text
                )
                acknowledgements += extra_acknowledgements
                self._generate_audio(text, generation)
            except Exception as e:
                logger.error("TTS worker error: %s", e)
            finally:
                for _ in range(acknowledgements):
                    self.tts_text_queue.task_done()
            if stop_after_item:
                break

    def _coalesce_tts_text(
        self,
        generation: int,
        text: str,
    ) -> tuple[str, int, bool]:
        """Merge adjacent fragments from one response before cloud synthesis."""
        wait_seconds = self._tts_text_coalesce_seconds
        if wait_seconds <= 0 or len(text) >= self._tts_text_coalesce_max_chars:
            return text, 0, False

        deadline = time.monotonic() + wait_seconds
        merged = text
        acknowledgements = 0
        stop_after_item = False
        merged_items = 1
        while len(merged) < self._tts_text_coalesce_max_chars:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                next_item = self.tts_text_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if next_item is None:
                acknowledgements += 1
                stop_after_item = True
                break

            next_generation, next_text = next_item
            if next_generation != generation:
                self.tts_text_queue.task_done()
                self.tts_text_queue.put(next_item)
                break

            acknowledgements += 1
            available = self._tts_text_coalesce_max_chars - len(merged)
            if len(next_text) > available:
                self.tts_text_queue.task_done()
                self.tts_text_queue.put(next_item)
                acknowledgements -= 1
                break
            merged += next_text
            merged_items += 1

        if merged_items > 1:
            logger.info("TTS: coalesced %d text fragments", merged_items)
        return merged, acknowledgements, stop_after_item

    def _generate_audio(self, text: str, generation: int) -> None:
        """Dispatch to local, edge, or minimax backend."""
        if not self._is_generation_current(generation):
            logger.debug("TTS: dropping stale request before synthesis")
            return

        logger.info("TTS [%s] generating: %r", self._backend, text[:80])

        if self._backend == "local":
            self._generate_local(text, generation)
        elif self._backend == "minimax":
            # If MiniMax is temporarily disabled due to consecutive failures,
            # skip directly to fallback without attempting the API call.
            if time.monotonic() < self._minimax_disabled_until:
                remaining = self._minimax_disabled_until - time.monotonic()
                logger.info(
                    "TTS: MiniMax temporarily disabled (%.0fs remaining), using fallback",
                    remaining,
                )
                self._use_minimax_fallback(text, generation)
            elif not self._run_async(self._generate_minimax_transport(text, generation)):
                # MiniMax failed — track and possibly disable temporarily
                self._minimax_fail_count += 1
                if self._minimax_fail_count >= self._MINIMAX_FAIL_THRESHOLD:
                    self._minimax_disabled_until = (
                        time.monotonic() + self._MINIMAX_BACKOFF_SECONDS
                    )
                    logger.warning(
                        "TTS: MiniMax failed %d consecutive times — "
                        "disabling for %.0f seconds",
                        self._minimax_fail_count,
                        self._MINIMAX_BACKOFF_SECONDS,
                    )
                else:
                    logger.warning(
                        "TTS: MiniMax failed (%d/%d), falling back",
                        self._minimax_fail_count,
                        self._MINIMAX_FAIL_THRESHOLD,
                    )
                self._use_minimax_fallback(text, generation)
            else:
                # Success — reset failure counter
                if self._minimax_fail_count > 0:
                    logger.info(
                        "TTS: MiniMax recovered after %d failure(s)",
                        self._minimax_fail_count,
                    )
                    self._minimax_fail_count = 0
        else:
            self._run_async(self._generate_edge(text, generation))
        self._queue_output_tail_silence(generation)

    def _queue_output_tail_silence(self, generation: int) -> None:
        """Keep USB playback fed briefly so the device does not swallow the final phoneme."""
        if not self._is_generation_current(generation):
            return
        sample_count = int(self._sample_rate * self._output_tail_silence_seconds)
        if sample_count <= 0:
            return
        with self._buffer_lock:
            self.tts_buffer.append(np.zeros(sample_count, dtype=np.float32))
        logger.debug("TTS: queued %.3fs output tail silence", self._output_tail_silence_seconds)

    def _use_minimax_fallback(self, text: str, generation: int) -> None:
        """Use local or edge TTS as a fallback when MiniMax is unavailable."""
        if (
            self._local_tts is None
            and self._fallback_backend == "local"
            and os.path.isdir(self._model_dir)
        ):
            self._init_local_tts()
        if self._local_tts is not None:
            self._generate_local(text, generation)
        else:
            self._run_async(self._generate_edge(text, generation))

    def _run_async(self, coro) -> bool:
        """Run an async coroutine in a dedicated event loop on this worker thread.

        Creates a fresh loop per call and does NOT call ``set_event_loop`` —
        that would pollute the global state and conflict with the main asyncio
        loop running in the parent thread (blueprint / RuntimeApp).
        """
        loop = None
        try:
            loop = asyncio.new_event_loop()
            # Do NOT call asyncio.set_event_loop(loop) — the main thread
            # already owns the process-wide event loop.
            result = loop.run_until_complete(coro)
            return True if result is None else bool(result)
        except Exception as exc:
            logger.error("TTS async error: %s", exc)
            return False
        finally:
            if loop is not None:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                loop.close()

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        """Resample audio via linear interpolation.

        Returns *samples* unchanged when rates match or input is too short.
        """
        if source_rate == target_rate or len(samples) <= 1:
            return samples
        ratio = target_rate / source_rate
        new_len = max(1, int(len(samples) * ratio))
        indices = np.linspace(0, len(samples) - 1, new_len)
        return np.interp(indices, np.arange(len(samples)), samples).astype(np.float32)

    # ------------------------------------------------------------------
    # Local backend (sherpa-onnx)
    # ------------------------------------------------------------------

    def _generate_local(self, text: str, generation: int) -> None:
        """Synthesise via sherpa-onnx OfflineTts — direct float32 samples."""
        if self._local_tts is None:
            return

        audio = self._local_tts.generate(text, sid=self._sid, speed=self._speed)
        if not self._is_generation_current(generation):
            return

        samples = np.array(audio.samples, dtype=np.float32)

        # Resample if model rate differs from playback rate
        samples = self._resample(samples, self._local_sample_rate, self._sample_rate)

        if self._is_generation_current(generation) and len(samples) > 0:
            with self._buffer_lock:
                self.tts_buffer.append(samples)

    # ------------------------------------------------------------------
    # Edge backend (network)
    # ------------------------------------------------------------------

    async def _generate_edge(self, text: str, generation: int) -> None:
        """Synthesise via Microsoft Edge TTS — MP3 stream → decode → queue."""
        import edge_tts
        import miniaudio

        communicate = edge_tts.Communicate(text, self._voice, rate=self._rate)
        mp3_acc = bytearray()

        async for chunk in communicate.stream():
            if not self._is_generation_current(generation):
                logger.debug("TTS: aborting stale edge request mid-stream")
                return
            if chunk["type"] == "audio":
                mp3_acc.extend(chunk["data"])

        if not mp3_acc or not self._is_generation_current(generation):
            return

        try:
            decoded = miniaudio.decode(bytes(mp3_acc), nchannels=1, sample_rate=self._sample_rate)
            samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
            if self._is_generation_current(generation):
                with self._buffer_lock:
                    self.tts_buffer.append(samples)
        except Exception as exc:
            logger.error("TTS edge decode error: %s", exc)

    # ------------------------------------------------------------------
    # MiniMax backend (SSE streaming, incremental playback)
    # ------------------------------------------------------------------

    async def _generate_minimax_transport(self, text: str, generation: int) -> bool:
        """Synthesize through the configured MiniMax streaming transport."""
        if self._minimax_tts_transport in {"websocket", "ws"}:
            return await self._generate_minimax_websocket(text, generation)
        return await self._generate_minimax(text, generation)

    def _minimax_voice_setting(self) -> dict[str, Any]:
        voice_setting: dict[str, Any] = {
            "voice_id": self._minimax_voice_id,
            "speed": self._minimax_speed,
            "vol": self._minimax_vol,
            "pitch": self._minimax_pitch,
        }
        if self._minimax_emotion:
            voice_setting["emotion"] = self._minimax_emotion
        return voice_setting

    def _minimax_audio_setting(self) -> dict[str, Any]:
        return {
            "sample_rate": self._minimax_sample_rate,
            "bitrate": self._minimax_bitrate,
            "format": self._minimax_audio_format,
            "channel": 1,
        }

    def _minimax_decode_audio_chunk(self, hex_audio: str) -> np.ndarray:
        """Decode a MiniMax hex audio chunk into playback-rate float32 samples."""
        audio_bytes = bytes.fromhex(hex_audio)
        if self._minimax_audio_format != "pcm":
            return self._decode_minimax_encoded_audio(audio_bytes)

        samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
        if self._minimax_sample_rate != self._sample_rate:
            samples = self._resample(
                samples, self._minimax_sample_rate, self._sample_rate,
            )
        return samples

    def _decode_minimax_encoded_audio(self, audio_bytes: bytes) -> np.ndarray:
        """Decode a complete non-PCM MiniMax audio payload."""
        try:
            import miniaudio
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"MiniMax {self._minimax_audio_format} decode requires miniaudio"
            ) from exc
        decoded = miniaudio.decode(
            audio_bytes,
            nchannels=1,
            sample_rate=self._sample_rate,
        )
        return np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0

    def _queue_minimax_encoded_audio(
        self,
        encoded_audio: bytearray,
        pending: list[np.ndarray],
        state: dict[str, int | bool],
    ) -> None:
        """Decode a complete encoded MiniMax stream and queue it."""
        if not encoded_audio:
            return
        samples = self._decode_minimax_encoded_audio(bytes(encoded_audio))
        self._queue_minimax_samples(samples, pending, state)
        if pending:
            self._flush_minimax_pending(pending, state)

    def _queue_minimax_samples(
        self,
        samples: np.ndarray,
        pending: list[np.ndarray],
        state: dict[str, int | bool],
    ) -> None:
        """Queue MiniMax decoded samples in playback-sized chunks."""
        if len(samples) <= 0:
            return
        pending.append(samples)
        state["pending_len"] = int(state["pending_len"]) + len(samples)
        if int(state["pending_len"]) >= self._MINIMAX_MIN_STREAM_SAMPLES:
            self._flush_minimax_pending(pending, state)

    def _flush_minimax_pending(
        self,
        pending: list[np.ndarray],
        state: dict[str, int | bool],
    ) -> None:
        if not pending:
            return
        chunk = np.concatenate(pending) if len(pending) > 1 else pending[0]
        if bool(state["first_flush"]):
            state["first_flush"] = False
            threshold = max(0.0, self._minimax_onset_threshold)
            nonzero = np.where(np.abs(chunk) > threshold)[0]
            if len(nonzero) > 0 and nonzero[0] > 100:
                preserve = int(
                    self._sample_rate
                    * max(0.0, self._minimax_leading_silence_preserve_seconds)
                )
                trim = max(0, nonzero[0] - preserve)
                chunk = chunk[trim:]
                logger.debug("TTS: trimmed %d leading silence samples", trim)
        with self._buffer_lock:
            self.tts_buffer.append(chunk)
        state["queued_samples"] = int(state["queued_samples"]) + len(chunk)
        pending.clear()
        state["pending_len"] = 0

    async def _generate_minimax(self, text: str, generation: int) -> bool:
        """Synthesise via MiniMax T2A v2 — SSE hex-PCM stream → incremental buffer."""
        import json as _json

        import httpx

        url = f"{self._minimax_tts_url}/t2a_v2"
        headers = {
            "Authorization": f"Bearer {self._minimax_api_key}",
            "Content-Type": "application/json",
        }
        voice_setting: dict[str, Any] = {"voice_id": self._minimax_voice_id}
        if self._minimax_speed != 1.0:
            voice_setting["speed"] = self._minimax_speed
        if self._minimax_vol != 1.0:
            voice_setting["vol"] = self._minimax_vol
        if self._minimax_pitch != 0:
            voice_setting["pitch"] = self._minimax_pitch
        if self._minimax_emotion:
            voice_setting["emotion"] = self._minimax_emotion

        body = {
            "model": self._minimax_tts_model,
            "text": text,
            "stream": True,
            "voice_setting": voice_setting,
            "audio_setting": {
                "sample_rate": self._minimax_sample_rate,
                "format": "pcm",
                "channel": 1,
            },
            "output_format": "hex",
        }

        need_resample = self._minimax_sample_rate != self._sample_rate
        # Minimum chunk size for immediate playback (150ms @ 16kHz = 2400 samples).
        # Smaller chunks cause excessive context-switching in the playback loop.
        _MIN_SAMPLES = 2400
        pending: list[np.ndarray] = []
        pending_len = 0
        queued_samples = 0
        _first_flush = True  # trim leading silence from MiniMax's first chunk
        encoded_audio = bytearray()

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    body_text = await resp.aread()
                    logger.error("MiniMax TTS HTTP %d: %s", resp.status_code, body_text[:200])
                    return False

                async for line in resp.aiter_lines():
                    if not self._is_generation_current(generation):
                        return True
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        payload = _json.loads(data_str)
                        data_field = payload.get("data", {})
                        # status=2 is the final summary event — MiniMax resends the
                        # complete audio here as a duplicate.  Skip it; we already
                        # have all the audio from the status=1 streaming chunks.
                        # Note: MiniMax may return status as int 2 or string "2".
                        if data_field.get("status") in (2, "2"):
                            continue
                        hex_audio = data_field.get("audio", "")
                        if not hex_audio:
                            continue
                        if self._minimax_audio_format != "pcm":
                            encoded_audio.extend(bytes.fromhex(hex_audio))
                            continue
                        pcm_bytes = bytes.fromhex(hex_audio)
                        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
                        if need_resample:
                            samples = self._resample(
                                samples, self._minimax_sample_rate, self._sample_rate,
                            )
                        if len(samples) > 0:
                            pending.append(samples)
                            pending_len += len(samples)
                            # Flush to playback buffer once we have enough samples
                            if pending_len >= _MIN_SAMPLES:
                                chunk = np.concatenate(pending) if len(pending) > 1 else pending[0]
                                # Trim leading silence from first chunk —
                                # MiniMax TTS prepends ~120ms of zeros that
                                # cause the first word to be inaudible.
                                if _first_flush:
                                    _first_flush = False
                                    threshold = max(0.0, self._minimax_onset_threshold)
                                    nonzero = np.where(np.abs(chunk) > threshold)[0]
                                    if len(nonzero) > 0 and nonzero[0] > 100:
                                        preserve = int(
                                            self._sample_rate
                                            * max(0.0, self._minimax_leading_silence_preserve_seconds)
                                        )
                                        trim = max(0, nonzero[0] - preserve)
                                        chunk = chunk[trim:]
                                        logger.debug("TTS: trimmed %d leading silence samples", trim)
                                with self._buffer_lock:
                                    self.tts_buffer.append(chunk)
                                queued_samples += len(chunk)
                                pending.clear()
                                pending_len = 0
                    except (_json.JSONDecodeError, ValueError) as exc:
                        logger.debug("MiniMax TTS chunk parse: %s", exc)

        # Flush any remaining samples
        if encoded_audio and self._is_generation_current(generation):
            samples = self._decode_minimax_encoded_audio(bytes(encoded_audio))
            if len(samples) > 0:
                pending.append(samples)
                pending_len += len(samples)
        if pending and self._is_generation_current(generation):
            chunk = np.concatenate(pending) if len(pending) > 1 else pending[0]
            with self._buffer_lock:
                self.tts_buffer.append(chunk)
            queued_samples += len(chunk)

        if not self._is_generation_current(generation):
            return True
        if queued_samples <= 0:
            logger.warning("MiniMax TTS produced no playable audio; using fallback")
            return False
        return True

    async def _generate_minimax_websocket(self, text: str, generation: int) -> bool:
        """Synthesise via MiniMax T2A WebSocket hex audio stream."""
        import json as _json

        try:
            import websocket
        except ModuleNotFoundError:
            logger.warning("MiniMax TTS WebSocket requires websocket-client package")
            return False

        pending: list[np.ndarray] = []
        state: dict[str, int | bool] = {
            "pending_len": 0,
            "queued_samples": 0,
            "first_flush": True,
        }
        encoded_audio = bytearray()
        ws = None

        try:
            create_connection = getattr(websocket, "create_connection", None)
            if create_connection is None:
                logger.warning("MiniMax TTS WebSocket requires websocket-client")
                return False
            ws = create_connection(
                self._minimax_tts_ws_url,
                header=[f"Authorization: Bearer {self._minimax_api_key}"],
                timeout=10,
            )
            connected = _json.loads(ws.recv())
            if connected.get("event") != "connected_success":
                logger.error("MiniMax TTS WS connect failed: %s", connected)
                return False

            ws.send(_json.dumps({
                "event": "task_start",
                "model": self._minimax_tts_model,
                "language_boost": "auto",
                "voice_setting": self._minimax_voice_setting(),
                "audio_setting": self._minimax_audio_setting(),
            }))
            started = _json.loads(ws.recv())
            if started.get("event") != "task_started":
                logger.error("MiniMax TTS WS start failed: %s", started)
                return False

            ws.send(_json.dumps({"event": "task_continue", "text": text}))

            while self._is_generation_current(generation):
                payload = _json.loads(ws.recv())
                base_resp = payload.get("base_resp", {})
                if int(base_resp.get("status_code", 0) or 0) != 0:
                    logger.error("MiniMax TTS WS error: %s", base_resp)
                    return False
                if payload.get("event") == "task_failed":
                    logger.error("MiniMax TTS WS task failed: %s", payload)
                    return False

                data = payload.get("data") or {}
                hex_audio = data.get("audio", "")
                if hex_audio:
                    if self._minimax_audio_format == "pcm":
                        samples = self._minimax_decode_audio_chunk(hex_audio)
                        self._queue_minimax_samples(samples, pending, state)
                    else:
                        encoded_audio.extend(bytes.fromhex(hex_audio))
                if payload.get("is_final") is True:
                    break

            try:
                ws.send(_json.dumps({"event": "task_finish"}))
                while True:
                    payload = _json.loads(ws.recv())
                    if payload.get("event") in {"task_finished", "task_failed"}:
                        break
            except Exception as exc:
                logger.debug("MiniMax TTS WS finish ignored: %s", exc)

        except (_json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning("MiniMax TTS WS failed: %s", exc)
            return False
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

        if encoded_audio and self._is_generation_current(generation):
            self._queue_minimax_encoded_audio(encoded_audio, pending, state)
        if pending and self._is_generation_current(generation):
            self._flush_minimax_pending(pending, state)
        if not self._is_generation_current(generation):
            return True
        if int(state["queued_samples"]) <= 0:
            logger.warning("MiniMax TTS WS produced no playable audio; using fallback")
            return False
        return True

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _kill_aplay(self) -> None:
        """Terminate any running aplay subprocess (immediate interruption)."""
        with self._aplay_lock:
            proc = self._aplay_proc
            self._aplay_proc = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception as exc:
                logger.debug("aplay terminate failed (ignored): %s", exc)

    def _wait_for_aplay_drain(self, proc: subprocess.Popen) -> bool:  # type: ignore[type-arg]
        """Let aplay drain queued USB audio before resorting to a forced kill."""
        try:
            proc.wait(timeout=self._aplay_drain_timeout_seconds)
            return True
        except subprocess.TimeoutExpired:
            logger.error(
                "aplay: drain timed out after %.1fs; forcing process stop",
                self._aplay_drain_timeout_seconds,
            )
        except Exception as exc:
            logger.error("aplay: drain wait failed; forcing process stop: %s", exc)
        try:
            proc.kill()
        except Exception as exc:
            logger.debug("aplay kill after drain failure ignored: %s", exc)
        return False

    def _kill_usb_audio(self) -> None:
        """Terminate any running MCP01 direct USB playback helper."""
        self._usb_direct_warming.clear()
        with self._usb_audio_lock:
            proc = self._usb_audio_proc
            self._usb_audio_proc = None
            stream_proc = self._usb_audio_stream_proc
            self._usb_audio_stream_proc = None
            self._usb_audio_stream_ready_at = 0.0
        if proc is not None:
            try:
                proc.terminate()
            except Exception as exc:
                logger.debug("MCP01 USB audio terminate failed (ignored): %s", exc)
        if stream_proc is not None:
            try:
                if stream_proc.stdin is not None:
                    stream_proc.stdin.close()
            except Exception as exc:
                logger.debug("MCP01 USB stream stdin close failed (ignored): %s", exc)
            try:
                stream_proc.terminate()
            except Exception as exc:
                logger.debug("MCP01 USB stream terminate failed (ignored): %s", exc)
            try:
                stream_proc.wait(timeout=0.8)
            except Exception:
                try:
                    stream_proc.kill()
                except Exception as exc:
                    logger.debug("MCP01 USB stream kill failed (ignored): %s", exc)

    def _usb_direct_source_path(self) -> Path:
        if self._usb_audio_source:
            return Path(self._usb_audio_source)
        return Path(__file__).resolve().parents[2] / "scripts" / "bench" / "mcp01_usb_audio_libusb.c"

    def _ensure_usb_audio_binary(self) -> str | None:
        if self._usb_audio_build_failed:
            return None

        source = self._usb_direct_source_path()
        binary = Path(tempfile.gettempdir()) / "askme_mcp01_usb_audio_libusb"
        if os.name == "nt":
            binary = binary.with_suffix(".exe")
        if self._usb_audio_binary:
            binary = Path(self._usb_audio_binary)
        self._usb_audio_binary = str(binary)

        try:
            if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
                return str(binary)
        except OSError:
            pass

        if not source.exists():
            logger.warning("MCP01 USB audio source not found: %s", source)
            self._usb_audio_build_failed = True
            return None
        if shutil.which("gcc") is None or shutil.which("pkg-config") is None:
            logger.warning("MCP01 USB audio fallback requires gcc and pkg-config")
            self._usb_audio_build_failed = True
            return None

        pkg = subprocess.run(
            ["pkg-config", "--cflags", "--libs", "libusb-1.0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if pkg.returncode != 0:
            logger.warning("MCP01 USB audio fallback requires libusb-1.0 development files")
            self._usb_audio_build_failed = True
            return None

        cmd = [
            "gcc",
            str(source),
            "-O2",
            "-Wall",
            "-Wextra",
            "-o",
            str(binary),
            *pkg.stdout.split(),
            "-lm",
        ]
        build = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if build.returncode != 0:
            logger.warning("MCP01 USB audio helper build failed: %s", build.stderr.strip())
            self._usb_audio_build_failed = True
            return None
        return str(binary)

    def _chunk_to_usb_stereo_pcm(self, chunk: np.ndarray) -> bytes:
        """Convert a mono float32 chunk to 48 kHz stereo S16_LE for MCP01."""
        samples = np.asarray(chunk, dtype=np.float32)
        if self._sample_rate != 48000:
            samples = self._resample(samples, self._sample_rate, 48000)
        pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        stereo = np.empty(pcm.size * 2, dtype=np.int16)
        stereo[0::2] = pcm
        stereo[1::2] = pcm
        return stereo.tobytes()

    def _play_chunk_usb_direct(self, chunk: np.ndarray) -> bool:
        """Play one chunk through MCP01 USB Audio without ALSA/PortAudio."""
        return self._play_chunk_usb_direct_scoped(chunk)

    def _play_chunk_usb_direct_with_preroll(self, chunk: np.ndarray) -> bool:
        """Play one chunk, prepending cold-DAC preroll under the USB session lock."""
        return self._play_chunk_usb_direct_scoped(chunk, preroll_if_cold=True)

    def _play_chunk_usb_direct_speech(self, chunk: np.ndarray) -> bool:
        """Play TTS speech through a fresh USB stream with a speech-safe lead-in."""
        return self._play_chunk_usb_direct_scoped(chunk, speech_leadin=True)

    def _play_chunk_usb_direct_warming(self, chunk: np.ndarray) -> bool:
        """Play an intentional preroll/prewarm chunk while advertising warm-up state."""
        return self._play_chunk_usb_direct_scoped(chunk, mark_warming=True)

    def _play_chunk_usb_direct_scoped(
        self,
        chunk: np.ndarray,
        *,
        preroll_if_cold: bool = False,
        speech_leadin: bool = False,
        mark_warming: bool = False,
    ) -> bool:
        """Serialize USB ownership, warm-state transition, optional preroll, and playback."""
        with self._usb_audio_session_lock:
            samples = np.asarray(chunk, dtype=np.float32)
            warming_set = False
            if speech_leadin:
                samples = self._apply_usb_direct_speech_gain(samples)
                speech_samples = samples
                leadin = self._usb_direct_speech_leadin_chunk(
                    warm=self._is_persistent_usb_stream_warm()
                )
                leadin_len = len(leadin)
                if len(leadin) > 0:
                    samples = np.concatenate([leadin, samples])
                cushion = self._usb_direct_speech_onset_cushion_chunk(speech_samples)
                if len(cushion) > 0:
                    samples = np.concatenate(
                        [samples[:leadin_len], cushion, samples[leadin_len:]]
                    )
            elif preroll_if_cold and not self._is_dac_warm():
                preroll = self._usb_direct_preroll_chunk()
                if len(preroll) > 0:
                    samples = np.concatenate([preroll, samples])
                    mark_warming = True
            elif preroll_if_cold:
                guard = self._usb_direct_stream_guard_chunk()
                if len(guard) > 0:
                    samples = np.concatenate([guard, samples])
            if mark_warming:
                self._usb_direct_warming.set()
                warming_set = True
            try:
                self._publish_echo_reference(samples)
                if self._audio_router is not None:
                    with self._audio_router.output_session():
                        return self._play_chunk_usb_direct_locked(samples)
                return self._play_chunk_usb_direct_locked(samples)
            finally:
                if warming_set:
                    self._usb_direct_warming.clear()

    def _apply_usb_direct_speech_gain(self, chunk: np.ndarray) -> np.ndarray:
        gain = max(0.0, self._usb_direct_speech_gain)
        if gain == 1.0 or len(chunk) == 0:
            return chunk
        boosted = np.asarray(chunk, dtype=np.float32) * gain
        if gain > 1.0:
            boosted = np.tanh(boosted)
        return np.clip(boosted, -1.0, 1.0).astype(np.float32)

    def _play_chunk_usb_direct_locked(self, chunk: np.ndarray) -> bool:
        """Play one chunk through MCP01 USB Audio without ALSA/PortAudio."""
        if self._usb_direct_persistent_stream:
            if self._play_chunk_usb_direct_stream_locked(chunk):
                return True
            logger.warning("MCP01 USB persistent stream failed; falling back to one-shot playback")
            self._kill_usb_audio()
        return self._play_chunk_usb_direct_one_shot_locked(chunk)

    def _watch_usb_audio_stream_stderr(
        self,
        proc: subprocess.Popen,  # type: ignore[type-arg]
        ready: threading.Event,
        startup_lines: list[str],
    ) -> None:
        """Drain MCP01 helper stderr and signal when the stream is configured."""
        stream = proc.stderr
        if stream is None:
            ready.set()
            return
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode(errors="replace").strip()
                if len(startup_lines) < 12:
                    startup_lines.append(line)
                if line:
                    logger.debug("MCP01 USB stream: %s", line)
                if "stdin stream ready" in line:
                    ready.set()
        except Exception as exc:
            logger.debug("MCP01 USB stream stderr reader ended: %s", exc)

    def _start_usb_audio_stream_locked(self) -> subprocess.Popen | None:
        """Start the persistent MCP01 helper while the session lock is held."""
        binary = self._ensure_usb_audio_binary()
        if binary is None:
            return None

        with self._usb_audio_lock:
            proc = self._usb_audio_stream_proc
            if proc is not None and proc.poll() is None:
                return proc
            self._usb_audio_stream_proc = None
            self._usb_audio_stream_ready_at = 0.0

        args = [binary, "--stdin-stream", "--capture-ms", "0"]
        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            logger.error("MCP01 USB persistent stream failed to start: %s", exc)
            return None

        ready = threading.Event()
        startup_lines: list[str] = []
        threading.Thread(
            target=self._watch_usb_audio_stream_stderr,
            args=(proc, ready, startup_lines),
            daemon=True,
        ).start()

        ready_timeout = max(0.05, self._usb_direct_stream_start_grace_seconds)
        ready.wait(timeout=ready_timeout)
        if proc.poll() is not None:
            _, stderr = proc.communicate(timeout=1.0)
            logger.error(
                "MCP01 USB persistent stream exited rc=%s startup=%s stderr=%s",
                proc.returncode,
                " | ".join(startup_lines),
                stderr.decode(errors="replace").strip(),
            )
            return None
        if not ready.is_set():
            logger.warning(
                "MCP01 USB persistent stream not ready after %.2fs; continuing startup=%s",
                ready_timeout,
                " | ".join(startup_lines),
            )

        with self._usb_audio_lock:
            self._usb_audio_stream_proc = proc
            self._usb_audio_stream_ready_at = time.monotonic()
        logger.info("MCP01 USB persistent stream started")
        return proc

    def _play_chunk_usb_direct_stream_locked(self, chunk: np.ndarray) -> bool:
        """Write PCM into a persistent MCP01 USB stream and wait for playback."""
        pcm_bytes = self._chunk_to_usb_stereo_pcm(chunk)
        if not pcm_bytes:
            return True

        proc = self._start_usb_audio_stream_locked()
        if proc is None or proc.stdin is None:
            return False

        duration = len(pcm_bytes) / (48000 * 2 * 2)
        started = time.monotonic()
        try:
            view = memoryview(pcm_bytes)
            block = 48000 * 2 * 2 // 10  # 100 ms of 48 kHz stereo S16_LE.
            for pos in range(0, len(view), block):
                if self._stop_requested.is_set() or proc.poll() is not None:
                    return False
                proc.stdin.write(view[pos : pos + block])
                proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            logger.error("MCP01 USB persistent stream write failed: %s", exc)
            return False

        deadline = started + duration + max(0.0, self._usb_direct_stream_drain_grace_seconds)
        while time.monotonic() < deadline:
            if self._stop_requested.is_set() or proc.poll() is not None:
                return False
            time.sleep(min(0.05, deadline - time.monotonic()))

        self._last_aplay_close = time.monotonic()
        return True

    def _play_chunk_usb_direct_one_shot_locked(self, chunk: np.ndarray) -> bool:
        """Play one chunk through the legacy one-shot MCP01 helper."""
        binary = self._ensure_usb_audio_binary()
        if binary is None:
            return False

        pcm_bytes = self._chunk_to_usb_stereo_pcm(chunk)
        if not pcm_bytes:
            return True

        args = [binary, "--stdin-play", "--capture-ms", "0"]
        timeout = max(5.0, len(pcm_bytes) / (48000 * 2 * 2) + 5.0)
        proc: subprocess.Popen | None = None  # type: ignore[type-arg]
        try:
            proc = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            with self._usb_audio_lock:
                self._usb_audio_proc = proc
            stdout, stderr = proc.communicate(pcm_bytes, timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_usb_audio()
            logger.error("MCP01 USB audio playback timed out")
            return False
        except OSError as exc:
            logger.error("MCP01 USB audio playback failed to start: %s", exc)
            return False
        finally:
            with self._usb_audio_lock:
                if proc is not None and self._usb_audio_proc is proc:
                    self._usb_audio_proc = None

        if proc is None:
            return False
        if proc.returncode != 0:
            logger.error(
                "MCP01 USB audio playback failed rc=%s stdout=%s stderr=%s",
                proc.returncode,
                stdout.decode(errors="replace").strip(),
                stderr.decode(errors="replace").strip(),
            )
            return False
        logger.info("MCP01 USB audio playback ok: %s", stdout.decode(errors="replace").strip())
        self._last_aplay_close = time.monotonic()
        return True

    def _is_plughw_output(self) -> bool:
        return self._output_device is not None and str(self._output_device).startswith("plughw:")

    def _alsa_output_available(self) -> bool:
        """Return False when ALSA clearly has no card for a plughw output."""
        if os.name != "posix" or not self._is_plughw_output():
            return True

        cards_path = Path("/proc/asound/cards")
        try:
            cards_text = cards_path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            return True
        if "no soundcards" in cards_text:
            return False

        match = re.match(r"plughw:(\d+)", str(self._output_device))
        if match and not Path(f"/proc/asound/card{match.group(1)}").exists():
            return False
        return True

    def _should_use_usb_direct(self) -> bool:
        if self._output_transport == "usb_direct":
            return True
        return (
            self._output_transport == "auto"
            and self._is_plughw_output()
            and not self._alsa_output_available()
        )

    def _should_try_usb_direct_fallback(self) -> bool:
        if self._output_transport == "usb_direct":
            return True
        if self._output_transport != "auto":
            return False
        return self._is_plughw_output()

    def _collect_usb_direct_chunk(self, first_chunk: np.ndarray) -> np.ndarray:
        """Coalesce the current streamed TTS burst into one MCP01 USB play."""
        chunks = [np.asarray(first_chunk, dtype=np.float32)]
        deadline = time.monotonic() + self._usb_direct_coalesce_timeout
        last_audio_at = time.monotonic()
        # MiniMax emits several chunks per sentence.  Stop after a short quiet
        # gap so the first sentence can play while later sentences are still
        # being generated.
        settle_seconds = 0.18

        while self._is_playing and not self._stop_requested.is_set():
            drained: list[np.ndarray] = []
            with self._buffer_lock:
                while self.tts_buffer:
                    drained.append(self.tts_buffer.popleft())

            if drained:
                chunks.extend(np.asarray(chunk, dtype=np.float32) for chunk in drained)
                last_audio_at = time.monotonic()

            now = time.monotonic()
            if now - last_audio_at >= settle_seconds:
                break
            if now >= deadline:
                logger.debug(
                    "MCP01 USB direct coalesce timeout after %.1fs",
                    self._usb_direct_coalesce_timeout,
                )
                break
            time.sleep(0.02)

        merged = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        logger.info(
            "MCP01 USB direct: %d chunks, %d samples = %.3fs",
            len(chunks),
            len(merged),
            len(merged) / self._sample_rate,
        )
        return merged.astype(np.float32, copy=False)

    def _is_dac_warm(self) -> bool:
        if self._usb_direct_warming.is_set():
            return True
        return (
            self._last_aplay_close > 0
            and (time.monotonic() - self._last_aplay_close) < self._preroll_warm_window
        )

    def _is_persistent_usb_stream_warm(self) -> bool:
        """Return True only when the live stream is allowed to imply speaker warmth."""
        if not self._usb_direct_persistent_stream:
            return False
        if not self._usb_direct_trust_persistent_warm_state:
            return False
        with self._usb_audio_lock:
            proc = self._usb_audio_stream_proc
            ready_at = self._usb_audio_stream_ready_at
        if proc is None or proc.poll() is not None or ready_at <= 0:
            return False
        return (
            self._last_aplay_close > 0
            and (time.monotonic() - self._last_aplay_close) < self._preroll_warm_window
        )

    def _usb_direct_preroll_chunk(self) -> np.ndarray:
        samples = int(self._sample_rate * self._usb_direct_preroll_seconds)
        if samples <= 0:
            return np.empty(0, dtype=np.float32)
        return self._usb_direct_wake_chunk(samples, seed=42)

    def _usb_direct_stream_guard_chunk(self) -> np.ndarray:
        samples = int(self._sample_rate * self._usb_direct_stream_guard_seconds)
        if samples <= 0:
            return np.empty(0, dtype=np.float32)
        return self._usb_direct_noise_chunk(samples, amp=160.0 / 32767.0, seed=43)

    def _usb_direct_speech_leadin_chunk(self, *, warm: bool = False) -> np.ndarray:
        seconds = (
            self._usb_direct_speech_warm_leadin_seconds
            if warm
            else self._usb_direct_speech_leadin_seconds
        )
        total_samples = int(self._sample_rate * seconds)
        if total_samples <= 0:
            return np.empty(0, dtype=np.float32)
        return self._usb_direct_wake_chunk(total_samples, seed=47 if warm else 44)

    def _usb_direct_wake_chunk(self, total_samples: int, *, seed: int) -> np.ndarray:
        if total_samples <= 0:
            return np.empty(0, dtype=np.float32)
        wake_samples = int(self._sample_rate * self._usb_direct_speech_wake_signal_seconds)
        gap_samples = int(self._sample_rate * self._usb_direct_speech_wake_gap_seconds)
        noise_gain = max(0.0, min(1.0, self._usb_direct_speech_wake_noise_gain))
        if wake_samples <= 0:
            return self._usb_direct_noise_chunk(total_samples, amp=noise_gain, seed=seed)

        wake_samples = min(wake_samples, total_samples)
        gap_samples = max(0, min(gap_samples, total_samples - wake_samples))
        hold_samples = max(0, total_samples - wake_samples - gap_samples)

        parts: list[np.ndarray] = []
        parts.append(self._usb_direct_tone_chunk(
            wake_samples,
            hz=self._usb_direct_speech_wake_signal_hz,
            gain=self._usb_direct_speech_wake_signal_gain,
        ))
        if gap_samples > 0:
            parts.append(self._usb_direct_noise_chunk(gap_samples, amp=noise_gain, seed=seed + 1))
        if hold_samples > 0:
            parts.append(self._usb_direct_noise_chunk(hold_samples, amp=noise_gain, seed=seed + 2))
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

    def _usb_direct_noise_chunk(self, samples: int, *, amp: float, seed: int) -> np.ndarray:
        if samples <= 0:
            return np.empty(0, dtype=np.float32)
        rng = np.random.RandomState(seed)
        return (rng.randn(samples) * amp).astype(np.float32)

    def _usb_direct_tone_chunk(self, samples: int, *, hz: float, gain: float) -> np.ndarray:
        if samples <= 0:
            return np.empty(0, dtype=np.float32)
        t = np.arange(samples, dtype=np.float32) / float(self._sample_rate)
        tone = (max(0.0, min(1.0, gain)) * np.sin(2.0 * np.pi * hz * t)).astype(np.float32)
        edge = min(samples // 2, int(self._sample_rate * 0.025))
        if edge > 1:
            ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
            tone[:edge] *= ramp
            tone[-edge:] *= ramp[::-1]
        return tone

    def _usb_direct_speech_onset_cushion_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Return a low-volume sacrificial copy of the first audible speech onset."""
        cushion_samples = int(self._sample_rate * self._usb_direct_speech_onset_cushion_seconds)
        if cushion_samples <= 0:
            return np.empty(0, dtype=np.float32)

        samples = np.asarray(chunk, dtype=np.float32)
        if len(samples) == 0:
            return np.empty(0, dtype=np.float32)

        audible = np.flatnonzero(np.abs(samples) > 0.002)
        if len(audible) > 0:
            preserve = int(self._sample_rate * 0.02)
            start = max(0, int(audible[0]) - preserve)
        else:
            start = 0

        excerpt = samples[start : start + cushion_samples].astype(np.float32, copy=True)
        if len(excerpt) == 0:
            return np.empty(0, dtype=np.float32)
        gain = max(0.0, min(1.0, self._usb_direct_speech_onset_cushion_gain))
        excerpt *= gain

        edge = min(len(excerpt) // 4, int(self._sample_rate * 0.02))
        if edge > 1:
            ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
            excerpt[:edge] *= ramp
            excerpt[-edge:] *= ramp[::-1]

        gap_samples = int(self._sample_rate * self._usb_direct_speech_onset_gap_seconds)
        if gap_samples > 0:
            gap = self._usb_direct_noise_chunk(gap_samples, amp=260.0 / 32767.0, seed=46)
            return np.concatenate([excerpt, gap])
        return excerpt

    def _playback_loop(self) -> None:
        """Drain tts_buffer one sentence at a time.

        On Linux with aplay available: pipe PCM to `aplay` via stdin.
        aplay plays exactly once — confirmed on PipeWire-managed ALSA (sunrise
        aarch64).  sounddevice (sd.OutputStream callback and sd.play()) both
        cause audio to play twice on this system due to a PortAudio/PipeWire
        interaction bug.

        On other platforms: fall back to sd.play() + sd.wait().
        """
        try:
            logger.info(
                "TTS playback: device=%s, sample_rate=%d, transport=%s, aplay=%s",
                self._output_device if self._output_device is not None else "default",
                self._sample_rate,
                self._output_transport,
                self._aplay_bin is not None,
            )

            _use_usb_direct = self._should_use_usb_direct()
            if _use_usb_direct and self._output_transport == "auto":
                logger.info("TTS playback: ALSA plughw output unavailable; using MCP01 USB direct")
            if _use_usb_direct and self._usb_direct_persistent_stream:
                with self._usb_audio_session_lock:
                    self._start_usb_audio_stream_locked()
            # Background prewarm remains opt-in.  With persistent USB streaming
            # it happens inside the same helper process that later carries
            # speech; with one-shot playback it still uses a separate stream.
            if (
                _use_usb_direct
                and self._usb_direct_background_prewarm
                and not self._has_buffered_audio()
                and not self._is_dac_warm()
            ):
                preroll = self._usb_direct_preroll_chunk()
                if len(preroll) > 0:
                    logger.info(
                        "MCP01 USB direct: prewarming speaker path %.3fs",
                        len(preroll) / self._sample_rate,
                    )
                    self._playback_busy.set()
                    try:
                        self._play_chunk_usb_direct_warming(preroll)
                    finally:
                        self._playback_busy.clear()

            # --- aplay persistent-process setup (computed once) ---
            _aplay_cmd: list[str] | None = None
            _preroll_bytes: bytes = b""
            if (
                self._aplay_bin
                and self._output_transport in {"auto", "aplay"}
                and not _use_usb_direct
            ):
                _aplay_cmd = [
                    self._aplay_bin,
                    "-r", str(self._sample_rate),
                    "-f", "S16_LE",
                    "-c", "1",
                    "-q",
                ]
                if self._output_device is not None:
                    _aplay_cmd += ["-D", str(self._output_device)]
                # Pre-roll: 1.5s low-volume white noise to wake USB DAC/amplifier.
                # The hardware has a ~1.5s soft-start that eats audio.
                # Low noise (~-40 dBFS) is inaudible but keeps the amp active.
                # Verified on sunrise MCP01: without this, first 3 characters lost.
                _preroll_n = int(self._sample_rate * 1.5)
                _rng = np.random.RandomState(42)  # deterministic for consistency
                _preroll_bytes = (_rng.randn(_preroll_n) * 200).astype(np.int16).tobytes()

            if _aplay_cmd is None and not _use_usb_direct:
                self._play_sounddevice_stream_until_stopped()
                return

            # Persistent aplay process state — one process per utterance,
            # all chunks piped into its stdin without restart.
            _proc: subprocess.Popen | None = None  # type: ignore[type-arg]
            _need_preroll = True
            _empty_polls = 0
            _prebuffer_wait_logged = False
            _MAX_EMPTY_POLLS = 250  # 250 × 20 ms = 5 s — keeps aplay warm between conversational turns, avoids re-paying 400ms pre-roll
            _router_ctx = None  # saved output_session() context manager

            # NOTE: aplay is NOT pre-started here. MCP01 is half-duplex — if
            # aplay opens the device before the mic, PortAudio sees 0 input
            # channels and recording fails. aplay starts on first audio chunk
            # (see _proc is None branch below).

            def _close_aplay() -> None:
                """Cleanly close the persistent aplay process."""
                nonlocal _proc, _need_preroll, _empty_polls, _router_ctx
                if _proc is None:
                    return
                try:
                    _proc.stdin.close()  # type: ignore[union-attr]
                except Exception:
                    pass
                self._wait_for_aplay_drain(_proc)
                with self._aplay_lock:
                    self._aplay_proc = None
                _proc = None
                _need_preroll = True
                _empty_polls = 0
                self._last_aplay_close = time.monotonic()  # track warm state
                # Give ALSA 200ms to fully release the device before reopening.
                # Without this, rapid close→open races cause "设备或资源忙".
                time.sleep(0.2)
                if _router_ctx is not None:
                    try:
                        _router_ctx.__exit__(None, None, None)
                    except Exception as exc:
                        logger.debug("audio router exit failed (ignored): %s", exc)
                    _router_ctx = None
                logger.info("aplay: done")

            while self._is_playing:
                # Check and clear stop request from barge-in
                if self._stop_requested.is_set():
                    self._stop_requested.clear()
                    self._clear_audio_buffer()
                    _close_aplay()
                    logger.info("TTS playback: stop_requested, skipping queued audio")
                    continue

                if (
                    _aplay_cmd is not None
                    and _proc is None
                    and self._aplay_prebuffer_pending()
                ):
                    if not _prebuffer_wait_logged:
                        if self._aplay_wait_for_synthesis_complete:
                            logger.info("aplay: buffering complete utterance before playback")
                        else:
                            logger.info(
                                "aplay: prebuffering up to %.2fs before streamed playback",
                                self._aplay_start_buffer_seconds,
                            )
                        _prebuffer_wait_logged = True
                    time.sleep(0.02)
                    continue
                if _prebuffer_wait_logged and _proc is None:
                    with self._buffer_lock:
                        buffered_samples = sum(len(item) for item in self.tts_buffer)
                    logger.info(
                        "aplay: prebuffer ready (%.2fs)",
                        buffered_samples / float(self._sample_rate),
                    )
                    _prebuffer_wait_logged = False

                chunk = None
                with self._buffer_lock:
                    if self.tts_buffer:
                        chunk = self.tts_buffer.popleft()
                        if len(chunk) > 0:
                            self._playback_busy.set()

                if chunk is not None and len(chunk) > 0:
                    _empty_polls = 0
                    if _use_usb_direct:
                        chunk = self._collect_usb_direct_chunk(chunk)
                        if self._stop_requested.is_set():
                            self._playback_busy.clear()
                            continue
                    # Apply volume
                    if self._volume != 1.0:
                        chunk = chunk * self._volume
                        np.clip(chunk, -1.0, 1.0, out=chunk)

                    if _use_usb_direct:
                        self._play_chunk_usb_direct_speech(chunk)
                    elif _aplay_cmd is not None:
                        pcm = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
                        dur = len(chunk) / self._sample_rate
                        logger.info(
                            "aplay: %d samples = %.3fs", len(chunk), dur
                        )
                        try:
                            # Start persistent aplay on first chunk (with retry on EBUSY)
                            if _proc is None:
                                if self._audio_router is not None:
                                    _router_ctx = self._audio_router.output_session()
                                    _router_ctx.__enter__()
                                for _retry in range(4):
                                    try:
                                        _proc = subprocess.Popen(
                                            _aplay_cmd, stdin=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL,
                                        )
                                        break
                                    except OSError:
                                        if _retry < 3:
                                            time.sleep(0.3)
                                        else:
                                            raise
                                with self._aplay_lock:
                                    self._aplay_proc = _proc

                            # Pre-roll only on first chunk after process start,
                            # AND only when DAC is cold (>5s since last playback).
                            if _need_preroll:
                                _dac_warm = (
                                    self._last_aplay_close > 0
                                    and (time.monotonic() - self._last_aplay_close) < self._preroll_warm_window
                                )
                                if not _dac_warm:
                                    _proc.stdin.write(_preroll_bytes)  # type: ignore[union-attr]
                                    self._publish_echo_reference(
                                        np.frombuffer(_preroll_bytes, dtype=np.int16).astype(np.float32)
                                        / 32767.0
                                    )
                                else:
                                    logger.debug("aplay: skipping pre-roll (DAC warm)")
                                _need_preroll = False

                            _proc.stdin.write(pcm.tobytes())  # type: ignore[union-attr]
                            _proc.stdin.flush()  # type: ignore[union-attr]
                            self._publish_echo_reference(pcm.astype(np.float32) / 32767.0)
                        except (BrokenPipeError, OSError):
                            # aplay killed externally (barge-in)
                            with self._aplay_lock:
                                self._aplay_proc = None
                            _proc = None
                            _need_preroll = True
                            if _router_ctx is not None:
                                try:
                                    _router_ctx.__exit__(None, None, None)
                                except Exception:
                                    pass
                                _router_ctx = None
                            if self._should_try_usb_direct_fallback():
                                logger.warning("aplay failed; trying MCP01 direct USB fallback")
                                self._play_chunk_usb_direct_speech(chunk)
                    else:
                        if self._audio_router is not None:
                            with self._audio_router.output_session():
                                self._publish_echo_reference(chunk)
                                sd.play(chunk, samplerate=self._sample_rate, device=self._output_device)
                                sd.wait()
                        else:
                            self._publish_echo_reference(chunk)
                            sd.play(chunk, samplerate=self._sample_rate, device=self._output_device)
                            sd.wait()
                    self._playback_busy.clear()
                else:
                    if _proc is not None:
                        _empty_polls += 1
                        if _empty_polls >= _MAX_EMPTY_POLLS:
                            _close_aplay()
                    time.sleep(0.02)
        except Exception as e:
            logger.error("Playback error: %s", e)
        finally:
            # Clean up persistent aplay if still running
            if _aplay_cmd is not None:
                _close_aplay()
            self._kill_usb_audio()
            with self._aplay_lock:
                self._aplay_proc = None
            self._playback_busy.clear()
            # Always clear _is_playing on exit — prevents start_playback() from
            # getting permanently blocked and wait_done() from deadlocking when
            # the audio device is unavailable or throws.
            with self._playback_lock:
                self._is_playing = False

    def _play_sounddevice_stream_until_stopped(self) -> None:
        """Play buffered audio through one continuous PortAudio stream."""
        stream_kwargs = {
            "samplerate": self._sample_rate,
            "device": self._output_device,
            "channels": 1,
            "dtype": "float32",
            "callback": self.play_audio_callback,
        }

        def _run() -> None:
            with sd.OutputStream(**stream_kwargs):
                while self._is_playing:
                    if self._stop_requested.is_set():
                        self._stop_requested.clear()
                        self._clear_audio_buffer()
                        self._playback_busy.clear()
                        logger.info("TTS playback: stop_requested, skipping queued audio")
                        continue
                    if self._has_buffered_audio():
                        self._playback_busy.set()
                    else:
                        self._playback_busy.clear()
                    time.sleep(0.02)

        if self._audio_router is not None:
            with self._audio_router.output_session():
                _run()
        else:
            _run()

    # ------------------------------------------------------------------
    # Generation tracking
    # ------------------------------------------------------------------

    def _advance_generation(self) -> int:
        with self._generation_lock:
            self._generation += 1
            return self._generation

    def _get_generation(self) -> int:
        with self._generation_lock:
            return self._generation

    def _is_generation_current(self, generation: int) -> bool:
        return generation == self._get_generation()

    def _has_buffered_audio(self) -> bool:
        with self._buffer_lock:
            return bool(self.tts_buffer)

    def _aplay_prebuffer_pending(self) -> bool:
        """Wait for a short network cushion before opening a streamed aplay utterance."""
        if self.tts_text_queue.unfinished_tasks <= 0:
            return False
        if self._aplay_wait_for_synthesis_complete:
            return True
        target = int(self._sample_rate * self._aplay_start_buffer_seconds)
        if target <= 0:
            return False
        with self._buffer_lock:
            buffered = sum(len(chunk) for chunk in self.tts_buffer)
        return buffered < target

    def _clear_audio_buffer(self) -> None:
        with self._buffer_lock:
            self.tts_buffer.clear()
