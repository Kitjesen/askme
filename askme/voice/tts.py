"""TTS Engine - three backends: local sherpa-onnx, edge-tts, or MiniMax streaming."""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import sounddevice as sd

if TYPE_CHECKING:
    from askme.voice.audio_router import AudioRouter

logger = logging.getLogger(__name__)


class TTSEngine:
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
        minimax_tts_model: str - TTS model name (default "speech-2.8-turbo")
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

    # Regex patterns for cleaning text before TTS
    _RE_EMOJI = re.compile(r'[\U00010000-\U0010ffff]')
    _RE_BOLD = re.compile(r'\*\*(.+?)\*\*')
    _RE_ITALIC = re.compile(r'\*(.+?)\*')
    _RE_CODE = re.compile(r'`(.+?)`')
    _RE_HEADER = re.compile(r'^#+\s*', flags=re.MULTILINE)
    _RE_LIST = re.compile(r'^[-*]\s+', flags=re.MULTILINE)
    _RE_IMG = re.compile(r'!\[.*?\]\(.*?\)')
    _RE_LINK = re.compile(r'\[(.+?)\]\(.*?\)')

    def __init__(self, config: dict[str, Any], *, audio_router: "AudioRouter | None" = None) -> None:
        self._backend: str = config.get("backend", "local")
        self._sample_rate: int = int(config.get("sample_rate", 24000))
        self._output_device: int | str | None = config.get("output_device")

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
        self._minimax_tts_model: str = config.get("minimax_tts_model", "speech-2.8-turbo")
        self._minimax_voice_id: str = config.get("minimax_voice_id", "male-qn-qingse")
        self._minimax_sample_rate: int = int(config.get("minimax_sample_rate", 24000))
        # Voice tuning: speed (0.5-2.0), vol (0-10), pitch (-12 to 12 semitones)
        self._minimax_speed: float = float(config.get("minimax_speed", 1.0))
        self._minimax_vol: float = min(10.0, max(0.0, float(config.get("minimax_vol", 1.0))))
        self._minimax_pitch: int = int(config.get("minimax_pitch", 0))
        # Emotion: "" (auto), happy, sad, angry, fearful, disgusted, surprised, calm
        self._minimax_emotion: str = config.get("minimax_emotion", "")

        # Consecutive failure tracking for MiniMax auto-disable
        self._minimax_fail_count: int = 0
        self._minimax_disabled_until: float = 0.0  # monotonic time

        # Volume multiplier applied to all PCM output (0.0–1.0)
        self._volume: float = float(config.get("volume", 1.0))

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
        # Immediate stop flag: checked by _playback_loop to abort mid-chunk
        self._stop_requested = threading.Event()

        # AudioRouter for device ownership coordination (optional)
        self._audio_router: AudioRouter | None = audio_router

        # Local TTS engine (lazy init)
        self._local_tts: Any | None = None
        self._local_sample_rate: int = 0

        # Auto-detect backend
        if self._backend == "minimax" and not self._minimax_api_key:
            logger.warning("MiniMax TTS: no API key configured, falling back to edge-tts")
            self._backend = "edge"
        if self._backend == "local" and not os.path.isdir(self._model_dir):
            logger.warning("Local TTS model not found at %s, falling back to edge-tts", self._model_dir)
            self._backend = "edge"

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
        if clean and len(clean) > 1:
            logger.info("speak queued: %r", clean[:60])
            self.tts_text_queue.put((self._get_generation(), clean))

    def start_playback(self) -> None:
        """Start the sounddevice output stream in a background thread."""
        with self._playback_lock:
            if self._is_playing:
                return
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
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def is_active(self) -> bool:
        """Return True if audio is buffered or playback is in progress."""
        with self._playback_lock:
            playing = self._is_playing
        return playing or self._has_buffered_audio()

    def wait_done(self, timeout: float = 30.0) -> None:
        """Block until all queued text has been synthesised and played.

        Args:
            timeout: Maximum seconds to wait for playback to finish after
                     synthesis is complete.  Prevents infinite blocking when
                     the audio device is unavailable.
        """
        self.tts_text_queue.join()
        deadline = time.monotonic() + timeout
        while self._has_buffered_audio():
            if time.monotonic() >= deadline:
                logger.warning("wait_done: timed out after %.1fs waiting for buffer drain", timeout)
                return
            time.sleep(0.05)
        # Wait for the last chunk to finish playing.
        # aplay: proc.communicate() is synchronous, but _aplay_proc is cleared
        # only after communicate() returns, so poll it.
        while self._aplay_proc is not None:
            if time.monotonic() >= deadline:
                logger.warning("wait_done: timed out after %.1fs waiting for aplay", timeout)
                return
            time.sleep(0.02)
        # Fallback: wait for any sounddevice stream (non-aplay systems).
        try:
            sd.wait()
        except Exception:
            pass

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

    def stop_immediately(self) -> None:
        """Signal the playback loop to abort the current chunk immediately.

        Unlike drain_buffers() which clears pending queues, this also
        interrupts the chunk currently being written to aplay/sounddevice.
        The _playback_loop checks _stop_requested and exits the current
        chunk early.  The flag is auto-cleared when _playback_loop resumes.
        """
        self._stop_requested.set()
        self._kill_aplay()

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
            try:
                self._generate_audio(text, generation)
            except Exception as e:
                logger.error("TTS worker error: %s", e)
            finally:
                self.tts_text_queue.task_done()

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
            elif not self._run_async(self._generate_minimax(text, generation)):
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

    def _use_minimax_fallback(self, text: str, generation: int) -> None:
        """Use local or edge TTS as a fallback when MiniMax is unavailable."""
        if self._local_tts is not None:
            self._generate_local(text, generation)
        else:
            self._run_async(self._generate_edge(text, generation))

    def _run_async(self, coro) -> bool:
        """Run an async coroutine in a new event loop. Returns True on success."""
        loop = None
        try:
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(coro)
            return True
        except Exception as exc:
            logger.error("TTS async error: %s", exc)
            return False
        finally:
            if loop is not None:
                # Properly close any lingering async generators before shutting
                # down the loop, suppressing the "Task was destroyed but pending"
                # warning that appears when httpx SSE generators are abandoned.
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                loop.close()

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
        if self._local_sample_rate != self._sample_rate and len(samples) > 0:
            ratio = self._sample_rate / self._local_sample_rate
            new_len = int(len(samples) * ratio)
            indices = np.linspace(0, len(samples) - 1, new_len)
            samples = np.interp(indices, np.arange(len(samples)), samples)

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

    async def _generate_minimax(self, text: str, generation: int) -> None:
        """Synthesise via MiniMax T2A v2 — SSE hex-PCM stream → incremental buffer."""
        import httpx
        import json as _json

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

        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    body_text = await resp.aread()
                    logger.error("MiniMax TTS HTTP %d: %s", resp.status_code, body_text[:200])
                    return

                async for line in resp.aiter_lines():
                    if not self._is_generation_current(generation):
                        return
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
                        pcm_bytes = bytes.fromhex(hex_audio)
                        samples = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
                        if need_resample and len(samples) > 1:
                            ratio = self._sample_rate / self._minimax_sample_rate
                            new_len = max(1, int(len(samples) * ratio))
                            indices = np.linspace(0, len(samples) - 1, new_len)
                            samples = np.interp(indices, np.arange(len(samples)), samples)
                        if len(samples) > 0:
                            pending.append(samples)
                            pending_len += len(samples)
                            # Flush to playback buffer once we have enough samples
                            if pending_len >= _MIN_SAMPLES:
                                chunk = np.concatenate(pending) if len(pending) > 1 else pending[0]
                                with self._buffer_lock:
                                    self.tts_buffer.append(chunk)
                                pending.clear()
                                pending_len = 0
                    except (_json.JSONDecodeError, ValueError) as exc:
                        logger.debug("MiniMax TTS chunk parse: %s", exc)

        # Flush any remaining samples
        if pending and self._is_generation_current(generation):
            chunk = np.concatenate(pending) if len(pending) > 1 else pending[0]
            with self._buffer_lock:
                self.tts_buffer.append(chunk)

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
            except Exception:
                pass

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
                "TTS playback: device=%s, sample_rate=%d, aplay=%s",
                self._output_device if self._output_device is not None else "default",
                self._sample_rate,
                self._aplay_bin is not None,
            )

            # --- aplay persistent-process setup (computed once) ---
            _aplay_cmd: list[str] | None = None
            _preroll_bytes: bytes = b""
            if self._aplay_bin:
                _aplay_cmd = [
                    self._aplay_bin,
                    "-r", str(self._sample_rate),
                    "-f", "S16_LE",
                    "-c", "1",
                    "-q",
                ]
                if self._output_device is not None:
                    _aplay_cmd += ["-D", str(self._output_device)]
                # Pre-roll: 80 Hz tone at -24 dBFS for 400 ms.
                # USB DAC pop-suppression needs a signal in the detection
                # passband (>50 Hz) before it unmutes.  80 Hz is inaudible
                # through small robot speakers (cutoff ~120 Hz).
                _n = int(self._sample_rate * 0.40)
                _t = np.arange(_n, dtype=np.float32) / self._sample_rate
                _pr = (np.sin(2 * np.pi * 80.0 * _t) * 2000.0).astype(np.int16)
                _fade = int(self._sample_rate * 0.03)
                _pr[-_fade:] = (
                    _pr[-_fade:].astype(np.float32)
                    * np.linspace(1.0, 0.0, _fade, dtype=np.float32)
                ).astype(np.int16)
                _preroll_bytes = _pr.tobytes()

            # Persistent aplay process state — one process per utterance,
            # all chunks piped into its stdin without restart.
            _proc: subprocess.Popen | None = None  # type: ignore[type-arg]
            _need_preroll = True
            _empty_polls = 0
            _MAX_EMPTY_POLLS = 50  # 50 × 20 ms = 1 s drain window (SSE gaps can be 200-500 ms)
            _router_ctx = None  # saved output_session() context manager

            def _close_aplay() -> None:
                """Cleanly close the persistent aplay process."""
                nonlocal _proc, _need_preroll, _empty_polls, _router_ctx
                if _proc is None:
                    return
                try:
                    _proc.stdin.close()  # type: ignore[union-attr]
                except Exception:
                    pass
                try:
                    _proc.wait(timeout=5)
                except Exception:
                    _proc.kill()
                with self._aplay_lock:
                    self._aplay_proc = None
                _proc = None
                _need_preroll = True
                _empty_polls = 0
                if _router_ctx is not None:
                    try:
                        _router_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
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

                chunk = None
                with self._buffer_lock:
                    if self.tts_buffer:
                        chunk = self.tts_buffer.popleft()

                if chunk is not None and len(chunk) > 0:
                    _empty_polls = 0
                    # Apply volume
                    if self._volume != 1.0:
                        chunk = chunk * self._volume
                        np.clip(chunk, -1.0, 1.0, out=chunk)

                    if _aplay_cmd is not None:
                        pcm = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
                        dur = len(chunk) / self._sample_rate
                        logger.info(
                            "aplay: %d samples = %.3fs", len(chunk), dur
                        )
                        try:
                            # Start persistent aplay on first chunk
                            if _proc is None:
                                if self._audio_router is not None:
                                    _router_ctx = self._audio_router.output_session()
                                    _router_ctx.__enter__()
                                _proc = subprocess.Popen(
                                    _aplay_cmd, stdin=subprocess.PIPE
                                )
                                with self._aplay_lock:
                                    self._aplay_proc = _proc

                            # Pre-roll only on first chunk after process start
                            if _need_preroll:
                                _proc.stdin.write(_preroll_bytes)  # type: ignore[union-attr]
                                _need_preroll = False

                            _proc.stdin.write(pcm.tobytes())  # type: ignore[union-attr]
                            _proc.stdin.flush()  # type: ignore[union-attr]
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
                    else:
                        if self._audio_router is not None:
                            with self._audio_router.output_session():
                                sd.play(chunk, samplerate=self._sample_rate, device=self._output_device)
                                sd.wait()
                        else:
                            sd.play(chunk, samplerate=self._sample_rate, device=self._output_device)
                            sd.wait()
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
            with self._aplay_lock:
                self._aplay_proc = None
            # Always clear _is_playing on exit — prevents start_playback() from
            # getting permanently blocked and wait_done() from deadlocking when
            # the audio device is unavailable or throws.
            with self._playback_lock:
                self._is_playing = False

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

    def _clear_audio_buffer(self) -> None:
        with self._buffer_lock:
            self.tts_buffer.clear()
