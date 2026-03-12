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
from typing import Any

import numpy as np
import sounddevice as sd

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

    # Regex patterns for cleaning text before TTS
    _RE_EMOJI = re.compile(r'[\U00010000-\U0010ffff]')
    _RE_BOLD = re.compile(r'\*\*(.+?)\*\*')
    _RE_ITALIC = re.compile(r'\*(.+?)\*')
    _RE_CODE = re.compile(r'`(.+?)`')
    _RE_HEADER = re.compile(r'^#+\s*', flags=re.MULTILINE)
    _RE_LIST = re.compile(r'^[-*]\s+', flags=re.MULTILINE)
    _RE_IMG = re.compile(r'!\[.*?\]\(.*?\)')
    _RE_LINK = re.compile(r'\[(.+?)\]\(.*?\)')

    def __init__(self, config: dict[str, Any]) -> None:
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
        self._minimax_vol: float = float(config.get("minimax_vol", 1.0))
        self._minimax_pitch: int = int(config.get("minimax_pitch", 0))
        # Emotion: "" (auto), happy, sad, angry, fearful, disgusted, surprised, calm
        self._minimax_emotion: str = config.get("minimax_emotion", "")

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
        self._aplay_bin: str | None = shutil.which("aplay")

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

    def wait_done(self) -> None:
        """Block until all queued text has been synthesised and played."""
        self.tts_text_queue.join()
        while self._has_buffered_audio():
            time.sleep(0.05)
        # Wait for the last chunk to finish playing.
        # aplay: proc.communicate() is synchronous, but _aplay_proc is cleared
        # only after communicate() returns, so poll it.
        while self._aplay_proc is not None:
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
            if not self._run_async(self._generate_minimax(text, generation)):
                # MiniMax failed — auto-fallback to local or edge
                logger.warning("TTS minimax failed, falling back")
                if self._local_tts is not None:
                    self._generate_local(text, generation)
                else:
                    self._run_async(self._generate_edge(text, generation))
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
        all_samples: list[np.ndarray] = []

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
                            all_samples.append(samples)
                    except (_json.JSONDecodeError, ValueError) as exc:
                        logger.debug("MiniMax TTS chunk parse: %s", exc)

        if all_samples and self._is_generation_current(generation):
            combined = np.concatenate(all_samples)
            with self._buffer_lock:
                self.tts_buffer.append(combined)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _kill_aplay(self) -> None:
        """Terminate any running aplay subprocess (immediate interruption)."""
        proc = self._aplay_proc
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
            while self._is_playing:
                chunk = None
                with self._buffer_lock:
                    if self.tts_buffer:
                        chunk = self.tts_buffer.popleft()

                if chunk is not None and len(chunk) > 0:
                    # Apply volume
                    if self._volume != 1.0:
                        chunk = chunk * self._volume
                        np.clip(chunk, -1.0, 1.0, out=chunk)

                    if self._aplay_bin:
                        pcm = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
                        dur = len(chunk) / self._sample_rate
                        logger.info(
                            "aplay: %d samples = %.3fs", len(chunk), dur
                        )
                        cmd = [
                            self._aplay_bin,
                            "-r", str(self._sample_rate),
                            "-f", "S16_LE",
                            "-c", "1",
                            "-q",
                        ]
                        self._aplay_proc = subprocess.Popen(
                            cmd, stdin=subprocess.PIPE
                        )
                        self._aplay_proc.communicate(input=pcm.tobytes())
                        self._aplay_proc = None
                        logger.info("aplay: done")
                    else:
                        sd.play(chunk, samplerate=self._sample_rate, device=self._output_device)
                        sd.wait()
                else:
                    time.sleep(0.02)
        except Exception as e:
            logger.error("Playback error: %s", e)
        finally:
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
