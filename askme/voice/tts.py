"""TTS Engine - dual backend: local sherpa-onnx (fast) or edge-tts (fallback)."""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import os
import queue
import re
import sys
import threading
import time
from typing import Any

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-speech engine with two backends:

    - **local** (default): sherpa-onnx VITS/MeloTTS — ~0.5-1s latency, no network.
    - **edge**: Microsoft Edge TTS — ~3s latency, requires internet.

    Config dict expected keys (under voice.tts)::

        backend: str          - "local" or "edge" (default "local")
        # Local backend
        model_dir: str        - path to sherpa-onnx TTS model directory
        num_threads: int      - inference threads (default 4)
        speed: float          - speech speed (default 1.0)
        sid: int              - speaker ID (default 0)
        # Edge backend
        voice: str            - Edge TTS voice name (default "zh-CN-YunxiNeural")
        rate: str             - Speed adjustment (default "+0%")
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

        # Queues and buffers
        self.tts_text_queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
        self.tts_buffer: deque[np.ndarray] = deque()
        self._buffer_lock = threading.Lock()
        self._generation_lock = threading.Lock()
        self._generation = 0

        # Playback state
        self._is_playing = False
        self._playback_thread: threading.Thread | None = None

        # Local TTS engine (lazy init)
        self._local_tts: Any | None = None
        self._local_sample_rate: int = 0

        # Auto-detect backend
        if self._backend == "local" and not os.path.isdir(self._model_dir):
            logger.warning("Local TTS model not found at %s, falling back to edge-tts", self._model_dir)
            self._backend = "edge"

        if self._backend == "local":
            self._init_local_tts()

        logger.info("TTS backend: %s", self._backend)

        # Start TTS worker thread
        self._worker_thread = threading.Thread(target=self._tts_loop, daemon=True)
        self._worker_thread.start()

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
            self.tts_text_queue.put((self._get_generation(), clean))

    def start_playback(self) -> None:
        """Start the sounddevice output stream in a background thread."""
        if self._is_playing:
            return
        self._is_playing = True
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()

    def stop_playback(self) -> None:
        """Stop the sounddevice output stream."""
        self._is_playing = False
        if self._playback_thread is not None and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=1.0)
            self._playback_thread = None

    def wait_done(self) -> None:
        """Block until all queued text has been synthesised and played."""
        self.tts_text_queue.join()
        while self._has_buffered_audio():
            time.sleep(0.1)
        time.sleep(0.5)  # grace period for the last audio chunk

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

    def shutdown(self) -> None:
        """Signal the worker thread to exit and stop playback."""
        self.drain_buffers()
        self.tts_text_queue.put(None)
        self.stop_playback()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

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
        """Dispatch to local or edge backend."""
        if not self._is_generation_current(generation):
            logger.debug("TTS: dropping stale request before synthesis")
            return

        logger.info("TTS [%s] generating: %r", self._backend, text[:80])

        if self._backend == "local":
            self._generate_local(text, generation)
        else:
            # asyncio.run() in a worker thread on Windows needs SelectorEventLoop
            # (IocpProactor is only available on the main thread)
            try:
                loop = asyncio.new_event_loop()
                if sys.platform == "win32":
                    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                    loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._generate_edge(text, generation))
            except Exception as exc:
                logger.error("TTS edge synthesis error: %s", exc)
            finally:
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
    # Playback
    # ------------------------------------------------------------------

    def _playback_loop(self) -> None:
        """Run ``sd.OutputStream`` until ``stop_playback`` is called."""
        try:
            logger.info(
                "TTS playback: device=%s, sample_rate=%d",
                self._output_device if self._output_device is not None else "default",
                self._sample_rate,
            )
            with sd.OutputStream(
                channels=1,
                callback=self.play_audio_callback,
                dtype="float32",
                samplerate=self._sample_rate,
                blocksize=1024,
                device=self._output_device,
            ):
                while self._is_playing:
                    sd.sleep(100)
        except Exception as e:
            logger.error("Playback error: %s", e)

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
