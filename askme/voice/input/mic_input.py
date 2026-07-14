"""Microphone input module — device management + audio capture.

Encapsulates sounddevice InputStream, chunk reading, peak calculation,
and pre-roll buffer. Extracted from audio_agent.py for independent testing.

Supports mics with non-standard native rates (e.g. 48 kHz only USB devices)
via automatic resampling + signal conditioning (high-pass filter, AGC).

Uses callback-based InputStream (not blocking reads) because some USB audio
devices on ALSA/aarch64 produce silence with blocking sd.InputStream.read().

Usage::

    mic = MicInput(device=0, sample_rate=16000)
    with mic.open():
        chunk = mic.read_chunk()         # float32 array at sample_rate
        peak = mic.get_peak(chunk)       # int peak from int16
        int16 = mic.to_int16(chunk)      # int16 conversion
"""

from __future__ import annotations

import collections
import logging
import os
import queue
import shutil
import subprocess
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from askme.config import project_root

try:
    import sounddevice as sd
except ModuleNotFoundError:
    class _SoundDeviceStub:
        InputStream = None
        @staticmethod
        def query_devices(device: object = None, kind: object = None) -> object:
            return {}
    sd = _SoundDeviceStub()  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default chunk duration: 100ms per read
_DEFAULT_CHUNK_MS = 100

# Pre-roll buffer: keep recent chunks so VAD latency doesn't lose speech onset
_DEFAULT_PRE_ROLL_CHUNKS = 5


class MicInput:
    """Microphone input device wrapper.

    Manages the sounddevice InputStream lifecycle and provides
    clean chunk-based audio reading with pre-roll buffering.

    When ``mic_native_rate`` differs from ``sample_rate``, the mic is opened
    at the native rate and each chunk is processed through:
    high-pass filter → AGC → polyphase resample → output at ``sample_rate``.

    Config keys (under ``voice``)::

        input_device: int|str|null      - Device index or ALSA name (null=default)
        input_transport: str            - "auto", "sounddevice", or "usb_direct"
        asr.sample_rate: int            - Target sample rate for ASR (default 16000)
        mic_native_rate: int|null       - Mic hardware sample rate (null=same as asr)
        mic_channels: int               - Mic hardware channels (default 1)
        mic_channel_select: int         - Which channel to use (default 0)
        mic_highpass_hz: int            - High-pass filter cutoff (default 80)
        mic_agc_target_rms: float       - AGC target RMS (default 0.15, 0=off)
    """

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 16000,
        chunk_ms: int = _DEFAULT_CHUNK_MS,
        pre_roll_chunks: int = _DEFAULT_PRE_ROLL_CHUNKS,
        audio_router: Any | None = None,
        *,
        mic_native_rate: int | None = None,
        mic_channels: int = 1,
        mic_channel_select: int = 0,
        mic_highpass_hz: int = 80,
        mic_agc_target_rms: float = 0.15,
        input_transport: str = "auto",
        usb_audio_binary: str | None = None,
        usb_audio_source: str | None = None,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._chunk_ms = chunk_ms
        self._chunk_samples = int(chunk_ms / 1000 * sample_rate)
        self._audio_router = audio_router
        self._stream: sd.InputStream | None = None
        self._input_transport = input_transport.lower()
        if self._input_transport not in {"auto", "sounddevice", "usb_direct"}:
            logger.warning("Unknown mic input_transport=%s; using auto", input_transport)
            self._input_transport = "auto"

        if mic_channels < 1:
            raise ValueError("mic_channels must be at least 1")
        if mic_channel_select < 0 or mic_channel_select >= mic_channels:
            raise ValueError(
                "mic_channel_select must be within configured mic_channels"
            )

        # Resampling pipeline config
        self._native_rate = mic_native_rate or sample_rate
        self._native_channels = mic_channels
        self._channel_select = mic_channel_select
        self._needs_resample = self._native_rate != self._sample_rate
        self._highpass_hz = mic_highpass_hz
        self._agc_target = mic_agc_target_rms

        # Native chunk size (at hardware rate)
        self._native_chunk = int(chunk_ms / 1000 * self._native_rate)

        # Streaming filter state (initialized on open)
        self._filter_state: np.ndarray | None = None
        self._filter_sos: np.ndarray | None = None
        self._agc_gain: float = 1.0

        # Callback-based audio queue (replaces blocking stream.read)
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._usb_audio_binary: str | None = usb_audio_binary
        self._usb_audio_source: str | None = usb_audio_source
        self._usb_audio_build_failed = False
        self._usb_audio_proc: subprocess.Popen | None = None  # type: ignore[type-arg]

        # Pre-roll buffer: recent chunks for catching speech onset
        self.pre_roll: collections.deque[np.ndarray] = collections.deque(
            maxlen=pre_roll_chunks
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples

    @property
    def is_open(self) -> bool:
        return self._stream is not None or self._usb_audio_proc is not None

    def _init_pipeline(self) -> None:
        """Initialize the signal processing pipeline for resampling mics."""
        if not self._needs_resample:
            return

        # High-pass IIR filter to remove DC offset
        if self._highpass_hz > 0:
            from scipy.signal import butter, sosfilt_zi
            self._filter_sos = butter(
                2, self._highpass_hz, btype="high",
                fs=self._native_rate, output="sos",
            )
            self._filter_state = sosfilt_zi(self._filter_sos)
            # Expand zi for our mono signal
            self._filter_state = self._filter_state * 0.0

        # AGC gain
        self._agc_gain = 1.0

        logger.info(
            "MicInput pipeline: %dHz %dch → HPF@%dHz → AGC(%.2f) → %dHz",
            self._native_rate, self._native_channels,
            self._highpass_hz, self._agc_target, self._sample_rate,
        )

    def _process_chunk(self, raw: np.ndarray) -> np.ndarray:
        """Process a native-rate chunk through the signal pipeline.

        Args:
            raw: float32 array from InputStream (may be multi-channel).

        Returns:
            float32 mono array at ``self._sample_rate``.
        """
        # Channel select
        if raw.ndim == 2:
            audio = raw[:, self._channel_select]
        else:
            audio = raw.reshape(-1)

        # High-pass filter (streaming, preserves state across chunks)
        if self._filter_sos is not None:
            from scipy.signal import sosfilt
            audio, self._filter_state = sosfilt(
                self._filter_sos, audio, zi=self._filter_state,
            )
            audio = audio.astype(np.float32)

        # AGC: smooth gain adjustment targeting self._agc_target RMS
        if self._agc_target > 0:
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms > 1e-5:
                desired_gain = self._agc_target / rms
                # Smooth gain changes (attack=fast, release=slow)
                if desired_gain < self._agc_gain:
                    self._agc_gain = 0.3 * desired_gain + 0.7 * self._agc_gain
                else:
                    self._agc_gain = 0.05 * desired_gain + 0.95 * self._agc_gain
                # Clamp gain to prevent saturation
                self._agc_gain = min(self._agc_gain, 5.0)
            audio = np.clip(audio * self._agc_gain, -1.0, 1.0)

        # Resample with scipy polyphase (deterministic, fixed output length)
        from math import gcd

        from scipy.signal import resample_poly
        up = self._sample_rate
        down = self._native_rate
        g = gcd(up, down)
        audio = resample_poly(audio, up // g, down // g).astype(np.float32)

        # Clip after resample: sinc interpolation can overshoot [-1, 1]
        return np.clip(audio, -1.0, 1.0)

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: Any, status: Any
    ) -> None:
        """InputStream callback — pushes raw audio chunks to the queue."""
        if status:
            logger.debug("MicInput callback status: %s", status)
        self._audio_queue.put(indata.copy())

    def start(self) -> None:
        """Open mic stream persistently. Pair with stop().

        The mic stays open across listen/speak cycles so VAD can detect
        barge-in during TTS playback and LLM processing.
        """
        if self.is_open:
            return  # already open

        if self._should_use_usb_direct():
            self._start_usb_direct()
            self.pre_roll.clear()
            if self._audio_router is not None:
                self._audio_router.wait_for_input_ready(timeout=10.0)
            logger.info("MicInput: started MCP01 direct USB capture")
            return

        self._init_pipeline()

        open_rate = self._native_rate if self._needs_resample else self._sample_rate
        open_channels = self._native_channels if self._needs_resample else 1
        blocksize = self._native_chunk if self._needs_resample else self._chunk_samples

        self._flush_queue()

        stream = sd.InputStream(
            device=self._device,
            channels=open_channels,
            dtype="float32",
            samplerate=open_rate,
            blocksize=blocksize,
            callback=self._audio_callback,
        )
        stream.start()
        self._stream = stream
        self.pre_roll.clear()

        if self._audio_router is not None:
            self._audio_router.wait_for_input_ready(timeout=10.0)

        logger.info("MicInput: started (persistent)")

    def stop(self) -> None:
        """Close mic stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("MicInput: stopped")
        if self._usb_audio_proc is not None:
            proc = self._usb_audio_proc
            self._usb_audio_proc = None
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                logger.exception("[MicInput] USB proc terminate failed, trying kill")
                try:
                    proc.kill()
                except Exception as exc:
                    logger.debug("MCP01 USB capture kill failed (ignored): %s", exc)
            logger.info("MicInput: stopped MCP01 direct USB capture")

    def _flush_queue(self) -> None:
        """Discard stale audio chunks from the callback queue."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    @contextmanager
    def open(self) -> Generator[MicInput, None, None]:
        """Open the microphone as a context manager.

        If the mic is already persistently open (via start()), yields
        without reopening. Otherwise opens/closes for backward compat.
        """
        already_open = self.is_open
        if not already_open:
            self.start()
        try:
            yield self
        finally:
            if not already_open:
                self.stop()

    def read_chunk(self) -> np.ndarray:
        """Read one chunk of audio from the microphone.

        Returns float32 array of shape ``(chunk_samples,)`` at ``sample_rate``.
        Raises RuntimeError if mic is not open.
        """
        if self._usb_audio_proc is not None:
            return self._read_usb_direct_chunk()

        if self._stream is None:
            raise RuntimeError("MicInput not open — use 'with mic.open():'")

        # Get raw audio from callback queue (blocks until available)
        try:
            raw = self._audio_queue.get(timeout=2.0)
        except queue.Empty:
            # Return silence if no data (shouldn't happen in normal operation)
            logger.warning("MicInput: no audio data received (timeout)")
            if self._needs_resample:
                return np.zeros(self._chunk_samples, dtype=np.float32)
            return np.zeros(self._chunk_samples, dtype=np.float32)

        if self._needs_resample:
            return self._process_chunk(raw)
        else:
            return raw.reshape(-1)

    def _usb_direct_source_path(self) -> Path:
        if self._usb_audio_source:
            return Path(self._usb_audio_source)
        return project_root() / "scripts" / "bench" / "mcp01_usb_audio_libusb.c"

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
            logger.warning("MCP01 USB input requires gcc and pkg-config")
            self._usb_audio_build_failed = True
            return None

        pkg = subprocess.run(
            ["pkg-config", "--cflags", "--libs", "libusb-1.0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if pkg.returncode != 0:
            logger.warning("MCP01 USB input requires libusb-1.0 development files")
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
            logger.warning("MCP01 USB input helper build failed: %s", build.stderr.strip())
            self._usb_audio_build_failed = True
            return None
        return str(binary)

    def _alsa_input_available(self) -> bool:
        """Return False when ALSA/PortAudio clearly has no input device."""
        if os.name != "posix":
            return True

        cards_path = Path("/proc/asound/cards")
        try:
            cards_text = cards_path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            cards_text = ""
        if "no soundcards" in cards_text:
            return False

        required_channels = max(1, int(self._native_channels or 1))
        if self._device is not None:
            try:
                device_info = sd.query_devices(self._device, kind="input")
            except Exception as exc:
                logger.debug(
                    "Configured input device is unavailable through PortAudio: %s",
                    exc,
                )
                return False
            if isinstance(device_info, dict):
                return int(device_info.get("max_input_channels", 0) or 0) >= required_channels
            return False

        try:
            devices = sd.query_devices()
        except Exception as exc:
            logger.debug("No input devices available through PortAudio: %s", exc)
            return False

        if isinstance(devices, dict):
            return int(devices.get("max_input_channels", 0) or 0) >= required_channels
        try:
            return any(
                int(device.get("max_input_channels", 0) or 0) >= required_channels
                for device in devices
            )
        except TypeError:
            return False

    def _mcp01_visible(self) -> bool:
        if os.name != "posix":
            return False
        if shutil.which("lsusb") is None:
            return True
        try:
            result = subprocess.run(
                ["lsusb"],
                capture_output=True,
                text=True,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and "17ef:a03b" in result.stdout.lower()

    def _should_use_usb_direct(self) -> bool:
        if self._input_transport == "usb_direct":
            return True
        if self._input_transport != "auto":
            return False
        return not self._alsa_input_available() and self._mcp01_visible()

    def _start_usb_direct(self) -> None:
        binary = self._ensure_usb_audio_binary()
        if binary is None:
            raise RuntimeError("MCP01 USB input helper is not available")

        self._usb_audio_proc = subprocess.Popen(
            [binary, "--play-ms", "0", "--capture-ms", "0", "--capture-stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _read_usb_direct_chunk(self) -> np.ndarray:
        proc = self._usb_audio_proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("MicInput not open")

        usb_rate = 16000
        usb_samples = int(self._chunk_ms / 1000 * usb_rate)
        byte_count = usb_samples * 2
        data = bytearray()
        while len(data) < byte_count:
            chunk = proc.stdout.read(byte_count - len(data))
            if not chunk:
                stderr = ""
                if proc.poll() is not None and proc.stderr is not None:
                    stderr = proc.stderr.read().decode(errors="replace").strip()
                self.stop()
                raise RuntimeError(f"MCP01 USB input stopped unexpectedly: {stderr}")
            data.extend(chunk)

        samples = np.frombuffer(bytes(data), dtype="<i2").astype(np.float32) / 32768.0
        if self._sample_rate != usb_rate:
            samples = self._resample_mono(samples, usb_rate, self._sample_rate)
        return np.clip(samples, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _resample_mono(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return samples.astype(np.float32)
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(source_rate, target_rate)
        return resample_poly(samples, target_rate // g, source_rate // g).astype(np.float32)

    def buffer_pre_roll(self, samples: np.ndarray) -> None:
        """Add a chunk to the pre-roll buffer (recent silence for speech onset)."""
        self.pre_roll.append(samples.copy())

    def flush_pre_roll(self) -> list[np.ndarray]:
        """Return and clear the pre-roll buffer contents."""
        chunks = list(self.pre_roll)
        self.pre_roll.clear()
        return chunks

    @staticmethod
    def to_int16(samples: np.ndarray) -> np.ndarray:
        """Convert float32 samples to int16."""
        return (samples * 32768).clip(-32768, 32767).astype(np.int16)

    @staticmethod
    def get_peak(samples_int16: np.ndarray) -> int:
        """Get the peak amplitude from int16 samples."""
        return int(np.max(np.abs(samples_int16)))

    @classmethod
    def from_config(cls, config: dict[str, Any], audio_router: Any = None) -> MicInput:
        """Create MicInput from askme voice config dict."""
        voice_cfg = config.get("voice", {})

        raw_input = voice_cfg.get("input_device", None)
        if raw_input is None:
            device = None
        elif isinstance(raw_input, int):
            device = raw_input
        else:
            try:
                device = int(raw_input)
            except (ValueError, TypeError):
                device = str(raw_input)

        sample_rate = int(voice_cfg.get("asr", {}).get("sample_rate", 16000))

        # Resampling pipeline config
        native_rate_raw = voice_cfg.get("mic_native_rate", None)
        native_rate = int(native_rate_raw) if native_rate_raw is not None else None
        mic_channels = int(voice_cfg.get("mic_channels", 1))
        channel_select = int(voice_cfg.get("mic_channel_select", 0))
        highpass_hz = int(voice_cfg.get("mic_highpass_hz", 80))
        agc_target = float(voice_cfg.get("mic_agc_target_rms", 0.15))
        input_transport = str(voice_cfg.get("input_transport", "auto"))
        usb_audio_binary = voice_cfg.get("usb_audio_binary")
        usb_audio_source = voice_cfg.get("usb_audio_source")

        return cls(
            device=device,
            sample_rate=sample_rate,
            audio_router=audio_router,
            mic_native_rate=native_rate,
            mic_channels=mic_channels,
            mic_channel_select=channel_select,
            mic_highpass_hz=highpass_hz,
            mic_agc_target_rms=agc_target,
            input_transport=input_transport,
            usb_audio_binary=usb_audio_binary,
            usb_audio_source=usb_audio_source,
        )
