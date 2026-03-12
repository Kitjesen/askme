"""AudioRouter — exclusive audio device ownership and error classification.

On ALSA single-device systems (e.g. sunrise aarch64 with MCP01 USB Audio),
the output stream (aplay subprocess) and input stream (sd.InputStream) share
the same physical hardware.  Without coordination, the mic stream accumulates
an XRUN buffer overrun while aplay holds the device, then fails on the next
open.

AudioRouter serialises them: the mic waits for TTS to finish before opening
a new stream, eliminating the XRUN cascade.  It also classifies exceptions
from the audio subsystem so the voice loop can apply the right recovery
policy instead of treating every error as a user-visible fault.

Ownership contract
------------------
- OUTPUT holds the device inside ``output_session()``.
- INPUT calls ``wait_for_input_ready()`` before opening ``sd.InputStream``.
- Both sides are optional: passing ``audio_router=None`` to TTSEngine /
  AudioAgent disables coordination (legacy behaviour, safe for tests).

Thread-safety
-------------
``threading.Event`` guarantees visibility across threads without polling.
The re-entrancy depth counter (``_depth``) handles the case where a single
playback sequence calls multiple chunk-level output_sessions — the event is
cleared exactly once (on first acquire) and set exactly once (on last
release).
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from enum import Enum
from typing import Generator

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Error taxonomy
# ──────────────────────────────────────────────────────────────────────────────


class AudioErrorKind(Enum):
    """Classification of audio-subsystem errors by recovery policy.

    Attributes
    ----------
    XRUN:
        ALSA buffer overrun/underrun or post-aplay stream-state corruption.
        Recovery: close stream, reopen immediately (silent retry).  No user
        notification — this is a normal transient on half-duplex hardware.
    DEVICE_LOST:
        USB card disconnected or kernel removed the device node.
        Recovery: long backoff (5 s), then retry.  Notify user once.
    DEVICE_BUSY:
        Another OS process holds exclusive ALSA access, or channel/rate
        mismatch (often caused by a competing process).
        Recovery: short backoff (2 s), retry silently.
    TTS_FAIL:
        TTS backend (MiniMax / edge-tts / sherpa-onnx / aplay) failed.
        Recovery: retry quickly (0.5 s).  Microphone is unaffected.
    UNKNOWN:
        Unclassified error — apply standard consecutive-error escalation.
    """

    XRUN = "xrun"
    DEVICE_LOST = "device_lost"
    DEVICE_BUSY = "device_busy"
    TTS_FAIL = "tts_fail"
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Ownership manager
# ──────────────────────────────────────────────────────────────────────────────


class AudioRouter:
    """Coordinates exclusive audio device access between TTS output and mic input.

    Usage
    -----
    Construct once and inject into both TTSEngine and AudioAgent::

        router = AudioRouter()
        tts    = TTSEngine(cfg, audio_router=router)
        agent  = AudioAgent(cfg, audio_router=router)

    In TTSEngine (inside the playback worker, per-chunk)::

        with router.output_session():
            aplay_proc.communicate(input=pcm_bytes)

    In AudioAgent (inside listen_loop, before opening the mic stream)::

        router.wait_for_input_ready()
        with sd.InputStream(...) as mic:
            ...
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth: int = 0            # re-entrancy depth counter
        # Set   = output is idle  → safe to open mic
        # Clear = output is active → mic must wait
        self._input_ready = threading.Event()
        self._input_ready.set()         # start idle

    # ── Output ownership ──────────────────────────────────────────────────

    @contextmanager
    def output_session(self) -> Generator[None, None, None]:
        """Hold exclusive OUTPUT ownership for the duration of the block.

        Re-entrant: nested ``output_session`` calls are reference-counted so
        ``_input_ready`` is cleared on the *first* entry and set on the
        *last* exit only.  A ``try/finally`` inside the context manager
        guarantees the event is always released even if the body raises.
        """
        with self._lock:
            self._depth += 1
            if self._depth == 1:
                self._input_ready.clear()
                logger.debug("AudioRouter: OUTPUT acquired (depth=1)")
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1
                if self._depth == 0:
                    self._input_ready.set()
                    logger.debug("AudioRouter: OUTPUT released (depth=0)")

    @property
    def is_output_active(self) -> bool:
        """True while an ``output_session`` is currently held."""
        return not self._input_ready.is_set()

    # ── Input readiness ───────────────────────────────────────────────────

    def wait_for_input_ready(self, timeout: float = 10.0) -> bool:
        """Block until the output (TTS) has finished and the device is free.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.  10 s is generous for any reasonable
            TTS utterance; callers should log a warning and proceed anyway
            on timeout rather than hanging indefinitely.

        Returns
        -------
        bool
            ``True`` if the device became free within *timeout* seconds,
            ``False`` if the wait timed out.
        """
        ready = self._input_ready.wait(timeout=timeout)
        if not ready:
            logger.warning(
                "AudioRouter: output did not finish within %.0fs — "
                "opening mic stream anyway (best-effort recovery).",
                timeout,
            )
        return ready

    # ── Error classification ──────────────────────────────────────────────

    _XRUN_PATTERNS: tuple[str, ...] = (
        "xrun",
        "paerrorcode -9999",
        "unanticipated host error",
        "illegal combination",
        "paerrorcode -9993",
        "alsa_snd_pcm_start",
        "alsarestart",
        "handle_xrun",
        "waitforframes",
    )

    _DEVICE_LOST_PATTERNS: tuple[str, ...] = (
        "cannot get card index",
        "no such device",
        "no such file or directory",
        "card index",
    )

    _DEVICE_BUSY_PATTERNS: tuple[str, ...] = (
        "device busy",
        "resource busy",
        "ebusy",
        "invalid number of channels",
        "paerrorcode -9998",
        "invalid sample rate",
        "paerrorcode -9997",
    )

    _TTS_FAIL_PATTERNS: tuple[str, ...] = (
        "minimax",
        "edge_tts",
        "edge-tts",
        "aplay",
        "tts",
        "t2a",
        "speech",
    )

    @classmethod
    def classify_error(cls, exc: BaseException) -> AudioErrorKind:
        """Classify an audio exception into a recovery category.

        The classification is heuristic (substring matching on the exception
        message).  XRUN is checked first because its patterns are the most
        specific and most frequent on sunrise aarch64.

        Parameters
        ----------
        exc:
            The exception caught by the voice loop or TTS worker.

        Returns
        -------
        AudioErrorKind
            The recovery category.  Callers should use this to select
            backoff duration, log level, and whether to notify the user.
        """
        msg = str(exc).lower()
        for pat in cls._XRUN_PATTERNS:
            if pat in msg:
                return AudioErrorKind.XRUN
        for pat in cls._DEVICE_LOST_PATTERNS:
            if pat in msg:
                return AudioErrorKind.DEVICE_LOST
        for pat in cls._DEVICE_BUSY_PATTERNS:
            if pat in msg:
                return AudioErrorKind.DEVICE_BUSY
        for pat in cls._TTS_FAIL_PATTERNS:
            if pat in msg:
                return AudioErrorKind.TTS_FAIL
        return AudioErrorKind.UNKNOWN
