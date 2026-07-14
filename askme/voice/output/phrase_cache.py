"""Persistent PCM cache for deterministic voice replies."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CachedPhrase:
    samples: np.ndarray
    sample_rate: int


class PhraseAudioCache:
    """Small disk-backed cache with an in-memory hot set."""

    def __init__(self, directory: str | Path, *, enabled: bool = True) -> None:
        self._enabled = bool(enabled)
        self._directory = Path(directory).expanduser()
        self._memory: dict[str, CachedPhrase] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def directory(self) -> Path:
        return self._directory

    def get(self, key: str) -> CachedPhrase | None:
        if not self._enabled or not key:
            return None
        with self._lock:
            cached = self._memory.get(key)
        if cached is not None:
            return CachedPhrase(cached.samples.copy(), cached.sample_rate)

        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as payload:
                samples = np.asarray(payload["samples"], dtype=np.float32)
                sample_rate = int(payload["sample_rate"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if samples.ndim != 1 or len(samples) == 0 or sample_rate <= 0:
            return None
        if not np.all(np.isfinite(samples)):
            return None
        stored = CachedPhrase(samples.copy(), sample_rate)
        with self._lock:
            self._memory[key] = stored
        return CachedPhrase(stored.samples.copy(), stored.sample_rate)

    def put(self, key: str, samples: np.ndarray, sample_rate: int) -> bool:
        if not self._enabled or not key or sample_rate <= 0:
            return False
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1 or len(audio) == 0 or not np.all(np.isfinite(audio)):
            return False
        stored = CachedPhrase(audio.copy(), int(sample_rate))
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            target = self._path_for(key)
            temporary = target.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                samples=stored.samples,
                sample_rate=np.asarray(stored.sample_rate, dtype=np.int32),
            )
            os.replace(temporary, target)
        except OSError:
            return False
        with self._lock:
            self._memory[key] = stored
        return True

    def _path_for(self, key: str) -> Path:
        safe = "".join(ch for ch in key if ch.isalnum() or ch in {"-", "_"})
        return self._directory / f"{safe}.npz"


__all__ = ["CachedPhrase", "PhraseAudioCache"]
