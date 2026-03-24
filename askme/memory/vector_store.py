"""Minimal local vector store with sentence-transformers embedding.

Graceful degradation: works without sentence-transformers installed —
``available`` returns False and all queries return empty results.

Persistence: JSON file at ``data/memory/vectors/store.json``.

Thread-safe: mutations guarded by ``threading.Lock``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy availability check — do NOT import sentence_transformers at module level
# because it pulls in torch/transformers which can take 30+ seconds.
_ST_AVAILABLE: bool | None = None  # None = not yet checked


def _check_st_available() -> bool:
    """Check if sentence-transformers is importable (cached after first call)."""
    global _ST_AVAILABLE
    if _ST_AVAILABLE is not None:
        return _ST_AVAILABLE
    try:
        import importlib.util

        _ST_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None
    except Exception:
        _ST_AVAILABLE = False
    return _ST_AVAILABLE


class VectorStore:
    """Lightweight vector store using sentence-transformers + numpy cosine similarity."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        store_path: str | Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._model: Any = None  # lazy-loaded SentenceTransformer
        self._store_path = Path(store_path) if store_path else None

        self._texts: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._embeddings: np.ndarray | None = None  # shape (N, dim)

        self._lock = threading.Lock()

        if self._store_path:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # -- Properties -----------------------------------------------------------

    @property
    def available(self) -> bool:
        """Whether sentence-transformers is installed and usable."""
        return _check_st_available()

    @property
    def size(self) -> int:
        """Number of stored entries."""
        return len(self._texts)

    # -- Model ----------------------------------------------------------------

    def _get_model(self) -> Any:
        """Lazy-load the embedding model."""
        if self._model is None:
            if not _check_st_available():
                raise RuntimeError("sentence-transformers is not installed")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embedding vectors."""
        model = self._get_model()
        return model.encode(texts, normalize_embeddings=True)

    # -- Public API -----------------------------------------------------------

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a text entry with optional metadata.

        No-ops when sentence-transformers is unavailable.
        """
        if not _check_st_available():
            return
        if not text.strip():
            return

        try:
            vec = self._encode([text])[0]  # shape (dim,)
        except Exception as exc:
            logger.warning("[VectorStore] Encoding failed: %s", exc)
            return

        with self._lock:
            self._texts.append(text)
            self._metadata.append(metadata or {})
            if self._embeddings is None:
                self._embeddings = vec.reshape(1, -1)
            else:
                self._embeddings = np.vstack([self._embeddings, vec.reshape(1, -1)])

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for the most similar entries to *query*.

        Returns a list of dicts: ``{"text": ..., "score": ..., "metadata": ...}``.
        Returns empty list when unavailable or empty.
        """
        if not _check_st_available() or self._embeddings is None or not query.strip():
            return []

        try:
            q_vec = self._encode([query])[0]  # shape (dim,)
        except Exception as exc:
            logger.warning("[VectorStore] Query encoding failed: %s", exc)
            return []

        with self._lock:
            if self._embeddings is None or len(self._texts) == 0:
                return []
            # Cosine similarity (embeddings are already L2-normalized)
            scores = self._embeddings @ q_vec
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                idx_int = int(idx)
                results.append({
                    "text": self._texts[idx_int],
                    "score": float(scores[idx_int]),
                    "metadata": self._metadata[idx_int],
                })
            return results

    # -- Persistence ----------------------------------------------------------

    def save(self) -> None:
        """Persist texts, metadata, and embeddings to JSON."""
        if self._store_path is None:
            return
        with self._lock:
            data = {
                "model": self._model_name,
                "texts": self._texts,
                "metadata": self._metadata,
                "embeddings": self._embeddings.tolist() if self._embeddings is not None else [],
                "saved_at": time.time(),
            }
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            logger.debug("[VectorStore] Saved %d entries to %s", len(self._texts), self._store_path)
        except Exception as exc:
            logger.warning("[VectorStore] Save failed: %s", exc)

    def _load(self) -> None:
        """Load from JSON persistence file."""
        if self._store_path is None or not self._store_path.exists():
            return
        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._texts = data.get("texts", [])
            self._metadata = data.get("metadata", [])
            emb_list = data.get("embeddings", [])
            if emb_list:
                self._embeddings = np.array(emb_list, dtype=np.float32)
            else:
                self._embeddings = None
            logger.info("[VectorStore] Loaded %d entries from %s", len(self._texts), self._store_path)
        except Exception as exc:
            logger.warning("[VectorStore] Load failed: %s", exc)
