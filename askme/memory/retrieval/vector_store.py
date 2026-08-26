"""Minimal local vector store with ONNX-based embedding via fastembed.

Uses ``fastembed.TextEmbedding`` (ONNX Runtime, no PyTorch) instead of the
heavy ``sentence_transformers`` (which pulls in torch/transformers — ~17 s
import alone). Runtime loading is strict ``local_files_only``: deployments must
pre-populate the ``Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q``
snapshot in ``FASTEMBED_CACHE_PATH`` or the system temporary directory's
``fastembed_cache``. The snapshot must contain the ONNX model plus its config
and tokenizer artifacts; startup and retrieval never download them.

Graceful degradation: works without fastembed installed —
``available`` returns False and all queries return empty results.

Persistence: JSON file at ``data/memory/vectors/store.json``.

Thread-safe: mutations guarded by ``threading.Lock``.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Lazy availability check — do NOT import fastembed at module level so the
# embedding backend is only loaded when actually needed.
_FE_AVAILABLE: bool | None = None  # None = not yet checked

# Global model cache keyed by model name (fastembed canonical form).
_MODEL_CACHE: dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()

# fastembed canonical model name for the same architecture.
_FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_FASTEMBED_HF_REPO = "qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
_FASTEMBED_REQUIRED_ARTIFACTS = (
    "config.json",
    "model_optimized.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _check_fastembed_available() -> bool:
    """Check if fastembed is importable (cached after first call)."""
    global _FE_AVAILABLE
    if _FE_AVAILABLE is not None:
        return _FE_AVAILABLE
    try:
        import importlib.util

        _FE_AVAILABLE = importlib.util.find_spec("fastembed") is not None
    except ImportError:
        _FE_AVAILABLE = False
    return _FE_AVAILABLE


def _fastembed_model_status(
    *,
    cache_dir: str | Path | None = None,
    dependency_installed: bool | None = None,
) -> dict[str, Any]:
    """Return a network-free snapshot of local FastEmbed model readiness.

    FastEmbed stores this model in the Hugging Face cache rooted at its own
    cache directory. Resolve the standard cache layout directly so readiness
    checks do not depend on a particular ``huggingface_hub`` API version.
    Requiring every artifact consumed by FastEmbed's ONNX text loader avoids
    treating an installed wheel as a runnable embedding backend.
    """

    installed = (
        _check_fastembed_available()
        if dependency_installed is None
        else bool(dependency_installed)
    )
    resolved_cache = Path(
        cache_dir
        or os.getenv("FASTEMBED_CACHE_PATH")
        or (Path(tempfile.gettempdir()) / "fastembed_cache")
    ).expanduser()
    status: dict[str, Any] = {
        "dependency_installed": installed,
        "model": _FASTEMBED_MODEL,
        "source_repo": _FASTEMBED_HF_REPO,
        "cache_dir": str(resolved_cache),
        "ready": False,
        "cached": False,
        "model_path": "",
        "missing_artifacts": list(_FASTEMBED_REQUIRED_ARTIFACTS),
        "reason": "dependency_missing" if not installed else "model_artifacts_missing",
        "check_mode": "huggingface_local_cache",
        "network_checked": False,
    }
    if not installed:
        return status
    if _FASTEMBED_MODEL in _MODEL_CACHE:
        status.update(
            ready=True,
            cached=True,
            missing_artifacts=[],
            reason="model_loaded",
        )
        return status

    resolved: dict[str, Path] = {}
    try:
        repo_cache = resolved_cache / f"models--{_FASTEMBED_HF_REPO.replace('/', '--')}"
        revision = (repo_cache / "refs" / "main").read_text(encoding="utf-8").strip()
        if revision and Path(revision).name == revision and revision not in {".", ".."}:
            snapshot = repo_cache / "snapshots" / revision
            for filename in _FASTEMBED_REQUIRED_ARTIFACTS:
                artifact = snapshot / filename
                if artifact.is_file() and artifact.stat().st_size > 0:
                    resolved[filename] = artifact
    except FileNotFoundError:
        pass
    except OSError as exc:
        status["reason"] = f"local_cache_error:{type(exc).__name__}"
        return status

    missing = [
        name for name in _FASTEMBED_REQUIRED_ARTIFACTS if name not in resolved
    ]
    status["missing_artifacts"] = missing
    if missing:
        return status
    status.update(
        ready=True,
        cached=True,
        model_path=str(resolved["model_optimized.onnx"].parent),
        reason="local_model_ready",
    )
    return status


def _top_score_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Return indices for the highest scores in descending score order.

    Uses ``argpartition`` for bounded top-k selection so large stores avoid
    sorting every row when callers only need a small retrieval window.
    """
    safe_top_k = int(top_k)
    if safe_top_k <= 0 or scores.size == 0:
        return np.array([], dtype=np.int64)

    result_count = min(safe_top_k, int(scores.size))
    if result_count == int(scores.size):
        return np.argsort(scores)[::-1]

    candidate_indices = np.argpartition(scores, -result_count)[-result_count:]
    ordered_candidates = np.argsort(scores[candidate_indices])[::-1]
    return candidate_indices[ordered_candidates]


class VectorStore:
    """Lightweight vector store using ONNX embedding + numpy cosine similarity.

    Uses ``fastembed`` (ONNX Runtime) by default, falling back gracefully
    when it is not installed.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        store_path: str | Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._model: Any = None  # lazy-loaded TextEmbedding instance
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
    def dependency_installed(self) -> bool:
        """Whether the FastEmbed Python dependency is installed."""
        return _check_fastembed_available()

    @property
    def model_status(self) -> dict[str, Any]:
        """Return network-free local model readiness evidence."""
        return _fastembed_model_status(
            dependency_installed=self.dependency_installed,
        )

    @property
    def available(self) -> bool:
        """Whether FastEmbed and its required local model are usable."""
        return bool(self.model_status["ready"])

    @property
    def size(self) -> int:
        """Number of stored entries."""
        return len(self._texts)

    # -- Model ----------------------------------------------------------------

    def _get_model(self) -> Any:
        """Lazy-load the ONNX embedding model via fastembed.

        The model is cached globally so multiple VectorStore instances share
        the same underlying ONNX session.
        """
        global _MODEL_CACHE
        if self._model is not None:
            return self._model

        if not _check_fastembed_available():
            raise RuntimeError("fastembed is not installed (pip install fastembed)")

        cache_key = _FASTEMBED_MODEL
        with _MODEL_LOCK:
            if cache_key not in _MODEL_CACHE:
                # Defer the import until we actually need the model so startup
                # stays fast when the vector store is never queried.
                from fastembed import TextEmbedding

                t0 = time.perf_counter()
                # Product deployments use CPU for the compact embedding model.
                # Explicit providers prevent onnxruntime-gpu from probing a
                # partially installed CUDA stack and printing an error per boot.
                _MODEL_CACHE[cache_key] = TextEmbedding(
                    cache_key,
                    providers=["CPUExecutionProvider"],
                    cuda=False,
                    local_files_only=True,
                )
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info(
                    "[VectorStore] ONNX embedding model loaded in %.0f ms", elapsed
                )
            self._model = _MODEL_CACHE[cache_key]
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to L2-normalised embedding vectors."""
        model = self._get_model()
        # fastembed returns List[np.ndarray]; each row is already normalised
        embeddings = list(model.embed(texts))
        if not embeddings:
            return np.empty((0, 384), dtype=np.float32)
        return np.asarray(embeddings, dtype=np.float32)

    # -- Public API -----------------------------------------------------------

    def add(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a text entry with optional metadata.

        No-ops when the embedding backend is unavailable.
        """
        if not _check_fastembed_available():
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
        if not _check_fastembed_available() or self._embeddings is None or not query.strip():
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
            top_indices = _top_score_indices(scores, top_k)

            results = []
            for idx in top_indices:
                idx_int = int(idx)
                results.append({
                    "text": self._texts[idx_int],
                    "score": float(scores[idx_int]),
                    "metadata": self._metadata[idx_int],
                })
            return results

    def list_records(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """Return stored records for admin/catalog views.

        This is intentionally metadata-preserving and does not filter deleted or
        draft records. Retrieval filtering lives in MemoryBridge.
        """
        safe_limit = max(1, min(int(limit), 500))
        safe_offset = max(0, int(offset))
        with self._lock:
            records: list[dict[str, Any]] = []
            for idx in range(safe_offset, min(len(self._texts), safe_offset + safe_limit)):
                metadata = dict(self._metadata[idx] or {})
                records.append({
                    "index": idx,
                    "text": self._texts[idx],
                    "metadata": metadata,
                })
            return records

    def update_metadata(self, record_id: str, patch: dict[str, Any]) -> bool:
        """Update metadata for a stored record by stable ``record_id``."""
        target = str(record_id or "").strip()
        if not target:
            return False
        with self._lock:
            for idx, metadata in enumerate(self._metadata):
                current = metadata if isinstance(metadata, dict) else {}
                if str(current.get("record_id") or "") != target:
                    continue
                current.update(patch)
                self._metadata[idx] = current
                return True
        return False

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
            tmp_path = self._store_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
                f.flush()
                import os as _os
                _os.fsync(f.fileno())
            tmp_path.replace(self._store_path)
            logger.debug("[VectorStore] Saved %d entries to %s", len(self._texts), self._store_path)
        except Exception as exc:
            logger.warning("[VectorStore] Save failed: %s", exc)

    def _load(self) -> None:
        """Load from JSON persistence file."""
        if self._store_path is None or not self._store_path.exists():
            return
        try:
            with open(self._store_path, encoding="utf-8") as f:
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
