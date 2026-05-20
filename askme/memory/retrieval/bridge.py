"""
Memory bridge — L4 vector memory with pluggable backends.

Supported backends (``memory.backend`` in config.yaml):

- ``"auto"``      - probe installed local backends and select the first ready one
- ``"mempalace"`` - MemPalace SDK

- ``"mem0"``      — Mem0 (default, backward-compatible)
- ``"robotmem"``  — robotmem SDK (pip install robotmem)
- ``"vector"``    — local VectorStore (sentence-transformers, no server)

Lazy initialization: the selected backend is only instantiated on first
use.  If it is unavailable (import error, config error, etc.), the bridge
falls back to the local ``VectorStore``.

Graceful degradation: all operations return empty / no-op on failure.

Usage::

    from askme.memory.retrieval.bridge import MemoryBridge

    mem = MemoryBridge()            # does NOT block or crash
    context = await mem.retrieve("今天天气怎么样")
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from askme.config import get_config, project_root
from askme.memory.retrieval.catalog import KnowledgeCatalog
from askme.memory.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)
_UTC = timezone(timedelta(0))
_SUPPORTED_BACKENDS = {"auto", "mem0", "robotmem", "mempalace", "vector"}
_DEFAULT_AUTO_BACKEND_ORDER = ("robotmem", "vector", "mem0", "mempalace")
_BACKEND_DEPENDENCIES = {
    "robotmem": {
        "module": "robotmem",
        "packages": ("robotmem",),
        "label": "RobotMem SDK",
    },
    "mempalace": {
        "module": "mempalace",
        "packages": ("mempalace",),
        "label": "MemPalace SDK",
    },
    "mem0": {
        "module": "mem0",
        "packages": ("mem0ai", "mem0"),
        "label": "Mem0 SDK",
    },
    "vector": {
        "module": "sentence_transformers",
        "packages": ("sentence-transformers",),
        "label": "Local vector store",
    },
}


class MemoryBridge:
    """L4 vector memory — pluggable backend with VectorStore fallback."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        data_dir: str | Path | None = None,
        knowledge_catalog: KnowledgeCatalog | None = None,
    ) -> None:
        """Create a MemoryBridge.

        Args:
            config: Full config dict (e.g. from get_config()).  If None, the
                global config.yaml is read via get_config().  Pass a dict in
                tests to avoid filesystem reads.
            data_dir: Override the data directory for the VectorStore fallback.
                If None, read from ``config["app"]["data_dir"]``.
        """
        cfg = config if config is not None else get_config()
        self._brain_cfg: dict[str, Any] = cfg.get("brain", {})
        self._mem_cfg: dict[str, Any] = cfg.get("memory", {})

        self._enabled: bool = self._mem_cfg.get("enabled", True)
        self._embed_model: str = self._mem_cfg.get(
            "embed_model", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._retrieve_timeout: float = self._mem_cfg.get("retrieve_timeout", 2.0)
        self._vector_min_similarity: float = float(
            self._mem_cfg.get("vector_min_similarity", 0.5)
        )

        # Backend selection for customer-facing answer evidence.
        # ``backend`` is kept as a legacy alias, while
        # ``customer_knowledge_backend`` is the product contract. Robot
        # behavior memory is reported separately and must not silently become
        # the customer RAG source.
        self._legacy_backend_config: str = (
            str(self._mem_cfg.get("backend", "mem0")).strip().lower()
        )
        self._customer_knowledge_backend: str = (
            str(
                self._mem_cfg.get(
                    "customer_knowledge_backend",
                    self._legacy_backend_config,
                )
            )
            .strip()
            .lower()
        )
        self._configured_backend: str = self._customer_knowledge_backend
        self._robot_behavior_memory_backend: str = (
            str(self._mem_cfg.get("robot_behavior_memory_backend", "robotmem"))
            .strip()
            .lower()
        )
        self._robot_behavior_memory_enabled: bool = bool(
            self._mem_cfg.get("robot_behavior_memory_enabled", False)
        )
        self._auto_backend_order = self._normalize_auto_backend_order(
            self._mem_cfg.get("auto_backend_order")
        )
        self._backend_selection_reason = ""
        self._backend: str = self._select_backend(self._configured_backend)
        self._retrieve_cache_ttl_s: float = max(
            0.0,
            float(self._mem_cfg.get("retrieve_cache_ttl_s", 1.0)),
        )
        self._retrieve_cache_max_entries: int = max(
            1,
            int(self._mem_cfg.get("retrieve_cache_max_entries", 128)),
        )
        self._retrieve_cache_empty_results: bool = bool(
            self._mem_cfg.get("retrieve_cache_empty_results", False)
        )
        self._robotmem_fallback_backend: str = self._mem_cfg.get(
            "robotmem_fallback_backend", "mem0"
        )
        self._mempalace_fallback_backend: str = self._mem_cfg.get(
            "mempalace_fallback_backend", "vector"
        )

        # Mem0 instance — lazy init via _ensure_mem0()
        self._mem0: Any = None
        self._mem0_failed: bool = False  # True after init failure, skip retries

        # RobotMem backend — lazy init via _ensure_robotmem()
        self._robotmem: Any = None  # RobotMemBackend instance
        self._robotmem_failed: bool = False

        # MemPalace backend - lazy init via _ensure_mempalace()
        self._mempalace: Any = None
        self._mempalace_failed: bool = False

        # Fallback: local VectorStore (lazy — only init when actually needed)
        if data_dir is not None:
            resolved = Path(data_dir)
        else:
            raw = cfg.get("app", {}).get("data_dir", "data")
            resolved = Path(raw)
            if not resolved.is_absolute():
                resolved = project_root() / resolved
        self._store_path = resolved / "memory" / "vectors" / "store.json"
        self._knowledge_catalog = knowledge_catalog or KnowledgeCatalog(config=cfg, data_dir=resolved)
        self._store: VectorStore | None = None
        self._warmup_active = False
        self._retrieve_count = 0
        self._retrieve_error_count = 0
        self._fallback_count = 0
        self._last_retrieve_ms: float | None = None
        self._last_retrieved_items = 0
        self._last_backend: str | None = None
        self._last_fallback_reason: str = ""
        self._last_evidence: list[dict[str, Any]] = []
        self._last_dropped_evidence: list[dict[str, Any]] = []
        self._retrieve_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._retrieve_inflight: dict[str, asyncio.Task[str]] = {}
        self._retrieve_cache_lock = asyncio.Lock()
        self._retrieve_cache_hits = 0
        self._retrieve_cache_misses = 0
        self._retrieve_coalesced_count = 0
        self._last_retrieve_cache_hit = False
        self._last_retrieve_coalesced = False
        self._rag_allowed_statuses = {
            str(status).strip().lower()
            for status in self._mem_cfg.get(
                "rag_allowed_approval_statuses",
                ["", "published", "approved", "active"],
            )
        }
        self._rag_enforce_expiry = bool(self._mem_cfg.get("rag_enforce_expiry", True))

        if not self._enabled:
            logger.info("[Memory] Memory disabled in config.")
        else:
            logger.info(
                "[Memory] MemoryBridge ready (backend=%s configured=%s).",
                self._backend,
                self._configured_backend,
            )

    def _select_backend(self, configured_backend: str) -> str:
        configured = (
            configured_backend if configured_backend in _SUPPORTED_BACKENDS else "vector"
        )
        if configured != "auto":
            if configured != configured_backend:
                self._backend_selection_reason = (
                    f"invalid_configured_backend:{configured_backend}"
                )
            else:
                self._backend_selection_reason = "explicit_backend"
            return configured

        for candidate in self._auto_backend_order:
            if self._candidate_backend_available(candidate):
                self._backend_selection_reason = f"auto_selected:{candidate}"
                return candidate
        self._backend_selection_reason = "auto_no_available_backend_vector_noop"
        return "vector"

    @staticmethod
    def _normalize_auto_backend_order(value: Any) -> tuple[str, ...]:
        if isinstance(value, (list, tuple)):
            raw_items = [str(item).strip().lower() for item in value]
        else:
            raw_items = list(_DEFAULT_AUTO_BACKEND_ORDER)
        cleaned: list[str] = []
        for item in raw_items:
            if item in _SUPPORTED_BACKENDS - {"auto"} and item not in cleaned:
                cleaned.append(item)
        return tuple(cleaned or _DEFAULT_AUTO_BACKEND_ORDER)

    @staticmethod
    def _candidate_backend_available(backend: str) -> bool:
        if backend == "robotmem":
            return importlib.util.find_spec("robotmem") is not None
        if backend == "mempalace":
            return importlib.util.find_spec("mempalace") is not None
        if backend == "mem0":
            return importlib.util.find_spec("mem0") is not None
        if backend == "vector":
            return importlib.util.find_spec("sentence_transformers") is not None
        return False

    def _backend_selection_snapshot(self) -> dict[str, Any]:
        candidates = [
            {"backend": backend, "available": self._candidate_backend_available(backend)}
            for backend in self._auto_backend_order
        ]
        return {
            "configured_backend": self._configured_backend,
            "selected_backend": self._backend,
            "reason": self._backend_selection_reason,
            "auto_order": list(self._auto_backend_order),
            "candidates": candidates,
            "fallbacks": {
                "robotmem": self._robotmem_fallback_backend,
                "mempalace": self._mempalace_fallback_backend,
            },
        }

    @classmethod
    def _backend_dependency_snapshot(cls) -> dict[str, dict[str, Any]]:
        """Return install/version evidence for each supported memory backend."""

        return {
            backend: cls._one_backend_dependency_snapshot(backend)
            for backend in ("mempalace", "vector", "robotmem", "mem0")
        }

    @classmethod
    def _one_backend_dependency_snapshot(cls, backend: str) -> dict[str, Any]:
        spec = _BACKEND_DEPENDENCIES.get(backend, {})
        module_name = str(spec.get("module") or backend)
        packages = tuple(str(item) for item in spec.get("packages", ()) if str(item))
        installed = importlib.util.find_spec(module_name) is not None
        version = ""
        package_name = ""
        version_error = ""
        for candidate in packages or (backend,):
            try:
                version = importlib_metadata.version(candidate)
                package_name = candidate
                version_error = ""
                break
            except importlib_metadata.PackageNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover - metadata can fail on broken installs
                package_name = candidate
                version_error = type(exc).__name__
                break
        if not package_name and packages:
            package_name = packages[0]
        return {
            "backend": backend,
            "label": str(spec.get("label") or backend),
            "module": module_name,
            "package": package_name,
            "installed": installed,
            "version": version,
            "version_available": bool(version),
            "version_error": version_error,
        }

    # ------------------------------------------------------------------
    # VectorStore lazy initialization (fallback)
    # ------------------------------------------------------------------

    def _ensure_store(self) -> VectorStore | None:
        """Lazy-init VectorStore only when actually needed as fallback."""
        if self._store is not None:
            return self._store
        try:
            self._store = VectorStore(
                model_name=self._embed_model,
                store_path=self._store_path,
            )
            return self._store
        except Exception as e:
            logger.debug("[Memory] VectorStore init failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Mem0 lazy initialization
    # ------------------------------------------------------------------

    def _ensure_mem0(self) -> bool:
        """Try to initialise the Mem0 instance. Returns True if ready."""
        if self._mem0 is not None:
            return True
        if not self._enabled or self._mem0_failed:
            return False
        try:
            from mem0 import Memory

            brain_cfg = self._brain_cfg
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "askme",
                        "path": str(project_root() / "data" / "memory" / "mem0_store"),
                    },
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "api_key": brain_cfg.get("api_key", ""),
                        "openai_base_url": brain_cfg.get("base_url", ""),
                        "model": brain_cfg.get("model", "MiniMax-M2.7-highspeed"),
                    },
                },
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": self._embed_model,
                    },
                },
            }
            self._mem0 = Memory.from_config(config)
            logger.info("[Memory] Mem0 initialised successfully.")
            return True
        except Exception as e:
            logger.warning("[Memory] Mem0 init failed, using VectorStore fallback: %s", e)
            self._mem0_failed = True
            return False

    # ------------------------------------------------------------------
    # RobotMem lazy initialization
    # ------------------------------------------------------------------

    def _ensure_robotmem(self) -> bool:
        """Try to initialise the RobotMem backend. Returns True if ready."""
        if self._robotmem is not None and self._robotmem.available:
            return True
        if not self._enabled or self._robotmem_failed:
            return False
        try:
            from askme.memory.backends.robotmem_backend import RobotMemBackend

            self._robotmem = RobotMemBackend(self._mem_cfg, self._brain_cfg)
            inited = self._robotmem._ensure_robotmem()
            if not inited:
                self._robotmem_failed = True
                self._robotmem = None
                return False
            logger.info("[Memory] RobotMem backend ready.")
            return True
        except Exception as e:
            logger.warning("[Memory] RobotMem init failed, falling back: %s", e)
            self._robotmem_failed = True
            return False

    # ------------------------------------------------------------------
    # MemPalace lazy initialization
    # ------------------------------------------------------------------

    def _ensure_mempalace(self) -> bool:
        """Try to initialise the MemPalace backend. Returns True if ready."""
        if self._mempalace is not None and self._mempalace.available:
            return True
        if not self._enabled or self._mempalace_failed:
            return False
        try:
            from askme.memory.backends.mempalace_backend import MemPalaceBackend

            self._mempalace = MemPalaceBackend(self._mem_cfg, self._brain_cfg)
            inited = self._mempalace._ensure_mempalace()
            if not inited:
                self._mempalace_failed = True
                self._mempalace = None
                return False
            logger.info("[Memory] MemPalace backend ready.")
            return True
        except Exception as e:
            logger.warning("[Memory] MemPalace init failed, falling back: %s", e)
            self._mempalace_failed = True
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Pre-load memory backends in the background without blocking turns."""
        if not self._enabled or self._warmup_active:
            return
        self._warmup_active = True
        try:
            await self._warmup_once()
        finally:
            self._warmup_active = False

    async def _warmup_once(self) -> None:
        """Pre-load the embedding model in a background thread."""
        if not self._enabled:
            return

        # Warm up the configured backend
        if self._backend == "robotmem":
            try:
                inited = await asyncio.to_thread(self._ensure_robotmem)
                if inited:
                    await self._robotmem.warmup()
                    logger.info("[Memory] RobotMem warmup complete.")
                    return
            except Exception:
                logger.debug("[Memory] RobotMem warmup failed, trying fallback.")

        if self._backend == "mempalace":
            try:
                inited = await asyncio.to_thread(self._ensure_mempalace)
                if inited:
                    await self._mempalace.warmup()
                    logger.info("[Memory] MemPalace warmup complete.")
                    return
            except Exception:
                logger.debug("[Memory] MemPalace warmup failed, trying fallback.")

        if self._backend in ("mem0", "robotmem", "mempalace"):
            # Try Mem0 (primary for mem0 backend, fallback for robotmem)
            try:
                if self._backend == "mempalace" and self._mempalace_fallback_backend != "mem0":
                    inited = False
                else:
                    inited = await asyncio.to_thread(self._ensure_mem0)
                if inited:
                    logger.info("[Memory] Mem0 warmup complete.")
                    return
            except Exception:
                logger.debug("[Memory] Mem0 warmup failed, trying VectorStore.")

        # Fallback: warm up VectorStore
        store = self._ensure_store()
        if store and store.available:
            try:
                await asyncio.to_thread(store.search, "warmup", 1)
                logger.info("[Memory] VectorStore warmup complete.")
            except Exception:
                logger.debug("[Memory] VectorStore warmup triggered model load (expected).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(self, text: str) -> str:
        """Retrieve relevant memory context for *text*.

        Returns a formatted context string (one ``- item`` per line),
        or an empty string if memory is unavailable / finds nothing.
        """
        if not self._enabled:
            self._last_evidence = []
            self._record_retrieve_result("", backend="disabled", elapsed_ms=0.0)
            return ""
        started = time.perf_counter()
        cache_key = self._retrieve_cache_key(text)
        cached = await self._get_cached_retrieve(cache_key)
        if cached is not None:
            self._restore_cached_retrieve(cached)
            self._record_retrieve_result(
                str(cached.get("result") or ""),
                backend=str(cached.get("backend") or self._backend),
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                cache_hit=True,
            )
            return str(cached.get("result") or "")

        task: asyncio.Task[str] | None = None
        owns_task = False
        coalesced = False
        if cache_key and self._retrieve_cache_ttl_s > 0:
            async with self._retrieve_cache_lock:
                task = self._retrieve_inflight.get(cache_key)
                if task is None:
                    task = asyncio.create_task(self._retrieve_with_fallbacks(text))
                    task.add_done_callback(self._consume_retrieve_task_exception)
                    self._retrieve_inflight[cache_key] = task
                    owns_task = True
                    self._retrieve_cache_misses += 1
                else:
                    coalesced = True
                    self._retrieve_coalesced_count += 1
        if task is None:
            task = asyncio.create_task(self._retrieve_with_fallbacks(text))
            task.add_done_callback(self._consume_retrieve_task_exception)
            owns_task = True
        try:
            result = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._retrieve_timeout,
            )
            if owns_task:
                await self._store_cached_retrieve(cache_key, result)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._record_retrieve_result(
                result,
                backend=self._last_backend or self._backend,
                elapsed_ms=elapsed_ms,
                coalesced=coalesced,
            )
            return result
        except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
            logger.warning("[Memory] retrieval timed out (%.1fs).", self._retrieve_timeout)
            self._retrieve_error_count += 1
            self._record_retrieve_result(
                "",
                backend=self._last_backend or self._backend,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                fallback_reason="retrieve_timeout",
                coalesced=coalesced,
            )
            return ""
        except Exception as exc:
            logger.debug("[Memory] retrieve failed: %s", exc)
            self._retrieve_error_count += 1
            self._record_retrieve_result(
                "",
                backend=self._last_backend or self._backend,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                fallback_reason=type(exc).__name__,
                coalesced=coalesced,
            )
            return ""
        finally:
            if owns_task and cache_key:
                async with self._retrieve_cache_lock:
                    if self._retrieve_inflight.get(cache_key) is task:
                        self._retrieve_inflight.pop(cache_key, None)

    async def _retrieve_with_fallbacks(self, text: str) -> str:
        """Run backend retrieval under the public retrieve() time budget."""
        self._last_backend = None
        self._last_fallback_reason = ""
        self._last_evidence = []
        self._last_dropped_evidence = []
        if (
            self._warmup_active
            and self._robotmem is None
            and self._mempalace is None
            and self._mem0 is None
            and self._store is None
        ):
            logger.debug("[Memory] Warmup in progress; skipping retrieval for this turn.")
            self._last_backend = "warmup"
            self._last_fallback_reason = "warmup_in_progress"
            return ""

        # Try configured backend first — use to_thread so lazy init (ONNX load)
        # never blocks the event loop on cold start.
        if self._backend == "robotmem":
            try:
                robotmem_ready = await asyncio.wait_for(
                    asyncio.to_thread(self._ensure_robotmem),
                    timeout=self._retrieve_timeout,
                )
            except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
                logger.warning(
                    "[Memory] RobotMem init timed out (%.1fs).",
                    self._retrieve_timeout,
                )
                robotmem_ready = False
                self._fallback_count += 1
                self._last_fallback_reason = "robotmem_init_timeout"
            except Exception as exc:
                logger.debug("[Memory] RobotMem init failed: %s", exc)
                robotmem_ready = False
                self._fallback_count += 1
                self._last_fallback_reason = "robotmem_init_failed"
            if robotmem_ready:
                self._last_backend = "robotmem"
                if hasattr(type(self._robotmem), "retrieve_items"):
                    items = await self._robotmem.retrieve_items(text)
                    items = self._filter_evidence_items(items, backend="robotmem")
                    self._set_evidence(items, backend="robotmem")
                    return self._format_evidence(items)
                result = await self._robotmem.retrieve(text)
                items = [
                    {
                        "text": line.strip().lstrip("- ").strip(),
                        "backend": "robotmem",
                        "source": "robotmem",
                        "category": "",
                        "score": None,
                        "metadata": {},
                    }
                    for line in str(result).splitlines()
                    if line.strip()
                ]
                items = self._filter_evidence_items(items, backend="robotmem")
                self._set_evidence(items, backend="robotmem")
                return self._format_evidence(items)
            if not self._last_fallback_reason:
                self._fallback_count += 1
                self._last_fallback_reason = "robotmem_unavailable"

        if self._backend == "mempalace":
            try:
                mempalace_ready = await asyncio.wait_for(
                    asyncio.to_thread(self._ensure_mempalace),
                    timeout=self._retrieve_timeout,
                )
            except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
                logger.warning(
                    "[Memory] MemPalace init timed out (%.1fs).",
                    self._retrieve_timeout,
                )
                mempalace_ready = False
                self._fallback_count += 1
                self._last_fallback_reason = "mempalace_init_timeout"
            except Exception as exc:
                logger.debug("[Memory] MemPalace init failed: %s", exc)
                mempalace_ready = False
                self._fallback_count += 1
                self._last_fallback_reason = "mempalace_init_failed"
            if mempalace_ready:
                self._last_backend = "mempalace"
                try:
                    raw_items = await self._mempalace.retrieve_items(text)
                except Exception as exc:
                    logger.warning("[Memory] MemPalace retrieval failed, falling back: %s", exc)
                    raw_items = []
                    self._fallback_count += 1
                    self._last_fallback_reason = "mempalace_retrieve_failed"
                items = self._filter_evidence_items(raw_items, backend="mempalace")
                if items or self._last_dropped_evidence:
                    self._set_evidence(items, backend="mempalace")
                    return self._format_evidence(items)
                if not self._last_fallback_reason:
                    self._fallback_count += 1
                    self._last_fallback_reason = "mempalace_empty"
            if not self._last_fallback_reason:
                self._fallback_count += 1
                self._last_fallback_reason = "mempalace_unavailable"

        use_mem0 = self._backend == "mem0" or (
            self._backend == "robotmem" and self._robotmem_fallback_backend == "mem0"
        ) or (
            self._backend == "mempalace" and self._mempalace_fallback_backend == "mem0"
        )
        if use_mem0 and await asyncio.to_thread(self._ensure_mem0):
            self._last_backend = "mem0"
            return await self._retrieve_mem0(text)

        # Fallback to VectorStore
        if self._backend != "vector":
            self._fallback_count += 1
            if not self._last_fallback_reason:
                self._last_fallback_reason = "vector_fallback"
        self._last_backend = "vector"
        return await self._retrieve_vector_store(text)

    async def save(self, user_text: str, assistant_text: str) -> None:
        """Persist a conversation exchange to L4 memory.

        Silently no-ops when the backend is unavailable.
        """
        if not self._enabled:
            return
        await self._clear_retrieve_cache()

        # Try configured backend first — use to_thread so lazy init never blocks.
        if self._backend == "robotmem" and await asyncio.to_thread(self._ensure_robotmem):
            await self._robotmem.save(user_text, assistant_text)
            return

        if self._backend == "mempalace" and await asyncio.to_thread(self._ensure_mempalace):
            await self._mempalace.save(user_text, assistant_text)
            return

        use_mem0 = self._backend == "mem0" or (
            self._backend == "robotmem" and self._robotmem_fallback_backend == "mem0"
        ) or (
            self._backend == "mempalace" and self._mempalace_fallback_backend == "mem0"
        )
        if use_mem0 and await asyncio.to_thread(self._ensure_mem0):
            await self._save_mem0(user_text, assistant_text)
            return

        # Fallback to VectorStore
        await self._save_vector_store(user_text, assistant_text)

    async def save_fact(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Persist a curated knowledge fact with optional metadata."""
        if not self._enabled:
            return
        clean = str(text or "").strip()
        if not clean:
            return
        await self._clear_retrieve_cache()

        if self._backend == "robotmem" and await asyncio.to_thread(self._ensure_robotmem):
            await self._robotmem.save_fact(clean, metadata or {})
            return

        if self._backend == "mempalace" and await asyncio.to_thread(self._ensure_mempalace):
            await self._mempalace.save_fact(clean, metadata or {})
            return

        use_mem0 = self._backend == "mem0" or (
            self._backend == "robotmem" and self._robotmem_fallback_backend == "mem0"
        ) or (
            self._backend == "mempalace" and self._mempalace_fallback_backend == "mem0"
        )
        if use_mem0 and await asyncio.to_thread(self._ensure_mem0):
            await self._save_mem0(clean, f"[knowledge_import] {metadata or {}}")
            return

        store = self._ensure_store()
        if not store or not store.available:
            return
        await asyncio.to_thread(store.add, clean, {"type": "knowledge", **(metadata or {})})
        await asyncio.to_thread(store.save)

    async def list_knowledge(self, *, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        """List locally indexed knowledge records for admin UI.

        External providers such as RobotMem and Mem0 do not currently expose a
        stable list/delete contract here, so this view is scoped to the local
        vector catalog.
        """
        if not self._enabled:
            return {"backend": "disabled", "records": [], "total": 0}
        store = self._ensure_store()
        if not store:
            return {"backend": "vector", "records": [], "total": 0}
        records = await asyncio.to_thread(store.list_records, limit=limit, offset=offset)
        return {
            "backend": "vector",
            "records": [self._knowledge_catalog_record(record) for record in records],
            "total": getattr(store, "size", len(records)),
        }

    async def update_knowledge_metadata(
        self,
        record_id: str,
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        """Patch local knowledge metadata and persist the catalog."""
        if not self._enabled:
            return {"updated": False, "error": "memory_disabled"}
        await self._clear_retrieve_cache()
        store = self._ensure_store()
        if not store:
            return {"updated": False, "error": "vector_store_unavailable"}
        allowed = {
            "approval_status",
            "category",
            "source",
            "owner",
            "updated_at",
            "expires_at",
            "deleted_at",
            "deleted_reason",
            "restored_at",
        }
        clean_patch = {k: v for k, v in patch.items() if k in allowed}
        if not clean_patch:
            return {"updated": False, "error": "empty_patch"}
        updated = await asyncio.to_thread(store.update_metadata, record_id, clean_patch)
        if updated:
            await asyncio.to_thread(store.save)
        return {"updated": bool(updated), "record_id": record_id, "patch": clean_patch}

    def import_existing_data(self) -> int:
        """Scan L3 knowledge/digests and import into vector store.

        Returns the number of entries imported. Runs synchronously.
        No-ops when unavailable.
        """
        store = self._ensure_store()
        if not self._enabled or not store or not store.available:
            return 0
        self._retrieve_cache.clear()

        # _store_path = <data_dir>/memory/vectors/store.json → parent.parent = <data_dir>/memory
        memory_dir = self._store_path.parent.parent

        imported = 0

        # Import knowledge .md files (line by line)
        knowledge_dir = memory_dir / "knowledge"
        if knowledge_dir.exists():
            for md_file in knowledge_dir.glob("*.md"):
                try:
                    for line in md_file.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and line.startswith("- "):
                            store.add(line[2:], {"type": "knowledge", "source": md_file.name})
                            imported += 1
                except Exception as exc:
                    logger.warning("[Memory] Import %s failed: %s", md_file.name, exc)

        # Import digest .md files (whole file)
        digest_dir = memory_dir / "digests"
        if digest_dir.exists():
            for md_file in digest_dir.glob("*.md"):
                try:
                    content = md_file.read_text(encoding="utf-8").strip()
                    if content:
                        store.add(content, {"type": "digest", "source": md_file.name})
                        imported += 1
                except Exception as exc:
                    logger.warning("[Memory] Import %s failed: %s", md_file.name, exc)

        if imported:
            store.save()
            logger.info("[Memory] Imported %d entries from existing L3 data.", imported)
        return imported

    def health(self) -> dict[str, Any]:
        """Return runtime-observable RAG backend status."""
        available = self.available
        store = self._store
        robotmem_available = bool(self._robotmem is not None and self._robotmem.available)
        mempalace_available = bool(
            self._mempalace is not None and self._mempalace.available
        )
        mem0_available = self._mem0 is not None
        vector_available = bool(store and store.available)
        backend_ready = {
            "robotmem": robotmem_available,
            "mempalace": mempalace_available,
            "mem0": mem0_available,
            "vector": vector_available,
        }
        selected_backend_ready = bool(backend_ready.get(self._backend, False))
        fallback_backend = ""
        if self._backend == "robotmem":
            fallback_backend = self._robotmem_fallback_backend
        elif self._backend == "mempalace":
            fallback_backend = self._mempalace_fallback_backend
        fallback_ready = bool(backend_ready.get(fallback_backend, False)) if fallback_backend else False
        backend_dependencies = self._backend_dependency_snapshot()
        selected_dependency = backend_dependencies.get(self._backend, {})
        fallback_dependency = backend_dependencies.get(fallback_backend, {}) if fallback_backend else {}
        return {
            "enabled": self._enabled,
            "backend": self._backend,
            "configured_backend": self._configured_backend,
            "legacy_backend_config": self._legacy_backend_config,
            "customer_knowledge_backend": self._customer_knowledge_backend,
            "robot_behavior_memory_backend": self._robot_behavior_memory_backend,
            "robot_behavior_memory_enabled": self._robot_behavior_memory_enabled,
            "selected_backend_ready": selected_backend_ready,
            "selected_backend_installed": self._candidate_backend_available(self._backend),
            "selected_backend_dependency": selected_dependency,
            "fallback_backend": fallback_backend,
            "fallback_ready": fallback_ready,
            "fallback_backend_dependency": fallback_dependency,
            "backend_dependencies": backend_dependencies,
            "backend_selection": self._backend_selection_snapshot(),
            "product_memory_roles": {
                "customer_knowledge": {
                    "purpose": "auditable answer evidence for customer questions",
                    "configured_backend": self._customer_knowledge_backend,
                    "selected_backend": self._backend,
                    "ready": selected_backend_ready,
                    "installed": self._candidate_backend_available(self._backend),
                    "dependency": selected_dependency,
                    "vector_store_path": str(self._store_path),
                },
                "robot_behavior": {
                    "purpose": "long-term robot behavior and interaction memory",
                    "configured_backend": self._robot_behavior_memory_backend,
                    "enabled": self._robot_behavior_memory_enabled,
                    "ready": bool(
                        self._robot_behavior_memory_enabled
                        and backend_ready.get(self._robot_behavior_memory_backend, False)
                    ),
                    "dependency": backend_dependencies.get(
                        self._robot_behavior_memory_backend,
                        {},
                    ),
                },
            },
            "available": available,
            "robotmem_ready": robotmem_available,
            "mempalace_ready": mempalace_available,
            "mempalace_path": (
                self._mempalace.palace_path if self._mempalace is not None else ""
            ),
            "mem0_ready": mem0_available,
            "vector_ready": vector_available,
            "vector_store_path": str(self._store_path),
            "vector_min_similarity": self._vector_min_similarity,
            "embed_model": self._embed_model,
            "retrieve_timeout_s": self._retrieve_timeout,
            "rag_enforce_expiry": self._rag_enforce_expiry,
            "rag_allowed_approval_statuses": sorted(self._rag_allowed_statuses),
            "vector_size": int(getattr(store, "size", 0) or 0) if store else 0,
            "retrieve_count": self._retrieve_count,
            "retrieve_error_count": self._retrieve_error_count,
            "fallback_count": self._fallback_count,
            "last_retrieve_ms": self._last_retrieve_ms,
            "last_retrieved_items": self._last_retrieved_items,
            "last_backend": self._last_backend or self._backend,
            "last_fallback_reason": self._last_fallback_reason,
            "retrieve_cache": {
                "enabled": self._retrieve_cache_ttl_s > 0,
                "ttl_s": self._retrieve_cache_ttl_s,
                "max_entries": self._retrieve_cache_max_entries,
                "size": len(self._retrieve_cache),
                "hits": self._retrieve_cache_hits,
                "misses": self._retrieve_cache_misses,
                "coalesced": self._retrieve_coalesced_count,
                "inflight": len(self._retrieve_inflight),
                "last_hit": self._last_retrieve_cache_hit,
                "last_coalesced": self._last_retrieve_coalesced,
            },
            "last_evidence": self._last_evidence,
            "last_dropped_evidence": self._last_dropped_evidence,
            "last_answer_policy": self._answer_policy_snapshot(),
        }

    def _record_retrieve_result(
        self,
        result: str,
        *,
        backend: str,
        elapsed_ms: float,
        fallback_reason: str = "",
        cache_hit: bool = False,
        coalesced: bool = False,
    ) -> None:
        self._retrieve_count += 1
        self._last_backend = backend
        self._last_retrieve_ms = round(float(elapsed_ms), 2)
        self._last_retrieved_items = len(
            [line for line in result.splitlines() if line.strip()]
        )
        self._last_retrieve_cache_hit = cache_hit
        self._last_retrieve_coalesced = coalesced
        if fallback_reason:
            self._last_fallback_reason = fallback_reason

    def _retrieve_cache_key(self, text: str) -> str:
        if self._retrieve_cache_ttl_s <= 0:
            return ""
        return " ".join(str(text or "").strip().split())

    async def _get_cached_retrieve(self, cache_key: str) -> dict[str, Any] | None:
        if not cache_key:
            return None
        now = time.monotonic()
        async with self._retrieve_cache_lock:
            entry = self._retrieve_cache.get(cache_key)
            if not entry:
                return None
            expires_at = float(entry.get("expires_at") or 0.0)
            if expires_at <= now:
                self._retrieve_cache.pop(cache_key, None)
                return None
            self._retrieve_cache.move_to_end(cache_key)
            self._retrieve_cache_hits += 1
            return dict(entry)

    async def _store_cached_retrieve(self, cache_key: str, result: str) -> None:
        if not cache_key or self._retrieve_cache_ttl_s <= 0:
            return
        if not result and not self._retrieve_cache_empty_results:
            return
        entry = {
            "result": result,
            "backend": self._last_backend or self._backend,
            "fallback_reason": self._last_fallback_reason,
            "last_evidence": [dict(item) for item in self._last_evidence],
            "last_dropped_evidence": [dict(item) for item in self._last_dropped_evidence],
            "expires_at": time.monotonic() + self._retrieve_cache_ttl_s,
        }
        async with self._retrieve_cache_lock:
            self._retrieve_cache[cache_key] = entry
            self._retrieve_cache.move_to_end(cache_key)
            while len(self._retrieve_cache) > self._retrieve_cache_max_entries:
                self._retrieve_cache.popitem(last=False)

    async def _clear_retrieve_cache(self) -> None:
        async with self._retrieve_cache_lock:
            self._retrieve_cache.clear()

    @staticmethod
    def _consume_retrieve_task_exception(task: asyncio.Task[str]) -> None:
        if task.cancelled():
            return
        try:
            task.exception()
        except Exception:
            return

    def _restore_cached_retrieve(self, entry: dict[str, Any]) -> None:
        self._last_backend = str(entry.get("backend") or self._backend)
        self._last_fallback_reason = str(entry.get("fallback_reason") or "")
        evidence = entry.get("last_evidence")
        dropped = entry.get("last_dropped_evidence")
        self._last_evidence = (
            [dict(item) for item in evidence if isinstance(item, dict)]
            if isinstance(evidence, list)
            else []
        )
        self._last_dropped_evidence = (
            [dict(item) for item in dropped if isinstance(item, dict)]
            if isinstance(dropped, list)
            else []
        )

    def _set_evidence(self, items: list[dict[str, Any]], *, backend: str) -> None:
        self._last_evidence = [
            {
                "text": str(item.get("text", ""))[:300],
                "backend": str(item.get("backend") or backend),
                "source": str(item.get("source") or backend),
                "category": str(item.get("category") or ""),
                "score": item.get("score"),
                "updated_at": (item.get("metadata") or {}).get("updated_at", ""),
                "expires_at": (item.get("metadata") or {}).get("expires_at", ""),
                "approval_status": (item.get("metadata") or {}).get("approval_status", ""),
                "record_id": (item.get("metadata") or {}).get("record_id", ""),
                "source_record_id": (item.get("metadata") or {}).get("record_id", ""),
                "lifecycle_state": (item.get("metadata") or {}).get("lifecycle_state", ""),
                "needs_reindex": bool((item.get("metadata") or {}).get("needs_reindex", False)),
                "evidence_version": (item.get("metadata") or {}).get("evidence_version", ""),
                "freshness_state": item.get("freshness_state", "unknown"),
                "used_in_prompt": True,
                "metadata": item.get("metadata") or {},
            }
            for item in items
            if str(item.get("text", "")).strip()
        ][:5]

    @staticmethod
    def _knowledge_catalog_record(record: dict[str, Any]) -> dict[str, Any]:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        return {
            "index": record.get("index"),
            "record_id": metadata.get("record_id") or "",
            "text": record.get("text") or "",
            "category": metadata.get("category") or metadata.get("type") or "",
            "source": metadata.get("source") or "",
            "owner": metadata.get("owner") or "",
            "updated_at": metadata.get("updated_at") or "",
            "expires_at": metadata.get("expires_at") or "",
            "approval_status": metadata.get("approval_status") or "",
            "content_hash": metadata.get("content_hash") or "",
            "metadata": metadata,
        }

    @staticmethod
    def _format_evidence(items: list[dict[str, Any]]) -> str:
        return "\n".join(
            f"- {item['text']}" for item in items if str(item.get("text", "")).strip()
        )

    def _filter_evidence_items(
        self,
        items: list[dict[str, Any]],
        *,
        backend: str,
    ) -> list[dict[str, Any]]:
        accepted: list[dict[str, Any]] = []
        dropped: list[dict[str, Any]] = []
        self._last_dropped_evidence = []
        now = datetime.now(_UTC)
        for raw in items:
            item = dict(raw)
            metadata = item.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            item["metadata"] = metadata
            item.setdefault("backend", backend)
            item.setdefault("source", metadata.get("source") or backend)
            item.setdefault("category", metadata.get("category") or metadata.get("type") or "")

            drop_reason = self._evidence_drop_reason(metadata, now=now)
            if drop_reason:
                dropped.append(self._dropped_evidence_snapshot(item, drop_reason))
                continue
            item["freshness_state"] = "fresh" if metadata.get("expires_at") else "no_expiry"
            accepted.append(item)

        accepted = self._drop_conflicting_evidence(accepted, dropped)
        self._last_dropped_evidence = dropped[:5]
        return accepted

    def _drop_conflicting_evidence(
        self,
        accepted: list[dict[str, Any]],
        dropped: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in accepted:
            metadata = item.get("metadata") or {}
            entity_key = self._normalized_metadata_value(metadata.get("entity_key"))
            fact_key = self._normalized_metadata_value(metadata.get("fact_key"))
            value = self._normalized_metadata_value(metadata.get("value"))
            if not entity_key or not fact_key or not value:
                continue
            item["_conflict_value"] = value
            groups.setdefault((entity_key, fact_key), []).append(item)

        conflicted_ids: set[int] = set()
        for (entity_key, fact_key), group in groups.items():
            values = {str(item.get("_conflict_value") or "") for item in group}
            if len(values) <= 1:
                continue
            reason = f"conflict:{entity_key}:{fact_key}"
            for item in group:
                conflicted_ids.add(id(item))
                item.pop("_conflict_value", None)
                dropped.append(self._dropped_evidence_snapshot(item, reason))

        filtered: list[dict[str, Any]] = []
        for item in accepted:
            item.pop("_conflict_value", None)
            if id(item) not in conflicted_ids:
                filtered.append(item)
        return filtered

    def _evidence_drop_reason(
        self,
        metadata: dict[str, Any],
        *,
        now: datetime,
    ) -> str:
        status = str(metadata.get("approval_status", "")).strip().lower()
        if status and status not in self._rag_allowed_statuses:
            return f"approval_status:{status}"
        if str(metadata.get("conflict_set_id") or "").strip():
            return f"conflict:{metadata.get('conflict_set_id')}"
        catalog_reason = self._knowledge_catalog.evidence_drop_reason(metadata)
        if catalog_reason:
            return catalog_reason
        if self._rag_enforce_expiry:
            expires_at = self._parse_evidence_time(metadata.get("expires_at"))
            if expires_at is not None and expires_at <= now:
                return "expired"
        return ""

    @staticmethod
    def _parse_evidence_time(value: Any) -> datetime | None:
        if value is None or str(value).strip() == "":
            return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=_UTC)
            except (OSError, OverflowError, ValueError):
                return None
        raw = str(value).strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=_UTC)
        return parsed.astimezone(_UTC)

    @staticmethod
    def _normalized_metadata_value(value: Any) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _dropped_evidence_snapshot(item: dict[str, Any], drop_reason: str) -> dict[str, Any]:
        metadata = item.get("metadata") or {}
        return {
            "text": str(item.get("text", ""))[:300],
            "backend": str(item.get("backend") or ""),
            "source": str(item.get("source") or ""),
            "category": str(item.get("category") or ""),
            "score": item.get("score"),
            "updated_at": metadata.get("updated_at", ""),
            "expires_at": metadata.get("expires_at", ""),
            "approval_status": metadata.get("approval_status", ""),
            "record_id": metadata.get("record_id", ""),
            "source_record_id": metadata.get("record_id", ""),
            "lifecycle_state": metadata.get("lifecycle_state", ""),
            "needs_reindex": bool(metadata.get("needs_reindex", False)),
            "evidence_version": metadata.get("evidence_version", ""),
            "freshness_state": "expired" if drop_reason == "expired" else "rejected",
            "used_in_prompt": False,
            "drop_reason": drop_reason,
            "metadata": metadata,
        }

    def _answer_policy_snapshot(self) -> dict[str, Any]:
        if self._last_evidence:
            return {
                "state": "grounded",
                "action": "answer_with_evidence",
                "reason": "",
                "message": "Found prompt-eligible evidence.",
            }
        if not self._last_dropped_evidence:
            return {
                "state": "no_evidence",
                "action": "clarify_or_refuse",
                "reason": "no_retrieval_hits",
                "message": "No prompt-eligible evidence was found.",
            }
        reasons = [str(item.get("drop_reason") or "") for item in self._last_dropped_evidence]
        if any("conflict" in reason for reason in reasons):
            return {
                "state": "conflict",
                "action": "clarify",
                "required_operator_action": "resolve_conflict",
                "reason": ",".join(sorted(set(reasons))),
                "message": "Relevant knowledge is conflicting; do not give a definitive answer.",
            }
        if any("expired" in reason or "version" in reason or "stale" in reason for reason in reasons):
            return {
                "state": "stale",
                "action": "refuse_and_request_update",
                "required_operator_action": "refresh_knowledge",
                "reason": ",".join(sorted(set(reasons))),
                "message": "Relevant knowledge is stale or expired; request an update before answering.",
            }
        if any("approval_status" in reason or "catalog_status" in reason for reason in reasons):
            return {
                "state": "unapproved",
                "action": "refuse",
                "required_operator_action": "approve_or_publish",
                "reason": ",".join(sorted(set(reasons))),
                "message": "Relevant knowledge is not approved for answering.",
            }
        return {
            "state": "filtered",
            "action": "clarify_or_refuse",
            "required_operator_action": "review_knowledge",
            "reason": ",".join(sorted(set(reasons))),
            "message": "Retrieved evidence was filtered and cannot be used in the prompt.",
        }

    @property
    def available(self) -> bool:
        """Whether the memory service is initialised and usable."""
        if not self._enabled:
            return False
        if self._robotmem is not None and self._robotmem.available:
            return True
        if self._mempalace is not None and self._mempalace.available:
            return True
        if self._mem0 is not None:
            return True
        store = self._ensure_store()
        return store.available if store else False

    @property
    def vector_store(self) -> VectorStore | None:
        """Direct access to the underlying VectorStore (for AssociationGraph)."""
        return self._ensure_store()

    # ------------------------------------------------------------------
    # Mem0 backend
    # ------------------------------------------------------------------

    async def _retrieve_mem0(self, text: str) -> str:
        """Search Mem0 for relevant memories."""
        try:
            logger.debug("[Memory] Mem0 searching for: %s", text[:60])
            results = await asyncio.wait_for(
                asyncio.to_thread(self._mem0.search, text, user_id="robot"),
                timeout=self._retrieve_timeout,
            )
            if not results or not results.get("results"):
                logger.debug("[Memory] Mem0 no relevant memories found.")
                return ""
            memories = [r.get("memory", "") for r in results["results"][:5]]
            items = [
                {
                    "text": memory,
                    "backend": "mem0",
                    "source": "mem0",
                    "category": "",
                    "score": None,
                    "metadata": {},
                }
                for memory in memories
                if memory
            ]
            if items:
                logger.info("[Memory] Mem0 found %d items.", len(items))
                items = self._filter_evidence_items(items, backend="mem0")
                self._set_evidence(items, backend="mem0")
                return self._format_evidence(items)
            return ""
        except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
            logger.warning("[Memory] Mem0 retrieval timed out (%.1fs).", self._retrieve_timeout)
            return ""
        except Exception as exc:
            logger.debug("[Memory] Mem0 retrieve failed: %s", exc)
            return ""

    async def _save_mem0(self, user_text: str, assistant_text: str) -> None:
        """Add conversation turn to Mem0 (auto-extracts facts)."""
        try:
            text = f"用户: {user_text}\n回复: {assistant_text[:200]}"
            await asyncio.to_thread(self._mem0.add, text, user_id="robot")
            logger.debug("[Memory] Mem0 saved conversation turn.")
        except Exception as exc:
            logger.debug("[Memory] Mem0 save failed: %s", exc)

    # ------------------------------------------------------------------
    # VectorStore fallback
    # ------------------------------------------------------------------

    async def _retrieve_vector_store(self, text: str) -> str:
        """Retrieve from local VectorStore."""
        store = self._ensure_store()
        if not store or not store.available:
            return ""
        try:
            logger.debug("[Memory] VectorStore searching for: %s", text[:60])
            results = await asyncio.wait_for(
                asyncio.to_thread(store.search, text, 5),
                timeout=self._retrieve_timeout,
            )
            if results:
                logger.info("[Memory] VectorStore found %d items.", len(results))
                items = []
                for item in results:
                    score = item.get("score", 0)
                    if score < self._vector_min_similarity:
                        continue
                    metadata = item.get("metadata") or {}
                    items.append(
                        {
                            "text": item.get("text", ""),
                            "backend": "vector",
                            "source": metadata.get("source") or "vector",
                            "category": metadata.get("category") or metadata.get("type") or "",
                            "score": score,
                            "metadata": metadata,
                        }
                    )
                items = self._filter_evidence_items(items, backend="vector")
                self._set_evidence(items, backend="vector")
                return self._format_evidence(items)
            logger.debug("[Memory] VectorStore no relevant memories found.")
            return ""
        except (asyncio.TimeoutError, TimeoutError):  # noqa: UP041
            logger.warning("[Memory] VectorStore retrieval timed out (%.1fs).", self._retrieve_timeout)
            return ""
        except Exception as exc:
            logger.warning("[Memory] VectorStore retrieval error: %s", exc)
            return ""

    async def _save_vector_store(self, user_text: str, assistant_text: str) -> None:
        """Persist to local VectorStore."""
        store = self._ensure_store()
        if not store or not store.available:
            return
        content = f"用户: {user_text}\n助手: {assistant_text[:200]}"
        try:
            await asyncio.to_thread(store.add, content, {
                "type": "conversation",
                "ts": __import__("time").time(),
            })
            # Periodic save (every 10 new entries)
            if store.size % 10 == 0:
                await asyncio.to_thread(store.save)
            logger.debug("[Memory] VectorStore saved conversation turn.")
        except Exception as exc:
            logger.warning("[Memory] VectorStore save failed: %s", exc)
