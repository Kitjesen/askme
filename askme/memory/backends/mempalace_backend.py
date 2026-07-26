"""MemPalace backend for askme long-term memory.

MemPalace is a local-first Chroma-based memory store. This adapter keeps it
optional: if the ``mempalace`` package is missing or the palace cannot be
opened, callers get an empty result and MemoryBridge can fall back to another
backend.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from askme.config import project_root

logger = logging.getLogger(__name__)


def _first_or_empty(results: Any, key: str) -> list[Any]:
    outer = getattr(results, key, None) if not isinstance(results, dict) else results.get(key)
    if not outer:
        return []
    return outer[0] or []


class MemPalaceBackend:
    """Async wrapper around MemPalace's local palace collections."""

    def __init__(self, mem_cfg: dict[str, Any], brain_cfg: dict[str, Any] | None = None) -> None:
        self._mem_cfg = mem_cfg
        self._brain_cfg = brain_cfg or {}
        self._failed = False
        self._get_collection: Any = None

        raw_path = mem_cfg.get("mempalace_palace_path", "data/memory/mempalace")
        palace_path = Path(str(raw_path)).expanduser()
        if not palace_path.is_absolute():
            palace_path = project_root() / palace_path
        self._palace_path = palace_path

        self._wing = str(mem_cfg.get("mempalace_wing", "askme"))
        self._room = str(mem_cfg.get("mempalace_room", "robot"))
        self._collection_name = mem_cfg.get("mempalace_collection_name") or None
        self._n_results = int(mem_cfg.get("mempalace_n_results", 5))
        self._min_similarity = float(mem_cfg.get("mempalace_min_similarity", 0.3))
        self._retrieve_timeout = float(mem_cfg.get("retrieve_timeout", 2.0))

    @property
    def available(self) -> bool:
        """Whether the MemPalace Python package has been loaded."""
        return self._get_collection is not None and not self._failed

    @property
    def palace_path(self) -> str:
        return str(self._palace_path)

    def _ensure_mempalace(self) -> bool:
        if self.available:
            return True
        if self._failed:
            return False
        try:
            from mempalace.palace import get_collection

            self._palace_path.mkdir(parents=True, exist_ok=True)
            self._get_collection = get_collection
            return True
        except Exception as exc:
            logger.warning("[Memory] MemPalace init failed: %s", exc)
            self._failed = True
            return False

    def _open_collection(self, *, create: bool) -> Any:
        if not self._ensure_mempalace():
            return None
        return self._get_collection(
            str(self._palace_path),
            collection_name=self._collection_name,
            create=create,
        )

    async def warmup(self) -> None:
        """Open the palace collection in a worker thread."""
        try:
            await asyncio.to_thread(self._open_collection, create=True)
            logger.info("[Memory] MemPalace warmup complete.")
        except Exception as exc:
            logger.debug("[Memory] MemPalace warmup failed: %s", exc)

    async def retrieve(self, text: str) -> str:
        items = await self.retrieve_items(text)
        return "\n".join(f"- {item['text']}" for item in items)

    async def retrieve_items(
        self,
        text: str,
        *,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not str(text or "").strip() or not self._ensure_mempalace():
            return []
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._retrieve_items_sync, text, metadata_filter),
                timeout=self._retrieve_timeout,
            )
        except TimeoutError:
            logger.warning("[Memory] MemPalace retrieval timed out (%.1fs).", self._retrieve_timeout)
            return []
        except Exception as exc:
            logger.debug("[Memory] MemPalace retrieve failed: %s", exc)
            return []

    def _retrieve_items_sync(
        self,
        text: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        collection = self._open_collection(create=False)
        if collection is None:
            return []
        where = self._where_filter(metadata_filter)
        kwargs: dict[str, Any] = {
            "query_texts": [text],
            "n_results": max(self._n_results * 2, self._n_results),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        docs = _first_or_empty(results, "documents")
        metas = _first_or_empty(results, "metadatas")
        dists = _first_or_empty(results, "distances")

        items: list[dict[str, Any]] = []
        for doc, metadata, distance in zip(docs, metas, dists):
            clean = str(doc or "").strip()
            if not clean:
                continue
            try:
                similarity = max(0.0, 1.0 - float(distance))
            except (TypeError, ValueError):
                similarity = None
            if similarity is not None and similarity < self._min_similarity:
                continue
            meta = dict(metadata or {})
            if metadata_filter and any(
                str(meta.get(key) or "") != str(value)
                for key, value in metadata_filter.items()
            ):
                continue
            source_file = str(meta.get("source_file") or "")
            items.append(
                {
                    "text": clean,
                    "backend": "mempalace",
                    "source": meta.get("source") or Path(source_file).name or "mempalace",
                    "category": meta.get("category") or meta.get("type") or meta.get("room") or "",
                    "score": round(similarity, 4) if similarity is not None else None,
                    "metadata": meta,
                }
            )
            if len(items) >= self._n_results:
                break
        return items

    def _where_filter(
        self,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clauses: list[dict[str, Any]] = []
        if self._wing:
            clauses.append({"wing": self._wing})
        if self._room:
            clauses.append({"room": self._room})
        for key, value in (metadata_filter or {}).items():
            clean_key = str(key or "").strip()
            if not clean_key or clean_key.startswith("$") or value is None:
                continue
            clauses.append({clean_key: value})
        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    async def save(self, user_text: str, assistant_text: str) -> None:
        user = str(user_text or "").strip()
        assistant = str(assistant_text or "").strip()
        if not user and not assistant:
            return
        content = f"User: {user}\nAssistant: {assistant[:500]}"
        await self.save_fact(
            content,
            {
                "type": "conversation",
                "source": "conversation",
                "owner": "askme",
            },
        )

    async def save_fact(self, text: str, metadata: dict[str, Any] | None = None) -> bool:
        """Persist one fact and report only a confirmed collection upsert.

        Initialization unavailability and empty input are explicit ``False``
        results.  Collection write failures propagate so callers cannot report
        durable progress for a failed write.
        """
        clean = str(text or "").strip()
        if not clean or not self._ensure_mempalace():
            return False
        return await asyncio.to_thread(self._save_fact_sync, clean, metadata or {})

    def _save_fact_sync(self, text: str, metadata: dict[str, Any]) -> bool:
        collection = self._open_collection(create=True)
        if collection is None:
            return False
        now = datetime.now(UTC).isoformat()
        source = str(metadata.get("source") or metadata.get("record_id") or "askme_memory")
        drawer_id = self._drawer_id(text, metadata)
        merged_metadata = self._sanitize_metadata(
            {
                "wing": self._wing,
                "room": self._room,
                "source_file": source,
                "chunk_index": 0,
                "added_by": "askme",
                "filed_at": now,
                "updated_at": metadata.get("updated_at") or now,
                **metadata,
            }
        )
        collection.upsert(
            documents=[text],
            ids=[drawer_id],
            metadatas=[merged_metadata],
        )
        return True

    def _drawer_id(self, text: str, metadata: dict[str, Any]) -> str:
        """Build an idempotent MemPalace drawer id.

        Catalog-backed facts should update the same drawer when their text or
        lifecycle metadata changes, so ``record_id`` is the preferred stable
        key. Free-form conversation memories remain content-addressed.
        """
        record_id = str(metadata.get("record_id") or "").strip()
        if record_id:
            seed = f"{self._wing}:{self._room}:record:{record_id}"
        else:
            source = str(metadata.get("source") or "askme_memory")
            seed = f"{self._wing}:{self._room}:content:{source}:{text}"
        return "drawer_askme_" + hashlib.sha256(seed.encode()).hexdigest()[:24]

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        clean: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, str | int | float | bool):
                clean[str(key)] = value
            else:
                clean[str(key)] = str(value)
        return clean

    def close(self) -> None:
        if not self._get_collection:
            return
        try:
            from mempalace import palace

            backend = getattr(palace, "_DEFAULT_BACKEND", None)
            close_palace = getattr(backend, "close_palace", None)
            if callable(close_palace):
                close_palace(str(self._palace_path))
        except Exception:
            pass
        self._get_collection = None
