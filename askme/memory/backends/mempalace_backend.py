"""MemPalace backend for askme long-term memory.

MemPalace is a local-first Chroma-based memory store. This adapter keeps it
optional: if the ``mempalace`` package is missing or the palace cannot be
opened, callers get an empty result and MemoryBridge can fall back to another
backend.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
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
        self._transport = str(mem_cfg.get("mempalace_transport", "python")).strip().lower()
        self._http_url = str(
            mem_cfg.get("mempalace_url", "http://127.0.0.1:8766")
        ).strip().rstrip("/")
        self._http_timeout = max(
            0.1,
            float(mem_cfg.get("mempalace_http_timeout", mem_cfg.get("retrieve_timeout", 2.0))),
        )
        self._http_healthy = False
        self._last_error = ""
        self._remote_count = 0
        self._mempalace_version = ""
        self._request_count = 0

        raw_path = mem_cfg.get("mempalace_palace_path", "data/memory/mempalace")
        palace_path = Path(str(raw_path)).expanduser()
        if not palace_path.is_absolute():
            palace_path = project_root() / palace_path
        self._palace_path = palace_path

        self._wing = str(mem_cfg.get("mempalace_wing", "askme")).strip()
        self._room = str(mem_cfg.get("mempalace_room", "robot")).strip()
        self._collection_name = mem_cfg.get("mempalace_collection_name") or None
        self._n_results = int(mem_cfg.get("mempalace_n_results", 5))
        self._min_similarity = float(mem_cfg.get("mempalace_min_similarity", 0.3))
        self._retrieve_timeout = float(mem_cfg.get("retrieve_timeout", 2.0))
        if self._transport not in {"http", "python"}:
            self._failed = True
            self._last_error = f"unsupported transport: {self._transport}"
        if self._transport == "http":
            parsed = urllib.parse.urlparse(self._http_url)
            if parsed.scheme != "http" or parsed.hostname not in {
                "127.0.0.1",
                "localhost",
                "::1",
            }:
                self._failed = True
                self._last_error = "mempalace_url must be a loopback HTTP URL"

    @property
    def available(self) -> bool:
        """Whether the configured MemPalace transport is ready."""
        if self._transport == "http":
            return self._http_healthy and not self._failed
        return self._get_collection is not None and not self._failed

    @property
    def palace_path(self) -> str:
        return str(self._palace_path)

    @property
    def health_snapshot(self) -> dict[str, Any]:
        """Return transport-level readiness, count and last error evidence."""
        return {
            "transport": self._transport,
            "available": self.available,
            "url": self._http_url if self._transport == "http" else "",
            "wing": self._wing,
            "room": self._room,
            "count": self._remote_count,
            "mempalace_version": self._mempalace_version,
            "request_count": self._request_count,
            "last_error": self._last_error,
        }

    def _ensure_mempalace(self) -> bool:
        if self.available:
            return True
        if self._failed:
            return False
        if self._transport == "http":
            try:
                response = self._request_http_sync("GET", "/healthz")
            except Exception as exc:
                self._record_http_failure(exc)
                return False
            self._accept_http_response(response)
            self._http_healthy = bool(response.get("ok") and response.get("ready"))
            if not self._http_healthy:
                self._last_error = str(response.get("error") or "sidecar not ready")
            return self._http_healthy
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
        """Open the palace collection or prove the sidecar is ready."""
        try:
            if self._transport == "http":
                ready = await asyncio.to_thread(self._ensure_mempalace)
                if not ready:
                    raise RuntimeError(self._last_error or "MemPalace sidecar unavailable")
            else:
                await asyncio.to_thread(self._open_collection, create=True)
            logger.info("[Memory] MemPalace warmup complete.")
        except Exception as exc:
            self._last_error = str(exc)
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
        clean_text = str(text or "").strip()
        if not clean_text:
            return []
        if not self.available and not await asyncio.to_thread(self._ensure_mempalace):
            return []
        try:
            if self._transport == "http":
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._request_http_sync,
                        "POST",
                        "/v1/search",
                        {
                            "query": clean_text,
                            "wing": self._wing,
                            "room": self._room,
                            "n_results": max(self._n_results * 2, self._n_results),
                        },
                    ),
                    timeout=self._retrieve_timeout,
                )
                self._accept_http_response(response)
                return self._http_response_items(response, metadata_filter)
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._retrieve_items_sync,
                    clean_text,
                    metadata_filter,
                ),
                timeout=self._retrieve_timeout,
            )
        except TimeoutError:
            self._record_http_failure("retrieval timeout")
            logger.warning("[Memory] MemPalace retrieval timed out (%.1fs).", self._retrieve_timeout)
            return []
        except Exception as exc:
            self._record_http_failure(exc)
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
        return self._build_items(docs, metas, dists, metadata_filter)

    def _http_response_items(
        self,
        response: dict[str, Any],
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        raw_items = response.get("items")
        if not isinstance(raw_items, list):
            return []
        docs: list[Any] = []
        metas: list[Any] = []
        dists: list[Any] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            docs.append(raw_item.get("text"))
            metas.append(raw_item.get("metadata"))
            dists.append(raw_item.get("distance"))
        return self._build_items(docs, metas, dists, metadata_filter)

    def _build_items(
        self,
        docs: list[Any],
        metas: list[Any],
        dists: list[Any],
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
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

    async def save(self, user_text: str, assistant_text: str) -> bool:
        user = str(user_text or "").strip()
        assistant = str(assistant_text or "").strip()
        if not user and not assistant:
            return False
        content = f"User: {user}\nAssistant: {assistant[:500]}"
        return await self.save_fact(
            content,
            {
                "type": "conversation",
                "source": "conversation",
                "owner": "askme",
            },
        )

    async def save_fact(self, text: str, metadata: dict[str, Any] | None = None) -> bool:
        """Persist one fact and report only a confirmed backend upsert."""
        clean = str(text or "").strip()
        if not clean:
            return False
        if not self.available and not await asyncio.to_thread(self._ensure_mempalace):
            return False
        drawer_id, merged_metadata = self._fact_record(clean, metadata or {})
        try:
            if self._transport == "http":
                response = await asyncio.to_thread(
                    self._request_http_sync,
                    "POST",
                    "/v1/upsert",
                    {
                        "id": drawer_id,
                        "text": clean,
                        "wing": self._wing,
                        "room": self._room,
                        "metadata": merged_metadata,
                    },
                )
                self._accept_http_response(response)
                return bool(response.get("ok"))
            return await asyncio.to_thread(
                self._save_fact_sync,
                clean,
                drawer_id,
                merged_metadata,
            )
        except Exception as exc:
            self._record_http_failure(exc)
            logger.debug("[Memory] MemPalace save_fact failed: %s", exc)
            return False

    def _fact_record(
        self, text: str, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        now = datetime.now(UTC).isoformat()
        source = str(metadata.get("source") or metadata.get("record_id") or "askme_memory")
        drawer_id = self._drawer_id(text, metadata)
        merged_metadata = self._sanitize_metadata(
            {
                "source_file": source,
                "chunk_index": 0,
                "added_by": "askme",
                "filed_at": now,
                "updated_at": metadata.get("updated_at") or now,
                **metadata,
                "wing": self._wing,
                "room": self._room,
            }
        )
        return drawer_id, merged_metadata

    def _save_fact_sync(
        self,
        text: str,
        drawer_id: str,
        merged_metadata: dict[str, Any],
    ) -> bool:
        collection = self._open_collection(create=True)
        if collection is None:
            return False
        collection.upsert(
            documents=[text],
            ids=[drawer_id],
            metadatas=[merged_metadata],
        )
        return True
    async def update_metadata(
        self, record_id: str, metadata: dict[str, Any] | None = None
    ) -> bool:
        clean_record_id = str(record_id or "").strip()
        if not clean_record_id:
            return False
        clean_id = (
            clean_record_id
            if clean_record_id.startswith("drawer_askme_")
            else self._drawer_id("", {"record_id": clean_record_id})
        )
        if not self.available and not await asyncio.to_thread(self._ensure_mempalace):
            return False
        merged = self._sanitize_metadata(
            {
                **(metadata or {}),
                "wing": self._wing,
                "room": self._room,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        try:
            if self._transport == "http":
                response = await asyncio.to_thread(
                    self._request_http_sync,
                    "POST",
                    "/v1/update",
                    {
                        "id": clean_id,
                        "wing": self._wing,
                        "room": self._room,
                        "metadata": merged,
                    },
                )
                self._accept_http_response(response)
                return bool(response.get("ok"))
            return await asyncio.to_thread(
                self._update_metadata_sync,
                clean_id,
                merged,
            )
        except Exception as exc:
            self._record_http_failure(exc)
            logger.debug("[Memory] MemPalace update_metadata failed: %s", exc)
            return False

    def _update_metadata_sync(
        self, drawer_id: str, metadata: dict[str, Any]
    ) -> bool:
        collection = self._open_collection(create=False)
        if collection is None:
            return False
        result = collection.get(ids=[drawer_id], include=["metadatas"])
        ids = result.get("ids") if isinstance(result, dict) else getattr(result, "ids", None)
        metadatas = (
            result.get("metadatas")
            if isinstance(result, dict)
            else getattr(result, "metadatas", None)
        )
        if not ids or not metadatas:
            return False
        existing = dict(metadatas[0] or {})
        if existing.get("wing") != self._wing or existing.get("room") != self._room:
            return False
        merged = {**existing, **metadata, "wing": self._wing, "room": self._room}
        collection.update(ids=[drawer_id], metadatas=[merged])
        return True

    async def stats(self) -> dict[str, Any]:
        if not self.available and not await asyncio.to_thread(self._ensure_mempalace):
            return self.health_snapshot
        try:
            if self._transport == "http":
                response = await asyncio.to_thread(
                    self._request_http_sync,
                    "POST",
                    "/v1/stats",
                    {"wing": self._wing, "room": self._room},
                )
                self._accept_http_response(response)
            else:
                self._remote_count = await asyncio.to_thread(self._stats_sync)
        except Exception as exc:
            self._record_http_failure(exc)
        return self.health_snapshot

    def _stats_sync(self) -> int:
        collection = self._open_collection(create=False)
        if collection is None:
            return 0
        result = collection.get(where=self._where_filter(), include=["metadatas"])
        ids = result.get("ids") if isinstance(result, dict) else getattr(result, "ids", None)
        return len(ids or [])

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

    def _request_http_sync(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._failed:
            raise RuntimeError(self._last_error or "MemPalace backend is disabled")
        if path not in {
            "/healthz",
            "/v1/search",
            "/v1/upsert",
            "/v1/update",
            "/v1/stats",
        }:
            raise ValueError("unsupported MemPalace sidecar endpoint")
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self._http_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        self._request_count += 1
        try:
            with urllib.request.urlopen(request, timeout=self._http_timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                error_payload = json.loads(raw.decode()) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                error_payload = {}
            detail = (
                error_payload.get("error")
                if isinstance(error_payload, dict)
                else ""
            )
            raise RuntimeError(
                f"MemPalace sidecar HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"MemPalace sidecar unavailable: {exc.reason}") from exc
        try:
            decoded = json.loads(raw.decode())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("MemPalace sidecar returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("MemPalace sidecar returned a non-object response")
        if decoded.get("ok") is False:
            raise RuntimeError(str(decoded.get("error") or "MemPalace sidecar error"))
        return decoded

    def _accept_http_response(self, response: dict[str, Any]) -> None:
        if response.get("ok") is False:
            raise RuntimeError(str(response.get("error") or "MemPalace sidecar error"))
        if response.get("ready") is False:
            raise RuntimeError(str(response.get("error") or "MemPalace sidecar not ready"))
        self._http_healthy = True
        self._last_error = ""
        if "count" in response:
            try:
                self._remote_count = max(0, int(response["count"]))
            except (TypeError, ValueError):
                pass
        if "mempalace_version" in response:
            self._mempalace_version = str(response["mempalace_version"] or "")

    def _record_http_failure(self, error: Exception | str) -> None:
        self._last_error = str(error)
        if self._transport == "http":
            self._http_healthy = False

    def close(self) -> None:
        if self._transport == "http":
            self._http_healthy = False
            return
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
