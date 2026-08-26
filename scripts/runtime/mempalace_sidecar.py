#!/usr/bin/env python3
"""Single-writer loopback HTTP sidecar for MemPalace.

MemPalace and Chroma own process-level state that is unsafe to share across
AskMe worker threads. This process serializes every storage operation through
the stdlib HTTPServer and is intentionally reachable only over loopback.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import os
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from time import monotonic
from typing import Any

logger = logging.getLogger("askme.mempalace_sidecar")

_MAX_BODY_BYTES = 1_048_576
_PROBE_TEXT = "MemPalace 中文启动探针"
_PROBE_DRAWER_ID = "__askme_mempalace_write_probe__"
_QUERY_CACHE_SIZE = 32
_QUERY_CACHE_TTL_S = 2.0


def _result_value(results: Any, key: str) -> Any:
    if isinstance(results, dict):
        return results.get(key)
    return getattr(results, key, None)


def _first_batch(results: Any, key: str) -> list[Any]:
    outer = _result_value(results, key)
    if not outer:
        return []
    return list(outer[0] or [])


def _required_string(payload: dict[str, Any], key: str, *, limit: int = 256) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    if len(value) > limit:
        raise ValueError(f"{key} is too long")
    return value


def _sanitize_metadata(metadata: Any) -> dict[str, str | int | float | bool]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        clean_key = str(key)
        if isinstance(value, str | int | float | bool):
            clean[clean_key] = value
        else:
            clean[clean_key] = str(value)
    return clean


def _scope(payload: dict[str, Any]) -> tuple[str, str]:
    return (
        _required_string(payload, "wing", limit=128),
        _required_string(payload, "room", limit=128),
    )


def _where(wing: str, room: str) -> dict[str, Any]:
    return {"$and": [{"wing": wing}, {"room": room}]}


class MemPalaceStore:
    """Synchronous storage facade; one HTTPServer thread is the sole caller."""

    def __init__(self, collection: Any, *, palace_path: str, collection_name: str) -> None:
        self.collection = collection
        self.palace_path = palace_path
        self.collection_name = collection_name
        self._query_embeddings: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._write_ready = False
        try:
            self.mempalace_version = importlib.metadata.version("mempalace")
        except importlib.metadata.PackageNotFoundError:
            self.mempalace_version = ""

    def probe(self) -> None:
        """Force the configured embedding model to load before opening a socket."""
        self.collection.query(
            query_texts=[_PROBE_TEXT],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        self.collection.upsert(
            documents=[_PROBE_TEXT],
            ids=[_PROBE_DRAWER_ID],
            metadatas=[{
                "wing": "askme_health",
                "room": "startup_probe",
                "probe": True,
            }],
        )
        self.collection.delete(ids=[_PROBE_DRAWER_ID])
        self._write_ready = True

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "ready": True,
            "write_ready": self._write_ready,
            "palace_path": self.palace_path,
            "collection": self.collection_name,
            "count": int(self.collection.count()),
            "mempalace_version": self.mempalace_version,
            "embedding_model": os.getenv("MEMPALACE_EMBEDDING_MODEL", ""),
            "embedding_device": os.getenv("MEMPALACE_EMBEDDING_DEVICE", ""),
        }

    def search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = _required_string(payload, "query", limit=20_000)
        wing, room = _scope(payload)
        try:
            n_results = int(payload.get("n_results", 5))
        except (TypeError, ValueError) as exc:
            raise ValueError("n_results must be an integer") from exc
        n_results = max(1, min(n_results, 50))
        query_args: dict[str, Any] = {
            "n_results": n_results,
            "where": _where(wing, room),
            "include": ["documents", "metadatas", "distances"],
        }
        query_embedding = self._query_embedding(query)
        if query_embedding is None:
            query_args["query_texts"] = [query]
        else:
            query_args["query_embeddings"] = query_embedding
        results = self.collection.query(**query_args)
        documents = _first_batch(results, "documents")
        metadatas = _first_batch(results, "metadatas")
        distances = _first_batch(results, "distances")
        items = [
            {
                "text": str(document or ""),
                "metadata": dict(metadata or {}),
                "distance": distance,
            }
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]
        return {
            "ok": True,
            "items": items,
            "count": self._scoped_count(wing, room),
        }

    def _query_embedding(self, query: str) -> Any:
        embed = getattr(self.collection, "_embed", None)
        if not callable(embed):
            inner_collection = getattr(self.collection, "_collection", None)
            embed = getattr(inner_collection, "_embed", None)
        if not callable(embed):
            return None
        now = monotonic()
        cached = self._query_embeddings.pop(query, None)
        if cached is not None:
            created_at, embedding = cached
            if now - created_at <= _QUERY_CACHE_TTL_S:
                self._query_embeddings[query] = cached
                return embedding
        embedding = embed([query], is_query=True)
        if embedding is None:
            return None
        self._query_embeddings[query] = (now, embedding)
        while len(self._query_embeddings) > _QUERY_CACHE_SIZE:
            self._query_embeddings.popitem(last=False)
        return embedding

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        drawer_id = _required_string(payload, "id", limit=512)
        text = _required_string(payload, "text", limit=200_000)
        wing, room = _scope(payload)
        metadata = _sanitize_metadata(payload.get("metadata"))
        metadata["wing"] = wing
        metadata["room"] = room
        self.collection.upsert(
            documents=[text],
            ids=[drawer_id],
            metadatas=[metadata],
        )
        return {
            "ok": True,
            "id": drawer_id,
            "count": self._scoped_count(wing, room),
        }

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        drawer_id = _required_string(payload, "id", limit=512)
        wing, room = _scope(payload)
        current = self.collection.get(ids=[drawer_id], include=["metadatas"])
        ids = list(_result_value(current, "ids") or [])
        metadatas = list(_result_value(current, "metadatas") or [])
        if not ids or not metadatas:
            raise KeyError(drawer_id)
        existing = dict(metadatas[0] or {})
        if existing.get("wing") != wing or existing.get("room") != room:
            raise PermissionError("drawer does not belong to requested wing/room")
        merged = {**existing, **_sanitize_metadata(payload.get("metadata"))}
        merged["wing"] = wing
        merged["room"] = room
        self.collection.update(ids=[drawer_id], metadatas=[merged])
        return {
            "ok": True,
            "id": drawer_id,
            "count": self._scoped_count(wing, room),
        }

    def stats(self, payload: dict[str, Any]) -> dict[str, Any]:
        wing, room = _scope(payload)
        return {
            "ok": True,
            "wing": wing,
            "room": room,
            "count": self._scoped_count(wing, room),
        }

    def _scoped_count(self, wing: str, room: str) -> int:
        results = self.collection.get(
            where=_where(wing, room),
            include=["metadatas"],
        )
        return len(list(_result_value(results, "ids") or []))


def build_handler(store: MemPalaceStore) -> type[BaseHTTPRequestHandler]:
    """Create a request handler bound to one initialized store."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - stdlib callback name
            if self.path != "/healthz":
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return
            self._send_json(HTTPStatus.OK, store.health())

        def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
            routes = {
                "/v1/search": store.search,
                "/v1/upsert": store.upsert,
                "/v1/update": store.update,
                "/v1/stats": store.stats,
            }
            operation = routes.get(self.path)
            if operation is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                return
            try:
                payload = self._read_json()
                result = operation(payload)
            except json.JSONDecodeError:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": "invalid_json"},
                )
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": str(exc)},
                )
            except PermissionError as exc:
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {"ok": False, "error": str(exc)},
                )
            except KeyError:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": "drawer_not_found"},
                )
            except Exception:
                logger.exception("MemPalace sidecar request failed: %s", self.path)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": "storage_error"},
                )
            else:
                self._send_json(HTTPStatus.OK, result)

        def _read_json(self) -> dict[str, Any]:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid content length") from exc
            if content_length <= 0:
                raise ValueError("request body is required")
            if content_length > _MAX_BODY_BYTES:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            return payload

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            logger.info("%s - %s", self.client_address[0], format % args)

    return Handler


def open_store(*, palace_path: str, collection_name: str) -> MemPalaceStore:
    """Open MemPalace and prove embeddings work before returning."""
    from mempalace.palace import get_collection

    Path(palace_path).mkdir(parents=True, exist_ok=True)
    collection = get_collection(
        palace_path,
        collection_name=collection_name,
        create=True,
    )
    store = MemPalaceStore(
        collection,
        palace_path=palace_path,
        collection_name=collection_name,
    )
    store.probe()
    return store


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--palace-path",
        default=os.getenv(
            "MEMPALACE_PALACE_PATH",
            "/home/sunrise/data/inovxio/askme/data/memory/mempalace/palace",
        ),
    )
    parser.add_argument(
        "--collection-name",
        default=os.getenv("MEMPALACE_COLLECTION", "askme"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("Refusing non-loopback MemPalace bind address")
    logging.basicConfig(
        level=os.getenv("MEMPALACE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    store = open_store(
        palace_path=str(Path(args.palace_path).expanduser().resolve()),
        collection_name=str(args.collection_name),
    )
    server = HTTPServer((args.host, args.port), build_handler(store))
    logger.info(
        "MemPalace sidecar ready at http://%s:%d (palace=%s collection=%s count=%d)",
        args.host,
        args.port,
        store.palace_path,
        store.collection_name,
        store.health()["count"],
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("MemPalace sidecar stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
