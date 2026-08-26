"""Contract tests for the local single-writer MemPalace sidecar."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from http.server import HTTPServer
from threading import Thread
from unittest.mock import patch
from urllib.request import Request, urlopen

import pytest

from scripts.runtime.mempalace_sidecar import MemPalaceStore, build_handler


@dataclass
class _FakeCollection:
    rows: dict[str, tuple[str, dict]] = field(default_factory=dict)
    query_calls: list[dict] = field(default_factory=list)
    embed_calls: list[tuple[list[str], bool]] = field(default_factory=list)
    probe_error: Exception | None = None
    write_error: Exception | None = None
    deleted_ids: list[str] = field(default_factory=list)

    def _embed(self, texts, *, is_query):
        self.embed_calls.append((list(texts), is_query))
        return [[0.1, 0.2, 0.3]]

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        if self.probe_error is not None:
            raise self.probe_error
        if kwargs.get("query_texts") == ["MemPalace 中文启动探针"]:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        matching = [
            (document, metadata)
            for document, metadata in self.rows.values()
            if _matches(metadata, kwargs.get("where", {}))
        ]
        return {
            "documents": [[row[0] for row in matching]],
            "metadatas": [[row[1] for row in matching]],
            "distances": [[0.1 for _ in matching]],
        }

    def upsert(self, *, documents, ids, metadatas):
        if self.write_error is not None:
            raise self.write_error
        self.rows[ids[0]] = (documents[0], dict(metadatas[0]))


    def delete(self, *, ids):
        self.deleted_ids.extend(ids)
        for drawer_id in ids:
            self.rows.pop(drawer_id, None)

    def get(self, *, ids=None, where=None, include=None):
        rows = self.rows.items()
        if ids is not None:
            rows = [(key, value) for key, value in rows if key in ids]
        if where:
            rows = [
                (key, value)
                for key, value in rows
                if _matches(value[1], where)
            ]
        return {
            "ids": [key for key, _ in rows],
            "metadatas": [value[1] for _, value in rows],
        }

    def update(self, *, ids, metadatas):
        document, _ = self.rows[ids[0]]
        self.rows[ids[0]] = (document, dict(metadatas[0]))

    def count(self):
        return len(self.rows)


def _matches(metadata: dict, where: dict) -> bool:
    if "$and" in where:
        return all(_matches(metadata, item) for item in where["$and"])
    return all(metadata.get(key) == value for key, value in where.items())


def test_probe_uses_chinese_query_and_propagates_embedding_failure():
    collection = _FakeCollection()
    store = MemPalaceStore(collection, palace_path="/palace", collection_name="askme")

    store.probe()

    assert collection.query_calls[0]["query_texts"] == ["MemPalace 中文启动探针"]

    collection.probe_error = RuntimeError("embedding unavailable")
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        store.probe()



def test_probe_verifies_write_lock_and_removes_probe_record():
    collection = _FakeCollection()
    store = MemPalaceStore(collection, palace_path="/palace", collection_name="askme")

    store.probe()

    assert collection.rows == {}
    assert len(collection.deleted_ids) == 1

    collection = _FakeCollection(write_error=OSError("read-only lock directory"))
    store = MemPalaceStore(collection, palace_path="/palace", collection_name="askme")
    with pytest.raises(OSError, match="read-only lock directory"):
        store.probe()
def test_health_reports_sidecar_mempalace_version():
    collection = _FakeCollection()
    with patch(
        "scripts.runtime.mempalace_sidecar.importlib.metadata.version",
        return_value="3.5.0",
    ):
        store = MemPalaceStore(
            collection, palace_path="/palace", collection_name="askme"
        )

    assert store.health()["mempalace_version"] == "3.5.0"


def test_upsert_and_update_cannot_escape_requested_lane():
    collection = _FakeCollection()
    store = MemPalaceStore(collection, palace_path="/palace", collection_name="askme")

    result = store.upsert(
        {
            "id": "drawer-1",
            "text": "The user prefers concise answers.",
            "wing": "askme",
            "room": "robot_behavior",
            "metadata": {
                "wing": "attacker",
                "room": "customer_knowledge",
                "source": "admission",
            },
        }
    )

    assert result == {"ok": True, "id": "drawer-1", "count": 1}
    metadata = collection.rows["drawer-1"][1]
    assert metadata["wing"] == "askme"
    assert metadata["room"] == "robot_behavior"

    with pytest.raises(PermissionError):
        store.update(
            {
                "id": "drawer-1",
                "wing": "askme",
                "room": "customer_knowledge",
                "metadata": {"source": "tamper"},
            }
        )

    updated = store.update(
        {
            "id": "drawer-1",
            "wing": "askme",
            "room": "robot_behavior",
            "metadata": {
                "wing": "attacker",
                "room": "customer_knowledge",
                "source": "operator",
            },
        }
    )
    assert updated["ok"] is True
    metadata = collection.rows["drawer-1"][1]
    assert metadata["wing"] == "askme"
    assert metadata["room"] == "robot_behavior"
    assert metadata["source"] == "operator"


def test_search_and_stats_are_scoped_to_requested_lane():
    collection = _FakeCollection(
        rows={
            "customer": ("FAQ answer", {"wing": "askme", "room": "customer_knowledge"}),
            "behavior": ("User preference", {"wing": "askme", "room": "robot_behavior"}),
        }
    )
    store = MemPalaceStore(collection, palace_path="/palace", collection_name="askme")

    result = store.search(
        {
            "query": "answer",
            "wing": "askme",
            "room": "customer_knowledge",
            "n_results": 5,
        }
    )

    assert [item["text"] for item in result["items"]] == ["FAQ answer"]
    assert result["count"] == 1
    assert store.stats({"wing": "askme", "room": "robot_behavior"})["count"] == 1


def test_same_query_across_rooms_reuses_one_query_embedding():
    collection = _FakeCollection(
        rows={
            "customer": ("FAQ answer", {"wing": "askme", "room": "customer_knowledge"}),
            "behavior": ("User preference", {"wing": "askme", "room": "robot_behavior"}),
        }
    )
    store = MemPalaceStore(collection, palace_path="/palace", collection_name="askme")

    customer = store.search(
        {
            "query": "同一个问题",
            "wing": "askme",
            "room": "customer_knowledge",
            "n_results": 5,
        }
    )
    behavior = store.search(
        {
            "query": "同一个问题",
            "wing": "askme",
            "room": "robot_behavior",
            "n_results": 5,
        }
    )

    assert collection.embed_calls == [(["同一个问题"], True)]
    assert "query_embeddings" in collection.query_calls[-2]
    assert "query_embeddings" in collection.query_calls[-1]
    assert [item["text"] for item in customer["items"]] == ["FAQ answer"]
    assert [item["text"] for item in behavior["items"]] == ["User preference"]


def test_search_falls_back_to_query_texts_without_private_embed_api():
    collection = _FakeCollection(
        rows={
            "customer": ("FAQ answer", {"wing": "askme", "room": "customer_knowledge"}),
        }
    )
    collection._embed = None
    store = MemPalaceStore(collection, palace_path="/palace", collection_name="askme")

    store.search(
        {
            "query": "fallback query",
            "wing": "askme",
            "room": "customer_knowledge",
        }
    )

    assert collection.query_calls[-1]["query_texts"] == ["fallback query"]


def test_http_server_exposes_health_and_json_endpoints():
    collection = _FakeCollection()
    store = MemPalaceStore(collection, palace_path="/palace", collection_name="askme")
    server = HTTPServer(("127.0.0.1", 0), build_handler(store))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = "http://127.0.0.1:" + str(server.server_port)
    try:
        with urlopen(base_url + "/healthz", timeout=1) as response:
            health = json.load(response)
        assert health["ready"] is True
        assert health["count"] == 0

        request = Request(
            base_url + "/v1/upsert",
            data=json.dumps(
                {
                    "id": "drawer-http",
                    "text": "HTTP memory",
                    "wing": "askme",
                    "room": "robot_behavior",
                    "metadata": {"room": "customer_knowledge"},
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=1) as response:
            result = json.load(response)
        assert result["ok"] is True
        assert collection.rows["drawer-http"][1]["room"] == "robot_behavior"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
