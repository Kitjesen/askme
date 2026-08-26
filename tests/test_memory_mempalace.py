"""Tests for the optional MemPalace memory backend."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from askme.memory.bridge import MemoryBridge
from askme.memory.mempalace_backend import MemPalaceBackend


class _FakeCollection:
    def __init__(self) -> None:
        self.query_calls: list[dict] = []
        self.upsert_calls: list[dict] = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return SimpleNamespace(
            documents=[["A区卫生间在东侧", "过期知识"]],
            metadatas=[
                [
                    {
                        "wing": "askme",
                        "room": "robot",
                        "source": "site-map.md",
                        "approval_status": "published",
                        "expires_at": "2099-01-01T00:00:00+00:00",
                    },
                    {
                        "wing": "askme",
                        "room": "robot",
                        "source": "old.md",
                        "approval_status": "published",
                        "expires_at": "2000-01-01T00:00:00+00:00",
                    },
                ]
            ],
            distances=[[0.12, 0.2]],
        )

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)


def _patch_mempalace(collection: _FakeCollection):
    palace_mod = ModuleType("mempalace.palace")
    palace_mod.get_collection = MagicMock(return_value=collection)
    pkg = ModuleType("mempalace")
    pkg.palace = palace_mod
    return patch.dict(sys.modules, {"mempalace": pkg, "mempalace.palace": palace_mod})


def _bridge_config(backend: str = "mempalace") -> dict:
    return {
        "memory": {
            "enabled": True,
            "backend": backend,
            "retrieve_timeout": 2.0,
            "mempalace_fallback_backend": "vector",
            "mempalace_wing": "askme",
            "mempalace_room": "robot",
        },
        "app": {"data_dir": "data"},
        "brain": {},
    }


class TestMemPalaceBackend:
    @pytest.mark.asyncio
    async def test_retrieve_items_preserves_trust_metadata(self, tmp_path):
        collection = _FakeCollection()
        backend = MemPalaceBackend(
            {
                "mempalace_palace_path": str(tmp_path / "palace"),
                "mempalace_wing": "askme",
                "mempalace_room": "robot",
                "mempalace_min_similarity": 0.3,
            }
        )

        with _patch_mempalace(collection):
            items = await backend.retrieve_items("卫生间在哪里")

        assert len(items) == 2
        assert items[0]["backend"] == "mempalace"
        assert items[0]["source"] == "site-map.md"
        assert items[0]["metadata"]["approval_status"] == "published"
        assert collection.query_calls[0]["where"] == {
            "$and": [{"wing": "askme"}, {"room": "robot"}]
        }

    @pytest.mark.asyncio
    async def test_save_fact_upserts_drawer_with_sanitized_metadata(self, tmp_path):
        collection = _FakeCollection()
        backend = MemPalaceBackend(
            {
                "mempalace_palace_path": str(tmp_path / "palace"),
                "mempalace_wing": "askme",
                "mempalace_room": "robot",
            }
        )

        with _patch_mempalace(collection):
            saved = await backend.save_fact(
                "三号设备在A区东侧",
                {"source": "sop.md", "tags": ["巡检"], "approval_status": "published"},
            )

        assert len(collection.upsert_calls) == 1
        call = collection.upsert_calls[0]
        assert call["documents"] == ["三号设备在A区东侧"]
        assert call["ids"][0].startswith("drawer_askme_")
        assert call["metadatas"][0]["source"] == "sop.md"
        assert call["metadatas"][0]["tags"] == "['巡检']"
        assert saved is True

    @pytest.mark.asyncio
    async def test_catalog_record_id_produces_stable_drawer_id(self, tmp_path):
        collection = _FakeCollection()
        backend = MemPalaceBackend(
            {
                "mempalace_palace_path": str(tmp_path / "palace"),
                "mempalace_wing": "askme",
                "mempalace_room": "robot",
            }
        )

        with _patch_mempalace(collection):
            await backend.save_fact(
                "A区卫生间在东侧",
                {"record_id": "know_restroom_a", "source": "site-map-v1.md"},
            )
            await backend.save_fact(
                "A区卫生间在东南侧",
                {"record_id": "know_restroom_a", "source": "site-map-v2.md"},
            )

        assert len(collection.upsert_calls) == 2
        assert collection.upsert_calls[0]["ids"][0] == collection.upsert_calls[1]["ids"][0]
        assert collection.upsert_calls[1]["documents"] == ["A区卫生间在东南侧"]

    @pytest.mark.asyncio
    async def test_without_record_id_content_address_changes_with_text(self, tmp_path):
        collection = _FakeCollection()
        backend = MemPalaceBackend(
            {
                "mempalace_palace_path": str(tmp_path / "palace"),
                "mempalace_wing": "askme",
                "mempalace_room": "robot",
            }
        )

        with _patch_mempalace(collection):
            await backend.save_fact("第一条自由记忆", {"source": "free"})
            await backend.save_fact("第二条自由记忆", {"source": "free"})

        assert collection.upsert_calls[0]["ids"][0] != collection.upsert_calls[1]["ids"][0]

    def test_unavailable_when_package_missing(self, tmp_path):
        backend = MemPalaceBackend({"mempalace_palace_path": str(tmp_path / "palace")})

        with patch("builtins.__import__", side_effect=ImportError("no mempalace")):
            assert backend._ensure_mempalace() is False

        assert backend.available is False

    @pytest.mark.asyncio
    async def test_save_fact_reports_local_write_failure(self, tmp_path):
        collection = _FakeCollection()
        collection.upsert = MagicMock(side_effect=RuntimeError("disk full"))
        backend = MemPalaceBackend(
            {"mempalace_palace_path": str(tmp_path / "palace")}
        )

        with _patch_mempalace(collection):
            saved = await backend.save_fact("cannot persist", {"source": "test"})

        assert saved is False

    @pytest.mark.asyncio
    async def test_http_transport_routes_search_to_configured_lane(self):
        backend = MemPalaceBackend(
            {
                "mempalace_transport": "http",
                "mempalace_url": "http://127.0.0.1:8766/",
                "mempalace_wing": "askme",
                "mempalace_room": "customer_knowledge",
                "mempalace_n_results": 3,
                "mempalace_min_similarity": 0.3,
            }
        )
        backend._http_healthy = True
        response = {
            "ok": True,
            "items": [
                {
                    "text": "Visitor registration is at reception.",
                    "metadata": {"source": "faq.md", "room": "customer_knowledge"},
                    "distance": 0.08,
                }
            ],
            "count": 7,
        }

        with patch.object(
            backend,
            "_request_http_sync",
            return_value=response,
        ) as request:
            items = await backend.retrieve_items("visitor registration")

        assert items[0]["text"] == "Visitor registration is at reception."
        assert items[0]["score"] == 0.92
        request.assert_called_once_with(
            "POST",
            "/v1/search",
            {
                "query": "visitor registration",
                "wing": "askme",
                "room": "customer_knowledge",
                "n_results": 6,
            },
        )
        assert backend.health_snapshot["count"] == 7

    @pytest.mark.asyncio
    async def test_http_transport_upsert_forces_configured_lane_and_returns_status(self):
        backend = MemPalaceBackend(
            {
                "mempalace_transport": "http",
                "mempalace_url": "http://127.0.0.1:8766",
                "mempalace_wing": "askme",
                "mempalace_room": "robot_behavior",
            }
        )
        backend._http_healthy = True

        with patch.object(
            backend,
            "_request_http_sync",
            return_value={"ok": True, "count": 4},
        ) as request:
            saved = await backend.save_fact(
                "The user prefers concise answers.",
                {"wing": "other", "room": "customer_knowledge", "source": "admission"},
            )

        assert saved is True
        payload = request.call_args.args[2]
        assert request.call_args.args[:2] == ("POST", "/v1/upsert")
        assert payload["wing"] == "askme"
        assert payload["room"] == "robot_behavior"
        assert payload["metadata"]["wing"] == "askme"
        assert payload["metadata"]["room"] == "robot_behavior"
        assert backend.health_snapshot["count"] == 4

    def test_http_transport_health_probe_reports_count_and_error(self):
        backend = MemPalaceBackend(
            {
                "mempalace_transport": "http",
                "mempalace_url": "http://127.0.0.1:8766",
                "mempalace_wing": "askme",
                "mempalace_room": "customer_knowledge",
            }
        )

        with patch.object(
            backend,
            "_request_http_sync",
            return_value={
                "ok": True,
                "ready": True,
                "count": 9,
                "mempalace_version": "3.5.0",
            },
        ):
            assert backend._ensure_mempalace() is True

        assert backend.available is True
        assert backend.health_snapshot["count"] == 9
        assert backend.health_snapshot["last_error"] == ""
        assert backend.health_snapshot["mempalace_version"] == "3.5.0"

        failed = MemPalaceBackend(
            {
                "mempalace_transport": "http",
                "mempalace_url": "http://127.0.0.1:8766",
            }
        )
        with patch.object(
            failed,
            "_request_http_sync",
            side_effect=ConnectionError("connection refused"),
        ):
            assert failed._ensure_mempalace() is False

        assert failed.available is False
        assert "connection refused" in failed.health_snapshot["last_error"]

    @pytest.mark.asyncio
    async def test_http_transport_updates_metadata_with_lane_scope(self):
        backend = MemPalaceBackend(
            {
                "mempalace_transport": "http",
                "mempalace_url": "http://127.0.0.1:8766",
                "mempalace_wing": "askme",
                "mempalace_room": "robot_behavior",
            }
        )
        backend._http_healthy = True

        with patch.object(
            backend,
            "_request_http_sync",
            return_value={"ok": True, "count": 2},
        ) as request:
            updated = await backend.update_metadata(
                "know_123",
                {"wing": "other", "room": "customer_knowledge", "status": "active"},
            )

        assert updated is True
        assert request.call_args.args[:2] == ("POST", "/v1/update")
        payload = request.call_args.args[2]
        assert payload["id"] == backend._drawer_id(
            "", {"record_id": "know_123"}
        )
        assert payload["wing"] == "askme"
        assert payload["room"] == "robot_behavior"
        assert payload["metadata"]["wing"] == "askme"
        assert payload["metadata"]["room"] == "robot_behavior"


class TestMemoryBridgeMemPalace:
    @pytest.mark.asyncio
    async def test_bridge_retrieve_routes_to_mempalace_and_filters_expired(self, tmp_path):
        bridge = MemoryBridge(config=_bridge_config(), data_dir=tmp_path)
        backend = MagicMock()
        backend.available = True
        backend.palace_path = str(tmp_path / "palace")
        backend.retrieve_items = AsyncMock(
            return_value=[
                {
                    "text": "A区卫生间在东侧",
                    "backend": "mempalace",
                    "source": "site-map.md",
                    "category": "location",
                    "score": 0.9,
                    "metadata": {
                        "approval_status": "published",
                        "expires_at": "2099-01-01T00:00:00+00:00",
                    },
                },
                {
                    "text": "旧答案",
                    "backend": "mempalace",
                    "source": "old.md",
                    "score": 0.8,
                    "metadata": {
                        "approval_status": "published",
                        "expires_at": "2000-01-01T00:00:00+00:00",
                    },
                },
            ]
        )
        bridge._mempalace = backend

        with patch.object(bridge, "_ensure_mempalace", return_value=True):
            result = await bridge.retrieve("卫生间")

        health = bridge.health()
        assert "A区卫生间在东侧" in result
        assert "旧答案" not in result
        assert health["last_backend"] == "mempalace"
        assert health["mempalace_ready"] is True
        assert health["last_dropped_evidence"][0]["drop_reason"] == "expired"

    @pytest.mark.asyncio
    async def test_bridge_falls_back_to_vector_when_mempalace_unavailable(self, tmp_path):
        bridge = MemoryBridge(config=_bridge_config(), data_dir=tmp_path)
        store = MagicMock()
        store.available = True
        store.size = 1
        store.search = MagicMock(return_value=[
            {"text": "vector fallback", "score": 0.9, "metadata": {}},
        ])
        bridge._store = store

        with patch.object(bridge, "_ensure_mempalace", return_value=False):
            result = await bridge.retrieve("test")

        assert "vector fallback" in result
        assert bridge.health()["last_backend"] == "vector"
        assert bridge.health()["last_fallback_reason"] == "mempalace_unavailable"

    @pytest.mark.asyncio
    async def test_bridge_falls_back_to_vector_when_mempalace_has_no_hits(self, tmp_path):
        bridge = MemoryBridge(config=_bridge_config(), data_dir=tmp_path)
        backend = MagicMock()
        backend.available = True
        backend.retrieve_items = AsyncMock(return_value=[])
        bridge._mempalace = backend
        store = MagicMock()
        store.available = True
        store.size = 1
        store.search = MagicMock(return_value=[
            {"text": "vector answer", "score": 0.91, "metadata": {}},
        ])
        bridge._store = store

        with patch.object(bridge, "_ensure_mempalace", return_value=True):
            result = await bridge.retrieve("test")

        assert "vector answer" in result
        assert bridge.health()["last_backend"] == "vector"
        assert bridge.health()["last_fallback_reason"] == "mempalace_empty"

    @pytest.mark.asyncio
    async def test_bridge_falls_back_to_vector_when_mempalace_query_fails(self, tmp_path):
        bridge = MemoryBridge(config=_bridge_config(), data_dir=tmp_path)
        backend = MagicMock()
        backend.available = True
        backend.retrieve_items = AsyncMock(side_effect=RuntimeError("query failed"))
        bridge._mempalace = backend
        store = MagicMock()
        store.available = True
        store.size = 1
        store.search = MagicMock(return_value=[
            {"text": "vector recovery", "score": 0.88, "metadata": {}},
        ])
        bridge._store = store

        with patch.object(bridge, "_ensure_mempalace", return_value=True):
            result = await bridge.retrieve("test")

        assert "vector recovery" in result
        assert bridge.health()["last_backend"] == "vector"
        assert bridge.health()["last_fallback_reason"] == "mempalace_retrieve_failed"

    @pytest.mark.asyncio
    async def test_bridge_save_routes_to_mempalace(self, tmp_path):
        bridge = MemoryBridge(config=_bridge_config(), data_dir=tmp_path)
        backend = MagicMock()
        backend.available = True
        backend.save = AsyncMock()
        bridge._mempalace = backend

        with patch.object(bridge, "_ensure_mempalace", return_value=True):
            await bridge.save("你好", "你好，需要什么帮助")

        backend.save.assert_awaited_once_with("你好", "你好，需要什么帮助")
