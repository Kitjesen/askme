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
            await backend.save_fact(
                "三号设备在A区东侧",
                {"source": "sop.md", "tags": ["巡检"], "approval_status": "published"},
            )

        assert len(collection.upsert_calls) == 1
        call = collection.upsert_calls[0]
        assert call["documents"] == ["三号设备在A区东侧"]
        assert call["ids"][0].startswith("drawer_askme_")
        assert call["metadatas"][0]["source"] == "sop.md"
        assert call["metadatas"][0]["tags"] == "['巡检']"

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
