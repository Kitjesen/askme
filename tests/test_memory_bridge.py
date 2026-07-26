"""Tests for MemoryBridge — Mem0 primary + VectorStore fallback."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from askme.memory.bridge import MemoryBridge
from askme.memory.catalog import KnowledgeCatalog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_config(enabled=True):
    """Patch get_config to return a minimal config dict."""
    cfg = {
        "memory": {"enabled": enabled, "embed_model": "test-model", "retrieve_timeout": 2.0},
        "app": {"data_dir": "data"},
        "brain": {"api_key": "test-key", "base_url": "http://test", "model": "test-model"},
    }
    return patch("askme.memory.bridge.get_config", return_value=cfg)


def _patch_vector_store():
    """Patch VectorStore to a mock that reports unavailable."""
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_instance.available = False
    mock_instance.size = 0
    mock_cls.return_value = mock_instance
    return patch("askme.memory.bridge.VectorStore", mock_cls), mock_instance


def _make_bridge(enabled=True):
    """Create a MemoryBridge with mocked dependencies."""
    vs_patch, vs_mock = _patch_vector_store()
    with _patch_config(enabled=enabled), vs_patch:
        bridge = MemoryBridge()
    bridge._store = vs_mock
    return bridge, vs_mock


def _make_mem0_mock(search_results=None):
    """Create a mock Mem0 Memory instance."""
    mock = MagicMock()
    mock.search = MagicMock(return_value=search_results or {"results": []})
    mock.add = MagicMock(return_value={"results": []})
    return mock


# ---------------------------------------------------------------------------
# Tests: Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_disabled_bridge(self):
        bridge, _ = _make_bridge(enabled=False)
        assert bridge._enabled is False
        assert bridge.available is False

    def test_enabled_no_mem0_no_vectorstore(self):
        bridge, vs = _make_bridge(enabled=True)
        vs.available = False
        # Mem0 not initialised yet, VectorStore unavailable
        assert bridge.available is False

    def test_available_when_mem0_ready(self):
        bridge, _ = _make_bridge(enabled=True)
        bridge._mem0 = _make_mem0_mock()
        assert bridge.available is True

    def test_available_when_vectorstore_ready(self):
        bridge, vs = _make_bridge(enabled=True)
        vs.available = True
        # No Mem0, but VectorStore works
        assert bridge.available is True

    def test_auto_backend_selects_first_available_candidate(self):
        cfg = {
            "memory": {
                "enabled": True,
                "backend": "auto",
                "auto_backend_order": ["robotmem", "vector", "mem0"],
                "embed_model": "test-model",
                "retrieve_timeout": 2.0,
            },
            "app": {"data_dir": "data"},
            "brain": {"api_key": "test-key", "base_url": "http://test", "model": "test-model"},
        }

        def fake_find_spec(name):
            return object() if name == "fastembed" else None

        vs_patch, vs_mock = _patch_vector_store()
        with patch("askme.memory.bridge.get_config", return_value=cfg), \
             patch("askme.memory.bridge.importlib.util.find_spec", side_effect=fake_find_spec), \
             vs_patch:
            bridge = MemoryBridge()
        bridge._store = vs_mock
        health = bridge.health()

        assert health["configured_backend"] == "auto"
        assert health["backend"] == "vector"
        assert health["backend_selection"]["reason"] == "auto_selected:vector"
        assert health["backend_selection"]["auto_order"] == ["robotmem", "vector", "mem0"]

    def test_customer_knowledge_backend_is_separate_from_robot_behavior_memory(self):
        cfg = {
            "memory": {
                "enabled": True,
                "backend": "robotmem",
                "customer_knowledge_backend": "vector",
                "robot_behavior_memory_backend": "robotmem",
                "robot_behavior_memory_enabled": True,
                "embed_model": "test-model",
                "retrieve_timeout": 2.0,
            },
            "app": {"data_dir": "data"},
            "brain": {"api_key": "test-key", "base_url": "http://test", "model": "test-model"},
        }

        vs_patch, vs_mock = _patch_vector_store()
        with patch("askme.memory.bridge.get_config", return_value=cfg), vs_patch:
            bridge = MemoryBridge()
        bridge._store = vs_mock
        health = bridge.health()

        assert health["legacy_backend_config"] == "robotmem"
        assert health["configured_backend"] == "vector"
        assert health["customer_knowledge_backend"] == "vector"
        assert health["robot_behavior_memory_backend"] == "robotmem"
        assert health["robot_behavior_memory_enabled"] is True
        assert health["product_memory_roles"]["customer_knowledge"]["selected_backend"] == "vector"
        assert health["product_memory_roles"]["robot_behavior"]["configured_backend"] == "robotmem"
        assert health["product_memory_roles"]["robot_behavior"]["enabled"] is True

    def test_health_exposes_memory_backend_dependency_versions(self):
        cfg = {
            "memory": {
                "enabled": True,
                "backend": "mempalace",
                "customer_knowledge_backend": "mempalace",
                "mempalace_fallback_backend": "vector",
                "robot_behavior_memory_backend": "robotmem",
                "robot_behavior_memory_enabled": False,
                "embed_model": "test-model",
                "retrieve_timeout": 2.0,
            },
            "app": {"data_dir": "data"},
            "brain": {"api_key": "test-key", "base_url": "http://test", "model": "test-model"},
        }

        def fake_find_spec(name):
            return object() if name in {"mempalace", "fastembed"} else None

        def fake_version(package):
            versions = {
                "mempalace": "3.3.5",
                "fastembed": "0.8.0",
            }
            if package not in versions:
                from importlib.metadata import PackageNotFoundError

                raise PackageNotFoundError(package)
            return versions[package]

        vs_patch, vs_mock = _patch_vector_store()
        with patch("askme.memory.bridge.get_config", return_value=cfg), \
             patch("askme.memory.bridge.importlib.util.find_spec", side_effect=fake_find_spec), \
             patch("askme.memory.bridge.importlib_metadata.version", side_effect=fake_version), \
             vs_patch:
            bridge = MemoryBridge()
        bridge._store = vs_mock
        health = bridge.health()

        assert health["selected_backend_dependency"]["backend"] == "mempalace"
        assert health["selected_backend_dependency"]["installed"] is True
        assert health["selected_backend_dependency"]["version"] == "3.3.5"
        assert health["fallback_backend_dependency"]["backend"] == "vector"
        assert health["fallback_backend_dependency"]["version"] == "0.8.0"
        assert health["backend_dependencies"]["robotmem"]["installed"] is False
        assert (
            health["product_memory_roles"]["customer_knowledge"]["dependency"]["backend"]
            == "mempalace"
        )


# ---------------------------------------------------------------------------
# Tests: _ensure_mem0
# ---------------------------------------------------------------------------

class TestEnsureMem0:
    def test_returns_true_when_already_initialised(self):
        bridge, _ = _make_bridge()
        bridge._mem0 = _make_mem0_mock()
        assert bridge._ensure_mem0() is True

    def test_returns_false_when_disabled(self):
        bridge, _ = _make_bridge(enabled=False)
        assert bridge._ensure_mem0() is False

    def test_returns_false_after_failure(self):
        bridge, _ = _make_bridge()
        bridge._mem0_failed = True
        assert bridge._ensure_mem0() is False

    def test_initialises_mem0_on_first_call(self):
        bridge, _ = _make_bridge()
        mock_mem0 = _make_mem0_mock()
        mock_memory_cls = MagicMock()
        mock_memory_cls.from_config = MagicMock(return_value=mock_mem0)

        with patch.dict("sys.modules", {"mem0": MagicMock(Memory=mock_memory_cls)}):
            result = bridge._ensure_mem0()

        assert result is True
        assert bridge._mem0 is mock_mem0

    def test_sets_failed_flag_on_import_error(self):
        bridge, _ = _make_bridge()

        with patch("builtins.__import__", side_effect=ImportError("no mem0")):
            result = bridge._ensure_mem0()

        assert result is False
        assert bridge._mem0_failed is True

    def test_does_not_retry_after_failure(self):
        bridge, _ = _make_bridge()
        bridge._mem0_failed = True
        # Should not attempt import
        assert bridge._ensure_mem0() is False


# ---------------------------------------------------------------------------
# Tests: retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self):
        bridge, _ = _make_bridge(enabled=False)
        result = await bridge.retrieve("test query")
        assert result == ""

    @pytest.mark.asyncio
    async def test_mem0_retrieve_success(self):
        bridge, _ = _make_bridge()
        mock_mem0 = _make_mem0_mock(search_results={
            "results": [
                {"memory": "仓库A温度异常"},
                {"memory": "仓库B正常"},
            ]
        })
        bridge._mem0 = mock_mem0

        result = await bridge.retrieve("仓库情况")
        assert "仓库A温度异常" in result
        assert "仓库B正常" in result
        mock_mem0.search.assert_called_once_with("仓库情况", user_id="robot")
        health = bridge.health()
        assert health["last_backend"] == "mem0"
        assert health["last_retrieved_items"] == 2
        assert health["retrieve_count"] == 1
        assert health["last_evidence"][0]["backend"] == "mem0"
        assert "仓库A" in health["last_evidence"][0]["text"]

    @pytest.mark.asyncio
    async def test_mem0_retrieve_empty_results(self):
        bridge, _ = _make_bridge()
        bridge._mem0 = _make_mem0_mock(search_results={"results": []})

        result = await bridge.retrieve("nothing")
        assert result == ""

    @pytest.mark.asyncio
    async def test_mem0_retrieve_no_results_key(self):
        bridge, _ = _make_bridge()
        bridge._mem0 = _make_mem0_mock(search_results={})

        result = await bridge.retrieve("nothing")
        assert result == ""

    @pytest.mark.asyncio
    async def test_mem0_retrieve_filters_empty_memories(self):
        bridge, _ = _make_bridge()
        bridge._mem0 = _make_mem0_mock(search_results={
            "results": [
                {"memory": "有内容"},
                {"memory": ""},
                {"memory": "也有内容"},
            ]
        })

        result = await bridge.retrieve("test")
        assert "有内容" in result
        assert "也有内容" in result
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_fallback_to_vectorstore(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[
            {"text": "fallback result", "score": 0.8, "metadata": {"type": "knowledge", "approval_status": "published"}},
        ])

        result = await bridge.retrieve("test")
        assert "fallback result" in result
        vs.search.assert_called_once()
        health = bridge.health()
        assert health["last_backend"] == "vector"
        assert health["fallback_count"] >= 1
        assert health["last_evidence"][0]["backend"] == "vector"
        assert health["last_evidence"][0]["text"] == "fallback result"

    @pytest.mark.asyncio
    async def test_fallback_evidence_exposes_catalog_record_traceability(self, tmp_path):
        bridge, vs = _make_bridge()
        bridge._knowledge_catalog = KnowledgeCatalog(path=tmp_path / "records.json")
        bridge._knowledge_catalog.upsert_payloads([{
            "record_id": "route_gate_a",
            "text": "visitor desk is near gate A",
            "memory_text": "[route] visitor desk is near gate A",
            "approval_status": "published",
        }])
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[{
            "text": "visitor desk is near gate A",
            "score": 0.92,
            "metadata": {
                "record_id": "route_gate_a",
                "approval_status": "published",
                "source": "route.md",
                "category": "route",
                "evidence_version": 1,
            },
        }])

        await bridge.retrieve("gate A")
        evidence = bridge.health()["last_evidence"][0]

        assert evidence["record_id"] == "route_gate_a"
        assert evidence["source_record_id"] == "route_gate_a"
        assert evidence["evidence_version"] == 1

    @pytest.mark.asyncio
    async def test_fallback_filters_low_score(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[
            {"text": "low score", "score": 0.2, "metadata": {}},
        ])

        result = await bridge.retrieve("test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_fallback_filters_below_configured_vector_similarity_floor(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[
            {"text": "unrelated answer", "score": 0.49, "metadata": {}},
        ])

        result = await bridge.retrieve("expired route")
        health = bridge.health()

        assert result == ""
        assert health["vector_min_similarity"] == 0.5
        assert health["last_evidence"] == []

    @pytest.mark.asyncio
    async def test_fallback_filters_unapproved_knowledge(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[
            {
                "text": "draft location",
                "score": 0.8,
                "metadata": {"approval_status": "draft", "source": "site.md"},
            },
        ])

        result = await bridge.retrieve("location")
        health = bridge.health()

        assert result == ""
        assert health["last_evidence"] == []
        assert health["last_dropped_evidence"][0]["drop_reason"] == "approval_status:draft"
        assert health["last_dropped_evidence"][0]["used_in_prompt"] is False

    @pytest.mark.asyncio
    async def test_fallback_filters_deleted_knowledge(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[{
            "text": "deleted location",
            "score": 0.8,
            "metadata": {"approval_status": "deleted", "source": "site.md"},
        }])

        result = await bridge.retrieve("location")
        health = bridge.health()

        assert result == ""
        assert health["last_dropped_evidence"][0]["drop_reason"] == "approval_status:deleted"
        assert health["last_answer_policy"]["state"] == "unapproved"
        assert health["last_answer_policy"]["action"] == "refuse"
        assert health["last_answer_policy"]["required_operator_action"] == "approve_or_publish"

    @pytest.mark.asyncio
    async def test_fallback_filters_conflict_set_knowledge(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[{
            "text": "conflicted location",
            "score": 0.99,
            "metadata": {
                "approval_status": "published",
                "conflict_set_id": "conflict:device:a:location",
                "source": "site.md",
            },
        }])

        result = await bridge.retrieve("location")
        health = bridge.health()

        assert result == ""
        assert health["last_dropped_evidence"][0]["drop_reason"] == (
            "conflict:conflict:device:a:location"
        )
        assert health["last_answer_policy"]["required_operator_action"] == "resolve_conflict"

    @pytest.mark.asyncio
    async def test_fallback_filters_stale_catalog_version(self, tmp_path):
        bridge, vs = _make_bridge()
        bridge._knowledge_catalog = KnowledgeCatalog(path=tmp_path / "records.json")
        bridge._knowledge_catalog.upsert_payloads([{
            "record_id": "know_1",
            "text": "Restroom east",
            "memory_text": "[location] Restroom east",
            "approval_status": "published",
            "metadata": {"record_id": "know_1", "approval_status": "published"},
        }])
        bridge._knowledge_catalog.update_metadata(
            "know_1",
            {"expires_at": "2099-01-01T00:00:00+00:00"},
        )
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[{
            "text": "stale location",
            "score": 0.8,
            "metadata": {
                "record_id": "know_1",
                "approval_status": "published",
                "evidence_version": 1,
                "source": "site.md",
            },
        }])

        result = await bridge.retrieve("location")
        health = bridge.health()

        assert result == ""
        assert health["last_dropped_evidence"][0]["drop_reason"] == (
            "catalog_evidence_version:1->2"
        )
        assert health["last_answer_policy"]["state"] == "stale"
        assert health["last_answer_policy"]["required_operator_action"] == "refresh_knowledge"

    @pytest.mark.asyncio
    async def test_fallback_filters_catalog_deleted_record(self, tmp_path):
        bridge, vs = _make_bridge()
        bridge._knowledge_catalog = KnowledgeCatalog(path=tmp_path / "records.json")
        bridge._knowledge_catalog.upsert_payloads([{
            "record_id": "know_1",
            "text": "Restroom east",
            "memory_text": "[location] Restroom east",
            "approval_status": "published",
            "metadata": {"record_id": "know_1", "approval_status": "published"},
        }])
        bridge._knowledge_catalog.update_metadata("know_1", {"approval_status": "deleted"})
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[{
            "text": "deleted catalog location",
            "score": 0.8,
            "metadata": {
                "record_id": "know_1",
                "approval_status": "published",
                "evidence_version": 1,
                "source": "site.md",
            },
        }])

        result = await bridge.retrieve("location")
        health = bridge.health()

        assert result == ""
        assert health["last_dropped_evidence"][0]["drop_reason"] == "catalog_status:deleted"
        assert health["last_answer_policy"]["state"] == "unapproved"

    @pytest.mark.asyncio
    async def test_fallback_filters_expired_knowledge(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[
            {
                "text": "expired route",
                "score": 0.8,
                "metadata": {
                    "approval_status": "published",
                    "expires_at": "2000-01-01T00:00:00+00:00",
                    "source": "route.md",
                },
            },
        ])

        result = await bridge.retrieve("route")
        health = bridge.health()

        assert result == ""
        assert health["last_evidence"] == []
        assert health["last_dropped_evidence"][0]["drop_reason"] == "expired"
        assert health["last_answer_policy"]["required_operator_action"] == "refresh_knowledge"

    @pytest.mark.asyncio
    async def test_dropped_evidence_exposes_catalog_record_traceability(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[{
            "text": "expired route",
            "score": 0.8,
            "metadata": {
                "record_id": "route_old",
                "approval_status": "published",
                "expires_at": "2000-01-01T00:00:00+00:00",
                "source": "route.md",
            },
        }])

        await bridge.retrieve("route")
        dropped = bridge.health()["last_dropped_evidence"][0]

        assert dropped["record_id"] == "route_old"
        assert dropped["source_record_id"] == "route_old"

    @pytest.mark.asyncio
    async def test_fallback_keeps_published_unexpired_knowledge(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[
            {
                "text": "fresh route",
                "score": 0.8,
                "metadata": {
                    "approval_status": "published",
                    "expires_at": "2099-01-01T00:00:00+00:00",
                    "source": "route.md",
                    "category": "route",
                },
            },
        ])

        result = await bridge.retrieve("route")
        health = bridge.health()

        assert "fresh route" in result
        assert health["last_evidence"][0]["source"] == "route.md"
        assert health["last_evidence"][0]["category"] == "route"
        assert health["last_evidence"][0]["freshness_state"] == "fresh"
        assert health["last_answer_policy"]["state"] == "grounded"

    @pytest.mark.asyncio
    async def test_fallback_filters_conflicting_knowledge(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[
            {
                "text": "设备 A 在东门",
                "score": 0.9,
                "metadata": {
                    "approval_status": "published",
                    "entity_key": "equipment:a",
                    "fact_key": "location",
                    "value": "east_gate",
                    "source": "site-a.md",
                },
            },
            {
                "text": "设备 A 在西门",
                "score": 0.88,
                "metadata": {
                    "approval_status": "published",
                    "entity_key": "equipment:a",
                    "fact_key": "location",
                    "value": "west_gate",
                    "source": "site-b.md",
                },
            },
        ])

        result = await bridge.retrieve("设备 A 在哪里")
        health = bridge.health()

        assert result == ""
        assert health["last_evidence"] == []
        reasons = {item["drop_reason"] for item in health["last_dropped_evidence"]}
        assert reasons == {"conflict:equipment:a:location"}
        assert all(item["used_in_prompt"] is False for item in health["last_dropped_evidence"])
        assert health["last_answer_policy"]["state"] == "conflict"
        assert health["last_answer_policy"]["action"] == "clarify"

    @pytest.mark.asyncio
    async def test_fallback_keeps_consistent_duplicate_knowledge(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True
        vs.search = MagicMock(return_value=[
            {
                "text": "设备 A 在东门",
                "score": 0.9,
                "metadata": {
                    "approval_status": "published",
                    "entity_key": "equipment:a",
                    "fact_key": "location",
                    "value": "east_gate",
                    "source": "site-a.md",
                },
            },
            {
                "text": "设备 A 靠近东门",
                "score": 0.88,
                "metadata": {
                    "approval_status": "published",
                    "entity_key": "equipment:a",
                    "fact_key": "location",
                    "value": "east_gate",
                    "source": "site-b.md",
                },
            },
        ])

        result = await bridge.retrieve("设备 A 在哪里")
        health = bridge.health()

        assert "设备 A 在东门" in result
        assert "设备 A 靠近东门" in result
        assert health["last_dropped_evidence"] == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_both_unavailable(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = False

        result = await bridge.retrieve("test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_mem0_retrieve_exception_returns_empty(self):
        bridge, _ = _make_bridge()
        mock_mem0 = MagicMock()
        mock_mem0.search = MagicMock(side_effect=RuntimeError("network error"))
        bridge._mem0 = mock_mem0

        result = await bridge.retrieve("test")
        assert result == ""

    @pytest.mark.asyncio
    async def test_retrieve_uses_short_ttl_cache_for_repeated_queries(self):
        bridge, _ = _make_bridge()
        bridge._retrieve_cache_ttl_s = 30.0
        mock_mem0 = _make_mem0_mock(search_results={
            "results": [{"memory": "cached location"}],
        })
        bridge._mem0 = mock_mem0

        first = await bridge.retrieve("gate location")
        second = await bridge.retrieve(" gate   location ")
        health = bridge.health()

        assert first == second
        assert "cached location" in second
        mock_mem0.search.assert_called_once_with("gate location", user_id="robot")
        assert health["retrieve_count"] == 2
        assert health["retrieve_cache"]["hits"] == 1
        assert health["retrieve_cache"]["misses"] == 1
        assert health["retrieve_cache"]["last_hit"] is True

    @pytest.mark.asyncio
    async def test_retrieve_cache_can_be_disabled(self):
        bridge, _ = _make_bridge()
        bridge._retrieve_cache_ttl_s = 0.0
        mock_mem0 = _make_mem0_mock(search_results={
            "results": [{"memory": "uncached location"}],
        })
        bridge._mem0 = mock_mem0

        await bridge.retrieve("gate location")
        await bridge.retrieve("gate location")
        health = bridge.health()

        assert mock_mem0.search.call_count == 2
        assert health["retrieve_cache"]["enabled"] is False
        assert health["retrieve_cache"]["hits"] == 0

    @pytest.mark.asyncio
    async def test_retrieve_coalesces_same_query_in_flight_work(self):
        bridge, _ = _make_bridge()
        bridge._retrieve_cache_ttl_s = 30.0
        mock_mem0 = _make_mem0_mock(search_results={
            "results": [{"memory": "shared location"}],
        })

        def slow_search(*args, **kwargs):
            time.sleep(0.03)
            return {"results": [{"memory": "shared location"}]}

        mock_mem0.search = MagicMock(side_effect=slow_search)
        bridge._mem0 = mock_mem0

        first, second = await asyncio.gather(
            bridge.retrieve("same query"),
            bridge.retrieve("same query"),
        )
        health = bridge.health()

        assert first == second
        assert "shared location" in first
        assert mock_mem0.search.call_count == 1
        assert health["retrieve_count"] == 2
        assert health["retrieve_cache"]["coalesced"] >= 1

    @pytest.mark.asyncio
    async def test_retrieve_with_context_isolates_concurrent_turn_evidence(self):
        bridge, _ = _make_bridge()

        async def fake_retrieve(text: str) -> str:
            bridge._last_backend = "vector"
            bridge._last_evidence = [{"record_id": text, "text": f"fact:{text}"}]
            bridge._last_dropped_evidence = []
            await asyncio.sleep(0.01)
            return f"- fact:{text}"

        bridge._retrieve_with_fallbacks_unlocked = fake_retrieve

        first, second = await asyncio.gather(
            bridge.retrieve_with_context("turn-a"),
            bridge.retrieve_with_context("turn-b"),
        )

        assert first.context == "- fact:turn-a"
        assert first.evidence[0]["record_id"] == "turn-a"
        assert second.context == "- fact:turn-b"
        assert second.evidence[0]["record_id"] == "turn-b"
        assert first.rag["turn_scoped"] is True
        assert second.rag["turn_scoped"] is True


# ---------------------------------------------------------------------------
# Tests: save
# ---------------------------------------------------------------------------

class TestSave:
    @pytest.mark.asyncio
    async def test_save_disabled_noop(self):
        bridge, _ = _make_bridge(enabled=False)
        await bridge.save("user", "assistant")  # should not raise

    @pytest.mark.asyncio
    async def test_mem0_save(self):
        bridge, _ = _make_bridge()
        mock_mem0 = _make_mem0_mock()
        bridge._mem0 = mock_mem0

        await bridge.save("你好", "你好，有什么任务？")
        mock_mem0.add.assert_called_once()
        call_args = mock_mem0.add.call_args
        assert "你好" in call_args[0][0]
        assert call_args[1]["user_id"] == "robot"

    @pytest.mark.asyncio
    async def test_mem0_save_truncates_reply(self):
        bridge, _ = _make_bridge()
        mock_mem0 = _make_mem0_mock()
        bridge._mem0 = mock_mem0

        long_reply = "x" * 500
        await bridge.save("q", long_reply)
        call_text = mock_mem0.add.call_args[0][0]
        # Reply should be truncated to 200 chars
        assert len(call_text) < 250  # user line + truncated reply

    @pytest.mark.asyncio
    async def test_mem0_save_exception_swallowed(self):
        bridge, _ = _make_bridge()
        mock_mem0 = MagicMock()
        mock_mem0.add = MagicMock(side_effect=RuntimeError("save error"))
        bridge._mem0 = mock_mem0

        await bridge.save("user", "assistant")  # should not raise

    @pytest.mark.asyncio
    async def test_fallback_save_to_vectorstore(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True

        await bridge.save("user", "assistant")
        vs.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_noop_when_both_unavailable(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = False

        await bridge.save("user", "assistant")  # should not raise
        vs.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_fact_to_vectorstore(self):
        bridge, vs = _make_bridge()
        bridge._mem0 = None
        bridge._mem0_failed = True
        vs.available = True

        await bridge.save_fact("配电室在二楼", {"category": "location", "source": "site.md"})

        vs.add.assert_called_once()
        text, metadata = vs.add.call_args[0]
        assert text == "配电室在二楼"
        assert metadata["type"] == "knowledge"
        assert metadata["category"] == "location"
        vs.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_knowledge_reads_vector_catalog(self):
        bridge, vs = _make_bridge()
        vs.size = 1
        vs.list_records.return_value = [{
            "index": 0,
            "text": "site fact",
            "metadata": {
                "record_id": "know_1",
                "category": "location",
                "source": "site.md",
                "approval_status": "published",
            },
        }]

        payload = await bridge.list_knowledge(limit=20)

        assert payload["backend"] == "vector"
        assert payload["total"] == 1
        assert payload["records"][0]["record_id"] == "know_1"
        vs.list_records.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_knowledge_metadata_patches_and_saves(self):
        bridge, vs = _make_bridge()
        vs.update_metadata.return_value = True

        payload = await bridge.update_knowledge_metadata(
            "know_1",
            {"approval_status": "deleted", "ignored": True},
        )

        assert payload["updated"] is True
        assert payload["patch"] == {"approval_status": "deleted"}
        vs.update_metadata.assert_called_once_with("know_1", {"approval_status": "deleted"})
        vs.save.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: warmup
# ---------------------------------------------------------------------------

class TestWarmup:
    @pytest.mark.asyncio
    async def test_warmup_disabled_noop(self):
        bridge, _ = _make_bridge(enabled=False)
        await bridge.warmup()  # should not raise

    @pytest.mark.asyncio
    async def test_warmup_with_mem0(self):
        bridge, _ = _make_bridge()
        mock_mem0 = _make_mem0_mock()

        # Make _ensure_mem0 succeed and set mem0
        def side_effect():
            bridge._mem0 = mock_mem0
            return True

        with patch.object(bridge, "_ensure_mem0", side_effect=side_effect):
            await bridge.warmup()

    @pytest.mark.asyncio
    async def test_warmup_falls_back_to_vectorstore(self):
        bridge, vs = _make_bridge()
        vs.available = True
        bridge._mem0_failed = True

        with patch.object(bridge, "_ensure_mem0", return_value=False):
            await bridge.warmup()
            vs.search.assert_called_once_with("warmup", 1)


# ---------------------------------------------------------------------------
# Tests: properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_vector_store_property(self):
        bridge, vs = _make_bridge()
        assert bridge.vector_store is vs

    def test_import_existing_data_noop_when_disabled(self):
        bridge, _ = _make_bridge(enabled=False)
        assert bridge.import_existing_data() == 0

    def test_import_existing_data_noop_when_unavailable(self):
        bridge, vs = _make_bridge()
        vs.available = False
        assert bridge.import_existing_data() == 0
