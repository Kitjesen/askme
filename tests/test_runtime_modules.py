"""Tests for runtime module concrete classes: LLMModule, MemoryModule, HealthModule, LEDModule."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from askme.runtime.module import ModuleRegistry

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_registry() -> ModuleRegistry:
    return ModuleRegistry()


# ── LLMModule ─────────────────────────────────────────────────────────────────

class TestLLMModule:
    def _make_module(self, cfg=None):
        from askme.runtime.modules.llm_module import LLMModule
        mod = LLMModule()
        with patch("askme.runtime.modules.llm_module.LLMClient") as mock_cls, \
             patch("askme.runtime.modules.llm_module.OTABridgeMetrics"), \
             patch("askme.runtime.modules.llm_module.LLMConfig.from_cfg") as mock_cfg, \
             patch("askme.runtime.modules.llm_module.LLMConfig.validate_and_warn"):
            mock_client = MagicMock()
            mock_client.model = "test-model"
            mock_cls.return_value = mock_client
            mock_cfg.return_value = MagicMock()
            mock_cfg.return_value.validate_and_warn = MagicMock()
            mod.build(cfg or {}, _make_registry())
        return mod

    def test_build_creates_client(self):
        from askme.runtime.modules.llm_module import LLMModule
        mod = LLMModule()
        with patch("askme.runtime.modules.llm_module.LLMClient") as mock_cls, \
             patch("askme.runtime.modules.llm_module.OTABridgeMetrics"), \
             patch("askme.runtime.modules.llm_module.LLMConfig") as mock_cfg_cls:
            mock_client = MagicMock()
            mock_client.model = "model"
            mock_cls.return_value = mock_client
            mock_llm_cfg = MagicMock()
            mock_cfg_cls.from_cfg.return_value = mock_llm_cfg
            mod.build({}, _make_registry())
        assert mod.client is not None

    def test_health_returns_ok(self):
        mod = self._make_module()
        h = mod.health()
        assert h["status"] == "ok"
        assert "model" in h

    def test_llm_client_property(self):
        mod = self._make_module()
        assert mod.llm_client is mod.client

    @pytest.mark.asyncio
    async def test_stop_cancels_warmup_task(self):
        mod = self._make_module()
        # Create a fake "running" task
        async def _long_task():
            await asyncio.sleep(100)

        task = asyncio.create_task(_long_task())
        mod._warmup_task = task
        await mod.stop()
        # Give the event loop a tick to process cancellation
        await asyncio.sleep(0)
        assert task.cancelled() or task.cancelling() > 0


# ── MemoryModule ──────────────────────────────────────────────────────────────

class TestMemoryModule:
    def _make_module(self):
        from askme.runtime.modules.memory_module import MemoryModule
        mod = MemoryModule()
        # Patch all the heavy memory classes
        with patch("askme.runtime.modules.memory_module.SessionMemory"), \
             patch("askme.runtime.modules.memory_module.ConversationManager") as mock_conv, \
             patch("askme.runtime.modules.memory_module.MemoryBridge"), \
             patch("askme.runtime.modules.memory_module.EpisodicMemory") as mock_epi, \
             patch("askme.runtime.modules.memory_module.MemorySystem"):
            mock_conv_inst = MagicMock()
            mock_conv_inst.history = []
            mock_conv.return_value = mock_conv_inst
            mock_epi_inst = MagicMock()
            mock_epi_inst._buffer = []
            mock_epi.return_value = mock_epi_inst
            mod.llm_client = None  # no LLMModule wired
            mod.build({}, _make_registry())
        return mod

    def _make_module_with_catalog(self, tmp_path):
        from askme.memory.catalog import KnowledgeCatalog
        from askme.memory.index_jobs import KnowledgeIndexJobStore

        mod = self._make_module()
        mod._knowledge_catalog = KnowledgeCatalog(path=tmp_path / "records.json")
        mod._knowledge_job_store = KnowledgeIndexJobStore(path=tmp_path / "index_jobs.json")
        return mod

    def test_build_creates_memory_components(self):
        mod = self._make_module()
        assert mod.conversation is not None
        assert mod.session_memory is not None
        assert mod.episodic is not None
        assert mod.memory_bridge is not None

    def test_build_passes_runtime_config_to_memory_bridge(self):
        from askme.runtime.modules.memory_module import MemoryModule

        cfg = {"memory": {"enabled": False, "backend": "vector"}}
        mod = MemoryModule()
        with patch("askme.runtime.modules.memory_module.SessionMemory"), \
             patch("askme.runtime.modules.memory_module.ConversationManager") as mock_conv, \
             patch("askme.runtime.modules.memory_module.MemoryBridge") as mock_bridge, \
             patch("askme.runtime.modules.memory_module.EpisodicMemory") as mock_epi, \
             patch("askme.runtime.modules.memory_module.MemorySystem"):
            mock_conv.return_value.history = []
            mock_epi.return_value._buffer = []
            mod.llm_client = None
            mod.build(cfg, _make_registry())

        assert mock_bridge.call_count == 1
        _, kwargs = mock_bridge.call_args
        assert kwargs["config"] == cfg
        assert kwargs["knowledge_catalog"] is mod._knowledge_catalog

    def test_health_returns_ok(self):
        mod = self._make_module()
        h = mod.health()
        assert h["status"] == "ok"
        assert "conversation_len" in h

    @pytest.mark.asyncio
    async def test_knowledge_preview_payload_parses_inline_content(self):
        mod = self._make_module()

        payload = await mod.preview_payload({
            "filename": "site.md",
            "content": "# Floor 1\n- Restroom east",
            "category": "location",
        })

        assert payload["parsed"] == 1
        assert payload["records"][0]["category"] == "location"
        assert payload["records"][0]["text"] == "Floor 1: Restroom east"

    @pytest.mark.asyncio
    async def test_knowledge_preview_payload_exposes_product_taxonomy(self):
        mod = self._make_module()

        payload = await mod.preview_payload({
            "filename": "merchant.md",
            "content": "- 梵木咖啡在 2 号楼一层",
            "category": "merchant",
            "owner": "交付工程师",
        })

        assert payload["records"][0]["category"] == "merchant"
        assert payload["records"][0]["category_label"] == "商户与服务"
        assert payload["records"][0]["owner"] == "交付工程师"
        assert payload["category_taxonomy"]["schema_version"] == "askme.knowledge_taxonomy.v1"
        assert {
            item["id"]
            for item in payload["category_taxonomy"]["categories"]
        } >= {"route", "merchant", "incident", "safety", "sensor", "contact"}

    @pytest.mark.asyncio
    async def test_knowledge_preview_payload_applies_governance_metadata(self):
        mod = self._make_module()

        payload = await mod.preview_payload({
            "filename": "fanmu-routes.csv",
            "content": "text,category\nCoffee is in building 2,merchant\n",
            "category": "merchant",
            "quality_status": "internal",
            "visibility": "internal",
            "customer_id": "fanmu",
            "project_id": "fanmu-phase-1",
            "product_area": "space",
            "workstream": "wayfinding",
            "linked_object_type": "park_point",
            "linked_object_id": "poi-fanmu-coffee",
        })

        record = payload["records"][0]
        assert payload["document_profile"]["document_type"] == "csv"
        assert record["quality_status"] == "internal"
        assert record["visibility"] == "internal"
        assert record["customer_id"] == "fanmu"
        assert record["project_id"] == "fanmu-phase-1"
        assert record["product_area"] == "space"
        assert record["workstream"] == "wayfinding"
        assert record["linked_object_type"] == "park_point"
        assert record["linked_object_id"] == "poi-fanmu-coffee"

    @pytest.mark.asyncio
    async def test_knowledge_preview_payload_rejects_unsupported_binary_documents(self):
        mod = self._make_module()

        payload = await mod.preview_payload({
            "filename": "site.pdf",
            "content": "%PDF-1.4",
            "category": "route",
        })

        assert payload["parsed"] == 0
        assert payload["document_profile"]["supported"] is False
        assert payload["errors"] == ["unsupported_file_type:.pdf"]

    @pytest.mark.asyncio
    async def test_knowledge_import_payload_returns_document_profile_on_rejection(self):
        mod = self._make_module()

        payload = await mod.import_payload({
            "filename": "site.pdf",
            "content": "%PDF-1.4",
            "category": "route",
        })

        assert payload["imported"] == 0
        assert payload["document_profile"]["supported"] is False
        assert payload["document_profile"]["reason"] == "unsupported_file_type:.pdf"

    @pytest.mark.asyncio
    async def test_knowledge_import_payload_saves_preview_records(self):
        mod = self._make_module()
        mod.memory_bridge.save_fact = AsyncMock()
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": False})
        mod.memory_bridge.health.return_value = {"backend": "vector"}

        payload = await mod.import_payload({
            "filename": "site.md",
            "content": "- Restroom east",
            "category": "location",
        })

        assert payload["imported"] == 1
        mod.memory_bridge.save_fact.assert_awaited_once()
        text, metadata = mod.memory_bridge.save_fact.await_args.args
        assert text == "[location] Restroom east"
        assert metadata["category"] == "location"
        assert metadata["record_id"].startswith("know_")

    @pytest.mark.asyncio
    async def test_knowledge_import_payload_catalogs_internal_records_without_public_indexing(
        self,
        tmp_path,
    ):
        mod = self._make_module_with_catalog(tmp_path)
        mod.memory_bridge.save_fact = AsyncMock()
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": True})
        mod.memory_bridge.health.return_value = {"backend": "vector"}

        payload = await mod.import_payload({
            "filename": "internal-sop.md",
            "content": "- Staff-only shutdown note",
            "category": "inspection",
            "quality_status": "internal",
            "visibility": "internal",
            "customer_id": "fanmu",
            "project_id": "fanmu-phase-1",
        })

        assert payload["cataloged"] == 1
        assert payload["indexed"] == 0
        mod.memory_bridge.save_fact.assert_not_awaited()
        listed = await mod.list_knowledge_payload({"limit": 50})
        record = listed["records"][0]
        assert record["lifecycle_state"] == "internal_only"
        assert record["prompt_eligible"] is False
        assert listed["catalog"]["by_visibility"]["internal"] == 1
        assert listed["catalog"]["by_customer"]["fanmu"] == 1

    @pytest.mark.asyncio
    async def test_knowledge_import_payload_upserts_catalog_before_syncing_eligible_records(
        self,
        tmp_path,
    ):
        mod = self._make_module_with_catalog(tmp_path)
        mod.memory_bridge.save_fact = AsyncMock()
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": True})
        mod.memory_bridge.health.return_value = {"backend": "vector"}

        payload = await mod.import_payload({
            "filename": "site.json",
            "content": '[{"record_id":"know_1","text":"Restroom east","category":"location"}]',
        })

        assert payload["imported"] == 1
        assert payload["catalog"]["total"] == 1
        mod.memory_bridge.save_fact.assert_awaited_once()
        text, metadata = mod.memory_bridge.save_fact.await_args.args
        assert text == "[location] Restroom east"
        assert metadata["record_id"] == "know_1"
        assert metadata["evidence_version"] == 1
        assert metadata["source_version"] == "1"

    @pytest.mark.asyncio
    async def test_knowledge_import_payload_does_not_sync_conflicted_records(self, tmp_path):
        mod = self._make_module_with_catalog(tmp_path)
        mod.memory_bridge.save_fact = AsyncMock()
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": True})
        mod.memory_bridge.health.return_value = {"backend": "vector"}

        payload = await mod.import_payload({
            "filename": "site.json",
            "content": (
                "["
                '{"record_id":"know_a","text":"Device A east","entity_key":"device:a",'
                '"fact_key":"location","value":"east","approval_status":"published"},'
                '{"record_id":"know_b","text":"Device A west","entity_key":"device:a",'
                '"fact_key":"location","value":"west","approval_status":"published"}'
                "]"
            ),
        })

        assert payload["imported"] == 0
        assert payload["skipped"] == 2
        mod.memory_bridge.save_fact.assert_not_awaited()
        listed = mod._knowledge_catalog.list_records()["records"]
        assert {record["conflict_set_id"] for record in listed} == {
            "conflict:device:a:location"
        }

    @pytest.mark.asyncio
    async def test_knowledge_import_payload_catalogs_when_memory_backend_disabled(self, tmp_path):
        mod = self._make_module_with_catalog(tmp_path)
        mod.memory_bridge.save_fact = AsyncMock()
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={
            "updated": False,
            "error": "memory_disabled",
        })
        mod.memory_bridge.health.return_value = {"enabled": False, "backend": "disabled"}

        payload = await mod.import_payload({
            "filename": "site.json",
            "content": '[{"record_id":"know_1","text":"3号楼在主通道尽头","category":"route"}]',
        })

        assert payload["cataloged"] == 1
        assert payload["imported"] == 0
        assert payload["indexed"] == 0
        assert payload["skipped"] == 1
        mod.memory_bridge.save_fact.assert_not_awaited()
        listed = await mod.list_knowledge_payload({"limit": 50})
        assert listed["records"][0]["record_id"] == "know_1"

    @pytest.mark.asyncio
    async def test_knowledge_list_payload_returns_catalog(self, tmp_path):
        mod = self._make_module_with_catalog(tmp_path)
        mod._knowledge_catalog.upsert_payloads([{
            "record_id": "know_1",
            "text": "Restroom east",
            "memory_text": "[location] Restroom east",
            "approval_status": "published",
        }])
        mod.memory_bridge.health.return_value = {"backend": "vector"}

        payload = await mod.list_knowledge_payload({"limit": 50})

        assert payload["records"][0]["record_id"] == "know_1"
        assert payload["total"] == 1
        assert payload["backend"] == "catalog"
        assert payload["catalog"]["prompt_eligible"] == 1
        assert payload["catalog"]["needs_reindex"] == 1
        assert payload["records"][0]["lifecycle_state"] == "needs_reindex"
        assert payload["records"][0]["lifecycle_label"] == "需重建索引"
        assert payload["category_taxonomy"]["default_category"] == "faq"

    @pytest.mark.asyncio
    async def test_memory_search_payload_exposes_answer_policy(self):
        mod = self._make_module()
        mod.memory_bridge.retrieve = AsyncMock(return_value="")
        mod.memory_bridge.health.return_value = {
            "enabled": True,
            "backend": "vector",
            "last_backend": "vector",
            "last_evidence": [],
            "last_dropped_evidence": [{"drop_reason": "expired", "text": "old"}],
            "last_answer_policy": {
                "state": "stale",
                "action": "refuse_and_request_update",
            },
        }

        payload = await mod.search_payload({"query": "route"})

        assert payload["results"] == []
        assert payload["rag"]["answer_policy"]["state"] == "stale"

    @pytest.mark.asyncio
    async def test_memory_search_payload_falls_back_to_catalog_records(self, tmp_path):
        mod = self._make_module_with_catalog(tmp_path)
        mod._knowledge_catalog.upsert_payloads([{
            "record_id": "know_route_3",
            "text": "3号楼在主通道尽头左转80米",
            "memory_text": "[route] 3号楼在主通道尽头左转80米",
            "category": "route",
            "approval_status": "published",
        }])
        mod.memory_bridge.retrieve = AsyncMock(return_value="")
        mod.memory_bridge.health.return_value = {
            "enabled": True,
            "backend": "vector",
            "last_backend": "vector",
            "last_evidence": [],
            "last_dropped_evidence": [],
            "last_answer_policy": {},
        }

        payload = await mod.search_payload({"query": "3号楼怎么走"})

        assert payload["results"][0]["record_id"] == "know_route_3"
        assert payload["results"][0]["match_reason"] == "catalog_keyword_fallback"
        assert payload["rag"]["last_backend"] == "catalog"

    @pytest.mark.asyncio
    async def test_memory_search_payload_catalog_fallback_exposes_expired_policy(
        self,
        tmp_path,
    ):
        mod = self._make_module_with_catalog(tmp_path)
        mod._knowledge_catalog.upsert_payloads([
            {
                "record_id": "know_coffee",
                "text": "Fanmu coffee is on the first floor of Building 2.",
                "memory_text": "[route] Fanmu coffee is on the first floor of Building 2.",
                "category": "route",
                "approval_status": "published",
            },
            {
                "record_id": "know_old_hall",
                "text": "Old Hall route starts at the north gate.",
                "memory_text": "[route] Old Hall route starts at the north gate.",
                "category": "route",
                "approval_status": "published",
                "expires_at": "2000-01-01T00:00:00+00:00",
            },
        ])
        mod.memory_bridge.retrieve = AsyncMock(return_value="")
        mod.memory_bridge.health.return_value = {
            "enabled": True,
            "backend": "vector",
            "last_backend": "vector",
            "last_evidence": [],
            "last_dropped_evidence": [],
            "last_answer_policy": {
                "state": "no_evidence",
                "action": "clarify_or_refuse",
            },
        }

        payload = await mod.search_payload({"query": "Old Hall route"})

        assert payload["results"] == []
        assert payload["rag"]["dropped_evidence"][0]["record_id"] == "know_old_hall"
        assert payload["rag"]["dropped_evidence"][0]["drop_reason"] == "expired"
        assert payload["rag"]["answer_policy"]["state"] == "stale"
        assert payload["rag"]["answer_policy"]["action"] == "refuse_and_request_update"

    @pytest.mark.asyncio
    async def test_memory_health_payload_separates_customer_knowledge_from_robot_behavior(
        self,
        tmp_path,
    ):
        mod = self._make_module_with_catalog(tmp_path)
        mod._memory_cfg = {
            "backend": "vector",
            "customer_knowledge_backend": "vector",
            "robot_behavior_memory_backend": "robotmem",
            "robot_behavior_memory_enabled": False,
        }
        mod._knowledge_catalog.upsert_payloads([{
            "record_id": "know_coffee",
            "text": "梵木咖啡在2号楼一层",
            "memory_text": "[route] 梵木咖啡在2号楼一层",
            "category": "route",
            "approval_status": "published",
        }])
        mod.memory_bridge.health.return_value = {
            "enabled": True,
            "available": True,
            "backend": "vector",
            "configured_backend": "vector",
            "last_backend": "vector",
            "selected_backend_ready": True,
            "selected_backend_installed": True,
            "fallback_backend": "",
            "fallback_ready": False,
            "selected_backend_dependency": {
                "backend": "vector",
                "installed": True,
                "version": "5.2.3",
            },
            "fallback_backend_dependency": {},
            "backend_dependencies": {
                "vector": {
                    "backend": "vector",
                    "installed": True,
                    "version": "5.2.3",
                },
                "mempalace": {
                    "backend": "mempalace",
                    "installed": True,
                    "version": "3.3.5",
                },
                "robotmem": {
                    "backend": "robotmem",
                    "installed": False,
                    "version": "",
                },
            },
            "robotmem_ready": False,
            "vector_store_path": str(tmp_path / "store.json"),
            "vector_size": 1,
            "rag_enforce_expiry": True,
        }

        payload = await mod.health_payload({})

        assert payload["status"] == "ready"
        assert payload["ready"] is True
        assert payload["customer_status"] == "客户知识库可用于有证据回答。"
        assert payload["customer_next_step"] == "继续维护已发布知识，并在回答气泡展示引用证据。"
        assert payload["current_backend"] == "vector"
        assert payload["selected_backend_dependency"]["version"] == "5.2.3"
        assert payload["backend_dependencies"]["mempalace"]["version"] == "3.3.5"
        assert payload["memory_strategy"]["customer_knowledge"]["backend"] == "vector"
        assert payload["memory_strategy"]["customer_knowledge"]["enters_prompt"] is True
        robot_memory = payload["memory_strategy"]["robot_behavior_memory"]
        assert robot_memory["backend"] == "robotmem"
        assert robot_memory["enabled"] is False
        assert robot_memory["enters_prompt"] is False
        assert payload["paths"]["catalog"].endswith("records.json")
        assert payload["counts"]["catalog_total"] == 1
        assert payload["counts"]["prompt_eligible"] == 1
        assert payload["answer_contract"] == {
            "contract_type": "askme.customer_knowledge_answer_contract.v1",
            "evidence_required": True,
            "approved_knowledge_only": True,
            "current_knowledge_only": True,
            "conflict_free_knowledge_only": True,
            "show_evidence_in_answer": True,
            "refuse_when_no_evidence": True,
            "refuse_when_expired": True,
            "refuse_when_conflicting": True,
            "robot_behavior_memory_enters_customer_prompt": False,
        }

    @pytest.mark.asyncio
    async def test_memory_health_payload_warns_when_expiry_is_not_enforced(self, tmp_path):
        mod = self._make_module_with_catalog(tmp_path)
        mod.memory_bridge.health.return_value = {
            "enabled": True,
            "available": True,
            "backend": "vector",
            "configured_backend": "vector",
            "last_backend": "vector",
            "selected_backend_ready": True,
            "selected_backend_installed": True,
            "fallback_backend": "",
            "fallback_ready": False,
            "robotmem_ready": False,
            "rag_enforce_expiry": False,
        }

        payload = await mod.health_payload({})

        assert "rag_expiry_not_enforced" in payload["warnings"]
        assert payload["customer_status"] == "知识过期拦截未启用，不能作为客户回答依据。"
        assert payload["customer_next_step"] == "先启用知识过期拦截，再允许知识进入回答。"
        assert payload["answer_contract"]["current_knowledge_only"] is False
        assert payload["answer_contract"]["refuse_when_expired"] is False

    @pytest.mark.asyncio
    async def test_memory_health_payload_reports_catalog_only_when_runtime_disabled(
        self,
        tmp_path,
    ):
        mod = self._make_module_with_catalog(tmp_path)
        mod._memory_cfg = {
            "enabled": False,
            "backend": "mempalace",
            "customer_knowledge_backend": "mempalace",
            "robot_behavior_memory_backend": "robotmem",
            "robot_behavior_memory_enabled": False,
        }
        mod._knowledge_catalog.upsert_payloads([{
            "record_id": "know_restroom",
            "text": "Restroom is east of the main hall.",
            "memory_text": "[location] Restroom is east of the main hall.",
            "category": "location",
            "approval_status": "published",
            "visibility": "external",
        }])
        mod.memory_bridge.health.return_value = {
            "enabled": False,
            "available": False,
            "backend": "mempalace",
            "configured_backend": "mempalace",
            "last_backend": "mempalace",
            "selected_backend_ready": False,
            "selected_backend_installed": True,
            "fallback_backend": "vector",
            "fallback_ready": True,
            "selected_backend_dependency": {
                "backend": "mempalace",
                "installed": True,
                "version": "3.3.5",
            },
            "fallback_backend_dependency": {
                "backend": "vector",
                "installed": True,
                "version": "5.2.3",
            },
            "backend_dependencies": {},
            "robotmem_ready": False,
            "vector_store_path": str(tmp_path / "store.json"),
            "vector_size": 0,
            "rag_enforce_expiry": True,
        }

        payload = await mod.health_payload({})

        assert payload["status"] == "catalog_only"
        assert payload["ready"] is True
        assert payload["catalog_answer_ready"] is True
        assert payload["retrieval_runtime_ready"] is False
        assert payload["counts"]["prompt_eligible"] == 1
        assert "memory_runtime_disabled_catalog_only" in payload["warnings"]
        assert "customer_knowledge_catalog_only" in payload["warnings"]
        assert "memory_backend_not_ready" not in payload["warnings"]

    @pytest.mark.asyncio
    async def test_knowledge_update_payload_soft_deletes_record(self, tmp_path):
        mod = self._make_module_with_catalog(tmp_path)
        mod._knowledge_catalog.upsert_payloads([{
            "record_id": "know_1",
            "text": "Restroom east",
            "memory_text": "[location] Restroom east",
            "approval_status": "published",
        }])
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={
            "updated": True,
            "record_id": "know_1",
            "patch": {"approval_status": "deleted"},
        })
        mod.memory_bridge.health.return_value = {"backend": "vector"}

        payload = await mod.update_knowledge_payload({
            "record_id": "know_1",
            "action": "delete",
        })

        assert payload["updated"] is True
        assert payload["record"]["approval_status"] == "deleted"
        _, patch_arg = mod.memory_bridge.update_knowledge_metadata.await_args.args
        assert patch_arg["approval_status"] == "deleted"
        assert "deleted_at" in patch_arg

    @pytest.mark.asyncio
    async def test_knowledge_update_payload_bulk_patches_records(self, tmp_path):
        mod = self._make_module_with_catalog(tmp_path)
        mod._knowledge_catalog.upsert_payloads([
            {
                "record_id": "know_a",
                "text": "Restroom east",
                "memory_text": "[location] Restroom east",
                "approval_status": "published",
            },
            {
                "record_id": "know_b",
                "text": "Cafe west",
                "memory_text": "[location] Cafe west",
                "approval_status": "published",
            },
        ])
        mod.memory_bridge.save_fact = AsyncMock()
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": True})
        mod.memory_bridge.health.return_value = {"backend": "vector"}

        payload = await mod.update_knowledge_payload({
            "action": "bulk_update",
            "record_ids": ["know_a", "know_b"],
            "patch": {"owner": "ops"},
        })

        assert payload["updated"] == 2
        assert payload["failed"] == 0
        assert payload["sync"]["indexed"] == 2
        records = {record["record_id"]: record for record in payload["records"]}
        assert records["know_a"]["owner"] == "ops"
        assert mod.memory_bridge.save_fact.await_count == 2
        assert mod.memory_bridge.update_knowledge_metadata.await_count == 2

    @pytest.mark.asyncio
    async def test_knowledge_resolve_conflict_keeps_selected_record(self, tmp_path):
        mod = self._make_module_with_catalog(tmp_path)
        mod._knowledge_catalog.upsert_payloads([
            {
                "record_id": "know_a",
                "text": "Device A east",
                "memory_text": "[equipment] Device A east",
                "approval_status": "published",
                "entity_key": "device:a",
                "fact_key": "location",
                "value": "east",
            },
            {
                "record_id": "know_b",
                "text": "Device A west",
                "memory_text": "[equipment] Device A west",
                "approval_status": "published",
                "entity_key": "device:a",
                "fact_key": "location",
                "value": "west",
            },
        ])
        mod.memory_bridge.save_fact = AsyncMock()
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": True})
        mod.memory_bridge.health.return_value = {"backend": "vector"}

        payload = await mod.update_knowledge_payload({
            "action": "resolve_conflict",
            "keep_record_id": "know_a",
            "operator_id": "ops.lead",
            "review_note": "site verified east",
        })

        records = {
            record["record_id"]: record
            for record in mod._knowledge_catalog.list_records()["records"]
        }
        assert payload["action"] == "resolve_conflict"
        assert payload["keep_record_id"] == "know_a"
        assert payload["rejected_record_ids"] == ["know_b"]
        assert records["know_a"]["conflict_set_id"] == ""
        assert records["know_a"]["approval_status"] == "published"
        assert records["know_a"]["approved_by"] == "ops.lead"
        assert records["know_b"]["approval_status"] == "rejected"
        assert records["know_b"]["rejected_by"] == "ops.lead"
        assert payload["sync"]["indexed"] == 1

    @pytest.mark.asyncio
    async def test_rebuild_knowledge_index_payload_indexes_catalog_candidates(self, tmp_path):
        mod = self._make_module_with_catalog(tmp_path)
        mod._knowledge_catalog.upsert_payloads([
            {
                "record_id": "know_a",
                "text": "Restroom east",
                "memory_text": "[location] Restroom east",
                "approval_status": "published",
            },
            {
                "record_id": "know_b",
                "text": "Draft note",
                "memory_text": "[location] Draft note",
                "approval_status": "draft",
            },
        ])
        mod.memory_bridge.save_fact = AsyncMock()
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": True})
        mod.memory_bridge.health.return_value = {"backend": "vector"}

        payload = await mod.update_knowledge_payload({"action": "rebuild_index"})

        assert payload["job"]["type"] == "knowledge_rebuild_index"
        assert payload["job"]["status"] == "completed"
        assert payload["job"]["job_id"].startswith("knowledge_rebuild_")
        assert payload["scanned"] == 2
        assert payload["eligible"] == 1
        assert payload["indexed"] == 1
        assert payload["record_ids"] == ["know_a"]
        assert payload["index_jobs"][0]["job_id"] == payload["job"]["job_id"]
        assert payload["index_jobs"][0]["indexed"] == 1
        mod.memory_bridge.save_fact.assert_awaited_once()
        text, metadata = mod.memory_bridge.save_fact.await_args.args
        assert text == "[location] Restroom east"
        assert metadata["record_id"] == "know_a"

    @pytest.mark.asyncio
    async def test_rebuild_knowledge_index_job_history_persists(self, tmp_path):
        from askme.memory.index_jobs import KnowledgeIndexJobStore

        mod = self._make_module_with_catalog(tmp_path)
        mod._knowledge_catalog.upsert_payloads([{
            "record_id": "know_a",
            "text": "Gate A is beside the fountain",
            "memory_text": "[location] Gate A is beside the fountain",
            "approval_status": "published",
        }])
        mod.memory_bridge.save_fact = AsyncMock()
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": True})
        mod.memory_bridge.health.return_value = {"backend": "vector", "last_backend": "vector"}

        payload = await mod.update_knowledge_payload({
            "action": "rebuild_index",
            "operator_id": "ops.lead",
        })

        restarted = KnowledgeIndexJobStore(path=tmp_path / "index_jobs.json")
        jobs = restarted.list_jobs(limit=5)
        assert jobs[0]["job_id"] == payload["job"]["job_id"]
        assert jobs[0]["operator_id"] == "ops.lead"
        assert jobs[0]["status"] == "completed"
        assert jobs[0]["record_ids"] == ["know_a"]

    @pytest.mark.asyncio
    async def test_knowledge_list_payload_exposes_recent_index_jobs(self, tmp_path):
        mod = self._make_module_with_catalog(tmp_path)
        mod._knowledge_catalog.upsert_payloads([{
            "record_id": "know_a",
            "text": "Service desk is on floor one",
            "memory_text": "[location] Service desk is on floor one",
            "approval_status": "published",
        }])
        mod.memory_bridge.save_fact = AsyncMock()
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": True})
        mod.memory_bridge.health.return_value = {"backend": "vector"}

        await mod.update_knowledge_payload({"action": "rebuild_index"})
        payload = await mod.list_knowledge_payload({"limit": 50})

        assert payload["index_jobs"][0]["type"] == "knowledge_rebuild_index"
        assert payload["index_jobs"][0]["indexed"] == 1
        assert payload["operations"]["release_cadence"]["mode"] == "manual"
        assert payload["operations"]["release_cadence"]["next_release_window"]
        assert "scheduled_release_automation" in payload["operations"]["missing_product_capabilities"]

    @pytest.mark.asyncio
    async def test_knowledge_diff_and_rollback_restore_prior_answer_text(self, tmp_path):
        mod = self._make_module_with_catalog(tmp_path)
        mod._knowledge_catalog.upsert_payloads([{
            "record_id": "know_a",
            "text": "Gate A is east",
            "memory_text": "[location] Gate A is east",
            "approval_status": "published",
        }])
        mod.memory_bridge.save_fact = AsyncMock()
        mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": True})
        mod.memory_bridge.health.return_value = {"backend": "vector"}

        await mod.update_knowledge_payload({
            "record_id": "know_a",
            "action": "publish",
            "patch": {"text": "Gate A is west", "memory_text": "[location] Gate A is west"},
            "operator_id": "ops.lead",
        })
        diff = await mod.update_knowledge_payload({"record_id": "know_a", "action": "diff"})
        rollback = await mod.update_knowledge_payload({
            "record_id": "know_a",
            "action": "rollback",
            "operator_id": "ops.lead",
        })

        assert diff["found"] is True
        assert any(change["field"] == "text" for change in diff["changes"])
        assert rollback["updated"] is True
        assert rollback["record"]["text"] == "Gate A is east"
        assert rollback["action"] == "rollback"

    @pytest.mark.asyncio
    async def test_start_launches_memory_warmup_task(self):
        mod = self._make_module()
        mod.memory_bridge.warmup = AsyncMock()

        await mod.start()
        await asyncio.gather(mod._warmup_task)

        mod.memory_bridge.warmup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_no_llm_no_crash(self):
        mod = self._make_module()
        await mod.stop()  # should not raise


# ── HealthModule ──────────────────────────────────────────────────────────────

class TestMissionModule:
    def _make_module(self):
        from askme.runtime.modules.mission_module import MissionModule

        mod = MissionModule()
        mod.build({}, _make_registry())
        return mod

    def test_build_creates_service(self):
        mod = self._make_module()
        assert mod.mission_service is mod.service

    def test_health_returns_ok(self):
        mod = self._make_module()
        h = mod.health()
        assert h["status"] == "ok"
        assert h["submit_enabled"] is False

    def test_capabilities_expose_http_paths(self):
        mod = self._make_module()
        capabilities = mod.capabilities()
        assert capabilities["dry_run_default"] is True
        assert "POST /api/missions/draft" in capabilities["http_paths"]


class TestHealthModule:
    def _make_module(self):
        from askme.runtime.modules.health_module import HealthModule
        mod = HealthModule()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server), \
             patch("askme.runtime.modules.health_module.AskmeHealthServer",
                   return_value=mock_server, create=True):
            mod.build({}, _make_registry())
        return mod

    def test_build_creates_server(self):
        mod = self._make_module()
        assert mod.server is not None

    def test_health_returns_ok(self):
        mod = self._make_module()
        h = mod.health()
        assert h["status"] == "ok"
        assert h["port"] == 8080

    def test_runtime_health_provider_reports_ok_when_children_ok(self):
        from askme.runtime.modules.health_module import HealthModule

        class HealthyModule:
            name = "text"

            def health(self):
                return {"status": "ok", "ready": True}

        registry = _make_registry()
        registry.register(HealthyModule())

        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server) as server_cls:
            HealthModule().build({}, registry)

        provider = server_cls.call_args.kwargs["snapshot_provider"]
        snapshot = provider()

        assert snapshot["status"] == "ok"
        assert snapshot["service"] == "askme"
        assert snapshot["text"] == {"status": "ok", "ready": True}
        assert snapshot["voice_pipeline_status"]["pipeline_ok"] is True

    def test_runtime_health_provider_exposes_audio_input_snapshot(self):
        from askme.runtime.modules.health_module import HealthModule

        class TextModule:
            name = "text"

            def __init__(self):
                self.text_loop = MagicMock()
                self.text_loop._audio.status_snapshot.return_value = {
                    "mode": "text",
                    "enabled": False,
                    "output_ready": True,
                    "pipeline_ok": True,
                    "input": {
                        "run_id": "run-1",
                        "last_peak": 123,
                        "gate_state": "noise",
                    },
                }

            def health(self):
                return {"status": "ok"}

        registry = _make_registry()
        registry.register(TextModule())

        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server) as server_cls:
            HealthModule().build({}, registry)

        provider = server_cls.call_args.kwargs["snapshot_provider"]
        snapshot = provider()

        assert snapshot["voice_pipeline_status"]["input"]["run_id"] == "run-1"
        assert snapshot["voice_pipeline_status"]["input"]["last_peak"] == 123
        assert snapshot["voice_pipeline_status"]["input"]["gate_state"] == "noise"

    def test_runtime_health_provider_exposes_model_routing_and_skill_callability(self):
        from askme.runtime.modules.health_module import HealthModule

        class SkillModule:
            name = "skill"

            def __init__(self):
                self.skill_manager = MagicMock()
                self.skill_manager.get_enabled.return_value = [
                    MagicMock(name="skill_obj", name_attr=""),
                ]
                self.skill_manager.get_enabled.return_value[0].name = "get_time"
                self.skill_manager.get_agent_shell_skills.return_value = {"agent_task"}

            def health(self):
                return {"status": "ok"}

        class PipelineModule:
            name = "pipeline"

            def health(self):
                return {"status": "ok"}

        class ExecutorModule:
            name = "executor"

            def __init__(self):
                self.shell = MagicMock()
                self.shell._model = "MiniMax-M2.7-highspeed"
                self.shell._default_timeout = 120.0
                self.shell._iteration_limit = 5
                self.shell._profile.name = "field_operator"
                self.shell.deprecated_replacement = "ZeroClaw MCP Agent"

            def health(self):
                return {"status": "ok"}

        class VoiceModule:
            name = "voice"

            def __init__(self):
                self.audio = MagicMock()
                self.audio.status_snapshot.return_value = {
                    "mode": "voice",
                    "enabled": True,
                    "output_ready": True,
                    "pipeline_ok": True,
                    "tts_backend": "minimax",
                    "asr": {
                        "provider": "cloud+local",
                        "cloud": {"model": "paraformer-realtime-v2"},
                    },
                    "tts": {
                        "minimax": {
                            "model": "speech-2.8-turbo",
                            "active_profile": "visitor_friendly",
                        },
                    },
                }

            def health(self):
                return {"status": "ok"}

        registry = _make_registry()
        registry.register(SkillModule())
        registry.register(PipelineModule())
        registry.register(ExecutorModule())
        registry.register(VoiceModule())

        cfg = {
            "brain": {
                "provider": "minimax",
                "model": "MiniMax-M2.7-highspeed",
                "voice_model": "MiniMax-M2.7-highspeed",
            },
            "voice": {
                "cloud_asr": {"model": "paraformer-realtime-v2"},
                "tts": {"backend": "minimax", "minimax_tts_model": "speech-2.8-turbo"},
            },
        }
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server) as server_cls:
            HealthModule().build(cfg, registry)

        provider = server_cls.call_args.kwargs["snapshot_provider"]
        snapshot = provider()

        assert snapshot["model_routing"]["dialogue"]["llm_model"] == "MiniMax-M2.7-highspeed"
        assert snapshot["model_routing"]["dialogue"]["asr_model"] == "paraformer-realtime-v2"
        assert snapshot["model_routing"]["dialogue"]["tts_model"] == "speech-2.8-turbo"
        assert snapshot["model_routing"]["agent_shell"]["loaded"] is True
        assert snapshot["model_routing"]["agent_shell"]["enabled"] is False
        assert snapshot["model_routing"]["agent_shell"]["status"] == "deprecated"
        assert snapshot["model_routing"]["agent_shell"]["replacement"] == "ZeroClaw MCP Agent"
        assert snapshot["model_routing"]["agent_shell"]["model"] == "MiniMax-M2.7-highspeed"
        assert snapshot["skill_callability"]["callable"] is True
        assert snapshot["skill_callability"]["agent_shell_callable"] is False
        assert snapshot["skill_callability"]["agent_shell_status"] == "deprecated"
        assert (
            snapshot["skill_callability"]["agent_shell_replacement"] == "ZeroClaw MCP Agent"
        )
        assert snapshot["skill_callability"]["agent_shell_skills"] == ["agent_task"]

    def test_runtime_health_provider_exposes_rag_trust_report(self, tmp_path):
        from askme.runtime.modules.health_module import HealthModule

        report_path = tmp_path / "rag-trust.json"
        report_path.write_text(
            json.dumps({
                "suite": "askme-rag-trust",
                "status": "passed",
                "scenario_count": 2,
                "passed": 2,
                "failed": 0,
                "scenarios": [
                    {"name": "visitor_wayfinding_grounded", "passed": True},
                    {"name": "expired_knowledge_refused", "passed": True},
                ],
            }),
            encoding="utf-8",
        )

        registry = _make_registry()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server) as server_cls:
            HealthModule().build({"rag_trust": {"report_path": str(report_path)}}, registry)

        provider = server_cls.call_args.kwargs["snapshot_provider"]
        snapshot = provider()

        assert snapshot["rag_trust"]["status"] == "passed"
        assert snapshot["rag_trust"]["passed"] == 2
        assert snapshot["rag_trust"]["scenarios"][0]["name"] == "visitor_wayfinding_grounded"

    def test_runtime_health_provider_exposes_voice_e2e_report(self, tmp_path):
        from askme.runtime.modules.health_module import HealthModule

        report_path = tmp_path / "voice-e2e.json"
        report_path.write_text(
            json.dumps({
                "suite": "askme-voice-e2e",
                "status": "passed",
                "scenario_count": 2,
                "passed": 2,
                "failed": 0,
                "metrics": {
                    "false_respond_rate": 0,
                    "tts_first_audio_ms": 360,
                },
                "scenarios": [
                    {
                        "name": "visitor_wayfinding_grounded",
                        "passed": True,
                        "interaction_gate": {"action": "respond"},
                    },
                    {
                        "name": "noise_bystander_casual_recorded_only",
                        "passed": True,
                        "interaction_gate": {"action": "record_only"},
                    },
                ],
            }),
            encoding="utf-8",
        )

        registry = _make_registry()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server) as server_cls:
            HealthModule().build({"voice_e2e": {"report_path": str(report_path)}}, registry)

        provider = server_cls.call_args.kwargs["snapshot_provider"]
        snapshot = provider()

        assert snapshot["voice_e2e"]["status"] == "passed"
        assert snapshot["voice_e2e"]["passed"] == 2
        assert snapshot["voice_e2e"]["metrics"]["false_respond_rate"] == 0
        assert snapshot["voice_e2e"]["scenarios"][1]["gate_action"] == "record_only"

    def test_runtime_health_provider_exposes_field_operations_report(self, tmp_path):
        from askme.runtime.modules.health_module import HealthModule

        report_path = tmp_path / "field-ops.json"
        report_path.write_text(
            json.dumps({
                "suite": "askme-field-operations",
                "status": "passed",
                "scenario_count": 2,
                "passed": 2,
                "failed": 0,
                "external_services": False,
                "hardware_dispatch": False,
                "product_demo": {
                    "suite_name": "园区机器狗场景演示包",
                    "demo_ready": True,
                    "real_integration_ready": False,
                    "customer_scenario_count": 2,
                    "blocked_on_real_integrations": ["真实摄像头/VMS 事件流"],
                },
                "scenarios": [
                    {"name": "robot_immobilized_notifies_security", "passed": True},
                    {"name": "illegal_parking_camera_ingest", "passed": True},
                ],
            }),
            encoding="utf-8",
        )

        registry = _make_registry()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server) as server_cls:
            HealthModule().build(
                {"field_operations": {"scenario_report_path": str(report_path)}},
                registry,
            )

        provider = server_cls.call_args.kwargs["snapshot_provider"]
        snapshot = provider()

        assert snapshot["field_operations"]["status"] == "passed"
        assert snapshot["field_operations"]["passed"] == 2
        assert snapshot["field_operations"]["hardware_dispatch"] is False
        assert snapshot["field_operations"]["product_demo"]["demo_ready"] is True
        assert snapshot["field_operations"]["product_demo"]["real_integration_ready"] is False
        assert snapshot["field_operations"]["scenarios"][0]["name"] == (
            "robot_immobilized_notifies_security"
        )

    @pytest.mark.parametrize("child_status", ["degraded", "error"])
    def test_runtime_health_provider_degrades_when_child_unhealthy(self, child_status):
        from askme.runtime.modules.health_module import HealthModule

        class ChildModule:
            name = "pipeline"

            def health(self):
                return {"status": child_status, "detail": "not ready"}

        registry = _make_registry()
        registry.register(ChildModule())

        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server) as server_cls:
            HealthModule().build({}, registry)

        provider = server_cls.call_args.kwargs["snapshot_provider"]
        snapshot = provider()

        assert snapshot["status"] == "degraded"
        assert snapshot["pipeline"] == {"status": child_status, "detail": "not ready"}

    def test_runtime_health_provider_degrades_when_child_health_raises(self):
        from askme.runtime.modules.health_module import HealthModule

        class BrokenModule:
            name = "skill"

            def health(self):
                raise RuntimeError("health probe failed")

        registry = _make_registry()
        registry.register(BrokenModule())

        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server) as server_cls:
            HealthModule().build({}, registry)

        provider = server_cls.call_args.kwargs["snapshot_provider"]
        snapshot = provider()

        assert snapshot["status"] == "degraded"
        assert snapshot["skill"] == {"status": "error"}

    @pytest.mark.asyncio
    async def test_start_calls_server_start_when_enabled(self):
        mod = self._make_module()
        mod.server.enabled = True
        mod.server.start = AsyncMock()
        await mod.start()
        mod.server.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_skips_when_disabled(self):
        mod = self._make_module()
        mod.server.enabled = False
        mod.server.start = AsyncMock()
        await mod.start()
        mod.server.start.assert_not_called()

    def test_build_wires_runtime_http_providers(self):
        from askme.runtime.modules.health_module import HealthModule

        class FakeTextModule:
            name = "text"

            def __init__(self):
                self.text_loop = MagicMock()
                self.process_turn_calls: list[dict[str, object]] = []

                async def _process_turn(
                    text: str,
                    *,
                    speak: bool = False,
                    conversation_session_id: str | None = None,
                    planning_session_id: str | None = None,
                    runtime_policy: str = "disabled",
                ) -> str:
                    self.process_turn_calls.append({
                        "text": text,
                        "speak": speak,
                        "conversation_session_id": conversation_session_id,
                        "planning_session_id": planning_session_id,
                        "runtime_policy": runtime_policy,
                    })
                    return "reply"

                self.text_loop.process_turn = _process_turn
                self.text_loop.current_turn_rag = {
                    "evidence": [{"record_id": "turn-rec", "text": "turn fact"}],
                    "rag": {
                        "turn_scoped": True,
                        "answer_policy": {
                            "state": "grounded",
                            "action": "answer_with_evidence",
                        },
                        "used_in_answer": True,
                    },
                }
                self.text_loop._audio = MagicMock()
                self.text_loop._audio.is_busy = False
                self.text_loop._audio.wait_speaking_done.return_value = True

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {"mode": "text"}

        class FakeMemoryModule:
            name = "memory"
            conversation = MagicMock(
                history=[{"role": "user", "content": "hello"}],
            )

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {}

        class FakeSkillManager:
            def get_contracts(self):
                return [MagicMock(source="code"), MagicMock(source="legacy")]

            def openapi_document(self):
                return {
                    "info": {"title": "askme", "version": "1.0"},
                    "paths": {"/skills/test": {}},
                }

            def get_all(self):
                return {"test": object()}

            def get_enabled(self):
                return {"test": object()}

            def get_contract_catalog(self):
                return [{"name": "test"}]

        class FakeSkillModule:
            name = "skill"
            skill_manager = FakeSkillManager()

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {"skills": True}

        class FakeMissionModule:
            name = "mission"
            mission_service = MagicMock()

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {"dry_run_default": True}

        registry = _make_registry()
        text_mod = FakeTextModule()
        mission_mod = FakeMissionModule()
        memory_mod = FakeMemoryModule()
        registry.register(text_mod)
        registry.register(memory_mod)
        registry.register(FakeSkillModule())
        registry.register(mission_mod)

        mod = HealthModule()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server):
            mod.build({"app": {"name": "askme-test", "version": "9.9"}}, registry)

        mock_server.set_chat_handler.assert_called_once()
        mock_server.set_capabilities_provider.assert_called_once()
        mock_server.set_conversation_provider.assert_called_once()
        mock_server.set_mission_handler.assert_called_once_with(
            mission_mod.mission_service,
        )
        mock_server.set_memory_handler.assert_called_once_with(memory_mod)

        chat_handler = mock_server.set_chat_handler.call_args.args[0]
        chat_payload = asyncio.run(
            chat_handler(
                "hello",
                speak=True,
                conversation_session_id="conv-1",
                planning_session_id="plan-1",
                runtime_policy="runtime_first",
            )
        )
        assert chat_payload["reply"] == "reply"
        assert chat_payload["spoken"] is True
        assert chat_payload["evidence"][0]["record_id"] == "turn-rec"
        assert chat_payload["rag"]["used_in_answer"] is True
        memory_mod.conversation.update_last_assistant_metadata.assert_called_once_with(
            {
                "evidence": chat_payload["evidence"],
                "rag": chat_payload["rag"],
            },
            conversation_session_id="conv-1",
        )
        assert text_mod.process_turn_calls == [{
            "text": "hello",
            "speak": True,
            "conversation_session_id": "conv-1",
            "planning_session_id": "plan-1",
            "runtime_policy": "runtime_first",
        }]
        text_mod.text_loop._audio.speak.assert_not_called()
        text_mod.text_loop._audio.start_playback.assert_not_called()
        text_mod.text_loop._audio.wait_speaking_done.assert_not_called()
        text_mod.text_loop._audio.stop_playback.assert_not_called()

        capabilities = mock_server.set_capabilities_provider.call_args.args[0]()
        assert capabilities["app"]["name"] == "askme-test"
        assert capabilities["app"]["version"] == "9.9"
        assert capabilities["app"]["voice_mode"] is False
        assert capabilities["app"]["robot_mode"] is False
        assert capabilities["profile"]["name"] == "text"
        assert capabilities["profile"]["primary_loop"] == "text"
        assert capabilities["components"]["text"]["capabilities"] == {"mode": "text"}
        assert capabilities["mission_adapter"] == {"dry_run_default": True}
        assert capabilities["skills"]["contract_count"] == 2
        assert capabilities["skills"]["code_contract_count"] == 1
        assert capabilities["openapi"]["path_count"] == 1

        conversation = mock_server.set_conversation_provider.call_args.args[0]()
        assert conversation == [{"role": "user", "content": "hello"}]

    def test_chat_handler_includes_cognition_metadata_when_text_loop_handles_task(self):
        from askme.runtime.module import ModuleRegistry

        from askme.runtime.modules.health_module import HealthModule

        class FakeTextModule:
            name = "text"

            def __init__(self):
                self.text_loop = MagicMock()
                self.text_loop.last_cognition_result = {
                    "handled": True,
                    "plan": {
                        "planning_session_id": "session-1",
                        "interaction_state": "awaiting_confirmation",
                    },
                }

                async def _process_turn(text: str, *, speak: bool = False) -> str:
                    return "已生成巡检任务草案，请确认后再交给运行时仲裁器。"

                self.text_loop.process_turn = _process_turn

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {}

        registry = ModuleRegistry()
        registry.register(FakeTextModule())

        mod = HealthModule()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server):
            mod.build({}, registry)

        chat_handler = mock_server.set_chat_handler.call_args.args[0]
        chat_payload = asyncio.run(chat_handler("巡检 A 区", speak=False))

        assert chat_payload["reply"].startswith("已生成巡检任务草案")
        assert chat_payload["cognition"]["handled"] is True
        assert chat_payload["cognition"]["plan"]["planning_session_id"] == "session-1"

    def test_chat_handler_submits_ready_cognition_plan_to_runtime_handoff(self):
        from askme.runtime.module import ModuleRegistry

        from askme.runtime.modules.health_module import HealthModule

        class FakeTextModule:
            name = "text"

            def __init__(self):
                self.text_loop = MagicMock()
                self.text_loop.last_cognition_result = {
                    "handled": True,
                    "plan": {
                        "plan_id": "plan-1",
                        "planning_session_id": "session-1",
                        "interaction_state": "ready_for_arbiter",
                        "handoff_ready": True,
                    },
                }

                async def _process_turn(text: str, *, speak: bool = False) -> str:
                    return "ready"

                self.text_loop.process_turn = _process_turn

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {}

        class FakeRuntimeHandoffModule:
            name = "runtime_handoff"

            def __init__(self):
                self.seen_plan = None

            def submit_plan_payload(self, plan):
                self.seen_plan = dict(plan)
                return {
                    "accepted": True,
                    "run": {"run_id": "run-1", "current_state": "completed"},
                }

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {}

        registry = ModuleRegistry()
        registry.register(FakeTextModule())
        runtime = FakeRuntimeHandoffModule()
        registry.register(runtime)

        mod = HealthModule()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server):
            mod.build({}, registry)

        chat_handler = mock_server.set_chat_handler.call_args.args[0]
        chat_payload = asyncio.run(chat_handler("confirm", speak=False))

        assert runtime.seen_plan["plan_id"] == "plan-1"
        assert chat_payload["runtime"]["accepted"] is True
        assert chat_payload["cognition"]["runtime"]["run"]["run_id"] == "run-1"

    def test_chat_handler_wayfinding_question_does_not_submit_runtime_task(self):
        from askme.runtime.module import ModuleRegistry

        from askme.runtime.modules.health_module import HealthModule

        class FakeTextModule:
            name = "text"

            def __init__(self):
                self.text_loop = MagicMock()
                self.text_loop.last_cognition_result = {
                    "handled": True,
                    "intent_type": "visitor_wayfinding",
                    "plan": {
                        "planning_session_id": "session-wayfinding",
                        "interaction_state": "idle",
                        "handoff_ready": False,
                    },
                }

                async def _process_turn(text: str, *, speak: bool = False) -> str:
                    return "卫生间在 A 区东侧，请沿右侧走廊前行。"

                self.text_loop.process_turn = _process_turn

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {}

        class FakeRuntimeHandoffModule:
            name = "runtime_handoff"

            def __init__(self):
                self.control_texts: list[str] = []
                self.submitted = False

            def handle_chat_control(self, text):
                self.control_texts.append(text)
                return None

            def submit_plan_payload(self, plan):
                self.submitted = True
                return {"accepted": True}

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {}

        registry = ModuleRegistry()
        registry.register(FakeTextModule())
        runtime = FakeRuntimeHandoffModule()
        registry.register(runtime)

        mod = HealthModule()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server):
            mod.build({}, registry)

        chat_handler = mock_server.set_chat_handler.call_args.args[0]
        chat_payload = asyncio.run(chat_handler("游客问：卫生间在哪里？", speak=False))

        assert "卫生间" in chat_payload["reply"]
        assert "runtime" not in chat_payload
        assert chat_payload["cognition"]["intent_type"] == "visitor_wayfinding"
        assert runtime.control_texts == ["游客问：卫生间在哪里？"]
        assert runtime.submitted is False

    def test_build_falls_back_to_pipeline_chat_when_text_missing(self):
        from askme.runtime.modules.health_module import HealthModule

        class FakePipelineModule:
            name = "pipeline"

            def __init__(self):
                self.brain_pipeline = MagicMock()
                self.brain_pipeline.process = AsyncMock(return_value="reply")

            def health(self):
                return {"status": "ok"}

            def capabilities(self):
                return {}

        registry = _make_registry()
        pipeline_mod = FakePipelineModule()
        registry.register(pipeline_mod)

        mod = HealthModule()
        mock_server = MagicMock()
        mock_server.enabled = True
        mock_server.port = 8080
        with patch("askme.health_server.AskmeHealthServer", return_value=mock_server):
            mod.build({}, registry)

        mock_server.set_chat_handler.assert_called_once_with(
            pipeline_mod.brain_pipeline.process,
        )

    def test_runtime_profile_infers_mcp_and_edge_robot_modes(self):
        from askme.runtime.modules.health_module import HealthModule

        class FakeModule:
            def __init__(self, name):
                self.name = name

        mod = HealthModule()

        mcp_registry = _make_registry()
        for name in ("voice", "control", "executor", "safety"):
            mcp_registry.register(FakeModule(name))

        mcp_profile = mod._runtime_profile(mcp_registry)
        assert mcp_profile.name == "mcp"
        assert mcp_profile.primary_loop == "mcp"
        assert mcp_profile.http_chat is False

        edge_registry = _make_registry()
        for name in ("voice", "text", "control", "perception", "led"):
            edge_registry.register(FakeModule(name))

        edge_profile = mod._runtime_profile(edge_registry)
        assert edge_profile.name == "edge_robot"
        assert edge_profile.primary_loop == "voice"
        assert edge_profile.http_chat is True


# ── LEDModule ─────────────────────────────────────────────────────────────────

class TestLEDModule:
    def _make_module(self, led_base_url=""):
        from askme.runtime.modules.led_module import LEDModule
        mod = LEDModule()
        mock_controller = MagicMock()
        mock_bridge = MagicMock()
        mock_bridge.run = AsyncMock()
        with patch(
            "askme.runtime.modules.led_module.build_status_led",
            return_value=(mock_controller, mock_bridge),
        ):
            cfg = {"led": {"base_url": led_base_url}}
            mod.build(cfg, _make_registry())
        return mod

    def test_build_with_empty_url_uses_null_controller(self):
        mod = self._make_module(led_base_url="")
        # NullLedController is used when no URL provided
        assert mod.led_controller is not None
        assert mod.led_bridge is not None

    def test_health_returns_ok(self):
        mod = self._make_module()
        h = mod.health()
        assert h["status"] == "ok"

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        mod = self._make_module()

        async def _fake_run():
            await asyncio.sleep(100)

        mod._task = asyncio.create_task(_fake_run())
        await mod.stop()
        assert mod._task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_no_task_no_crash(self):
        mod = self._make_module()
        # No task set — should not raise
        await mod.stop()
