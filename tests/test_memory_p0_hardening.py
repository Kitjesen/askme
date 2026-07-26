"""P0 regression contracts for governed and observable memory."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.memory.core.turn_admission import TurnAdmissionResult
from askme.memory.retrieval.bridge import MemoryBridge
from askme.runtime.modules.memory_module import MemoryModule


def test_memory_system_receives_the_runtime_config() -> None:
    cfg = {"memory": {"enabled": False, "backend": "vector"}}
    module = MemoryModule()
    with (
        patch("askme.runtime.modules.memory_module.SessionMemory"),
        patch("askme.runtime.modules.memory_module.ConversationManager"),
        patch("askme.runtime.modules.memory_module.KnowledgeCatalog"),
        patch("askme.runtime.modules.memory_module.MemoryBridge"),
        patch("askme.runtime.modules.memory_module.EpisodicMemory"),
        patch("askme.runtime.modules.memory_module.MemorySystem") as memory_system,
        patch("askme.runtime.modules.memory_module.KnowledgeIndexJobStore"),
    ):
        module.llm_client = None
        module.build(cfg, MagicMock())

    assert memory_system.call_args.kwargs["config"] is cfg


def test_sync_health_is_not_ok_when_no_memory_backend_is_ready() -> None:
    module = MemoryModule.__new__(MemoryModule)
    module._memory_bridge = MagicMock()
    module._memory_bridge.health.return_value = {
        "enabled": True,
        "available": False,
        "selected_backend_ready": False,
        "fallback_ready": False,
    }
    module._knowledge_catalog = MagicMock()
    module._knowledge_catalog.health.return_value = {"prompt_eligible": 0}
    module._knowledge_job_store = MagicMock()
    module._knowledge_job_store.health.return_value = {"total": 0}
    module._conversation = SimpleNamespace(history=[])
    module._episodic = SimpleNamespace(_buffer=[])
    module._memory_cfg = {}

    health = module.health()

    assert health["status"] == "not_ready"
    assert health["ready"] is False


@pytest.mark.asyncio
async def test_request_scope_overrides_static_memory_scope(tmp_path) -> None:
    bridge = MemoryBridge(
        config={
            "memory": {
                "enabled": True,
                "backend": "vector",
                "customer_id": "configured-customer",
                "project_id": "configured-project",
                "user_id": "configured-user",
            },
            "brain": {},
        },
        data_dir=tmp_path,
    )
    bridge._turn_admission.classify = MagicMock(
        return_value=TurnAdmissionResult(False, rejected_reason="not_durable_memory")
    )

    await bridge.admit_turn(
        "普通聊天",
        customer_id="request-customer",
        project_id="request-project",
        user_id="request-user",
    )

    assert (
        bridge._turn_admission.classify.call_args.kwargs["customer_id"]
        == "request-customer"
    )
    assert (
        bridge._turn_admission.classify.call_args.kwargs["project_id"]
        == "request-project"
    )
    assert bridge._turn_admission.classify.call_args.kwargs["user_id"] == "request-user"


@pytest.mark.asyncio
async def test_failed_reindex_does_not_promote_stale_vector_metadata() -> None:
    module = MemoryModule.__new__(MemoryModule)
    module._memory_bridge = MagicMock()
    module._memory_bridge.health.return_value = {"enabled": True, "backend": "vector"}
    module._memory_bridge.update_knowledge_metadata = AsyncMock()
    module._memory_bridge.save_fact = AsyncMock(return_value=False)
    module._knowledge_catalog = MagicMock()
    module._knowledge_catalog.is_prompt_eligible.return_value = True

    result = await module._sync_catalog_records(
        [
            {
                "record_id": "know-route",
                "text": "新版路线",
                "approval_status": "published",
                "evidence_version": 2,
                "metadata": {
                    "record_id": "know-route",
                    "approval_status": "published",
                    "evidence_version": 2,
                },
            }
        ]
    )

    module._memory_bridge.update_knowledge_metadata.assert_not_awaited()
    module._knowledge_catalog.mark_indexed.assert_not_called()
    assert result["indexed"] == 0
    assert result["skipped"] == 1
