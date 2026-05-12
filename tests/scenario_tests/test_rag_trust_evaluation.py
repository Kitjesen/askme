from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from askme.memory.catalog import KnowledgeCatalog
from scripts.eval.evaluate_rag_trust_scenarios import evaluate_scenarios, write_report


def _memory_module(tmp_path: Path):
    from askme.runtime.modules.memory_module import MemoryModule

    mod = MemoryModule()
    with patch("askme.runtime.modules.memory_module.SessionMemory"), \
         patch("askme.runtime.modules.memory_module.ConversationManager") as mock_conv, \
         patch("askme.runtime.modules.memory_module.MemoryBridge"), \
         patch("askme.runtime.modules.memory_module.EpisodicMemory") as mock_epi, \
         patch("askme.runtime.modules.memory_module.MemorySystem"):
        mock_conv.return_value.history = []
        mock_epi.return_value._buffer = []
        mod.llm_client = None
        mod.build({"app": {"data_dir": str(tmp_path)}}, MagicMock())
    mod._knowledge_catalog = KnowledgeCatalog(path=tmp_path / "catalog.json")
    mod.memory_bridge.save_fact = AsyncMock()
    mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": True})
    mod.memory_bridge.health.return_value = {
        "enabled": True,
        "backend": "vector",
        "last_backend": "vector",
        "last_retrieve_ms": 1.0,
        "last_retrieved_items": 0,
        "last_evidence": [],
        "last_dropped_evidence": [],
        "last_answer_policy": {
            "state": "no_evidence",
            "action": "clarify_or_refuse",
        },
    }
    return mod


async def _import_knowledge(mod, records: list[dict]) -> dict:
    import json

    return await mod.import_payload({
        "filename": "scenario.json",
        "content": json.dumps(records),
    })


@pytest.mark.asyncio
async def test_rag_trust_scenario_visitor_wayfinding_uses_grounded_evidence(tmp_path):
    mod = _memory_module(tmp_path)

    imported = await _import_knowledge(mod, [
        {
            "record_id": "loc_restroom",
            "question": "洗手间在哪里",
            "answer": "一层东侧，靠近服务台",
            "category": "location",
            "approval_status": "published",
            "entity_key": "place:restroom",
            "fact_key": "location",
            "value": "floor1_east",
        }
    ])
    _, metadata = mod.memory_bridge.save_fact.await_args.args
    mod.memory_bridge.retrieve = AsyncMock(return_value="- 问: 洗手间在哪里\n答: 一层东侧，靠近服务台")
    mod.memory_bridge.health.return_value = {
        "enabled": True,
        "backend": "vector",
        "last_backend": "vector",
        "last_retrieve_ms": 2.0,
        "last_retrieved_items": 1,
        "last_evidence": [{
            "text": "问: 洗手间在哪里\n答: 一层东侧，靠近服务台",
            "source": "scenario.json",
            "category": "location",
            "score": 0.91,
            "metadata": metadata,
        }],
        "last_dropped_evidence": [],
        "last_answer_policy": {
            "state": "grounded",
            "action": "answer_with_evidence",
        },
    }

    result = await mod.search_payload({"query": "洗手间在哪里"})

    assert imported["imported"] == 1
    assert result["results"][0]["category"] == "location"
    assert result["rag"]["answer_policy"]["state"] == "grounded"
    assert result["rag"]["answer_policy"]["action"] == "answer_with_evidence"


@pytest.mark.asyncio
async def test_rag_trust_scenario_expired_knowledge_refuses_answer(tmp_path):
    mod = _memory_module(tmp_path)

    await _import_knowledge(mod, [
        {
            "record_id": "route_old",
            "text": "旧展厅路线从北门进入",
            "category": "route",
            "approval_status": "published",
            "expires_at": "2000-01-01T00:00:00+00:00",
        }
    ])
    mod.memory_bridge.retrieve = AsyncMock(return_value="")
    mod.memory_bridge.health.return_value = {
        "enabled": True,
        "backend": "vector",
        "last_backend": "vector",
        "last_retrieve_ms": 1.5,
        "last_retrieved_items": 0,
        "last_evidence": [],
        "last_dropped_evidence": [{
            "text": "旧展厅路线从北门进入",
            "drop_reason": "expired",
            "used_in_prompt": False,
        }],
        "last_answer_policy": {
            "state": "stale",
            "action": "refuse_and_request_update",
        },
    }

    result = await mod.search_payload({"query": "展厅怎么走"})

    assert result["results"] == []
    assert result["rag"]["dropped_evidence"][0]["drop_reason"] == "expired"
    assert result["rag"]["answer_policy"]["state"] == "stale"


@pytest.mark.asyncio
async def test_rag_trust_scenario_conflicting_device_location_clarifies(tmp_path):
    mod = _memory_module(tmp_path)

    imported = await _import_knowledge(mod, [
        {
            "record_id": "dev_a_east",
            "text": "设备 A 在东门",
            "category": "equipment",
            "approval_status": "published",
            "entity_key": "device:a",
            "fact_key": "location",
            "value": "east_gate",
        },
        {
            "record_id": "dev_a_west",
            "text": "设备 A 在西门",
            "category": "equipment",
            "approval_status": "published",
            "entity_key": "device:a",
            "fact_key": "location",
            "value": "west_gate",
        },
    ])
    mod.memory_bridge.retrieve = AsyncMock(return_value="")
    mod.memory_bridge.health.return_value = {
        "enabled": True,
        "backend": "vector",
        "last_backend": "vector",
        "last_retrieve_ms": 1.1,
        "last_retrieved_items": 0,
        "last_evidence": [],
        "last_dropped_evidence": [
            {"text": "设备 A 在东门", "drop_reason": "conflict:device:a:location"},
            {"text": "设备 A 在西门", "drop_reason": "conflict:device:a:location"},
        ],
        "last_answer_policy": {
            "state": "conflict",
            "action": "clarify",
        },
    }

    result = await mod.search_payload({"query": "设备 A 在哪里"})

    assert imported["imported"] == 0
    assert imported["skipped"] == 2
    assert result["rag"]["answer_policy"]["state"] == "conflict"
    assert result["rag"]["answer_policy"]["action"] == "clarify"


@pytest.mark.asyncio
async def test_rag_trust_scenario_deleted_knowledge_is_not_usable(tmp_path):
    mod = _memory_module(tmp_path)

    await _import_knowledge(mod, [
        {
            "record_id": "faq_deleted",
            "text": "临时出口在南侧",
            "category": "location",
            "approval_status": "published",
        }
    ])
    deleted = await mod.update_knowledge_payload({
        "record_id": "faq_deleted",
        "action": "delete",
    })
    mod.memory_bridge.retrieve = AsyncMock(return_value="")
    mod.memory_bridge.health.return_value = {
        "enabled": True,
        "backend": "vector",
        "last_backend": "vector",
        "last_retrieve_ms": 1.2,
        "last_retrieved_items": 0,
        "last_evidence": [],
        "last_dropped_evidence": [{
            "text": "临时出口在南侧",
            "drop_reason": "catalog_status:deleted",
        }],
        "last_answer_policy": {
            "state": "unapproved",
            "action": "refuse",
        },
    }

    result = await mod.search_payload({"query": "临时出口在哪"})

    assert deleted["updated"] is True
    assert deleted["record"]["approval_status"] == "deleted"
    assert result["results"] == []
    assert result["rag"]["answer_policy"]["action"] == "refuse"


@pytest.mark.asyncio
async def test_rag_trust_scenario_unknown_location_has_no_evidence_policy(tmp_path):
    mod = _memory_module(tmp_path)
    mod.memory_bridge.retrieve = AsyncMock(return_value="")

    result = await mod.search_payload({"query": "贵宾室在哪里"})

    assert result["results"] == []
    assert result["rag"]["answer_policy"]["state"] == "no_evidence"
    assert result["warnings"] == []


@pytest.mark.asyncio
async def test_rag_trust_scenario_evaluation_suite_writes_report(tmp_path):
    payload = await evaluate_scenarios()
    report = write_report(payload, tmp_path / "rag-trust.json")

    names = {item["name"] for item in payload["scenarios"]}
    assert payload["status"] == "passed"
    assert payload["external_services"] is False
    assert payload["failed"] == 0
    assert report.exists()
    assert {
        "visitor_wayfinding_grounded",
        "expired_knowledge_refused",
        "conflicting_device_location_clarifies",
        "deleted_knowledge_refused",
        "unknown_location_no_evidence",
    }.issubset(names)
