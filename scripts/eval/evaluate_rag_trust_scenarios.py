"""Evaluate deterministic RAG trust scenarios and write an auditable artifact."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from askme.memory.catalog import KnowledgeCatalog  # noqa: E402

DEFAULT_REPORT_PATH = Path("artifacts/rag_trust/scenario-evaluation.json")


async def evaluate_scenarios() -> dict[str, Any]:
    """Run deterministic RAG trust scenarios without external services."""
    with tempfile.TemporaryDirectory(prefix="askme-rag-trust-") as temp_dir:
        root = Path(temp_dir)
        scenarios = [
            await _scenario_grounded_wayfinding(root / "grounded"),
            await _scenario_expired_refusal(root / "expired"),
            await _scenario_conflict_clarification(root / "conflict"),
            await _scenario_deleted_refusal(root / "deleted"),
            await _scenario_no_evidence(root / "none"),
        ]
    passed = sum(1 for item in scenarios if item["passed"])
    return {
        "suite": "askme-rag-trust",
        "external_services": False,
        "scenario_count": len(scenarios),
        "passed": passed,
        "failed": len(scenarios) - passed,
        "status": "passed" if passed == len(scenarios) else "failed",
        "scenarios": scenarios,
        "generated_at": time.time(),
    }


def write_report(payload: dict[str, Any], path: Path = DEFAULT_REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


async def _scenario_grounded_wayfinding(root: Path) -> dict[str, Any]:
    mod = _memory_module(root)
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
    _set_health(
        mod,
        evidence=[{
            "text": "问: 洗手间在哪里\n答: 一层东侧，靠近服务台",
            "source": "scenario.json",
            "category": "location",
            "score": 0.91,
            "metadata": metadata,
        }],
        policy={"state": "grounded", "action": "answer_with_evidence"},
    )
    result = await mod.search_payload({"query": "洗手间在哪里"})
    return _verdict(
        "visitor_wayfinding_grounded",
        imported["imported"] == 1
        and len(result["results"]) == 1
        and result["rag"]["answer_policy"]["state"] == "grounded",
        observed={"imported": imported, "rag": result["rag"]},
    )


async def _scenario_expired_refusal(root: Path) -> dict[str, Any]:
    mod = _memory_module(root)
    await _import_knowledge(mod, [{
        "record_id": "route_old",
        "text": "旧展厅路线从北门进入",
        "category": "route",
        "approval_status": "published",
        "expires_at": "2000-01-01T00:00:00+00:00",
    }])
    mod.memory_bridge.retrieve = AsyncMock(return_value="")
    _set_health(
        mod,
        dropped=[{"text": "旧展厅路线从北门进入", "drop_reason": "expired"}],
        policy={"state": "stale", "action": "refuse_and_request_update"},
    )
    result = await mod.search_payload({"query": "展厅怎么走"})
    return _verdict(
        "expired_knowledge_refused",
        result["results"] == [] and result["rag"]["answer_policy"]["state"] == "stale",
        observed={"rag": result["rag"]},
    )


async def _scenario_conflict_clarification(root: Path) -> dict[str, Any]:
    mod = _memory_module(root)
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
    _set_health(
        mod,
        dropped=[{"text": "设备 A", "drop_reason": "conflict:device:a:location"}],
        policy={"state": "conflict", "action": "clarify"},
    )
    result = await mod.search_payload({"query": "设备 A 在哪里"})
    return _verdict(
        "conflicting_device_location_clarifies",
        imported["imported"] == 0
        and imported["skipped"] == 2
        and result["rag"]["answer_policy"]["action"] == "clarify",
        observed={"imported": imported, "rag": result["rag"]},
    )


async def _scenario_deleted_refusal(root: Path) -> dict[str, Any]:
    mod = _memory_module(root)
    await _import_knowledge(mod, [{
        "record_id": "faq_deleted",
        "text": "临时出口在南侧",
        "category": "location",
        "approval_status": "published",
    }])
    deleted = await mod.update_knowledge_payload({"record_id": "faq_deleted", "action": "delete"})
    mod.memory_bridge.retrieve = AsyncMock(return_value="")
    _set_health(
        mod,
        dropped=[{"text": "临时出口在南侧", "drop_reason": "catalog_status:deleted"}],
        policy={"state": "unapproved", "action": "refuse"},
    )
    result = await mod.search_payload({"query": "临时出口在哪"})
    return _verdict(
        "deleted_knowledge_refused",
        deleted["updated"] is True
        and result["results"] == []
        and result["rag"]["answer_policy"]["action"] == "refuse",
        observed={"deleted": deleted, "rag": result["rag"]},
    )


async def _scenario_no_evidence(root: Path) -> dict[str, Any]:
    mod = _memory_module(root)
    mod.memory_bridge.retrieve = AsyncMock(return_value="")
    result = await mod.search_payload({"query": "贵宾室在哪里"})
    return _verdict(
        "unknown_location_no_evidence",
        result["results"] == []
        and result["rag"]["answer_policy"]["state"] == "no_evidence",
        observed={"rag": result["rag"]},
    )


def _memory_module(root: Path):
    from askme.runtime.modules.memory_module import MemoryModule

    root.mkdir(parents=True, exist_ok=True)
    mod = MemoryModule()
    with patch("askme.runtime.modules.memory_module.SessionMemory"), \
         patch("askme.runtime.modules.memory_module.ConversationManager") as mock_conv, \
         patch("askme.runtime.modules.memory_module.MemoryBridge"), \
         patch("askme.runtime.modules.memory_module.EpisodicMemory") as mock_epi, \
         patch("askme.runtime.modules.memory_module.MemorySystem"):
        mock_conv.return_value.history = []
        mock_epi.return_value._buffer = []
        mod.llm_client = None
        mod.build({"app": {"data_dir": str(root)}}, MagicMock())
    mod._knowledge_catalog = KnowledgeCatalog(path=root / "catalog.json")
    mod.memory_bridge.save_fact = AsyncMock()
    mod.memory_bridge.update_knowledge_metadata = AsyncMock(return_value={"updated": True})
    _set_health(mod, policy={"state": "no_evidence", "action": "clarify_or_refuse"})
    return mod


async def _import_knowledge(mod: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    return await mod.import_payload({
        "filename": "scenario.json",
        "content": json.dumps(records, ensure_ascii=False),
    })


def _set_health(
    mod: Any,
    *,
    evidence: list[dict[str, Any]] | None = None,
    dropped: list[dict[str, Any]] | None = None,
    policy: dict[str, Any] | None = None,
) -> None:
    mod.memory_bridge.health.return_value = {
        "enabled": True,
        "backend": "vector",
        "last_backend": "vector",
        "last_retrieve_ms": 1.0,
        "last_retrieved_items": len(evidence or []),
        "last_evidence": evidence or [],
        "last_dropped_evidence": dropped or [],
        "last_answer_policy": policy or {},
    }


def _verdict(name: str, passed: bool, *, observed: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "observed": observed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    import asyncio

    payload = asyncio.run(evaluate_scenarios())
    report = write_report(payload, args.output)
    print(json.dumps({"status": payload["status"], "report": str(report)}, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
