from __future__ import annotations

import json

from askme.contracts.catalog import contract_catalog, contract_examples


def test_contract_catalog_describes_product_boundary() -> None:
    catalog = contract_catalog()

    assert catalog["version"]
    assert catalog["flow"][:4] == [
        "PerceptionInput",
        "IntentInput",
        "ActionDecision",
        "UserFacingOutput",
    ]
    assert "guide_by_voice" in catalog["enums"]["RobotActionType"]
    assert "PerceptionInput" in catalog["contracts"]
    assert "CapabilityPackageManifest" in catalog["contracts"]
    assert catalog["adapters"]["field_event"]["outputs"] == [
        "EvidenceRef",
        "ActionDecision",
        "UserFacingOutput",
    ]
    assert any(
        item["name"] == "evidence_refs"
        for item in catalog["contracts"]["ActionDecision"]
    )


def test_contract_examples_are_customer_scenario_payloads() -> None:
    examples = contract_examples()

    assert examples["perception_input"]["location"]["name"] == "西门问询点"
    assert examples["intent_input"]["intent_type"] == "ask_direction"
    assert examples["action_decision"]["action_type"] == "guide_by_voice"
    assert examples["user_facing_output"]["audit_id"] == "audit-demo-001"


def test_mcp_contract_resources_return_json() -> None:
    from askme.mcp.resources.contract_resources import contracts_examples, contracts_io

    catalog = json.loads(contracts_io())
    examples = json.loads(contracts_examples())

    assert catalog["contracts"]["UserFacingOutput"]
    assert examples["action_decision"]["skill_name"] == "answer_wayfinding"
