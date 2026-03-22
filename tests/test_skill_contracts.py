"""Tests for code-defined skill contracts and generated OpenAPI."""

from __future__ import annotations

from askme.skills.skill_manager import SkillManager


def _manager(tmp_path, monkeypatch) -> SkillManager:
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    manager = SkillManager(project_dir=tmp_path)
    manager.load()
    return manager


def test_code_defined_contract_overrides_markdown_metadata(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path, monkeypatch)

    contract = manager.get_contract("navigate")

    assert contract is not None
    assert contract.name == "navigate"
    assert contract.source == "code"
    assert contract.safety_level == "dangerous"
    assert [parameter.name for parameter in contract.parameters] == ["destination"]


def test_openapi_document_is_generated_from_loaded_contracts(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path, monkeypatch)

    document = manager.openapi_document()
    navigate_execute = document["paths"]["/api/v1/skills/navigate/execute"]["post"]

    assert document["info"]["title"] == "Askme Skill Runtime API"
    assert navigate_execute["x-askme-skill"]["contract_source"] == "code"
    assert "destination" in navigate_execute["requestBody"]["content"]["application/json"]["schema"]["properties"]


def test_contract_catalog_exposes_code_and_legacy_metadata(tmp_path, monkeypatch) -> None:
    manager = _manager(tmp_path, monkeypatch)

    catalog = {
        entry["name"]: entry
        for entry in manager.get_contract_catalog()
    }

    assert catalog["navigate"]["contract_source"] == "code"
    assert catalog["navigate"]["legacy_source"] == "builtin"
    assert "enabled" in catalog["navigate"]
    assert isinstance(catalog["navigate"]["enabled"], bool)
