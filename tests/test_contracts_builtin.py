"""Tests for built-in skill contracts — registration and structure."""

from __future__ import annotations

import pytest

import askme.skills.contracts_builtin  # noqa: F401 — triggers @skill_contract decorators
from askme.skills.contracts import SkillContractRegistry, registered_skill_contracts


class TestBuiltinContracts:
    def test_agent_task_registered(self):
        registry = registered_skill_contracts()
        assert "agent_task" in registry

    def test_navigate_registered(self):
        registry = registered_skill_contracts()
        assert "navigate" in registry

    def test_find_object_registered(self):
        registry = registered_skill_contracts()
        assert "find_object" in registry

    def test_find_person_registered(self):
        registry = registered_skill_contracts()
        assert "find_person" in registry

    def test_recall_memory_registered(self):
        registry = registered_skill_contracts()
        assert "recall_memory" in registry

    def test_solve_problem_registered(self):
        registry = registered_skill_contracts()
        assert "solve_problem" in registry

    def test_web_search_registered(self):
        registry = registered_skill_contracts()
        assert "web_search" in registry

    def test_navigate_safety_dangerous(self):
        registry = registered_skill_contracts()
        contract = registry["navigate"]
        assert contract.safety_level == "dangerous"

    def test_navigate_has_destination_param(self):
        registry = registered_skill_contracts()
        contract = registry["navigate"]
        param_names = [p.name for p in contract.parameters]
        assert "destination" in param_names

    def test_find_object_has_object_name_param(self):
        registry = registered_skill_contracts()
        contract = registry["find_object"]
        param_names = [p.name for p in contract.parameters]
        assert "object_name" in param_names

    def test_find_person_param_not_required(self):
        registry = registered_skill_contracts()
        contract = registry["find_person"]
        person_param = next(p for p in contract.parameters if p.name == "person_name")
        assert person_param.required is False

    def test_web_search_has_query_param_required(self):
        registry = registered_skill_contracts()
        contract = registry["web_search"]
        query_param = next(p for p in contract.parameters if p.name == "query")
        assert query_param.required is True

    def test_agent_task_execution_is_agent_shell(self):
        registry = registered_skill_contracts()
        contract = registry["agent_task"]
        assert contract.execution == "agent_shell"

    def test_navigate_has_robot_tag(self):
        registry = registered_skill_contracts()
        contract = registry["navigate"]
        assert "robot" in contract.tags

    def test_all_contracts_have_descriptions(self):
        registry = registered_skill_contracts()
        for name, contract in registry.items():
            assert contract.description, f"Contract '{name}' has empty description"

    def test_all_contracts_have_names(self):
        registry = registered_skill_contracts()
        for name, contract in registry.items():
            assert contract.name == name
