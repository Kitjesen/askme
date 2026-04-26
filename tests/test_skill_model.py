"""Tests for SkillDefinition, SlotSpec, build_prompt, to_contract, _slot_type_to_json_type."""

from __future__ import annotations

from askme.skills.contracts import SkillContract
from askme.skills.skill_model import (
    SkillDefinition,
    SlotSpec,
    _slot_type_to_json_type,
)

# ── SlotSpec defaults ─────────────────────────────────────────────────────────

class TestSlotSpec:
    def test_required_fields(self):
        slot = SlotSpec(name="destination")
        assert slot.name == "destination"

    def test_defaults(self):
        slot = SlotSpec(name="x")
        assert slot.type == "text"
        assert slot.prompt == ""
        assert slot.optional is False
        assert slot.default == ""

    def test_custom_values(self):
        slot = SlotSpec(name="speed", type="int", prompt="多快？", optional=True, default="5")
        assert slot.type == "int"
        assert slot.optional is True
        assert slot.default == "5"


# ── SkillDefinition defaults ──────────────────────────────────────────────────

class TestSkillDefinitionDefaults:
    def test_required_name(self):
        sd = SkillDefinition(name="patrol")
        assert sd.name == "patrol"

    def test_default_version(self):
        sd = SkillDefinition(name="x")
        assert sd.version == "1.0.0"

    def test_default_trigger(self):
        sd = SkillDefinition(name="x")
        assert sd.trigger == "manual"

    def test_default_safety_level(self):
        sd = SkillDefinition(name="x")
        assert sd.safety_level == "normal"

    def test_default_execution(self):
        sd = SkillDefinition(name="x")
        assert sd.execution == "skill_executor"

    def test_default_enabled(self):
        sd = SkillDefinition(name="x")
        assert sd.enabled is True

    def test_empty_lists(self):
        sd = SkillDefinition(name="x")
        assert sd.tags == []
        assert sd.depends == []
        assert sd.conflicts == []
        assert sd.required_slots == []


# ── build_prompt ──────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_no_context_returns_template(self):
        sd = SkillDefinition(name="x", prompt_template="Go to the room.")
        assert sd.build_prompt() == "Go to the room."

    def test_substitutes_placeholder(self):
        sd = SkillDefinition(name="x", prompt_template="导航到{{destination}}")
        result = sd.build_prompt({"destination": "仓库A"})
        assert result == "导航到仓库A"

    def test_multiple_placeholders(self):
        sd = SkillDefinition(name="x", prompt_template="从{{from}}到{{to}}")
        result = sd.build_prompt({"from": "A", "to": "B"})
        assert "A" in result
        assert "B" in result

    def test_unresolved_placeholder_removed(self):
        sd = SkillDefinition(name="x", prompt_template="任务：{{task}}")
        result = sd.build_prompt()  # no context
        assert "{{task}}" not in result
        assert "任务：" in result

    def test_none_context_no_crash(self):
        sd = SkillDefinition(name="x", prompt_template="Hello {{name}}")
        result = sd.build_prompt(None)
        assert isinstance(result, str)
        assert "{{name}}" not in result

    def test_empty_template_returns_empty(self):
        sd = SkillDefinition(name="x")
        assert sd.build_prompt({"k": "v"}) == ""


# ── to_contract ───────────────────────────────────────────────────────────────

class TestToContract:
    def test_returns_skill_contract(self):
        sd = SkillDefinition(name="patrol", description="巡逻任务")
        contract = sd.to_contract()
        assert isinstance(contract, SkillContract)

    def test_name_matches(self):
        sd = SkillDefinition(name="patrol")
        assert sd.to_contract().name == "patrol"

    def test_description_matches(self):
        sd = SkillDefinition(name="x", description="do stuff")
        assert sd.to_contract().description == "do stuff"

    def test_safety_level_matches(self):
        sd = SkillDefinition(name="x", safety_level="dangerous")
        assert sd.to_contract().safety_level == "dangerous"

    def test_tags_become_tuple(self):
        sd = SkillDefinition(name="x", tags=["robot", "nav"])
        contract = sd.to_contract()
        assert isinstance(contract.tags, tuple)
        assert "robot" in contract.tags

    def test_slots_become_parameters(self):
        sd = SkillDefinition(
            name="nav",
            required_slots=[
                SlotSpec(name="destination", type="location", prompt="去哪里？"),
                SlotSpec(name="speed", type="int", optional=True, default="3"),
            ],
        )
        contract = sd.to_contract()
        assert len(contract.parameters) == 2
        params = {p.name: p for p in contract.parameters}
        assert params["destination"].required is True
        assert params["speed"].required is False
        assert params["speed"].default == "3"

    def test_source_is_legacy_markdown(self):
        sd = SkillDefinition(name="x")
        assert sd.to_contract().source == "legacy_markdown"

    def test_confirm_before_execute(self):
        sd = SkillDefinition(name="x", confirm_before_execute=True)
        assert sd.to_contract().confirm_before_execute is True

    def test_empty_default_slot_maps_to_none(self):
        sd = SkillDefinition(
            name="x",
            required_slots=[SlotSpec(name="s", optional=True, default="")],
        )
        contract = sd.to_contract()
        # default="" should map to None in the parameter
        assert contract.parameters[0].default is None


# ── _slot_type_to_json_type ───────────────────────────────────────────────────

class TestSlotTypeToJsonType:
    def test_text_to_string(self):
        assert _slot_type_to_json_type("text") == "string"

    def test_location_to_string(self):
        assert _slot_type_to_json_type("location") == "string"

    def test_int_to_integer(self):
        assert _slot_type_to_json_type("int") == "integer"

    def test_float_to_number(self):
        assert _slot_type_to_json_type("float") == "number"

    def test_bool_to_boolean(self):
        assert _slot_type_to_json_type("bool") == "boolean"

    def test_unknown_type_defaults_to_string(self):
        assert _slot_type_to_json_type("exotic_type") == "string"

    def test_enum_to_string(self):
        assert _slot_type_to_json_type("enum") == "string"
