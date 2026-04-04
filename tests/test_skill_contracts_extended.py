"""Extended tests for SkillContract, SkillParameter, and build_skills_openapi."""

from __future__ import annotations

import pytest

from askme.skills.contracts import (
    SkillContract,
    SkillContractRegistry,
    SkillParameter,
    build_skills_openapi,
    skill_contract,
)


# ── SkillParameter ────────────────────────────────────────────────────────────

class TestSkillParameterJsonSchema:
    def test_minimal_schema_has_type(self):
        p = SkillParameter(name="x", type="string")
        schema = p.json_schema()
        assert schema["type"] == "string"

    def test_description_included_when_set(self):
        p = SkillParameter(name="x", type="string", description="The target location.")
        schema = p.json_schema()
        assert schema["description"] == "The target location."

    def test_description_excluded_when_empty(self):
        p = SkillParameter(name="x", type="string", description="")
        schema = p.json_schema()
        assert "description" not in schema

    def test_enum_included_when_set(self):
        p = SkillParameter(name="mode", type="string", enum=("fast", "slow"))
        schema = p.json_schema()
        assert schema["enum"] == ["fast", "slow"]

    def test_enum_excluded_when_empty(self):
        p = SkillParameter(name="x", type="string", enum=())
        schema = p.json_schema()
        assert "enum" not in schema

    def test_default_included_when_set(self):
        p = SkillParameter(name="x", type="string", default="hello")
        schema = p.json_schema()
        assert schema["default"] == "hello"

    def test_default_excluded_when_none(self):
        p = SkillParameter(name="x", type="string", default=None)
        schema = p.json_schema()
        assert "default" not in schema

    def test_default_excluded_when_empty_string(self):
        p = SkillParameter(name="x", type="string", default="")
        schema = p.json_schema()
        assert "default" not in schema

    def test_integer_type(self):
        p = SkillParameter(name="count", type="integer")
        assert p.json_schema()["type"] == "integer"


# ── SkillContract ─────────────────────────────────────────────────────────────

def _make_contract(name: str = "test_skill", **kwargs) -> SkillContract:
    defaults = {
        "description": "Test skill description",
        "version": "1.0.0",
        "safety_level": "normal",
        "execution": "skill_executor",
        "tags": (),
        "parameters": (),
        "confirm_before_execute": False,
        "source": "code",
    }
    defaults.update(kwargs)
    return SkillContract(name=name, **defaults)


class TestSkillContractWithFallbacks:
    def test_empty_description_takes_fallback(self):
        c = _make_contract(description="")
        result = c.with_fallbacks(description="fallback desc")
        assert result.description == "fallback desc"

    def test_set_description_not_overridden(self):
        c = _make_contract(description="original")
        result = c.with_fallbacks(description="fallback")
        assert result.description == "original"

    def test_empty_version_takes_fallback(self):
        c = _make_contract(version="")
        result = c.with_fallbacks(version="2.0.0")
        assert result.version == "2.0.0"

    def test_empty_version_defaults_to_1_0_0(self):
        c = _make_contract(version="")
        result = c.with_fallbacks()
        assert result.version == "1.0.0"

    def test_empty_safety_level_takes_fallback(self):
        c = _make_contract(safety_level="")
        result = c.with_fallbacks(safety_level="dangerous")
        assert result.safety_level == "dangerous"

    def test_empty_safety_level_defaults_to_normal(self):
        c = _make_contract(safety_level="")
        result = c.with_fallbacks()
        assert result.safety_level == "normal"

    def test_empty_tags_take_fallback(self):
        c = _make_contract(tags=())
        result = c.with_fallbacks(tags=["patrol", "robot"])
        assert "patrol" in result.tags

    def test_set_tags_not_overridden(self):
        c = _make_contract(tags=("original",))
        result = c.with_fallbacks(tags=["other"])
        assert result.tags == ("original",)

    def test_empty_tags_in_fallback_filtered(self):
        c = _make_contract(tags=())
        result = c.with_fallbacks(tags=["", "valid", ""])
        assert "" not in result.tags
        assert "valid" in result.tags

    def test_confirm_before_execute_or_logic(self):
        c = _make_contract(confirm_before_execute=False)
        result = c.with_fallbacks(confirm_before_execute=True)
        assert result.confirm_before_execute is True

    def test_confirm_before_execute_already_true_stays(self):
        c = _make_contract(confirm_before_execute=True)
        result = c.with_fallbacks(confirm_before_execute=False)
        assert result.confirm_before_execute is True

    def test_returns_new_contract_instance(self):
        c = _make_contract()
        result = c.with_fallbacks(description="new")
        assert result is not c


class TestSkillContractRequestSchema:
    def test_always_has_user_input(self):
        c = _make_contract()
        schema = c.request_schema()
        assert "user_input" in schema["properties"]
        assert "user_input" in schema["required"]

    def test_parameter_added_to_properties(self):
        param = SkillParameter(name="location", type="string", required=True)
        c = _make_contract(parameters=(param,))
        schema = c.request_schema()
        assert "location" in schema["properties"]

    def test_required_parameter_in_required_list(self):
        param = SkillParameter(name="location", type="string", required=True)
        c = _make_contract(parameters=(param,))
        schema = c.request_schema()
        assert "location" in schema["required"]

    def test_optional_parameter_not_in_required_list(self):
        param = SkillParameter(name="comment", type="string", required=False)
        c = _make_contract(parameters=(param,))
        schema = c.request_schema()
        assert "comment" not in schema["required"]

    def test_additional_properties_false(self):
        c = _make_contract()
        schema = c.request_schema()
        assert schema["additionalProperties"] is False

    def test_type_is_object(self):
        c = _make_contract()
        assert c.request_schema()["type"] == "object"


class TestSkillContractOpenApiPathItem:
    def test_has_post_key(self):
        c = _make_contract()
        item = c.openapi_path_item()
        assert "post" in item

    def test_summary_contains_name(self):
        c = _make_contract("my_skill")
        item = c.openapi_path_item()
        assert "my_skill" in item["post"]["summary"]

    def test_tags_contains_skills(self):
        c = _make_contract()
        item = c.openapi_path_item()
        assert "Skills" in item["post"]["tags"]

    def test_request_body_is_required(self):
        c = _make_contract()
        item = c.openapi_path_item()
        assert item["post"]["requestBody"]["required"] is True

    def test_200_response_present(self):
        c = _make_contract()
        item = c.openapi_path_item()
        assert "200" in item["post"]["responses"]

    def test_x_askme_skill_metadata(self):
        c = _make_contract("nav", version="2.0.0", safety_level="dangerous")
        item = c.openapi_path_item()
        meta = item["post"]["x-askme-skill"]
        assert meta["name"] == "nav"
        assert meta["version"] == "2.0.0"
        assert meta["safety_level"] == "dangerous"


class TestSkillContractSummary:
    def test_has_required_keys(self):
        c = _make_contract("patrol")
        s = c.summary()
        for key in ("name", "description", "version", "safety_level",
                    "execution", "tags", "parameter_count", "contract_source"):
            assert key in s

    def test_parameter_count_correct(self):
        params = (
            SkillParameter(name="a", type="string"),
            SkillParameter(name="b", type="integer"),
        )
        c = _make_contract(parameters=params)
        assert c.summary()["parameter_count"] == 2

    def test_tags_is_list(self):
        c = _make_contract(tags=("robot", "patrol"))
        s = c.summary()
        assert isinstance(s["tags"], list)
        assert "robot" in s["tags"]


# ── SkillContractRegistry ─────────────────────────────────────────────────────

class TestSkillContractRegistry:
    def test_register_and_get(self):
        reg = SkillContractRegistry()
        c = _make_contract("nav")
        reg.register(c)
        assert reg.get("nav") is c

    def test_get_nonexistent_returns_none(self):
        reg = SkillContractRegistry()
        assert reg.get("ghost") is None

    def test_all_returns_sorted(self):
        reg = SkillContractRegistry()
        reg.register(_make_contract("zebra"))
        reg.register(_make_contract("alpha"))
        reg.register(_make_contract("middle"))
        names = [c.name for c in reg.all()]
        assert names == sorted(names)

    def test_register_returns_contract(self):
        reg = SkillContractRegistry()
        c = _make_contract("nav")
        result = reg.register(c)
        assert result is c

    def test_overwrite_existing(self):
        reg = SkillContractRegistry()
        c1 = _make_contract("nav", description="old")
        c2 = _make_contract("nav", description="new")
        reg.register(c1)
        reg.register(c2)
        assert reg.get("nav").description == "new"


# ── skill_contract decorator ──────────────────────────────────────────────────

class TestSkillContractDecorator:
    def test_decorator_registers_contract(self):
        @skill_contract(name="test_deco_skill_unique_xyz", description="A test")
        def my_func():
            return "ok"

        from askme.skills.contracts import registered_skill_contracts
        contracts = registered_skill_contracts()
        assert "test_deco_skill_unique_xyz" in contracts

    def test_decorated_func_has_contract_attr(self):
        @skill_contract(name="test_attr_skill_unique_abc", description="B")
        def another_func():
            pass

        assert hasattr(another_func, "__askme_skill_contract__")

    def test_decorated_func_still_callable(self):
        @skill_contract(name="test_callable_skill_unique_qrs")
        def echo(x):
            return x

        assert echo(42) == 42


# ── build_skills_openapi ──────────────────────────────────────────────────────

class TestBuildSkillsOpenapi:
    def test_openapi_version(self):
        doc = build_skills_openapi([])
        assert doc["openapi"] == "3.1.0"

    def test_info_section(self):
        doc = build_skills_openapi([], title="My API", version="2.0")
        assert doc["info"]["title"] == "My API"
        assert doc["info"]["version"] == "2.0"

    def test_skills_list_path_always_present(self):
        doc = build_skills_openapi([])
        assert "/api/v1/skills" in doc["paths"]

    def test_skill_execute_path_added(self):
        c = _make_contract("patrol")
        doc = build_skills_openapi([c])
        assert "/api/v1/skills/patrol/execute" in doc["paths"]

    def test_paths_sorted_alphabetically(self):
        contracts = [_make_contract("zebra"), _make_contract("alpha")]
        doc = build_skills_openapi(contracts)
        execute_paths = [
            p for p in doc["paths"] if "execute" in p
        ]
        assert execute_paths == sorted(execute_paths)

    def test_empty_contracts_no_execute_paths(self):
        doc = build_skills_openapi([])
        execute_paths = [p for p in doc["paths"] if "execute" in p]
        assert execute_paths == []

    def test_tags_section_present(self):
        doc = build_skills_openapi([])
        assert "tags" in doc
        tag_names = [t["name"] for t in doc["tags"]]
        assert "Skills" in tag_names
