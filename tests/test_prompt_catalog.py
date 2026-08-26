from __future__ import annotations

import pytest

from askme.prompts import PromptTemplate, get_prompt_template


def test_product_prompt_catalog_returns_scenario_template() -> None:
    template = get_prompt_template("park_wayfinding")

    assert isinstance(template, PromptTemplate)
    assert template.name == "park_wayfinding"
    assert template.grounded is True
    assert "园区" in template.system


def test_product_prompt_catalog_rejects_unknown_template() -> None:
    with pytest.raises(KeyError, match="Unknown prompt template"):
        get_prompt_template("open_domain_chat")


@pytest.mark.parametrize(
    "name",
    [
        "field_incident_narration",
        "park_wayfinding",
        "task_handoff_summary",
    ],
)
def test_product_prompts_require_a_short_semantic_first_clause(name: str) -> None:
    template = get_prompt_template(name)

    assert "首句必须是10字以内" in template.system
    assert "10字以内" in template.system
    assert "强结束标点" in template.system
    assert "不先寒暄" in template.system
