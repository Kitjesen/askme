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
