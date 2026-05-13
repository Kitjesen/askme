from askme.pipeline.persona import persona_from_brain_config


def test_persona_generates_customer_configurable_identity() -> None:
    persona = persona_from_brain_config({
        "persona": {
            "robot_name": "Ranger",
            "product_name": "园区服务平台",
            "customer_name": "示范园区",
            "role": "巡检机器狗",
            "ownership_label": "",
        }
    })

    prompt = persona.build_system_prompt()
    seed = persona.build_prompt_seed()

    assert "Ranger" in prompt
    assert "示范园区" in prompt
    assert "不要主动声明厂商或归属" in prompt
    assert "穹沛科技" not in prompt
    assert "语言模型" in prompt
    assert seed[0]["role"] == "user"
    assert "示范园区" in seed[0]["content"]


def test_persona_allows_explicit_ownership_label() -> None:
    persona = persona_from_brain_config({
        "persona": {
            "robot_name": "Thunder",
            "ownership_label": "由客户自有运维体系管理",
        }
    })

    prompt = persona.build_system_prompt()

    assert "归属口径：由客户自有运维体系管理" in prompt
