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
    assert "首句必须是10字以内的有效结论、动作状态或澄清问题" in prompt
    assert "安全告警、拒绝和澄清不先寒暄" in prompt
    assert seed[0]["role"] == "user"
    assert "示范园区" in seed[0]["content"]
    assert "首句必须是10字以内的有效结论、动作状态或澄清问题" in seed[0]["content"]
    assert "首句必须是10字以内的有效结论/状态/问题" in persona.build_user_prefix()
    assert "[SILENT]" in prompt
    assert "地点、路线或活动查询都视为在跟你说话" in prompt


def test_persona_allows_explicit_ownership_label() -> None:
    persona = persona_from_brain_config({
        "persona": {
            "robot_name": "Thunder",
            "ownership_label": "由客户自有运维体系管理",
        }
    })

    prompt = persona.build_system_prompt()

    assert "归属口径：由客户自有运维体系管理" in prompt


def test_persona_uses_brand_neutral_defaults_when_config_values_are_blank() -> None:
    persona = persona_from_brain_config({
        "persona": {
            "robot_name": "   ",
            "product_name": "",
            "operator_audience": None,
            "role": " ",
            "speaking_style": "",
            "max_reply_chars": "",
        }
    })

    prompt = persona.build_system_prompt()
    prefix = persona.build_user_prefix()

    assert persona.robot_name == "现场机器人"
    assert persona.product_name == "现场任务平台"
    assert persona.role == "园区巡检与服务机器人"
    assert persona.max_reply_chars == 80
    assert "现场机器人" in prompt
    assert "不要主动声明厂商或归属" in prompt
    assert "80字以内" in prefix


def test_persona_clamps_invalid_reply_length_for_voice_safety() -> None:
    too_short = persona_from_brain_config({"persona": {"max_reply_chars": 1}})
    too_long = persona_from_brain_config({"persona": {"max_reply_chars": 9999}})
    invalid = persona_from_brain_config({"persona": {"max_reply_chars": "not-a-number"}})

    assert too_short.max_reply_chars == 20
    assert too_long.max_reply_chars == 300
    assert invalid.max_reply_chars == 80
