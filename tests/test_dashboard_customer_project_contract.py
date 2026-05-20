from __future__ import annotations

from pathlib import Path

DASHBOARD_APP = Path("askme/static/dashboard/app.js")
DASHBOARD_CSS = Path("askme/static/dashboard/app.css")


def _dashboard_source() -> str:
    return DASHBOARD_APP.read_text(encoding="utf-8")


def _function_body(source: str, name: str) -> str:
    start = source.index(f"function {name}")
    end = source.find("\nfunction ", start + 1)
    return source[start:] if end == -1 else source[start:end]


def test_dashboard_customer_project_visible_copy_is_valid_utf8_chinese() -> None:
    source = _dashboard_source()
    customer_project_slice = source[
        source.index("function renderCustomerProjectCreateResult") :
        source.index("function renderCustomerProjectCustomerSignoff")
    ]

    assert "客户项目创建失败" in customer_project_slice
    assert "客户现场验收清单" in customer_project_slice
    assert "系统自动采信" in customer_project_slice
    assert "现场验收证据" in customer_project_slice
    assert "现场凭证" in customer_project_slice

    mojibake_fragments = [
        "瀹㈡埛",
        "鐜板満",
        "楠屾敹",
        "琛ラ綈",
        "绯荤粺鑷",
        "\ufffd",
    ]
    for fragment in mojibake_fragments:
        assert fragment not in customer_project_slice


def test_dashboard_onsite_evidence_renderer_has_no_unreachable_duplicate_return() -> None:
    body = _function_body(_dashboard_source(), "renderCustomerProjectOnsiteEvidence")

    assert body.count("return `") == 2
    assert "missing ${esc" not in body
    assert '|| "receipt"' not in body
    assert "现场凭证" in body


def test_dashboard_acceptance_report_uses_customer_readable_evidence_fallbacks() -> None:
    body = _function_body(_dashboard_source(), "renderCustomerProjectAcceptanceReport")

    assert '|| "onsite evidence"' not in body
    assert "现场证据" in body


def test_dashboard_acceptance_report_surfaces_device_onboarding_gate() -> None:
    source = _dashboard_source()
    body = _function_body(source, "renderCustomerProjectAcceptanceReport")
    css = DASHBOARD_CSS.read_text(encoding="utf-8")

    assert "renderProjectDeviceOnboardingAcceptance(payload)" in body
    assert "field_device_onboarding" in source
    assert "data-project-device-onboarding" in source
    assert "project-device-onboarding" in css
    assert "project-device-onboarding-metrics" in css


def test_dashboard_projects_page_renders_customer_delivery_chain() -> None:
    source = _dashboard_source()
    body = _function_body(source, "renderProjectGoldenPathWorkbench")
    css = DASHBOARD_CSS.read_text(encoding="utf-8")

    assert "renderProjectDeliveryChain(workbench.delivery_chain)" in body
    assert "function renderProjectDeliveryChain" in source
    assert "data-project-delivery-chain" in source
    assert "project-delivery-chain" in css
    assert "project-delivery-chain-steps" in css


def test_dashboard_customer_visible_fallbacks_do_not_use_raw_english_placeholders() -> None:
    source = _dashboard_source()

    raw_placeholders = [
        "Site profile status unknown.",
        "audit status",
        "ASR provider",
        "LLM provider",
        "TTS provider",
    ]
    for placeholder in raw_placeholders:
        assert placeholder not in source


def test_dashboard_has_customer_readable_scenario_product_page() -> None:
    source = _dashboard_source()
    body = _function_body(source, "renderScenarios")

    assert "/dashboard/scenarios" in source
    assert "场景验收矩阵" in source
    assert "产品页不是广告页，是客户能逐条验收的场景清单" in body
    assert "问路" in source
    assert "带路" in source
    assert "违停" in source
    assert "火灾" in source
    assert "垃圾桶" in source
    assert "陌生人" in source
    assert "恶意挡路" in source
    assert "/api/scenario-intents/preview" in source


def test_dashboard_overview_has_customer_interface_principles() -> None:
    source = _dashboard_source()
    body = _function_body(source, "renderCustomerInterfacePrinciples")

    assert "客户接口原则" in body
    assert "每个页面都要让客户知道" in body
    assert "能做什么、依据什么、风险在哪里" in body
    assert "对话入口" in source
    assert "现场事件" in source
    assert "空间问路" in source
    assert "知识库" in source
    assert "交付审计" in source
