from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_layout_doc_tracks_confusing_roots() -> None:
    text = (ROOT / "docs" / "REPOSITORY_LAYOUT.md").read_text(encoding="utf-8")

    assert "Same-level folders do not mean same-level architecture authority" in text
    for path in (
        "askme/",
        "scripts/",
        "deploy/",
        "docker/",
        ".zeroclaw/",
        "native/",
        "prompts/",
        "data/",
        "models/",
        "artifacts/",
    ):
        assert f"`{path}`" in text

    assert "`video-lab/`" not in text
    assert "`config/`" not in text


def test_package_and_layout_guides_point_to_product_architecture_spine() -> None:
    guide_paths = (
        ROOT / "askme" / "README.md",
        ROOT / "askme" / "CODE_MAP.md",
        ROOT / "docs" / "REPOSITORY_LAYOUT.md",
    )

    for guide_path in guide_paths:
        text = guide_path.read_text(encoding="utf-8")
        for token in (
            "`docs/PRODUCT_REQUIREMENTS.md`",
            "`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`",
            "`docs/PRODUCT_ARCHITECTURE_TRACE.md`",
            "`docs/DEMAND_EVIDENCE_LEDGER.md`",
            "Field Delivery Domain",
            "Product/Admin/Platform/Internal",
            "Runtime / Safety / Hardware",
            "customer signoff != production readiness",
        ):
            assert token in text, guide_path

        for stale_phrase in (
            "机器人现场任务与智能交互平台",
            "现场任务平台",
            "现场服务终端",
            "完全无人值守生产上线",
        ):
            assert stale_phrase not in text, guide_path


def test_multi_agent_guides_use_product_architecture_spine_before_lane_assignment() -> None:
    guide_paths = (
        ROOT / "docs" / "MODULE_OWNERSHIP.md",
        ROOT / "docs" / "MULTI_AGENT_WORKFLOW.md",
        ROOT / "docs" / "MULTI_AGENT_REFACTOR_LANES.md",
    )

    for guide_path in guide_paths:
        text = guide_path.read_text(encoding="utf-8")
        for token in (
            "`docs/PRODUCT_REQUIREMENTS.md`",
            "`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`",
            "`docs/PRODUCT_ARCHITECTURE_TRACE.md`",
            "`docs/DEMAND_EVIDENCE_LEDGER.md`",
            "Field Delivery Domain",
            "Product/Admin/Platform/Internal",
            "Runtime / Safety / Hardware",
            "customer signoff != production readiness",
        ):
            assert token in text, guide_path


def test_architecture_v2_keeps_product_positioning_boundary() -> None:
    text = (ROOT / "docs" / "ARCHITECTURE_V2.md").read_text(encoding="utf-8")

    assert "现场运营交付中台" in text
    assert "机器人身体" not in text
    assert "Dashboard 只调用 Product 表面接口" not in text


def test_architecture_entry_points_to_product_architecture_spine() -> None:
    text = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")

    for token in (
        "现场运营交付中台",
        "机器人方案商/集成商交付中台",
        "`docs/PRODUCT_REQUIREMENTS.md`",
        "`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`",
        "`docs/PRODUCT_ARCHITECTURE_TRACE.md`",
        "`docs/DEMAND_EVIDENCE_LEDGER.md`",
        "Field Delivery Domain",
        "Product/Admin/Platform/Internal",
        "Runtime / Safety / Hardware",
        "customer signoff != production readiness",
        "不替代底盘控制",
        "不承诺无人值守生产上线",
    ):
        assert token in text


def test_root_readme_points_to_product_requirements_and_architecture_spine() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    for token in (
        "机器人方案商/集成商交付中台",
        "现场运营交付中台",
        "`docs/PRODUCT_REQUIREMENTS.md`",
        "`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`",
        "`docs/PRODUCT_ARCHITECTURE_TRACE.md`",
        "`docs/DEMAND_EVIDENCE_LEDGER.md`",
        "Field Delivery Domain",
        "Runtime / Safety / Hardware",
        "customer signoff != production readiness",
        "不替代底盘控制",
        "不承诺无人值守生产上线",
    ):
        assert token in text

    assert "机器人身体" not in text
    assert "现场任务平台" not in text


def test_market_research_prioritizes_field_delivery_demand() -> None:
    text = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")

    assert "第二版公开资料调研 + 仓库产品能力映射" in text
    assert "机器人方案商 / 集成商 / 交付团队" in text
    assert "把一次性机器人 Demo 变成可复制、可验收的客户项目交付包" in text
    assert "定制项目脚本/一次性 Demo" in text
    assert "试点验收包 / acceptance dossier" in text
    assert "底盘控制、VLA 运动控制、完全无人值守承诺" in text
    assert "机器人聊天工具" in text
    assert "机器人底盘控制系统" in text


def test_solution_provider_icp_connects_demand_to_architecture() -> None:
    icp = (ROOT / "docs" / "SOLUTION_PROVIDER_ICP.md").read_text(encoding="utf-8")
    market = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    architecture_v2 = (ROOT / "docs" / "ARCHITECTURE_V2.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "机器人方案商/集成商交付中台",
        "Demo-to-pilot",
        "acceptance dossier",
        "Field Delivery Domain",
        "`askme/pipeline/field`",
        "Runtime / Safety / Hardware",
        "不是通用聊天机器人",
        "不是底盘控制系统",
    ):
        assert token in icp

    for doc_text in (market, architecture, architecture_v2, docs_index):
        assert "`docs/SOLUTION_PROVIDER_ICP.md`" in doc_text


def test_product_architecture_trace_maps_icp_requirements_to_owned_surfaces() -> None:
    trace = (ROOT / "docs" / "PRODUCT_ARCHITECTURE_TRACE.md").read_text(encoding="utf-8")
    icp = (ROOT / "docs" / "SOLUTION_PROVIDER_ICP.md").read_text(encoding="utf-8")
    architecture_v2 = (ROOT / "docs" / "ARCHITECTURE_V2.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for requirement_id in ("R1", "R2", "R3", "R4", "R5", "R6", "R7"):
        assert f"| {requirement_id} " in trace

    for token in (
        "`docs/SOLUTION_PROVIDER_ICP.md`",
        "Field Delivery Domain",
        "`askme/pipeline/field`",
        "`askme/api/routes/field_*`",
        "`askme/api/services/field_*`",
        "`askme/static`",
        "Runtime / Safety / Hardware",
        "Product/Admin/Platform/Internal",
        "customer signoff != production readiness",
        "tests/test_field_operations.py",
        "tests/test_dashboard_http.py",
        "tests/test_product_launch_readiness.py",
    ):
        assert token in trace

    for doc_text in (icp, architecture_v2, docs_index):
        assert "`docs/PRODUCT_ARCHITECTURE_TRACE.md`" in doc_text


def test_competitive_replacement_matrix_keeps_positioning_boundaries() -> None:
    matrix = (ROOT / "docs" / "COMPETITIVE_REPLACEMENT_MATRIX.md").read_text(encoding="utf-8")
    market = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")
    icp = (ROOT / "docs" / "SOLUTION_PROVIDER_ICP.md").read_text(encoding="utf-8")
    skill_research = (ROOT / "docs" / "GITHUB_SKILL_RESEARCH.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "人工运营 + 微信/Excel",
        "OEM fleet/app",
        "VMS / AI 告警",
        "CMMS / 工单",
        "定制项目脚本 / 一次性 Demo",
        "通用 LLM Agent 平台",
        "AskMe",
        "方案商交付中台",
        "Field Delivery Domain",
        "Runtime / Safety / Hardware",
        "customer signoff != production readiness",
        "不替代底盘控制",
    ):
        assert token in matrix

    for doc_text in (market, icp, skill_research, docs_index):
        assert "`docs/COMPETITIVE_REPLACEMENT_MATRIX.md`" in doc_text


def test_github_skill_research_uses_verified_skill_sources() -> None:
    skill_research = (ROOT / "docs" / "GITHUB_SKILL_RESEARCH.md").read_text(
        encoding="utf-8"
    )

    for token in (
        "https://github.com/anthropics/skills",
        "template/SKILL.md",
        "https://github.com/openai/skills",
        "skills/.curated/notion-research-documentation/SKILL.md",
        "skills/.curated/security-threat-model/SKILL.md",
        "skills/.curated/pdf/SKILL.md",
        "Mehdibargach/claude-code-pm-skills",
        "skills/market-sizing/SKILL.md",
        "skills/competitor-scan/SKILL.md",
        "skills/user-interview-prep/SKILL.md",
        "skills/feedback-analyzer/SKILL.md",
        "skills/persona/SKILL.md",
        "deanpeters/Product-Manager-Skills",
        "skills/jobs-to-be-done/SKILL.md",
        "skills/prd-development/SKILL.md",
        "skills/positioning-statement/SKILL.md",
        "mohitagw15856/pm-claude-skills",
        "plugins/pm-discovery/skills/assumption-mapper/SKILL.md",
        "plugins/pm-discovery/skills/user-interview-synthesis/SKILL.md",
        "plugins/pm-engineering/skills/architecture-decision-record/SKILL.md",
        "w95/awesome-claude-corporate-skills",
        "04-marketing/market-research/SKILL.md",
        "09-product-management/user-research-synthesizer/SKILL.md",
        "12-procurement-supply-chain/vendor-evaluation/SKILL.md",
        "08-it-engineering/software-architecture/SKILL.md",
        "wshobson/agents",
        "docs/plugin-eval.md",
        "GEMINI_API_KEY",
        "不直接安装",
        "`docs/INTERVIEW_GUIDE_SOLUTION_PROVIDER.md`",
        "`docs/COMPETITIVE_REPLACEMENT_MATRIX.md`",
        "`docs/PRODUCT_REQUIREMENTS.md`",
        "`docs/DEMAND_EVIDENCE_LEDGER.md`",
    ):
        assert token in skill_research


def test_solution_provider_interview_guide_validates_icp_and_architecture_assumptions() -> None:
    guide = (ROOT / "docs" / "INTERVIEW_GUIDE_SOLUTION_PROVIDER.md").read_text(encoding="utf-8")
    market = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")
    icp = (ROOT / "docs" / "SOLUTION_PROVIDER_ICP.md").read_text(encoding="utf-8")
    matrix = (ROOT / "docs" / "COMPETITIVE_REPLACEMENT_MATRIX.md").read_text(encoding="utf-8")
    trace = (ROOT / "docs" / "PRODUCT_ARCHITECTURE_TRACE.md").read_text(encoding="utf-8")
    skill_research = (ROOT / "docs" / "GITHUB_SKILL_RESEARCH.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "20-30 人",
        "机器人方案商/集成商",
        "Demo-to-pilot",
        "R1-R7",
        "acceptance dossier",
        "替代物",
        "VMS",
        "CMMS",
        "OEM fleet",
        "付费意愿",
        "customer signoff != production readiness",
        "不承诺无人值守生产上线",
    ):
        assert token in guide

    for doc_text in (market, icp, matrix, trace, skill_research, docs_index):
        assert "`docs/INTERVIEW_GUIDE_SOLUTION_PROVIDER.md`" in doc_text


def test_industry_scenario_cards_keep_template_scope_narrow_and_verifiable() -> None:
    cards = (ROOT / "docs" / "INDUSTRY_SCENARIO_DEMAND_CARDS.md").read_text(encoding="utf-8")
    market = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")
    icp = (ROOT / "docs" / "SOLUTION_PROVIDER_ICP.md").read_text(encoding="utf-8")
    trace = (ROOT / "docs" / "PRODUCT_ARCHITECTURE_TRACE.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "INTERVIEW_GUIDE_SOLUTION_PROVIDER.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "每类只保留 3 个高价值场景",
        "园区",
        "厂区",
        "仓储",
        "景区",
        "PARK-1",
        "FACTORY-1",
        "WAREHOUSE-1",
        "SCENIC-1",
        "R4 场景验收卡",
        "acceptance dossier",
        "Field Delivery Domain",
        "Runtime / Safety / Hardware",
        "不扩展成宽泛行业模板市场",
    ):
        assert token in cards

    for doc_text in (market, icp, trace, guide, docs_index):
        assert "`docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md`" in doc_text


def test_pilot_acceptance_dossier_product_surface_keeps_signoff_and_readiness_separate() -> None:
    surface = (ROOT / "docs" / "PILOT_ACCEPTANCE_DOSSIER_PRODUCT_SURFACE.md").read_text(
        encoding="utf-8"
    )
    market = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")
    product = (ROOT / "docs" / "PRODUCT.md").read_text(encoding="utf-8")
    trace = (ROOT / "docs" / "PRODUCT_ARCHITECTURE_TRACE.md").read_text(encoding="utf-8")
    architecture_v2 = (ROOT / "docs" / "ARCHITECTURE_V2.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "试点验收包产品面",
        "对象目录",
        "场景清单",
        "证据",
        "缺口",
        "责任边界",
        "客户签收不等于生产上线",
        "customer signoff != production readiness",
        "acceptance dossier",
        "Field Delivery Domain",
        "Runtime / Safety / Hardware",
        "Product/Admin/Platform/Internal",
        "`GET /api/field/customer-projects/{identifier}/acceptance-dossier`",
        "`POST /api/field/customer-projects/acceptance-dossier/verify`",
        "`GET /api/field/customer-projects/{identifier}/acceptance-closure`",
        "`GET /api/field/customer-projects/{identifier}/customer-signoff`",
        "`POST /api/field/customer-projects/{identifier}/customer-signoff`",
        "/dashboard/delivery",
        "blocked_uses",
    ):
        assert token in surface

    for doc_text in (market, product, trace, architecture_v2, docs_index):
        assert "`docs/PILOT_ACCEPTANCE_DOSSIER_PRODUCT_SURFACE.md`" in doc_text


def test_external_system_contracts_define_minimum_fields_and_failure_boundaries() -> None:
    contracts = (ROOT / "docs" / "EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md").read_text(
        encoding="utf-8"
    )
    market = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")
    icp = (ROOT / "docs" / "SOLUTION_PROVIDER_ICP.md").read_text(encoding="utf-8")
    trace = (ROOT / "docs" / "PRODUCT_ARCHITECTURE_TRACE.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "INTERVIEW_GUIDE_SOLUTION_PROVIDER.md").read_text(
        encoding="utf-8"
    )
    matrix = (ROOT / "docs" / "COMPETITIVE_REPLACEMENT_MATRIX.md").read_text(
        encoding="utf-8"
    )
    pilot_surface = (
        ROOT / "docs" / "PILOT_ACCEPTANCE_DOSSIER_PRODUCT_SURFACE.md"
    ).read_text(encoding="utf-8")
    architecture_v2 = (ROOT / "docs" / "ARCHITECTURE_V2.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "外部系统最小字段合同",
        "VMS",
        "CMMS",
        "IAM",
        "地图",
        "OEM fleet",
        "通知系统",
        "SIEM/WORM",
        "source_event_id",
        "idempotency_key",
        "tenant_id",
        "customer_id",
        "project_id",
        "site_id",
        "managed_object_id",
        "failure_state",
        "retry_policy",
        "audit_export_id",
        "Field Delivery Domain",
        "Product/Admin/Platform/Internal",
        "Runtime / Safety / Hardware",
        "不能写成一次性同步脚本",
        "不能用默认项目兜底",
    ):
        assert token in contracts

    for doc_text in (
        market,
        icp,
        trace,
        guide,
        matrix,
        pilot_surface,
        architecture_v2,
        docs_index,
    ):
        assert "`docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md`" in doc_text


def test_site_launch_readiness_checklist_keeps_hardware_acceptance_out_of_customer_signoff() -> None:
    checklist = (ROOT / "docs" / "SITE_LAUNCH_READINESS_CHECKLIST.md").read_text(
        encoding="utf-8"
    )
    market = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")
    icp = (ROOT / "docs" / "SOLUTION_PROVIDER_ICP.md").read_text(encoding="utf-8")
    trace = (ROOT / "docs" / "PRODUCT_ARCHITECTURE_TRACE.md").read_text(encoding="utf-8")
    pilot_surface = (
        ROOT / "docs" / "PILOT_ACCEPTANCE_DOSSIER_PRODUCT_SURFACE.md"
    ).read_text(encoding="utf-8")
    product = (ROOT / "docs" / "PRODUCT.md").read_text(encoding="utf-8")
    architecture_v2 = (ROOT / "docs" / "ARCHITECTURE_V2.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "上线前硬件/现场验收 checklist",
        "site_acceptance_checklist",
        "launch_readiness",
        "production readiness",
        "customer signoff != production readiness",
        "客户签收不等于生产上线",
        "Runtime / Safety / Hardware",
        "Field Delivery Domain",
        "Product/Admin/Platform/Internal",
        "fake/sim/shadow/lab/prod",
        "onsite evidence",
        "device ingest",
        "live voice",
        "external notifications",
        "runtime roundtrip",
        "operator review",
        "takeover",
        "rollback",
        "blocked_uses",
        "`GET /api/field/customer-projects/{identifier}/acceptance-closure`",
        "`GET /api/field/customer-projects/{identifier}/acceptance-dossier`",
        "不能用 customer signoff 替代 production readiness",
        "不能把 lab 证据包装成现场验收",
    ):
        assert token in checklist

    for doc_text in (
        market,
        icp,
        trace,
        pilot_surface,
        product,
        architecture_v2,
        docs_index,
    ):
        assert "`docs/SITE_LAUNCH_READINESS_CHECKLIST.md`" in doc_text


def test_pricing_packaging_hypotheses_bind_revenue_units_to_product_facts() -> None:
    pricing = (ROOT / "docs" / "PRICING_PACKAGING_HYPOTHESES.md").read_text(
        encoding="utf-8"
    )
    market = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")
    icp = (ROOT / "docs" / "SOLUTION_PROVIDER_ICP.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "INTERVIEW_GUIDE_SOLUTION_PROVIDER.md").read_text(
        encoding="utf-8"
    )
    trace = (ROOT / "docs" / "PRODUCT_ARCHITECTURE_TRACE.md").read_text(encoding="utf-8")
    pilot_surface = (
        ROOT / "docs" / "PILOT_ACCEPTANCE_DOSSIER_PRODUCT_SURFACE.md"
    ).read_text(encoding="utf-8")
    architecture_v2 = (ROOT / "docs" / "ARCHITECTURE_V2.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "定价包装假设",
        "项目费",
        "站点费",
        "机器人数量",
        "技能包",
        "交付包",
        "RaaS 运营报告",
        "forced choice",
        "Field Delivery Domain",
        "customer project",
        "site_acceptance_checklist",
        "acceptance dossier",
        "customer signoff != production readiness",
        "Runtime / Safety / Hardware",
        "不按底盘控制收费",
        "不把 production readiness 包装成签收加价项",
        "usage evidence",
        "pricing_signal",
    ):
        assert token in pricing

    for doc_text in (
        market,
        icp,
        guide,
        trace,
        pilot_surface,
        architecture_v2,
        docs_index,
    ):
        assert "`docs/PRICING_PACKAGING_HYPOTHESES.md`" in doc_text


def test_scenario_roi_model_binds_industry_value_to_field_delivery_evidence() -> None:
    roi = (ROOT / "docs" / "SCENARIO_ROI_MODEL.md").read_text(encoding="utf-8")
    market = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")
    icp = (ROOT / "docs" / "SOLUTION_PROVIDER_ICP.md").read_text(encoding="utf-8")
    cards = (ROOT / "docs" / "INDUSTRY_SCENARIO_DEMAND_CARDS.md").read_text(
        encoding="utf-8"
    )
    pricing = (ROOT / "docs" / "PRICING_PACKAGING_HYPOTHESES.md").read_text(
        encoding="utf-8"
    )
    trace = (ROOT / "docs" / "PRODUCT_ARCHITECTURE_TRACE.md").read_text(encoding="utf-8")
    architecture_v2 = (ROOT / "docs" / "ARCHITECTURE_V2.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "场景 ROI 模型",
        "园区",
        "厂区",
        "仓储",
        "景区",
        "PARK-1",
        "FACTORY-1",
        "WAREHOUSE-1",
        "SCENIC-1",
        "baseline",
        "target_delta",
        "value metric",
        "payback signal",
        "Field Delivery Domain",
        "usage evidence",
        "pricing_signal",
        "acceptance dossier",
        "customer signoff != production readiness",
        "Runtime / Safety / Hardware",
        "RaaS 运营报告",
        "项目费",
        "站点费",
        "交付包",
        "不把展示价值当 ROI",
        "不把 production readiness 当 ROI",
    ):
        assert token in roi

    for doc_text in (
        market,
        icp,
        cards,
        pricing,
        trace,
        architecture_v2,
        docs_index,
    ):
        assert "`docs/SCENARIO_ROI_MODEL.md`" in doc_text


def test_demand_evidence_ledger_keeps_hypotheses_separate_from_validated_claims() -> None:
    ledger = (ROOT / "docs" / "DEMAND_EVIDENCE_LEDGER.md").read_text(encoding="utf-8")
    market = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")
    icp = (ROOT / "docs" / "SOLUTION_PROVIDER_ICP.md").read_text(encoding="utf-8")
    guide = (ROOT / "docs" / "INTERVIEW_GUIDE_SOLUTION_PROVIDER.md").read_text(
        encoding="utf-8"
    )
    roi = (ROOT / "docs" / "SCENARIO_ROI_MODEL.md").read_text(encoding="utf-8")
    pricing = (ROOT / "docs" / "PRICING_PACKAGING_HYPOTHESES.md").read_text(
        encoding="utf-8"
    )
    trace = (ROOT / "docs" / "PRODUCT_ARCHITECTURE_TRACE.md").read_text(encoding="utf-8")
    architecture_v2 = (ROOT / "docs" / "ARCHITECTURE_V2.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "需求证据台账",
        "evidence_id",
        "source_type",
        "interview",
        "pilot",
        "artifact",
        "quote",
        "sample_count",
        "hypothesis_status",
        "research_pending",
        "validated",
        "contradicted",
        "confidence",
        "P0",
        "R1",
        "R4",
        "R7",
        "pricing_signal",
        "baseline",
        "target_delta",
        "acceptance dossier",
        "customer signoff != production readiness",
        "Field Delivery Domain",
        "Runtime / Safety / Hardware",
        "不能把访谈意向当已验证需求",
        "不能把单个样本升级成架构不变量",
        "redaction_required",
        "no secrets/no PII",
    ):
        assert token in ledger

    for doc_text in (
        market,
        icp,
        guide,
        roi,
        pricing,
        trace,
        architecture_v2,
        docs_index,
    ):
        assert "`docs/DEMAND_EVIDENCE_LEDGER.md`" in doc_text


def test_product_requirements_spine_connects_demand_evidence_to_architecture_contracts() -> None:
    requirements = (ROOT / "docs" / "PRODUCT_REQUIREMENTS.md").read_text(
        encoding="utf-8"
    )
    market = (ROOT / "docs" / "MARKET_RESEARCH.md").read_text(encoding="utf-8")
    icp = (ROOT / "docs" / "SOLUTION_PROVIDER_ICP.md").read_text(encoding="utf-8")
    product = (ROOT / "docs" / "PRODUCT.md").read_text(encoding="utf-8")
    trace = (ROOT / "docs" / "PRODUCT_ARCHITECTURE_TRACE.md").read_text(encoding="utf-8")
    architecture_v2 = (ROOT / "docs" / "ARCHITECTURE_V2.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "产品需求主干",
        "P0",
        "机器人方案商/集成商交付中台",
        "Demo-to-pilot",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "evidence_id",
        "hypothesis_status",
        "validated",
        "research_pending",
        "baseline",
        "target_delta",
        "pricing_signal",
        "acceptance dossier",
        "site_acceptance_checklist",
        "customer signoff != production readiness",
        "Field Delivery Domain",
        "`askme/pipeline/field`",
        "Product/Admin/Platform/Internal",
        "Runtime / Safety / Hardware",
        "不是通用聊天机器人",
        "不替代底盘控制",
        "不能把 research_pending 升级为 PRD 承诺",
        "`docs/DEMAND_EVIDENCE_LEDGER.md`",
        "`docs/PRODUCT_ARCHITECTURE_TRACE.md`",
        "`docs/SCENARIO_ROI_MODEL.md`",
        "`docs/PRICING_PACKAGING_HYPOTHESES.md`",
        "`docs/SITE_LAUNCH_READINESS_CHECKLIST.md`",
    ):
        assert token in requirements

    for doc_text in (market, icp, product, trace, architecture_v2, docs_index):
        assert "`docs/PRODUCT_REQUIREMENTS.md`" in doc_text


def test_product_manual_uses_prd_architecture_positioning_boundary() -> None:
    product = (ROOT / "docs" / "PRODUCT.md").read_text(encoding="utf-8")

    for token in (
        "机器人方案商/集成商交付中台",
        "现场运营交付中台",
        "Demo-to-pilot",
        "acceptance dossier",
        "`docs/PRODUCT_REQUIREMENTS.md`",
        "`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`",
        "`docs/PRODUCT_ARCHITECTURE_TRACE.md`",
        "`docs/DEMAND_EVIDENCE_LEDGER.md`",
        "Field Delivery Domain",
        "Product/Admin/Platform/Internal",
        "Runtime / Safety / Hardware",
        "customer signoff != production readiness",
        "不替代底盘控制",
        "不承诺无人值守生产上线",
    ):
        assert token in product

    for stale_phrase in (
        "机器人现场任务与智能交互平台",
        "现场任务平台",
        "现场服务终端",
        "完全无人值守生产上线",
    ):
        assert stale_phrase not in product


def test_software_architecture_blueprint_maps_bounded_contexts_to_product_requirements() -> None:
    blueprint = (ROOT / "docs" / "SOFTWARE_ARCHITECTURE_BLUEPRINT.md").read_text(
        encoding="utf-8"
    )
    product_requirements = (ROOT / "docs" / "PRODUCT_REQUIREMENTS.md").read_text(
        encoding="utf-8"
    )
    trace = (ROOT / "docs" / "PRODUCT_ARCHITECTURE_TRACE.md").read_text(
        encoding="utf-8"
    )
    architecture = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    architecture_v2 = (ROOT / "docs" / "ARCHITECTURE_V2.md").read_text(
        encoding="utf-8"
    )
    module_ownership = (ROOT / "docs" / "MODULE_OWNERSHIP.md").read_text(
        encoding="utf-8"
    )
    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for token in (
        "高级软件架构蓝图",
        "bounded contexts",
        "Context map",
        "System Context",
        "Container / Package Map",
        "Architecture decision gates",
        "`docs/PRODUCT_REQUIREMENTS.md`",
        "`docs/PRODUCT_ARCHITECTURE_TRACE.md`",
        "`docs/MODULE_OWNERSHIP.md`",
        "Field Delivery Domain",
        "Interaction & Knowledge Domain",
        "Runtime Handoff Domain",
        "Safety / Hardware Boundary",
        "Integration Contracts",
        "Observability / Audit",
        "Capability Governance",
        "Product/Admin/Platform/Internal",
        "`askme/pipeline/field`",
        "`askme/api/routes/field_*`",
        "`askme/api/services/field_*`",
        "`askme/runtime`",
        "`askme/robot`",
        "`askme/providers`",
        "`askme/ports`",
        "`askme/memory`",
        "`askme/skills`",
        "`askme/static`",
        "customer signoff != production readiness",
        "Runtime / Safety / Hardware owns execution truth",
        "No default-project fallback",
        "No Dashboard-only fact source",
        "No raw hardware control",
        "evidence_id",
        "hypothesis_status",
        "tests/test_six_layer_package_boundaries.py",
        "tests/test_product_launch_readiness.py",
    ):
        assert token in blueprint

    for doc_text in (
        product_requirements,
        trace,
        architecture,
        architecture_v2,
        module_ownership,
        docs_index,
    ):
        assert "`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`" in doc_text


def test_module_ownership_tracks_field_delivery_domain_boundary() -> None:
    text = (ROOT / "docs" / "MODULE_OWNERSHIP.md").read_text(encoding="utf-8")
    coordination_text = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    workflow_text = (ROOT / "docs" / "MULTI_AGENT_WORKFLOW.md").read_text(encoding="utf-8")
    layout_text = (ROOT / "docs" / "REPOSITORY_LAYOUT.md").read_text(encoding="utf-8")
    refactor_text = (ROOT / "docs" / "MULTI_AGENT_REFACTOR_LANES.md").read_text(encoding="utf-8")
    package_text = (ROOT / "askme" / "README.md").read_text(encoding="utf-8")

    active_boundary_docs = (text, coordination_text, workflow_text, layout_text, refactor_text, package_text)
    for doc_text in active_boundary_docs:
        assert "Field Delivery Domain" in doc_text
        assert "Product workflows" not in doc_text
        assert "Field workflows / tests / compatibility" not in doc_text
    assert "现场运营交付中台" in text
    assert "现场运营交付中台" in package_text
    assert "`askme/pipeline/field`" in text
    assert "`askme/api/routes/field_*`" in text
    assert "transport composition" in text


def test_askme_package_readme_classifies_every_top_level_package_dir() -> None:
    package_root = ROOT / "askme"
    text = (package_root / "README.md").read_text(encoding="utf-8")

    ignored = {"__pycache__"}
    directories = {
        path.name
        for path in package_root.iterdir()
        if path.is_dir() and path.name not in ignored
    }

    missing = sorted(name for name in directories if f"`{name}/`" not in text)

    assert "Product Composition" in text
    assert "Voice And Interaction" in text
    assert "Contracts And Boundaries" in text
    assert "Provider And Edge Implementations" in text
    assert "External Surfaces" in text
    assert missing == []


def test_compatibility_and_parking_dirs_are_marked_as_not_new_code_homes() -> None:
    text = (ROOT / "askme" / "README.md").read_text(encoding="utf-8")

    for marker in (
        "`compat/` | compatibility only",
        "`interaction/` | compatibility only",
        "`data/` | parking only",
    ):
        assert marker in text


def test_legacy_interaction_package_stays_facade_only() -> None:
    interaction_root = ROOT / "askme" / "interaction"
    violations: list[str] = []

    for path in sorted(interaction_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            if isinstance(node, ast.ImportFrom) and (
                node.module == "askme.robot_interaction"
                or node.module.startswith("askme.robot_interaction.")
            ):
                continue
            if isinstance(node, ast.Assign) and all(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                continue
            violations.append(f"{path.relative_to(ROOT)} contains {type(node).__name__}")

    assert violations == []


def test_non_compat_code_does_not_import_legacy_interaction_facade() -> None:
    allowed = {
        ROOT / "tests" / "test_package_migration_compat.py",
        ROOT / "tests" / "test_six_layer_package_boundaries.py",
    }
    violations: list[str] = []

    for root in (ROOT / "askme", ROOT / "tests"):
        for path in sorted(root.rglob("*.py")):
            if path in allowed or (ROOT / "askme" / "interaction") in path.parents:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    if module_name == "askme.interaction" or module_name.startswith(
                        "askme.interaction."
                    ):
                        violations.append(f"{path.relative_to(ROOT)} imports {module_name}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "askme.interaction" or alias.name.startswith(
                            "askme.interaction."
                        ):
                            violations.append(f"{path.relative_to(ROOT)} imports {alias.name}")

    assert violations == []


def test_non_compat_code_does_not_import_llm_intent_router_alias() -> None:
    allowed = {
        ROOT / "tests" / "test_package_migration_compat.py",
        ROOT / "tests" / "test_six_layer_package_boundaries.py",
    }
    violations: list[str] = []

    for root in (ROOT / "askme", ROOT / "scripts", ROOT / "tests"):
        for path in sorted(root.rglob("*.py")):
            if path in allowed:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    if module_name == "askme.llm.intent_router":
                        violations.append(f"{path.relative_to(ROOT)} imports {module_name}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "askme.llm.intent_router":
                            violations.append(f"{path.relative_to(ROOT)} imports {alias.name}")

    assert violations == []


def test_prompt_assets_stay_outside_prompt_registry_package() -> None:
    package_prompt_docs = {path.name for path in (ROOT / "askme" / "prompts").glob("*.md")}

    assert (ROOT / "prompts" / "SOUL.md").is_file()
    assert package_prompt_docs <= {"README.md"}


def test_package_local_data_is_not_importable_runtime_state() -> None:
    package_data = ROOT / "askme" / "data"

    assert not (package_data / "__init__.py").exists()
    assert list(package_data.glob("*.py")) == []
    assert (ROOT / "data").is_dir()


def test_cognition_owner_subpackages_are_not_empty_facades() -> None:
    cognition_root = ROOT / "askme" / "cognition"
    expected = {
        "memory": {"working_memory.py"},
        "perception": {"active_perception.py", "perception_sync.py"},
        "planning": {"planner.py", "planning_session.py"},
        "world": {"world_state.py"},
    }

    for package, files in expected.items():
        existing = {path.name for path in (cognition_root / package).glob("*.py")}
        assert files <= existing


def test_cognition_legacy_root_modules_alias_owner_subpackages() -> None:
    import importlib

    pairs = {
        "askme.cognition.active_perception": "askme.cognition.perception.active_perception",
        "askme.cognition.perception_sync": "askme.cognition.perception.perception_sync",
        "askme.cognition.planner": "askme.cognition.planning.planner",
        "askme.cognition.planning_session": "askme.cognition.planning.planning_session",
        "askme.cognition.working_memory": "askme.cognition.memory.working_memory",
        "askme.cognition.world_state": "askme.cognition.world.world_state",
    }

    for legacy, owner in pairs.items():
        assert importlib.import_module(legacy) is importlib.import_module(owner)
