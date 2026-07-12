# Product Requirements

日期：2026-06-05

状态：产品需求主干。本文把市场调研、方案商 ICP、需求证据台账、场景 ROI、定价包装、上线准入和高级架构追踪收束成一个 PRD 级入口。它不是路线图愿望清单，也不是客户已签约承诺；每个需求升级都必须能追到证据、事实源、架构 owner 和验证命令。

## Reading Contract

先读本文，再跳转到细分文档：

- 市场和定位：`docs/MARKET_RESEARCH.md`
- P0 客户和交付流程：`docs/SOLUTION_PROVIDER_ICP.md`
- 需求证据状态：`docs/DEMAND_EVIDENCE_LEDGER.md`
- R1-R7 架构映射：`docs/PRODUCT_ARCHITECTURE_TRACE.md`
- 行业场景和 ROI：`docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md`、`docs/SCENARIO_ROI_MODEL.md`
- 试点验收和上线准入：`docs/PILOT_ACCEPTANCE_DOSSIER_PRODUCT_SURFACE.md`、`docs/SITE_LAUNCH_READINESS_CHECKLIST.md`
- 定价包装：`docs/PRICING_PACKAGING_HYPOTHESES.md`
- 高级软件架构蓝图：`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`
- 高级安全架构：`docs/ARCHITECTURE_V2.md`

## P0 Product Bet

P0 是机器人方案商/集成商交付中台，而不是通用聊天机器人、底盘控制系统或单个 Dashboard。

目标用户：

- 机器人方案商/集成商负责人：关心 Demo-to-pilot 能否复制、交付包能否复用、客户能否签收。
- 交付工程师/实施 PM：关心客户项目、对象目录、场景验收卡、证据和阻断项是否可管理。
- 售前/解决方案顾问：关心方案边界、替代物、外部系统合同和不能过度承诺的内容。
- 现场主管/客户负责人：关心事件闭环、证据、责任边界、acceptance dossier 和上线准入缺口。

产品承诺：

- 把一次性 Demo 变成可复制、可验收的客户项目交付包。
- 把现场异常变成可审计、可关闭、可导出的事件闭环。
- 让机器人能听懂人、能交接任务、能留痕，但不替代 Runtime / Safety / Hardware。

非目标：

- 不是通用聊天机器人。
- 不替代底盘控制、导航、电机、机械臂或原始 fleet app。
- 不把 customer signoff 说成 production readiness。
- 不能把 research_pending 升级为 PRD 承诺。

## Requirement Spine

| ID | Product requirement | Evidence gate | Architecture owner | Verification anchor |
| --- | --- | --- | --- | --- |
| R1 | 客户项目工作台：维护 tenant/customer/project/site 和交付阶段。 | 至少 3 个独立样本或 1 个试点 artifact 证明项目复制痛点。 | Field Delivery Domain: `askme/pipeline/field` | `tests/test_field_customer_project_workbench.py`、`tests/test_field_http.py` |
| R2 | 管理对象目录：把点位、区域、路线、设备和验收用例绑定到客户项目。 | evidence_id 记录对象散落、复制困难或跨客户风险。 | Field Delivery Domain + field route/service composition | `tests/test_field_operations.py`、`tests/test_field_site_profile.py` |
| R3 | 交付资源治理：管理视觉模型、传感器协议、技能包、审批、禁用和回滚。 | hypothesis_status 至少 validated 或明确来自安全/审计硬边界。 | Field Delivery Domain + Admin governance | `tests/test_field_resource_governance_notifications.py` |
| R4 | 场景验收卡：用园区、厂区、仓储、景区的高价值卡验证触发、证据、通知和关闭。 | baseline、target_delta 和 usage evidence 写入 `docs/DEMAND_EVIDENCE_LEDGER.md`。 | Field scenarios/events + Product scene catalog | `tests/test_dashboard_http.py`、`tests/test_capability_package_payloads.py` |
| R5 | acceptance dossier：导出项目、对象、场景、事件、知识、权限、运行时和审计证据。 | acceptance dossier 必须有 evidence_id、customer signoff 和 blocked_uses。 | Field acceptance routes/services | `tests/test_field_customer_project_acceptance_routes.py` |
| R6 | 运行时安全边界：所有执行请求经过 InteractionGate、RBAC、SkillGate、SafetyPreflight 和 runtime profile。 | site_acceptance_checklist、operator review、takeover、rollback 缺一则不能生产上线。 | Runtime / Safety / Hardware + Field Delivery Domain claim boundary | `tests/test_runtime_handoff.py`、`tests/test_product_launch_readiness.py` |
| R7 | 方案商交付 Dashboard：按负责人、交付工程师、现场主管和客户拆分视图。 | pricing_signal 和 usage evidence 只能展示 Field Delivery Domain 输出。 | Product/Admin/Platform/Internal surfaces | `tests/test_dashboard_customer_project_contract.py`、`tests/test_dashboard_http.py` |

## Evidence Promotion Gates

所有需求、ROI、定价和架构升级都先经过 `docs/DEMAND_EVIDENCE_LEDGER.md`：

1. 每条 claim 必须有 evidence_id、source_type、sample_count、hypothesis_status 和 confidence。
2. research_pending 只能保留为假设，不能进入 PRD 承诺、销售承诺或架构不变量。
3. validated 至少需要 3 个独立样本，或 1 个试点 + 可复核 artifact。
4. contradicted 必须降级、删除或转成替代场景，不允许继续包装成 must-have。
5. quote、artifact 和 pilot evidence 必须 redaction_required，并确认 no secrets/no PII。
6. 安全、审计、权限和硬件边界可以作为 hard boundary，但仍要在 PRD 中标注来源。

## ROI And Packaging Binding

ROI 和定价只证明买点，不创建产品事实：

- baseline、target_delta、value metric、payback signal 和 pricing_signal 来自 `docs/SCENARIO_ROI_MODEL.md`。
- 项目费、站点费、机器人数量、技能包、交付包和 RaaS 运营报告来自 `docs/PRICING_PACKAGING_HYPOTHESES.md`。
- pricing_signal 可以帮助排序包装，但不能反向生成 acceptance dossier、customer signoff 或 production readiness。
- customer signoff != production readiness；客户签收只证明本次 acceptance dossier 覆盖了试点范围。

## Architecture Binding

产品需求到高级软件架构的 owner 规则：

- Field Delivery Domain owns customer project、managed object、field event、onsite evidence、acceptance dossier、customer signoff、readiness gaps 和 usage evidence。
- `askme/api/routes/field_*`、`askme/api/services/field_*` 和 `askme/static` 只能传输、组合和展示 Field Delivery Domain 输出，不能独立生成验收结论。
- Product/Admin/Platform/Internal 必须保持分层：Product 面向客户解释，Admin 面向治理和审批，Platform 面向健康和指标，Internal 只服务运行时/设备回调。
- Runtime / Safety / Hardware owns execution truth。AskMe 可以发起受控 handoff，但不替代底盘控制。
- `docs/PRODUCT_ARCHITECTURE_TRACE.md` 是 R1-R7 的架构映射表；`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md` 是 bounded contexts 和架构门禁入口；`docs/ARCHITECTURE_V2.md` 是高级安全架构入口。

## Release Gates

| Gate | Required evidence | Cannot claim |
| --- | --- | --- |
| Discovery | research_pending claim、访谈计划、替代物和证据台账字段 | validated demand |
| PRD-ready | validated claim、R1-R7 owner、non-goal、验证命令 | 已签约客户需求 |
| Pilot-ready | customer project、场景卡、acceptance dossier、baseline、usage evidence | production readiness |
| Launch-readiness review | site_acceptance_checklist、runtime roundtrip、operator review、takeover、rollback、外部系统失败状态 | 无人值守生产上线 |
| Packaging-ready | pricing_signal、payback signal、usage evidence、forced choice 访谈 | 固定价格或合同条款 |

## Verification Contract

当前主干只定义产品需求和高级架构边界，不证明客户已验证或现场已上线。每次修改本文件或其依赖文档时至少运行：

```powershell
python -m pytest tests/test_repository_layout.py -q
python -m pytest tests/test_repository_layout.py tests/test_package_migration_compat.py::test_multi_agent_docs_reference_existing_verification_targets -q
python -m ruff check tests/test_repository_layout.py
```

涉及 Field Delivery Domain、Dashboard、运行时或包边界时，再运行 `docs/PRODUCT_ARCHITECTURE_TRACE.md` 中对应的行为测试。
