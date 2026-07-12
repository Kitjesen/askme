# Product Architecture Trace

日期：2026-06-05

状态：需求到架构追踪表。本文从 `docs/PRODUCT_REQUIREMENTS.md` 和 `docs/SOLUTION_PROVIDER_ICP.md` 出发，把 P0 方案商交付需求映射到代码所有权、API 表面、运行时边界和验证测试；高级软件架构蓝图见 `docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`，行业场景卡见 `docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md`，场景 ROI 模型见 `docs/SCENARIO_ROI_MODEL.md`，试点验收包产品面见 `docs/PILOT_ACCEPTANCE_DOSSIER_PRODUCT_SURFACE.md`，外部系统合同见 `docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md`，上线准入清单见 `docs/SITE_LAUNCH_READINESS_CHECKLIST.md`，定价包装假设见 `docs/PRICING_PACKAGING_HYPOTHESES.md`，访谈验证路径见 `docs/INTERVIEW_GUIDE_SOLUTION_PROVIDER.md`，需求证据台账见 `docs/DEMAND_EVIDENCE_LEDGER.md`。它不是路线图愿望清单；每行都必须能回答“这个需求的事实源在哪里、谁只是表面、什么不能越界、用什么测试守住”。

## 架构原则

1. Field Delivery Domain 是方案商交付需求的产品事实源。客户项目、对象目录、交付资源、现场事件、验收证据、客户签收和上线准入规则必须落在 `askme/pipeline/field`。
2. HTTP route/service 和 Dashboard 只是表面。`askme/api/routes/field_*`、`askme/api/services/field_*`、`askme/static` 可以传输和展示字段，但不能独立生成验收结论。
3. Runtime handoff 是执行边界。AskMe 可以生成受控任务和证据，但真实执行必须交给 Runtime / Safety / Hardware；底盘、导航、电机和机械臂控制不属于 Field Delivery Domain。
4. API 表面必须保持 Product/Admin/Platform/Internal 分层。客户说明页不能依赖内部机器人控制接口，治理页面不能伪装成客户签收，Internal 不能驱动客户交付口径。
5. customer signoff != production readiness。客户签收、现场证据、acceptance dossier 和生产上线准入是不同状态，不能被一个 ready 字段合并。

## R1-R7 Trace Matrix

| Requirement | Product demand from ICP | Field Delivery Domain owner | Surface and integration | Must not own | Verification anchors |
| --- | --- | --- | --- | --- | --- |
| R1 | 客户项目工作台：从模板创建客户项目，维护客户、现场、项目、租户、命名空间和交付阶段。 | `askme/pipeline/field` customer project/site profile logic; delivery readiness rollups. | `askme/api/routes/field_*`, `askme/api/services/field_*`, `askme/static` project/workbench pages. | `askme/static` must not invent project readiness; API transport must not own domain rules. | `tests/test_field_customer_project_workbench.py`, `tests/test_field_http.py`, `tests/test_product_launch_readiness.py` |
| R2 | 管理对象目录：现场对象必须绑定资源、技能和验收用例，不能只存在于脚本或文案。 | `askme/pipeline/field` managed-object catalog, scope checks, acceptance gate logic. | Field routes/services expose catalog and exports; Dashboard renders object readiness. | Provider/robot packages must not become customer-object truth sources. | `tests/test_field_operations.py`, `tests/test_field_site_profile.py`, `tests/test_field_http.py` |
| R3 | 交付资源治理：复用视觉模型、传感器协议、技能包和验收用例，并保留版本、状态、禁用/回滚和审批记录。 | `askme/pipeline/field` delivery resource registry and governance request state. | Field governance routes and Dashboard panels show impact before review. | Shared resources cannot be silently disabled or rolled back from UI-only state. | `tests/test_field_resource_governance_notifications.py`, `tests/test_field_site_profile.py`, `tests/test_field_http.py` |
| R4 | 场景验收卡：首批高价值场景来自 `docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md`，需要触发来源、判断证据、通知对象、技能包、播报、归档和未完成项。 | `askme/pipeline/field` field scenarios/events; skill package handoff contracts. | Product/API scene catalog and Dashboard scenario acceptance surfaces. | Scenario UI cannot bypass SkillGate, SafetyPreflight, or event audit. | `tests/test_dashboard_http.py`, `tests/test_capability_package_payloads.py`, `tests/test_field_operations.py` |
| R5 | acceptance dossier：产品面见 `docs/PILOT_ACCEPTANCE_DOSSIER_PRODUCT_SURFACE.md`，合并项目、对象、场景、事件、知识、权限、运行时和审计证据，说明阻断项。 | `askme/pipeline/field` acceptance dossier/customer signoff state; audit evidence links. | Field acceptance routes/services export customer-readable dossier and proposal bundles. | customer signoff != production readiness; exported HTML/JSON cannot claim production go-live alone. | `tests/test_field_customer_project_acceptance_routes.py`, `tests/test_customer_project_package_rules.py`, `tests/test_audit_query.py` |
| R6 | 运行时安全边界：任务 handoff 必须经过 InteractionGate、RBAC、SkillGate、SafetyPreflight、runtime profile、`docs/SITE_LAUNCH_READINESS_CHECKLIST.md` 和 Runtime / Safety / Hardware。 | Field Delivery Domain owns customer-facing claim boundaries; runtime owns execution state and hardware handoff. | Product/Admin/Platform/Internal surfaces expose only the right level of claim and control. | Field route/service glue must not call raw robot control, and Internal must not drive customer UI. | `tests/test_runtime_handoff.py`, `tests/test_health.py`, `tests/test_product_launch_readiness.py`, `tests/test_dashboard_http.py` |
| R7 | 方案商交付 Dashboard：按角色拆分交付总门禁、项目/对象/资源、事件证据和客户验收阻断项。 | Field Delivery Domain produces the readiness facts; product launch readiness composes them. | `askme/static` renders pages; dashboard APIs expose scoped payloads. | Dashboard must not become a second business-rule engine or flatten all roles into one claim. | `tests/test_dashboard_http.py`, `tests/test_dashboard_customer_project_contract.py`, `tests/test_product_launch_readiness.py` |

## Boundary Contracts

### Field Delivery Domain

`askme/pipeline/field` owns the product facts that matter to the ICP:

- customer projects, site profiles, delivery namespaces, and tenant/customer/project/site scope;
- managed objects, resource bindings, acceptance checks, and action plans;
- field events, onsite evidence, notifications, close/review state, and audit links;
- acceptance dossier, customer signoff, production-readiness gaps, and delivery claims.

### API And Dashboard Surfaces

`askme/api/routes/field_*` and `askme/api/services/field_*` are transport and composition surfaces around Field Delivery Domain contracts. They may normalize request/response payloads, enforce operator identity, and compose customer-readable payloads, but any acceptance, readiness, or production claim must come from the domain layer.

`askme/static` renders the resulting contracts. It can explain blocked/manual_check/ready states, but it must not compute independent acceptance conclusions or use Internal robot-control routes as customer-facing proof.

### Runtime / Safety / Hardware

Runtime / Safety / Hardware owns execution and hardware state. Field Delivery Domain may request a gated handoff and record the customer-facing evidence boundary; it must not own low-level robot motion, navigation, motor, serial, or mechanical-arm behavior.

### API Surface Split

The product architecture keeps Product/Admin/Platform/Internal separate:

- Product: customer-visible explanation, scenarios, knowledge, capabilities, and field-event reports.
- Admin: governance, approvals, audit, delivery resources, acceptance review, and launch-readiness gates.
- Platform: health, metrics, readiness summary, and service state without customer business authority.
- Internal: runtime/device/vision callbacks and low-level integration, never customer UI or sales claim source.

### External System Contracts

`docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md` owns the product-level contract
for VMS, CMMS, IAM, map, OEM fleet, notification, and SIEM/WORM connections.
Those systems provide input facts, status receipts, or audit delivery evidence;
Field Delivery Domain still owns the customer-project scope, field event,
evidence, failure state, and acceptance/readiness conclusion.

### Pricing And Packaging Hypotheses

`docs/PRICING_PACKAGING_HYPOTHESES.md` owns product-level packaging hypotheses
for 项目费、站点费、机器人数量、技能包、交付包 and RaaS 运营报告.
Each revenue unit must map to Field Delivery Domain facts and usage evidence,
not to Dashboard copy or Runtime / Safety / Hardware ownership. Billing signals
can observe customer project, site_acceptance_checklist, acceptance dossier,
customer signoff, runtime receipt, or report export, but they must not create
acceptance, readiness, or hardware-control conclusions.

### Scenario ROI Model

`docs/SCENARIO_ROI_MODEL.md` owns product-level ROI hypotheses for the first
园区、厂区、仓储、景区 scenario cards. It is a read model over Field Delivery
Domain facts: baseline, target_delta, value metric, payback signal,
pricing_signal, and usage evidence must be traceable to customer projects,
field events, onsite evidence, report exports, or acceptance dossier records.
ROI payloads must not become a second acceptance engine, a Dashboard-only
calculation, or a production readiness source.

### Demand Evidence Ledger

`docs/DEMAND_EVIDENCE_LEDGER.md` owns research evidence status. It does not
create runtime facts; R1-R7 architecture constraints can be promoted from
customer research only after validated evidence or safety/security/audit hard
boundaries. Field Delivery Domain remains the facts source.

### Software Architecture Blueprint

`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md` owns bounded contexts, container /
package map, API surface rules, architecture invariants, and architecture
decision gates. This trace owns R1-R7 row-level mapping; the blueprint owns the
cross-context architecture contract.

## Verification Strategy

Use focused checks before any broad cleanup:

1. Product-positioning docs: `python -m pytest tests/test_repository_layout.py -q`.
2. Field delivery behavior: `python -m pytest tests/test_field_operations.py tests/test_field_ingest_adapters.py tests/test_field_contracts.py tests/test_field_customer_project_acceptance_routes.py -q`.
3. Dashboard/customer claim surfaces: `python -m pytest tests/test_dashboard_http.py tests/test_dashboard_customer_project_contract.py tests/test_product_launch_readiness.py -q`.
4. API and health surface split: `python -m pytest tests/test_health.py tests/test_dashboard_http.py -q`.
5. Cross-package boundary safety: `python -m pytest tests/test_six_layer_package_boundaries.py tests/test_package_migration_compat.py -q`.

Full `pytest tests` remains the final confidence gate after the dirty worktree is ready for a broad verification pass.
