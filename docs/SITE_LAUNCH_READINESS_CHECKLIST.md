# Site Launch Readiness Checklist

日期：2026-06-05

状态：产品需求和高级架构契约。本文把市场调研中的“上线前硬件/现场验收 checklist 产品化表达”落成生产准入清单，服务方案商 Demo-to-pilot 到受控生产试运行的交付判断。

## Goal

上线前硬件/现场验收 checklist 要回答：

> 客户已经签收试点材料后，还缺什么才能把项目推进到受控生产 readiness。

硬边界：

- customer signoff != production readiness。
- 客户签收不等于生产上线。
- 不能用 customer signoff 替代 production readiness。
- 不能把 lab 证据包装成现场验收。
- `fake/sim/shadow/lab/prod` runtime profile 必须在客户可读材料里清楚标注。
- Runtime / Safety / Hardware 拥有真实执行、接管、回滚和硬件状态；Field Delivery Domain 只拥有客户交付事实、证据和 readiness 结论。

## Readiness State Model

| State | Meaning | Allowed claim | Blocked claim |
| --- | --- | --- | --- |
| `demo_ready` | 本地或演示证据可跑通 | 可演示能力链路 | 不能声称现场验收 |
| `pilot_ready` | 客户项目、对象、场景和验收包完整 | 可进入现场试点 | 不能声称 production readiness |
| `accepted_by_customer` | 客户签收试点范围和风险确认 | 客户接受本次试点材料 | 不能替代硬件现场验收 |
| `production_readiness_manual_check` | 仍需人工复核现场证据或外部系统 | 可列出缺口和下一步 | 不能上线 |
| `production_ready_controlled` | IAM、现场证据、runtime roundtrip、接管、回滚、审计和安全边界都通过 | 可进入受控生产试运行 | 不能声称无人值守 |

任何未满足现场硬件证据、外部系统证据或 operator review 的状态都必须进入 blocked_uses，阻断无人值守生产上线和越界销售口径。

## site_acceptance_checklist

`site_acceptance_checklist` 是 production readiness 的最小客户现场 checklist。它不是 acceptance dossier 的附件，而是 acceptance closure 和 launch_readiness 的 gate。

Canonical gate IDs: `site_profile`, `managed_object_bindings`, `device ingest`, `live voice`, `external notifications`, `runtime roundtrip`, `takeover`, `rollback`, `operator review`, `audit_export`.

| Gate | Required evidence | Owner | Failure state |
| --- | --- | --- | --- |
| Site profile | tenant/customer/project/site/object 作用域、现场地址、交付 namespace | Field Delivery Domain | missing scope -> blocked |
| Managed object bindings | 对象目录绑定视觉模型、传感器协议、技能包和验收用例 | Field Delivery Domain | missing binding -> manual_check/blocked |
| Device ingest | real camera/sensor/OEM callback receipt, source_event_id, managed_object_id | External contracts + Field Delivery Domain | lab/mock only -> blocked |
| live voice | 现场麦克风、播报、打断、拒答和权限确认的真实运行证据 | Product surface + Runtime / Safety / Hardware | recorded/local only -> manual_check |
| external notifications | 钉钉/短信/工单/通知系统真实发送和 ack 证据 | External contracts | no ack -> retrying/blocked |
| runtime roundtrip | task handoff、SafetyPreflight、Runtime Arbiter、callback final status | Runtime / Safety / Hardware | no final callback -> blocked |
| takeover | 人工接管、暂停、急停或高风险审批路径演练记录 | Runtime / Safety / Hardware | no operator path -> blocked |
| rollback | 项目配置、技能启停、runtime profile 或模板回滚演练证据 | Field Delivery Domain + Runtime / Safety / Hardware | no rollback proof -> blocked |
| operator review | 交付 owner、现场主管、客户代表的 operator review | Admin surface | no review -> manual_check |
| audit export | SIEM/WORM 或客户审计包 export hash/signature/HMAC | Audit/Admin | export failed -> blocked |

## Architecture Ownership

- Field Delivery Domain owns `site_acceptance_checklist`, launch_readiness, customer project scope, onsite evidence, blocked_uses, and acceptance closure facts.
- API routes/services expose the checklist from the domain; they cannot merge customer signoff, acceptance dossier, and production readiness into one `ready` flag.
- Product/Admin/Platform/Internal split remains mandatory. Product explains readiness to customers, Admin records reviews, Platform reports health, and Internal handles runtime/device callbacks only.
- Runtime / Safety / Hardware owns runtime execution, hardware safety, takeover, rollback, and final callback truth.
- External systems must follow `docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md`; their events are evidence inputs, not production readiness owners.

## API And Surface Anchors

The checklist must be visible through existing delivery surfaces rather than a parallel readiness engine:

- `GET /api/field/customer-projects/{identifier}/acceptance-closure` exposes acceptance closure, site_acceptance_checklist, blockers, next step, and blocked_uses.
- `GET /api/field/customer-projects/{identifier}/acceptance-dossier` carries checklist status into the customer-readable acceptance dossier.
- `/dashboard/delivery` should show checklist status as customer-site acceptance work items, not raw internal runtime callback logs.

The checklist can reference acceptance dossier and customer signoff evidence, but it must not treat them as production readiness proof.

## Verification Contract

| Claim | Evidence |
| --- | --- |
| customer signoff != production readiness remains explicit | `tests/test_repository_layout.py`, `tests/test_product_launch_readiness.py` |
| site_acceptance_checklist gates launch_readiness | `tests/test_field_customer_project_acceptance_routes.py`, `tests/test_product_launch_readiness.py` |
| runtime handoff cannot bypass hardware safety | `tests/test_runtime_handoff.py`, `tests/test_safety_checker.py`, `tests/test_safety_estop.py` |
| external evidence has scope, idempotency, and failure state | `tests/test_field_ingest_adapters.py`, `tests/test_field_contracts.py`, `tests/test_audit_query.py` |
| Dashboard is a surface, not a readiness engine | `tests/test_dashboard_http.py`, `tests/test_dashboard_customer_project_contract.py` |

Full `pytest tests` remains the final confidence gate. This document only defines the product and architecture contract for production readiness; it does not prove a real customer site has passed hardware acceptance.
