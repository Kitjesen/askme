# External System Integration Contracts

日期：2026-06-05

状态：产品需求和高级架构契约。本文把 `docs/MARKET_RESEARCH.md` 和 `docs/SOLUTION_PROVIDER_ICP.md` 中的“关键外部系统连接清单”落成外部系统最小字段合同。它不是适配器实现方案，也不是采购清单。

## Contract Goal

AskMe 的外部系统接入目标不是“能临时同步一下”，而是让 VMS、CMMS、IAM、地图、OEM fleet、通知系统、SIEM/WORM 的事实以可审计、可重放、可验收的方式进入 Field Delivery Domain。

硬边界：

- 不能写成一次性同步脚本。
- 不能用默认项目兜底。
- 不能把外部系统的告警、工单或设备状态直接当成客户验收结论。
- 不能让 Product/Admin/Platform/Internal 分层混用；客户页面不能调用 Internal 设备回调。
- Runtime / Safety / Hardware 仍拥有真实执行和硬件状态；外部系统合同只定义交付事实和 handoff 证据。

## Shared Envelope

所有外部系统事件、回调、导出或验证响应都必须带最小 envelope。缺少这些字段时，事件进入 failure_state，不能写入客户验收包。

| Field | Required | Why |
| --- | --- | --- |
| `source_system` | yes | 标明 VMS、CMMS、IAM、地图、OEM fleet、通知系统或 SIEM/WORM |
| `source_event_id` | yes | 保留外部系统原始事件或工单 ID，支持追溯 |
| `idempotency_key` | yes | 避免 webhook、重试和人工导入造成重复事件 |
| `tenant_id` | yes | 多租户/交付空间隔离 |
| `customer_id` | yes | 客户作用域 |
| `project_id` | yes | 客户项目作用域 |
| `site_id` | yes | 现场作用域 |
| `managed_object_id` | conditional | 绑定现场对象；不能解析时必须变成对象缺口 |
| `event_time` | yes | 事件时间线 |
| `received_at` | yes | AskMe 接收时间，支持延迟审计 |
| `operator_id` | conditional | 人工导入、审批、复核和签收必须有操作者 |
| `evidence_refs` | conditional | 图片、日志、工单、通知回执、审计导出等证据引用 |
| `failure_state` | yes | `accepted`、`manual_check`、`blocked`、`rejected`、`retrying` |
| `retry_policy` | conditional | 外部投递、导出和 webhook 失败时的重试合同 |
| `audit_export_id` | conditional | 进入 SIEM/WORM 或客户审计包时的导出 ID |

## System Contracts

| System | Minimum input/output | Field Delivery Domain use | Failure boundary |
| --- | --- | --- | --- |
| VMS | camera/source ID, detection type, confidence, snapshot/video reference, zone, event time | 转成 field event、证据、通知和关闭责任 | 低置信度、无对象绑定或无证据时只能 manual_check |
| CMMS | work order ID, asset/object ID, status, assignee, SLA, close reason | 把现场事件推到工单流程，并把工单状态回写到验收证据 | 工单创建失败不能吞掉；进入 retrying/blocked 并保留失败原因 |
| IAM | verified identity, roles, tenant/customer/project/site scopes, high-risk approval claims | 约束操作者权限、客户可见范围和签收资格 | 未验证身份或作用域不匹配必须 rejected/blocked |
| 地图 | site map version, route/zone/object IDs, restricted areas, coordinate frame | 绑定对象目录、场景卡和现场证据位置 | 地图版本不匹配不能用默认路线兜底 |
| OEM fleet | robot ID, runtime profile, task/handoff ID, health, callback status | 记录受控 handoff 和运行证据，不拥有底盘控制 | fleet 失败不代表 AskMe 可绕过 Runtime / Safety / Hardware |
| 通知系统 | channel, recipient group, message ID, delivery status, acknowledgement | 证明通知、响应、升级和关闭责任 | 发送失败必须进入 retry_policy 和事件时间线 |
| SIEM/WORM | audit_export_id, payload hash, signature/HMAC, delivery status, retention target | 为 acceptance dossier 和客户审计提供不可篡改证据 | 导出失败不能声称审计就绪 |

## Failure State Contract

| failure_state | Meaning | Product behavior |
| --- | --- | --- |
| `accepted` | 字段完整、作用域匹配、证据可追溯 | 可进入现场事件、验收证据或审计包 |
| `manual_check` | 字段可读但对象、证据、置信度或映射不完整 | 可展示给交付人员，不能进入客户签收结论 |
| `blocked` | 缺必需字段、作用域冲突、地图版本冲突或安全边界不满足 | 阻断验收和上线准入 |
| `rejected` | 身份未验证、签名错误、越权、跨客户项目或篡改 | 不写入客户项目事实，只保留审计 |
| `retrying` | 外部投递、导出或回调暂时失败 | 保留失败原因、retry_policy、下一次重试时间 |

## Ownership Rules

1. Field Delivery Domain 拥有客户项目、对象绑定、现场事件、验收证据、failure_state 和 readiness 结论。
2. 外部系统只提供输入事实、状态回执或审计投递结果；它们不是 acceptance dossier 的最终事实源。
3. API route/service 可以做 envelope 校验、作用域校验、幂等处理和响应映射，但不能越过 domain 生成客户验收结论。
4. Product/Admin/Platform/Internal 分层保持不变：Product 解释客户可读状态，Admin 管审批和治理，Platform 展示健康，Internal 接 runtime/device/vision callback。
5. Runtime / Safety / Hardware 只接受通过门禁的 handoff；外部系统集成不能直接触发底盘、导航、电机或机械臂动作。

## Verification Anchors

| Contract | Suggested tests |
| --- | --- |
| 外部事件 envelope 和字段校验 | `tests/test_field_ingest_adapters.py`, `tests/test_field_contracts.py` |
| VMS/传感器进入 field event | `tests/test_field_operations.py`, `tests/test_field_ingest_bridge.py` |
| CMMS/通知失败状态和重试 | `tests/test_field_operations.py`, `tests/test_audit_query.py` |
| IAM/作用域不匹配阻断 | `tests/test_field_customer_project_acceptance_routes.py`, `tests/test_product_launch_readiness.py` |
| SIEM/WORM 审计导出证据 | `tests/test_audit_query.py`, `tests/test_dashboard_http.py` |
| Runtime handoff 不越权 | `tests/test_runtime_handoff.py`, `tests/test_six_layer_package_boundaries.py` |

Full `pytest tests` 仍是最终信心门；外部系统合同变更至少要跑 `tests/test_repository_layout.py` 和对应的 field/audit/runtime 定向测试。
