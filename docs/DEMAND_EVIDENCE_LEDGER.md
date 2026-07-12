# Demand Evidence Ledger

日期：2026-06-05

状态：需求证据台账。本文定义如何记录访谈、试点、材料样例、ROI、定价和架构声明的证据状态；它不是客户数据库、CRM、销售承诺或生产验收记录。

## Purpose

AskMe 当前的大部分产品判断仍是 hypothesis。需求证据台账的目标是把“我们认为客户会买什么”和“客户/试点已经证明什么”分开，避免把产品推演、单个访谈、漂亮 Demo 或销售意向升级成已验证需求。

核心规则：

- 不能把访谈意向当已验证需求。
- 不能把单个样本升级成架构不变量。
- 不能把展示价值当 ROI。
- 不能把 customer signoff != production readiness 改写成上线证明。
- Field Delivery Domain 是客户项目、现场事件、证据、acceptance dossier、customer signoff 和 readiness gaps 的产品事实源；本文只是研究证据台账。
- Runtime / Safety / Hardware 仍拥有真实执行、硬件状态、接管、回滚和现场硬件验收。

## Evidence Record Schema

每条证据记录至少包含以下字段：

| Field | Required | Meaning |
| --- | --- | --- |
| `evidence_id` | yes | 稳定编号，例如 `EV-P0-001`、`EV-R4-003` |
| `source_type` | yes | `interview`、`pilot`、`artifact`、`quote`、`field_event`、`report_export`、`win_loss` |
| `source_date` | yes | 证据日期 |
| `segment` | yes | P0 方案商、P1 FM、P1 厂区/仓储、P2 园区/景区、P3 OEM |
| `sample_count` | yes | 支撑该判断的样本数量 |
| `claim_id` | yes | 关联 P0/P1/P2/P3、R1-R7、PARK-1、FACTORY-1、WAREHOUSE-1、SCENIC-1、pricing_signal 或 ROI metric |
| `hypothesis_status` | yes | `research_pending`、`validated`、`contradicted`、`mixed`、`not_applicable` |
| `confidence` | yes | `low`、`medium`、`high`，必须说明原因 |
| `evidence_summary` | yes | 脱敏摘要，不放原始敏感信息 |
| `linked_artifact` | optional | 脱敏访谈记录、项目时间线、验收目录、事件台账或截图索引 |
| `redaction_required` | yes | 是否需要脱敏；默认 yes |
| `no secrets/no PII` | yes | 不记录密钥、真实个人联系方式、支付信息、客户敏感照片或未授权现场数据 |
| `next_action` | yes | 继续访谈、删除/降级需求、进入 PRD、补试点、补架构约束 |

## Hypothesis Status Rules

| Status | Meaning | Minimum evidence |
| --- | --- | --- |
| `research_pending` | 仍是产品推演或二手资料 | 无真实样本，或只有公开资料 |
| `validated` | 可作为下一轮产品/架构输入 | 至少 3 个独立样本或 1 个试点 + 可复核 artifact |
| `contradicted` | 当前假设被证据反驳 | 样本明确不痛、不付费、或替代物更强 |
| `mixed` | 分群差异明显 | 证据支持一个 segment，但反驳另一个 segment |
| `not_applicable` | 不适合当前 ICP | 与 Field Delivery Domain、R1-R7 或方案商交付无关 |

Architecture invariants need stricter evidence than interview preferences. A claim can influence R1-R7 only when it is validated across relevant segments or grounded in safety, security, privacy, audit, or Runtime / Safety / Hardware boundaries.

## Claim Mapping

| Claim family | What evidence must prove | Example evidence |
| --- | --- | --- |
| P0 ICP | 方案商/集成商比终端客户更痛、更快买、更能复用 | Demo-to-pilot timeline、交付材料目录、预算来源、采购触发 |
| R1 客户项目工作台 | 客户项目是交付事实源，不是配置文件 | 项目模板、客户/站点/对象目录、导出交付包 |
| R4 场景验收卡 | 客户愿意逐条验收触发、证据、通知、处理和未完成项 | `docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md` 中的场景选择和验收目录 |
| R5 acceptance dossier | 客户签收前确实需要证据包和缺口说明 | 人工整理验收材料小时数、acceptance dossier 样例、签收阻断原因 |
| R7 Dashboard | 不同角色真的需要不同视图 | 方案商负责人、交付工程师、现场主管、客户负责人访谈差异 |
| ROI | baseline、target_delta、value metric 和 payback signal 能被复核 | `docs/SCENARIO_ROI_MODEL.md` 的 usage evidence |
| Pricing | pricing_signal 与付费单位有关，而不是泛泛愿意买 AI | forced choice、预算来源、采购阻断、项目费/站点费/交付包/RaaS 运营报告偏好 |

## Architecture Binding

需求证据台账不创建运行时事实：

1. `docs/MARKET_RESEARCH.md` 和 `docs/SOLUTION_PROVIDER_ICP.md` 只能把 `validated` 或明确标注为 hypothesis 的证据写进需求结论。
2. `docs/INTERVIEW_GUIDE_SOLUTION_PROVIDER.md` 负责采集 evidence_id、source_type、sample_count、quote、artifact 和 redaction_required。
3. `docs/SCENARIO_ROI_MODEL.md` 读取 baseline、target_delta、value metric、payback signal 和 usage evidence，但不能用 research_pending 样本包装成 ROI。
4. `docs/PRICING_PACKAGING_HYPOTHESES.md` 读取 pricing_signal，但不能把 pricing_signal 写成 acceptance dossier、customer signoff 或 production readiness。
5. `docs/PRODUCT_ARCHITECTURE_TRACE.md` 只能把 validated evidence 或安全/审计硬边界升级为架构约束。
6. Field Delivery Domain 仍是 customer project、field event、onsite evidence、acceptance dossier、customer signoff 和 readiness gaps 的产品事实源。
7. Runtime / Safety / Hardware 仍是执行 truth，访谈偏好不能改变硬件接管、回滚、现场验收和安全门禁。

## Intake Template

```text
evidence_id:
source_type: interview | pilot | artifact | quote | field_event | report_export | win_loss
source_date:
segment:
sample_count:
claim_id:
hypothesis_status: research_pending | validated | contradicted | mixed | not_applicable
confidence: low | medium | high
evidence_summary:
linked_artifact:
redaction_required: yes
no secrets/no PII: confirmed
next_action:
```

## Review Gate

每次把需求写进 PRD、架构追踪或包装文档前，先检查：

1. 是否有 evidence_id。
2. source_type 是否不是单一口头意向。
3. sample_count 是否足够支撑该 claim。
4. 是否区分 research_pending、validated、contradicted。
5. 是否脱敏并确认 no secrets/no PII。
6. 是否会越过 customer signoff != production readiness。
7. 是否会让 Dashboard、pricing_signal 或 ROI 变成新的事实源。

如果任何一项不满足，该 claim 只能保留为 hypothesis，不能进入已验证需求或架构不变量。
