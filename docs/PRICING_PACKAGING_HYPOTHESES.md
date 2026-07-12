# Pricing Packaging Hypotheses

日期：2026-06-05

状态：定价包装假设。本文只定义访谈和试点要验证的收费单位、产品事实和禁止包装边界，不是公开报价、价格表或合同条款；场景 ROI 模型见 `docs/SCENARIO_ROI_MODEL.md`，需求证据台账见 `docs/DEMAND_EVIDENCE_LEDGER.md`。

## 核心原则

AskMe 应该为方案商交付结果和可复用运营证据收费，而不是为机器人底盘控制、通用聊天、模拟演示或内部 readiness 字段收费。

- 收费对象必须能追到 Field Delivery Domain 的客户项目、对象目录、现场事件、证据、验收包或运营报告。
- 所有 pricing_signal 都必须绑定 usage evidence，例如 customer project、site profile、site_acceptance_checklist、field event、acceptance dossier、customer signoff、RaaS report export。
- customer signoff != production readiness。客户签收只能证明本次试点范围被接受，不能自动证明生产上线。
- Runtime / Safety / Hardware 仍拥有真实执行、硬件状态、接管、回滚和现场硬件验收。

## Packaging Options

| 收费单位 | 适用客户 | 付费理由 | usage evidence | 不允许包装成 |
| --- | --- | --- | --- | --- |
| 项目费 | 机器人方案商、集成商、交付团队 | 创建和交付一个 customer project，节省 Demo-to-pilot 复制成本 | customer project、对象目录、交付阶段、验收阻断项、proposal bundle | 不按底盘控制收费 |
| 站点费 | 多园区、多厂区、多仓储或多景区项目 | 每个现场的对象、点位、路线、系统接入和 site_acceptance_checklist 不同 | site profile、site_acceptance_checklist、onsite evidence、launch readiness gaps | 不把 production readiness 包装成签收加价项 |
| 机器人数量 | 已有机器人机队的服务商或 OEM 渠道 | 运行管理、handoff receipt、事件证据和报告量随机器人增加 | runtime profile receipts、field events、robot receipts、operator review | 不把原始 fleet/app 或电机控制当 AskMe 收费项 |
| 技能包 | 有多行业复用需求的方案商 | 已审批行业模板、技能包、验收用例和知识证据可复用 | skill package approval、version、scope、audit trail、blocked uses | 不为未经审批的 LLM 草稿收费 |
| 交付包 | 方案商、售前、项目 PM | 客户需要可导出的 acceptance dossier、缺口清单、责任边界和签收记录 | acceptance dossier、proposal bundle、customer signoff、hash verification | 不把客户签收说成无人值守生产上线 |
| RaaS 运营报告 | 安保/FM 服务商、多甲方运营方 | 甲方愿意为透明报告、SLA 证据、事件闭环和审计留痕付费 | monthly field events、closure timeline、SLA evidence、report export | 不承诺未验收硬件的 production readiness |

## Architecture Binding

| Revenue unit | Field Delivery Domain facts | Required pricing_signal | Verification owner |
| --- | --- | --- | --- |
| 项目费 | customer project、delivery namespace、acceptance gates | project_created、project_exported、dossier_requested | `askme/pipeline/field` |
| 站点费 | site profile、managed objects、site_acceptance_checklist | site_created、site_checklist_completed、readiness_gap_closed | `docs/SITE_LAUNCH_READINESS_CHECKLIST.md` |
| 机器人数量 | runtime profile handoff、robot receipts、field events | robot_receipt_recorded、handoff_reviewed、event_closed | Runtime / Safety / Hardware + Field Delivery Domain |
| 技能包 | delivery resource registry、skill package、approval audit | skill_package_approved、package_enabled、package_rollback | Admin governance + audit |
| 交付包 | acceptance dossier、proposal bundle、customer signoff | dossier_exported、bundle_verified、signoff_recorded | Field acceptance routes/services |
| RaaS 运营报告 | field events、notification closure、audit export | report_exported、sla_evidence_attached、customer_report_viewed | Field events + audit |

这些 signal 只能用于产品包装和访谈验证，不能绕过验收、上线准入或硬件安全状态机。

`docs/SCENARIO_ROI_MODEL.md` 提供 baseline、target_delta、value metric 和 payback signal。定价包装可以读取这些 ROI 证据来验证项目费、站点费、交付包或 RaaS 运营报告，但不能反向把 pricing_signal 写成 acceptance dossier、customer signoff 或 production readiness 结论。

## Interview Validation

访谈必须用 forced choice 验证收费单位，而不是让客户泛泛评价“愿不愿意买 AI”：

1. 如果只能付费买一个能力，你选项目工作台、事件闭环、acceptance dossier、技能包、运行安全边界还是 RaaS 运营报告？
2. 你更能接受项目费、站点费、机器人数量、技能包、交付包还是 RaaS 运营报告？
3. 预算来自项目交付成本、试点服务费、RaaS 运营费、验收材料费、客户成功预算还是运维预算？
4. 哪个 usage evidence 能证明你愿意付费：少补多少材料、少跑多少现场、少返工多少次、客户签收提前多久？
5. 哪些集成成本会让你拒绝购买：VMS、CMMS、IAM、OEM fleet、通知系统、SIEM/WORM？

forced choice 的 pricing_signal 必须写入 `docs/DEMAND_EVIDENCE_LEDGER.md`，并区分 research_pending、validated、contradicted。

## Initial Recommendation

P0 方案商先测试“项目费 + 交付包”。如果一个方案商有多个客户现场，再加站点费；如果客户运营方要求月度甲方报告，再测试 RaaS 运营报告。机器人数量只能作为运行证据和报告量的乘数，不能作为底盘控制收费。技能包要等审批、版本、回滚和审计链条稳定后再包装。

## Non-Pricing Boundaries

- 不按底盘控制收费。
- 不把 production readiness 包装成签收加价项。
- 不为 fake/sim/shadow/lab 证据收生产上线费用。
- 不把未经现场验收的 customer signoff 用作硬件验收证明。
- 不把 Product/Admin/Platform/Internal 的内部能力混成一个客户收费面。
