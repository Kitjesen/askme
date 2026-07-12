# Scenario ROI Model

日期：2026-06-05

状态：场景 ROI 模型。本文把 `docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md` 的园区、厂区、仓储、景区场景卡转成访谈和试点要验证的价值指标。它不是财务预测，也不是承诺客户一定节省成本；所有 ROI 结论都必须来自 Field Delivery Domain 的 usage evidence。需求证据台账见 `docs/DEMAND_EVIDENCE_LEDGER.md`。

## Core Contract

场景 ROI 只回答一个问题：客户为什么愿意为一个可验收、可复用、可审计的现场交付闭环付费。

- 每个 ROI 假设必须有 baseline、target_delta、value metric 和 payback signal。
- baseline 来自客户现有项目材料、事件台账、工单截图、人工补验收材料耗时或现场主管复盘。
- target_delta 是试点目标，不是销售承诺；没有真实数据时只能标记为 research_pending。
- value metric 必须落到 field event closure、evidence completeness、manual material hours saved、response time、handoff review 或 report export。
- payback signal 必须能转成 pricing_signal，例如项目费、站点费、交付包、RaaS 运营报告或技能包续费。
- Field Delivery Domain 拥有 ROI 所需的 customer project、site profile、managed object、field event、onsite evidence、acceptance dossier、customer signoff 和 report export facts。
- customer signoff != production readiness；ROI 可以证明试点价值或交付效率，不能证明生产上线。
- Runtime / Safety / Hardware 仍拥有真实执行、硬件状态、接管、回滚和现场验收。

## Industry ROI Table

| Industry | Anchor cards | baseline | target_delta | value metric | payback signal | pricing_signal |
| --- | --- | --- | --- | --- | --- | --- |
| 园区 | PARK-1, PARK-2, PARK-3 | 高频问路/服务咨询数量、夜间异常处理时长、群聊补证据耗时 | 降低重复人工解释、缩短异常通知到关闭时间、提升验收材料完整度 | question resolution evidence、field event closure time、acceptance dossier completeness | 方案商能把问路、异常上报、公共设施异常包装成可复用场景包 | 项目费、站点费、交付包 |
| 厂区 | FACTORY-1, FACTORY-2, FACTORY-3 | 人工巡检小时、异常复核耗时、安全事件证据缺口、误报复盘成本 | 缩短巡检异常闭环时间、减少人工补图补日志、提高 SOP 复核可追溯性 | inspection event closure、operator review、onsite evidence coverage、audit trail completeness | 工业客户愿意为受控试点和证据链付费，但不因展示效果付费 | 项目费、站点费、交付包 |
| 仓储 | WAREHOUSE-1, WAREHOUSE-2, WAREHOUSE-3 | 主通道阻塞发现时间、装卸区异常台账、货架/托盘复核记录 | 缩短发现到通知、通知到整改、整改到复核的链路 | blockage closure time、notification receipt、managed object evidence、customer report export | 交付团队能证明 AskMe 补的是现场事件闭环，不替代 WMS/TMS | 站点费、RaaS 运营报告、交付包 |
| 景区 | SCENIC-1, SCENIC-2, SCENIC-3 | 游客高频咨询、广播审批记录、遗失物交接记录、服务点异常报告 | 提升知识引用可追溯、减少口头交接丢失、保留已审批广播证据 | knowledge citation coverage、approved broadcast receipt、handoff timeline、report export | 景区先买可演示、可试点、可验收的服务闭环，后续看运营报告价值 | 项目费、站点费、RaaS 运营报告 |

## Metric Shape

每个 ROI 记录至少包含以下字段：

| Field | Meaning |
| --- | --- |
| `scenario_card_id` | `PARK-1`、`FACTORY-1`、`WAREHOUSE-1`、`SCENIC-1` 等 R4 场景验收卡 |
| `customer_project_id` | Field Delivery Domain 中的 customer project |
| `site_id` | 站点或现场作用域 |
| `baseline` | 访谈或试点前的人工耗时、事件频次、关闭时长、证据缺口 |
| `target_delta` | 试点目标变化，例如 less manual material hours、faster closure、more complete evidence |
| `value_metric` | 直接衡量价值的指标 |
| `usage_evidence` | field event、onsite evidence、acceptance dossier、report export、customer signoff 等证据 |
| `payback_signal` | 能支持客户继续付费、扩大站点或购买交付包的证据 |
| `pricing_signal` | 与 `docs/PRICING_PACKAGING_HYPOTHESES.md` 对齐的收费信号 |
| `blocked_claims` | 不能宣称的内容，例如 production readiness 或无人值守上线 |

## Product Boundaries

- 不把展示价值当 ROI。一次漂亮 Demo 只能证明场景可讲清楚，不能证明客户会付费。
- 不把 production readiness 当 ROI。上线准入是安全/硬件/现场验收问题，不是价值指标。
- 不把 customer signoff 当生产证明。客户签收只说明 acceptance dossier 覆盖了本次试点范围。
- 不用单个 Dashboard 卡片计算 ROI。Dashboard 只能展示 Field Delivery Domain 产出的 value metric 和 usage evidence。
- 不把 fake/sim/shadow/lab 证据包装成现场 ROI。它们只能支持研发、演示或接入验证。

## Architecture Binding

ROI 模型是产品和架构之间的读模型，不是新事实源：

1. `askme/pipeline/field` owns customer projects, managed objects, field events, onsite evidence, acceptance dossier, customer signoff, report export, usage evidence, and readiness gaps.
2. `askme/api/routes/field_*`、`askme/api/services/field_*` 和 Dashboard 只能传输、聚合、展示 ROI 字段，不能独立创造 value metric。
3. `docs/PRICING_PACKAGING_HYPOTHESES.md` 使用 ROI 的 payback signal 和 pricing_signal 来验证收费单位，但不能反向修改验收结论。
4. `docs/SITE_LAUNCH_READINESS_CHECKLIST.md` 继续拥有 production readiness 的现场硬件证据要求。
5. Runtime / Safety / Hardware owns execution truth; ROI 只能引用受控 handoff receipts 和 operator review。

## Validation Plan

访谈和试点时按以下顺序验证：

1. 对每个受访客户先选最多 3 张场景卡，不扩成宽泛行业模板。
2. 为每张卡收 baseline：过去 3 个项目的人工材料小时、事件频次、关闭时长或证据缺口。
3. 在试点中收 usage evidence：事件时间线、通知、处理人、现场证据、报告导出、acceptance dossier。
4. 所有 baseline、target_delta 和 payback signal 必须有 evidence_id，并在 `docs/DEMAND_EVIDENCE_LEDGER.md` 中标注 hypothesis_status。
5. 把 payback signal 映射到项目费、站点费、交付包或 RaaS 运营报告。
6. 如果只有展示好看、没有 baseline 或 usage evidence，该场景不能进入首批付费包装。
