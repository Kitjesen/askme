# Competitive Replacement Matrix

日期：2026-06-05

状态：基于 `docs/MARKET_RESEARCH.md`、`docs/SOLUTION_PROVIDER_ICP.md` 和 GitHub skill 调研中 competitive-analysis 方法整理的替代矩阵。外部系统合同见 `docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md`，访谈验证路径见 `docs/INTERVIEW_GUIDE_SOLUTION_PROVIDER.md`。本文用于约束产品定位和高级架构边界，不等同于已完成客户访谈或商业尽调。

## 结论

AskMe 的短期竞争位置不是“机器人更聪明的聊天框”，也不是“替代 OEM fleet 或底盘控制”。更准确的切入点是：

> AskMe 是面向机器人方案商/集成商的方案商交付中台，主要替代人工运营 + 微信/Excel、定制项目脚本 / 一次性 Demo，以及割裂的验收材料整理流程；部分补强 OEM fleet/app、VMS / AI 告警、CMMS / 工单和通用 LLM Agent 平台缺少的现场交付治理。

架构后果：

- Field Delivery Domain 必须拥有客户项目、对象目录、现场事件、验收证据和 readiness 规则。
- API 和 Dashboard 只能展示和编排 Field Delivery Domain 事实，不能独立创造验收结论。
- Runtime / Safety / Hardware 仍然拥有真实执行和硬件状态；AskMe 不替代底盘控制。
- customer signoff != production readiness，不能因为客户签字就声明无人值守生产上线。

## Replacement Matrix

| 替代/竞品类型 | 客户今天为什么用它 | 强项 | 缺口 | AskMe 应替代或补强什么 | 架构边界 |
| --- | --- | --- | --- | --- | --- |
| 人工运营 + 微信/Excel | 最快、成本低、每个项目都能临时凑起来。 | 弹性高，现场人员容易理解。 | 难复制、难追责、证据散落、客户验收靠人工整理。 | 用客户项目、对象目录、事件时间线、证据包和 acceptance dossier 替代手工留痕。 | Field Delivery Domain 是事实源；Dashboard 只显示交付状态。 |
| OEM fleet/app | OEM 自带机器人管理、遥控、任务执行和设备状态。 | 硬件、导航、地图、遥控、设备健康强。 | 跨客户项目、知识证据、签收、审计、交付包和多系统运营弱。 | 补上层方案商交付中台，不抢 OEM fleet 入口。 | Runtime / Safety / Hardware 归 OEM/底层；AskMe 只做受控 handoff 和证据边界。 |
| VMS / AI 告警 | 已有摄像头和算法，能发现异常。 | 感知覆盖广，告警实时。 | 通常不能把机器人任务、人员通知、关闭责任和客户验收串成闭环。 | 把告警变成 field event、通知、复核、关闭和客户报告。 | VMS 是输入源，不是验收事实源；field event 才能进入交付证据。 |
| CMMS / 工单 | 管维护、派单、处理状态和工单闭环。 | 工单模型成熟，适合维修/设施团队。 | 对移动机器人、语音交互、现场对象和运行时 handoff 支持弱。 | 与工单互补，把机器人现场事件和验收证据推到工单/审计流程。 | CMMS 集成应通过合同字段和失败状态接入，不写一次性同步脚本。 |
| 定制项目脚本 / 一次性 Demo | 售前快，工程师能快速拼出效果。 | Demo 成本低，灵活。 | 难维护、难复用、难验收，换客户就重写。 | 这是 AskMe 的第一替代对象：把 Demo-to-pilot 流程产品化。 | 禁止把客户项目事实散落在脚本；R1-R7 必须落到 Field Delivery Domain 和验证测试。 |
| 通用 LLM Agent 平台 | 能做对话、规划、工具调用和自动化。 | 抽象能力强，生态丰富。 | 缺少现场对象、运行 profile、安全预检、客户验收、硬件 handoff 和审计边界。 | 让通用 Agent 只作为上层大脑或 MCP client；AskMe 提供现场交付工具和边界。 | ZeroClaw/Agent 不能直接控制硬件；必须经过 SkillGate、SafetyPreflight 和 Runtime / Safety / Hardware。 |
| AskMe | 方案商需要一个可复制、可验收、可审计的机器人现场交付平台。 | 客户项目、对象目录、现场事件、知识证据、运行时交接、验收包和审计闭环。 | 仍需真实客户访谈、外部系统合同、硬件联调和上线验收数据。 | 做方案商交付中台，压低每个客户项目的复制交付成本。 | 不替代底盘控制；不把 customer signoff 包装成 production readiness。 |

## Positioning Rules

1. 对方案商说：AskMe 降低 Demo-to-pilot 的复制交付成本，帮你把项目做成可验收、可复用、可审计的客户交付包。
2. 对终端客户说：AskMe 让机器人现场服务可解释、可复核、可交付；不是承诺无配置、无人工复核的全自动生产系统。
3. 对 OEM 说：AskMe 补上层场景运营和客户交付治理，不替代 OEM fleet/app、导航、遥控或硬件控制。
4. 对研发说：Field Delivery Domain 管产品事实，Runtime / Safety / Hardware 管执行事实，API/Dashboard 不能越权。
5. 对销售说：可以承诺可演示、可试点、可按现场验收推进；不能承诺未接硬件的无人值守生产上线。

## Architecture Implications

| 产品判断 | 架构要求 | 需要守住的测试 |
| --- | --- | --- |
| 替代定制脚本 / 一次性 Demo | 客户项目、对象目录、模板、资源和验收包必须是结构化事实源。 | `tests/test_field_customer_project_workbench.py`, `tests/test_field_http.py` |
| 补强 VMS / AI 告警 | 告警必须转成 field event，并进入通知、证据、关闭和审计链路。 | `tests/test_field_operations.py`, `tests/test_field_ingest_adapters.py` |
| 互补 CMMS / 工单 | 外部工单系统只能通过 `docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md` 的合同字段和失败状态集成。 | `tests/test_field_contracts.py`, `tests/test_api_route_dependency_injection.py` |
| 不抢 OEM fleet/app | Runtime handoff 必须保持受控，Field Delivery Domain 不拥有硬件动作。 | `tests/test_runtime_handoff.py`, `tests/test_six_layer_package_boundaries.py` |
| 不做通用聊天机器人 | 知识回答、场景意图和技能增长必须有证据、审批和客户可见范围。 | `tests/test_knowledge_route_payloads.py`, `tests/test_skill_governance.py` |
| customer signoff != production readiness | 客户签收、验收证据和上线准入必须是不同状态机。 | `tests/test_product_launch_readiness.py`, `tests/test_field_customer_project_acceptance_routes.py` |
| Product/Admin/Platform/Internal 分层 | 客户页面、治理页面、平台健康和内部设备回调不能混用。 | `tests/test_health.py`, `tests/test_dashboard_http.py` |

## Next Research Questions

1. 方案商当前最痛的是 Demo-to-pilot、验收材料、现场事件闭环，还是多客户项目复用？
2. 客户现在愿意替换的是人工/Excel、定制脚本、OEM app 旁路，还是 VMS/CMMS 的一部分流程？
3. 哪些外部系统是上线必需：VMS、CMMS、IAM、地图、OEM fleet、通知、SIEM/WORM？
4. 哪一类竞品最容易把 AskMe 锁在“只是 Demo 工具”的位置？
5. 如果只能先做一个差异点，客户是否愿意为 acceptance dossier 和 customer signoff 边界付费？
