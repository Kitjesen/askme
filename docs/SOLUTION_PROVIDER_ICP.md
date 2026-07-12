# AskMe Solution Provider ICP

日期：2026-06-05

状态：产品需求工作稿。本文把 `docs/MARKET_RESEARCH.md` 的 P0 市场判断落成需求、交付流程和架构约束；产品需求主干见 `docs/PRODUCT_REQUIREMENTS.md`，替代边界见 `docs/COMPETITIVE_REPLACEMENT_MATRIX.md`，需求到代码所有权、表面和测试的追踪见 `docs/PRODUCT_ARCHITECTURE_TRACE.md`，行业场景卡见 `docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md`，需求证据台账见 `docs/DEMAND_EVIDENCE_LEDGER.md`，场景 ROI 模型见 `docs/SCENARIO_ROI_MODEL.md`，外部系统合同见 `docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md`，上线准入清单见 `docs/SITE_LAUNCH_READINESS_CHECKLIST.md`，定价包装假设见 `docs/PRICING_PACKAGING_HYPOTHESES.md`，访谈验证见 `docs/INTERVIEW_GUIDE_SOLUTION_PROVIDER.md`。仍需客户访谈验证，不等同于已签约客户需求。

## 结论

AskMe 短期应优先做成机器人方案商/集成商交付中台。它不是通用聊天机器人，也不是底盘控制系统；它帮助方案商把一次性 Demo 变成可复制、可验收、可审计的客户项目交付包。

第一价值主张：

> 用客户项目、对象目录、现场事件、知识证据、运行时交接和 acceptance dossier，把 Demo-to-pilot 交付从人肉脚本和截图整理成标准闭环。

## ICP

### 目标客户

| 维度 | 定义 |
| --- | --- |
| 公司类型 | 机器人方案商、系统集成商、机器人渠道交付团队、做多客户现场项目的安防/FM 服务商 |
| 项目形态 | 园区、厂区、仓储、景区、商业综合体等现场机器人试点或 RaaS 项目 |
| 团队现状 | 有机器人或 OEM SDK，有 Demo，有客户，但每个项目都要手工改配置、补材料、整理证据 |
| 预算来源 | 项目交付成本、试点服务费、RaaS 运营费、验收材料费、客户成功/运维预算 |
| 购买人 | 方案商负责人、交付负责人、项目 PM、售前负责人、客户成功负责人 |
| 使用人 | 交付工程师、现场主管、售前、测试、客户负责人 |

### 不适合优先服务的客户

- 只想要游客闲聊或客服 FAQ 的客户。
- 只采购机器人底盘、导航、运动控制或 VLA 控制能力的客户。
- 只需要一次性展厅 Demo、没有试点验收和复制交付压力的客户。
- 要求无现场配置、无人工复核、直接承诺无人值守生产上线的客户。

## 购买触发

1. Demo 已经演示成功，但客户要求解释“怎么验收、怎么复制到下一个现场、哪些还不能上线”。
2. 方案商同时交付多个客户现场，项目 YAML、脚本、截图、日志和验收文档开始失控。
3. 客户要求交付验收材料、权限审计、现场证据、风险说明和上线准入清单。
4. OEM fleet/app 能管机器人，但不能覆盖客户项目、知识证据、签收、审计和多系统现场运营。
5. 现场已有 VMS、CMMS、人工巡检、通知群或工单系统，但事件闭环和机器人任务 handoff 割裂。

## Demo-to-pilot 流程

| 阶段 | 客户问题 | AskMe 产品对象 | 必须过的门 |
| --- | --- | --- | --- |
| 1. 售前 Demo | 这不是普通聊天吗？ | 场景清单、能力包、客户可读说明 | 不承诺生产上线，不暴露硬件控制 |
| 2. 项目建档 | 这是哪个客户、哪个现场？ | customer project、site profile、delivery namespace | 租户/客户/项目边界明确 |
| 3. 对象目录 | 现场有哪些真实对象？ | managed objects、路线、设备、点位、区域规则 | 对象必须能绑定资源和验收项 |
| 4. 资源绑定 | 视觉、传感器、技能和验收用例从哪里来？ | delivery resource registry、skill package、acceptance tests | 禁止 string-only 资源进入验收结论 |
| 5. 实验室演练 | 解析和 handoff 是否跑通？ | rehearsal、dry-run、shadow-post candidate evidence | 只证明接入解析，不证明客户签收 |
| 6. 现场试点 | 现场发生了什么、谁处理？ | field events、notification、onsite evidence、audit timeline | 事件必须可关闭、可复核、可导出 |
| 7. 客户签收 | 能否提交验收？ | acceptance dossier、customer signoff、gap list | 客户签收不等于生产上线 |
| 8. 上线准入 | 能否进入生产？ | product launch readiness、Runtime / Safety / Hardware handoff、`docs/SITE_LAUNCH_READINESS_CHECKLIST.md` | 必须补齐 IAM、硬件、接管、回滚和安全验收 |

## Must-have 需求

### R1. 客户项目工作台

交付团队必须能从行业模板创建客户项目，并维护客户、现场、项目、租户、命名空间和交付阶段。项目不是配置文件，而是交付事实源。

验收信号：

- 能列出所有客户项目和现场。
- 能从模板创建项目。
- 能导出项目交付包。
- 能显示当前项目 blocked/manual_check/ready 的原因。

### R2. 管理对象目录

每个客户现场的楼宇、路线、设备、货架、垃圾桶、消防点、服务点、禁行区等对象必须进入对象目录。对象必须绑定资源、技能和验收用例，不能只存在于脚本或页面文案。

验收信号：

- 每个对象有客户/项目/现场作用域。
- 每个对象能解释缺少哪个视觉模型、传感器协议、技能包或验收项。
- 对象目录能导出给交付、客户和 QA。

### R3. 交付资源治理

方案商需要复用视觉模型、传感器协议、技能包和验收用例，但复用资源必须有版本、状态、适用范围、禁用/回滚和审批记录。

验收信号：

- 资源 registry 是共享事实源。
- 禁用或回滚能显示影响哪些项目、对象和模板。
- 资源治理请求不能自审批。

### R4. 场景验收卡

每个高价值场景都要有验收卡，而不是只展示“系统能回答”。首批场景应限制在 3-5 个高价值项目：问路/导览、异常上报、夜间巡查、设备/通道异常、机器人故障。

验收信号：

- 每张卡说明触发来源、判断证据、通知对象、技能包、播报、归档和未完成项。
- 高风险场景不能由游客一句话直接触发真实硬件。
- 未命中、误命中和改口要进入增长候选，而不是静默丢失。

### R5. Acceptance dossier

客户要看到“本次试点做了什么、证据在哪里、哪些仍阻断上线”。acceptance dossier 必须合并项目、对象、场景、事件、知识、权限、运行时和审计证据。

验收信号：

- dossier 能导出给客户。
- pending review、缺证据、无客户签收、无现场硬件验收时不能声明通过。
- customer signoff 与 production readiness 是不同状态机。

### R6. 运行时安全边界

AskMe 可以理解任务并生成 handoff，但不能直接控制底层硬件。所有执行都必须经过 InteractionGate、RBAC、SkillGate、SafetyPreflight、runtime profile 和 Runtime / Safety / Hardware。

验收信号：

- `fake/sim/shadow/lab/prod` profile 对客户声明有明确边界。
- `dry_run` 和 `shadow_post` 证据不能被包装成生产验收。
- Product/Admin/Platform/Internal API 表面不混用客户 UI 和内部硬件控制。

### R7. 方案商交付 Dashboard

Dashboard 应按角色拆分，不应把所有功能塞进一个页面。方案商负责人看交付总门禁；交付工程师看项目、对象和资源；现场主管看事件和证据；客户负责人看验收范围和阻断项。

验收信号：

- 每个页面能回答“客户现在能验收什么”。
- 治理页面能解释为什么 blocked。
- 内部机器人控制接口不驱动客户交付口径。

## 高级架构约束

1. Field Delivery Domain 是产品需求的事实源。`askme/pipeline/field` owns customer projects, managed objects, field events, onsite evidence, customer signoff, and readiness gates。
2. API route/service 和 Dashboard 只是表面。`askme/api/routes/field_*`、`askme/api/services/field_*` 和 `askme/static` 不能独立创造验收结论。
3. Runtime handoff 是边界，不是所有权迁移。AskMe 只把通过门禁的任务交给 Runtime / Safety / Hardware，不拥有底盘、导航、电机或机械臂控制。
4. 审计和证据必须可追溯。客户交付材料必须能追到事件、操作员、资源版本、知识来源、运行 profile 和导出 hash。
5. 集成层按合同接入。VMS、CMMS、IAM、地图、OEM fleet、通知系统和 SIEM/WORM 必须通过明确字段、失败状态和审计记录接入，而不是写成一次性脚本。
6. 多客户隔离优先于演示便利。tenant/customer/project/site/object 作用域错误时，系统应该阻断验收，而不是用默认项目兜底。
7. 架构话术必须服务产品边界：可演示、可试点、可验收；不承诺未接硬件的无人值守生产上线。

## 访谈验证

必须验证的问题：

1. 最近 3 个 Demo-to-pilot 项目分别耗时多久，卡在哪里？
2. 每个项目人工补了哪些验收材料，花了多少小时？
3. 客户最常要求哪些上线前证据？
4. 当前替代物是什么：Excel、微信群、OEM app、VMS、CMMS、定制脚本还是工单系统？
5. 如果只买一个能力，客户愿意为客户项目工作台、现场事件闭环、验收包、知识证据还是运行安全边界付费？
6. 定价更适合项目费、站点费、机器人数量、技能包、交付包还是 RaaS 运营报告加成？候选包装见 `docs/PRICING_PACKAGING_HYPOTHESES.md`。
7. 哪些系统不集成就无法上线？

访谈成功标准：

- 至少 6 个方案商/集成商样本能复盘真实项目时间线。
- 至少 4 个样本愿意给出当前交付材料样例或脱敏目录。
- 至少 3 个样本明确愿意为验收包、项目复制或事件闭环付费。
- 能判断 P0 是否成立；若不成立，转向安保/FM 服务报告和现场事件闭环。

## 近期产品动作

1. 把现有客户项目、模板市场、对象目录、交付资源和 acceptance dossier 串成一个方案商交付路径。
2. 维护 `docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md` 和 `docs/SCENARIO_ROI_MODEL.md`：为园区、厂区、仓储、景区各保留 3 个高价值场景卡，并用 baseline、target_delta、value metric、payback signal 验证 ROI。
3. 在 Dashboard 上强化“验收阻断项”和“客户签收不等于生产上线”。
4. 维护 `docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md`，为 VMS/CMMS/IAM/OEM fleet/通知/SIEM 形成最小字段合同和失败状态。
5. 用访谈证据决定短期包装是否正式改为“机器人方案商交付中台”；进入 validated 状态前必须先维护 `docs/DEMAND_EVIDENCE_LEDGER.md`，再更新 `docs/PRICING_PACKAGING_HYPOTHESES.md` 的收费单位、usage evidence 和禁止包装边界。
