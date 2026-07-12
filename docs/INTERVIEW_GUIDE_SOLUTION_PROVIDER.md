# Solution Provider Interview Guide

日期：2026-06-05

状态：访谈验证工作稿。本文用于验证 `docs/SOLUTION_PROVIDER_ICP.md`、`docs/COMPETITIVE_REPLACEMENT_MATRIX.md`、`docs/PRODUCT_ARCHITECTURE_TRACE.md`、`docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md`、`docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md`、`docs/DEMAND_EVIDENCE_LEDGER.md` 和 `docs/PRICING_PACKAGING_HYPOTHESES.md` 中的 ICP、替代边界、R1-R7 需求、行业场景卡、外部系统合同、证据状态、定价包装和高级架构假设。它不是销售话术，也不是功能承诺。

## 调研目标

验证 AskMe 是否应该优先定位为机器人方案商/集成商交付中台，以及客户是否真的愿意为 Demo-to-pilot 复制交付、acceptance dossier、现场事件闭环和运行安全边界付费。

核心判断：

1. P0 是否成立：机器人方案商/集成商是否比终端客户更痛、更快买、更能复用。
2. 替代物是否明确：客户实际替换的是人工运营 + 微信/Excel、定制脚本、一部分 OEM fleet/app 旁路，还是 VMS/CMMS 流程。
3. R1-R7 是否是 Must-have：客户项目、对象目录、交付资源治理、场景验收卡、acceptance dossier、运行时安全边界、方案商 Dashboard 是否真能缩短交付。
4. 高级架构边界是否被客户接受：customer signoff != production readiness，不承诺无人值守生产上线，Runtime / Safety / Hardware 仍拥有真实执行。

## 样本设计

目标样本：20-30 人。先做 12 人快速验证，再决定是否扩展。

| 角色 | 建议数量 | 主要验证 |
| --- | --- | --- |
| 机器人方案商/集成商负责人 | 6-8 | 预算、采购触发、项目复制压力、付费意愿 |
| 交付工程师/实施 PM | 5-6 | Demo-to-pilot 卡点、项目资料、脚本、验收材料 |
| 售前/解决方案顾问 | 3-4 | 客户最常问什么、销售承诺风险、竞品替代 |
| 安保/FM/物业服务商项目负责人 | 3-4 | 多甲方服务报告、夜班巡检、人员流动、RaaS 运营 |
| OEM 渠道/生态负责人 | 2-3 | OEM fleet/app 边界、SDK 开放意愿、平台主导权 |
| 终端客户现场主管 | 3-5 | 现场事件闭环、验收标准、系统集成阻断 |

筛选条件：

- 最近 12 个月做过机器人、AI 现场服务、巡检、安防、导览、清洁或设施项目。
- 至少经历过一次 Demo-to-pilot 或试点验收。
- 能描述当前工具链：人工、微信/Excel、OEM app、VMS、CMMS、工单、定制脚本或 Agent 平台。
- 能给出脱敏项目材料、验收目录、事件记录、工单截图或交付 checklist 的任意一种。

排除条件：

- 只做纯底盘/运动控制研发，完全不负责客户交付。
- 只做展厅 Demo，没有试点或验收压力。
- 只想采购通用客服聊天机器人。

## 访谈结构

每场 45-60 分钟。不要先介绍 AskMe 功能；先让对方复盘真实项目。

### 1. 背景和角色

1. 你在最近一个机器人/AI 现场项目里负责什么？
2. 项目客户是谁，场景是什么：园区、厂区、仓储、景区、商业综合体、安保/FM，还是其他？
3. 这个项目最后停在 Demo、试点、客户签收、生产上线，还是失败？

### 2. Demo-to-pilot 复盘

1. 从第一次 Demo 到试点启动用了多久？
2. 卡得最久的三个环节是什么？
3. 哪些内容是每个客户都要重做的：客户项目、点位、路线、设备、知识、技能、通知、验收材料？
4. 哪些材料是交付团队手工补的：截图、日志、视频、Excel、PPT、报告、审计记录？
5. 如果下一个客户相似，当前能复用多少？不能复用的原因是什么？

### 3. 替代物和竞品

1. 你现在用什么管理项目和证据：人工运营 + 微信/Excel、OEM fleet/app、VMS / AI 告警、CMMS / 工单、定制项目脚本 / 一次性 Demo，还是通用 LLM Agent 平台？
2. 哪个工具是预算来源，哪个只是临时补丁？
3. OEM fleet/app 做得好的部分是什么？你不希望 AskMe 抢什么？
4. VMS 或 CMMS 已经覆盖了哪些流程？哪些还必须人工串起来？
5. 如果 AskMe 只替代一个东西，应该先替代什么？

### 4. R1-R7 需求验证

| 需求 | 访谈问题 | 强需求信号 |
| --- | --- | --- |
| R1 客户项目工作台 | 你是否需要一套客户/现场/项目/租户/命名空间目录？现在在哪里维护？ | 多客户项目混乱、复制靠人工、跨客户数据风险 |
| R2 管理对象目录 | 现场对象、路线、设备、点位和区域规则现在怎么维护？ | 对象散落在地图、Excel、脚本、PPT 或人员记忆 |
| R3 交付资源治理 | 视觉模型、传感器协议、技能包和验收用例是否复用？谁审批禁用/回滚？ | 资源版本不清、影响项目无法预估 |
| R4 场景验收卡 | 客户希望逐条验收 `docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md` 中哪些场景？ | 客户要求看触发、证据、通知、处理和未完成项 |
| R5 acceptance dossier | 客户签收前必须提交什么证据？ | 人工整理验收包耗时，且经常缺证据 |
| R6 运行时安全边界 | 哪些动作必须人工确认？哪些绝不能自动执行？ | 客户明确要求权限、二次确认、接管、回滚 |
| R7 方案商 Dashboard | 负责人、交付工程师、现场主管、客户分别想看什么？ | 同一页面无法回答不同角色问题 |

### 5. 高级架构假设验证

1. 你是否接受 customer signoff != production readiness？客户签收和生产上线在你们流程里是不是两个状态？
2. 你是否接受 AskMe 不替代底盘控制，只把通过门禁的任务交给 Runtime / Safety / Hardware？
3. 哪些系统不集成就无法上线：VMS、CMMS、IAM、地图、OEM fleet、通知、SIEM/WORM？这些系统是否能提供 `docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md` 中的最小字段、失败状态和审计证据？
4. 如果 tenant/customer/project/site/object 作用域不匹配，系统应该阻断，还是用默认项目兜底？
5. 销售最容易过度承诺的地方是什么？

### 6. 付费意愿

用 forced choice，不要只问“愿不愿意买”；候选收费单位和禁止包装边界见 `docs/PRICING_PACKAGING_HYPOTHESES.md`：

1. 如果只能付费买一个能力，你选哪个：客户项目工作台、对象目录、现场事件闭环、acceptance dossier、知识证据治理、运行安全边界、方案商 Dashboard？
2. 你更能接受哪种收费：项目费、站点费、机器人数量、技能包、交付包、RaaS 运营报告加成？
3. 这个工具帮你节省多少交付工程师时间才值得买？
4. 谁有预算，谁能拍板，谁会反对？

## 证据采集

访谈后尽量收集脱敏材料：

- Demo-to-pilot 项目时间线。
- 验收目录、PPT、报告、交付 checklist。
- 客户提出的上线前证据要求。
- 现场事件台账、工单截图、通知记录。
- 项目配置、脚本、Excel 或对象目录样例。
- 当前 OEM fleet/app、VMS、CMMS、IAM、地图、通知系统和 SIEM/WORM 的接入清单。

不要收集客户隐私、密钥、真实个人联系方式、支付信息或敏感现场照片。只要结构、字段和流程证据。

## 判断阈值

| 结论 | 达成条件 |
| --- | --- |
| P0 成立 | 至少 6 个方案商/集成商样本能复盘真实 Demo-to-pilot 卡点，且至少 3 个明确愿意为验收包、项目复制或事件闭环付费。 |
| R1/R2 是 Must-have | 超过半数样本存在客户项目/对象目录散落、复制困难或跨客户风险。 |
| R5 是强价值 | 至少 4 个样本有人工整理验收材料、客户签收阻断或证据缺口。 |
| R6 架构边界成立 | 多数样本认同不能直接控制硬件，且需要权限、二次确认、接管和回滚。 |
| P0 不成立 | 方案商不愿付费，或痛点主要在终端客户运营报告。转向安保/FM 服务报告和现场事件闭环。 |

## 访谈记录模板

```text
Interview ID:
Evidence ID:
Source type: interview
Role:
Company type:
Project type:
Stage reached: Demo / pilot / customer signoff / production readiness / failed

Current substitutes:
- Manual + WeChat/Excel:
- OEM fleet/app:
- VMS / AI alert:
- CMMS / work order:
- Custom scripts:
- Generic LLM Agent:

Top 3 Demo-to-pilot blockers:
1.
2.
3.

R1-R7 signal:
- R1 customer project:
- R2 managed object directory:
- R3 delivery resources:
- R4 scenario acceptance cards:
- R5 acceptance dossier:
- R6 runtime safety boundary:
- R7 delivery dashboard:

Architecture boundary reaction:
- customer signoff != production readiness:
- no raw chassis control:
- required integrations:
- scope mismatch should block:

Pricing forced choice:
- primary paid capability:
- preferred pricing unit:
- budget owner:
- objection:

Evidence offered:
Next follow-up:
Confidence: low / medium / high
Hypothesis status: research_pending / validated / contradicted / mixed / not_applicable
Sample count:
Redaction required: yes
No secrets/no PII: confirmed
```

## 输出要求

每条访谈记录都要进入 `docs/DEMAND_EVIDENCE_LEDGER.md`：没有 evidence_id、sample_count、hypothesis_status、redaction_required 和 no secrets/no PII 的记录，不能升级成 validated 需求。

完成 12 人快速访谈后，输出：

1. P0 是否继续成立。
2. R1-R7 的强/中/弱需求排序。
3. 替代物优先级：人工/Excel、定制脚本、OEM app、VMS、CMMS、通用 Agent。
4. 必须集成系统清单。
5. 付费模型优先级。
6. 需要修改的产品口径和架构边界。
