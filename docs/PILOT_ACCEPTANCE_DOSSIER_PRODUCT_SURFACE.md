# Pilot Acceptance Dossier Product Surface

日期：2026-06-05

状态：产品面与高级架构契约。本文承接 `docs/MARKET_RESEARCH.md` 第 4 个下一步动作，把“试点验收包产品面”定义为方案商交付路径，而不是单个报表、单个 Dashboard 卡片或销售附件；相关定价包装假设见 `docs/PRICING_PACKAGING_HYPOTHESES.md`。

## Product Job

试点验收包产品面要回答一个客户问题：

> 这次 Demo-to-pilot 到底交付了什么、证据在哪里、哪些缺口阻断上线、谁承担哪个责任边界、客户签收代表什么。

它必须把对象目录、场景清单、证据、缺口、责任边界和客户签收放在同一个客户可读路径里，同时明确客户签收不等于生产上线。英文契约保持为 customer signoff != production readiness。

## Reader Surfaces

| 读者 | 产品面 | 必须看到 | 不能看到或不能宣称 |
| --- | --- | --- | --- |
| 方案商负责人 | 交付总览 | 项目状态、验收阻断项、客户签收状态、blocked_uses | 不能把签收包装成无人值守生产上线 |
| 交付工程师 | 对象和证据工作台 | 对象目录、资源绑定、验收用例、现场证据、缺口 | 不能用页面状态替代 Field Delivery Domain 事实 |
| 现场主管 | 事件和责任闭环 | 场景清单、现场事件、通知、处理人、关闭记录 | 不能看到内部硬件控制接口 |
| 客户负责人 | 签收包 | acceptance dossier、责任边界、风险确认、签收历史 | 不能看到未脱敏内部日志或跨客户项目 |
| 研发/QA | 验证锚点 | API 响应、测试证据、manifest hash、HMAC/签名状态 | 不能通过测试替代真实现场验收 |

## Required Product Blocks

| Block | 来源事实 | 用户价值 | 验收边界 |
| --- | --- | --- | --- |
| 对象目录 | customer project, site profile, managed objects | 说明本次试点覆盖哪些楼栋、路线、设备、货架、服务点或区域 | 对象未绑定资源时只能 manual_check |
| 场景清单 | `docs/INDUSTRY_SCENARIO_DEMAND_CARDS.md` 和 field scenarios/events | 说明客户逐条验收哪些高价值场景 | 不新增宽泛行业模板 |
| 证据 | onsite evidence, audit evidence, readiness receipts | 证明发生过什么、谁处理、证据 hash 是什么 | local/mock/lab 证据不能冒充现场验收 |
| 缺口 | acceptance closure, readiness gaps, blocked_uses | 告诉客户哪些仍阻断上线或越界使用 | 缺口不能被 Dashboard 文案隐藏 |
| 责任边界 | delivery boundary, runtime profile, risk acknowledgement | 说明 AskMe、方案商、OEM、客户各自负责什么 | 不直接承诺底盘、导航、电机、机械臂控制 |
| 客户签收 | customer signoff record | 记录客户是否接受本次试点范围 | 客户签收不等于生产上线 |

## Architecture Ownership

Field Delivery Domain 是试点验收包产品面的事实源：

- `askme/pipeline/field` 拥有客户项目、对象目录、现场事件、证据、验收报告、acceptance dossier、acceptance closure、customer signoff 和 blocked_uses。
- `askme/api/routes/field_*` 与 `askme/api/services/field_*` 是传输和组合层，只能暴露 domain 给出的事实和状态。
- `/dashboard/delivery` 是客户项目与验收材料的读者路径和表单表面，不能独立生成验收结论，也不能把 Product/Admin/Platform/Internal 分层压扁成一个万能页面。
- Runtime / Safety / Hardware 拥有真实执行和硬件状态；试点验收包只能记录受控 handoff、runtime profile、人工确认和现场证据。
- Product/Admin/Platform/Internal 分层必须保持：Product 给客户解释，Admin 做审批与验收，Platform 展示健康和 readiness，Internal 接 runtime/device/vision callback。
- VMS、CMMS、IAM、地图、OEM fleet、通知系统和 SIEM/WORM 的外部证据必须符合 `docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md`，否则只能进入缺口或 manual_check。
- 上线前硬件/现场验收必须符合 `docs/SITE_LAUNCH_READINESS_CHECKLIST.md`；acceptance dossier 和 customer signoff 只能作为输入证据，不能替代 production readiness。
- `docs/PRICING_PACKAGING_HYPOTHESES.md` 可以把 acceptance dossier、交付包和 RaaS 运营报告作为收费假设，但这些 usage evidence 只能证明交付和签收活动，不能把 customer signoff 改写成 production readiness。

## API Contract Anchors

当前产品面应该围绕这些端点组织，而不是另起一套验收事实：

- `GET /api/field/customer-projects/{identifier}/acceptance-dossier` 导出客户可读 JSON/HTML dossier。
- `POST /api/field/customer-projects/acceptance-dossier/verify` 在客户签收前验证 dossier hash、签名和作用域。
- `GET /api/field/customer-projects/{identifier}/acceptance-closure` 汇总内部交付门禁、现场证据、dossier verification、proposal bundle verification、审计导出和下一步。
- `GET /api/field/customer-projects/{identifier}/customer-signoff` 展示客户签收历史和当前决策。
- `POST /api/field/customer-projects/{identifier}/customer-signoff` 记录 accepted、needs_fix 或 rejected，并保留风险确认和证据引用。

这些端点必须继续证明 customer signoff != production readiness：accepted_by_customer 只能证明客户接受了本次试点范围；blocked_uses 仍要阻断无人值守生产上线、越界场景和未经现场验收的硬件声明。

## Dashboard Contract

`/dashboard/delivery` 的试点验收包路径应按交付问题组织：

1. 项目和对象：这是哪个客户、哪个现场、哪些对象。
2. 场景和证据：哪些 R4 场景验收卡被覆盖，证据来自哪里。
3. 缺口和责任：哪些 gate 仍是 blocked/manual_check，谁需要处理。
4. 验收包：acceptance dossier 是否可导出、是否可验证、hash 是否匹配。
5. 客户签收：签收决策、风险确认、证据引用和 blocked_uses。

页面可以解释状态，但不能绕过 API、不能直接使用 Internal runtime/device 控制接口，也不能把 readiness summary 改写成销售承诺。

## Verification Contract

产品面完成度不能只看页面是否存在。最小验证链是：

| Claim | Evidence |
| --- | --- |
| 产品文档没有扩大上线承诺 | `tests/test_repository_layout.py` 产品定位和本文件契约测试 |
| acceptance dossier 可导出和校验 | `tests/test_field_customer_project_acceptance_routes.py` |
| 客户签收和生产上线保持分离 | `tests/test_product_launch_readiness.py` 与 field acceptance route tests |
| Dashboard 只是表面 | `tests/test_dashboard_customer_project_contract.py` 与 `tests/test_dashboard_http.py` |
| runtime handoff 没有变成硬件所有权 | `tests/test_runtime_handoff.py` |

Full `pytest tests` 仍是大脏工作区最终信心门，但本产品面每次调整至少要跑 `tests/test_repository_layout.py` 和 field/customer-project acceptance 相关定向测试。
