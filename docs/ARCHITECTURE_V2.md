# Askme 架构 v2 —— ZeroClaw 集成与企业安全架构

> 更新时间：2026-06-05
> 前置阅读：`docs/ARCHITECTURE.md`、`docs/PRODUCT_REQUIREMENTS.md`、`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`

---

## 1. 概述

v2 架构在 v1 基础上引入两个核心扩展：

1. **ZeroClaw 集成** —— 将 Askme 定位为现场运营交付中台，接受 ZeroClaw（Agent 大脑）通过 MCP 协议下达的任务指令。
2. **企业安全架构** —— 形式化四阶段安全门禁，确保 LLM 驱动的决策不越权控制硬件。

---

## 2. ZeroClaw 集成架构

### 2.1 定位

版本基线：`zeroclaw 0.1.7`。ZeroClaw 和 NanoClaw 不作为两套不同 Agent 系统维护，而是同一运行时家族的两种形态：

- `ZeroClaw`：标准 Agent/Gateway profile，负责桌面、服务器和 NERVA Studio 侧的规划、MCP 工具编排和长上下文决策。
- `NanoClaw`：预留的边缘 profile，面向真机/边缘盒子的轻量守护、状态回报、弱网兜底和固定安全动作。

因此 Askme 只需要对接一个 ZeroClaw/NanoClaw 兼容的 MCP/API 合同；NanoClaw 不重新定义工具、权限、记忆或硬件控制边界。

```
ZeroClaw (Agent 大脑)
    │  任务规划、上下文推理、子 Agent 编排
    │
    │ MCP 协议 (askme-mcp server)
    ▼
Askme (现场运营交付中台)
    │  语音交互、知识检索、现场事件、客户项目、证据、运行交接
    │
    ▼
Runtime / Safety / Hardware
    运动控制、导航、机械臂、传感器
```

ZeroClaw 作为高阶 Agent 大脑，负责任务分解和抽象推理。Askme 作为现场运营交付中台，提供受控工具、客户项目上下文、现场证据和运行交接给 MCP 客户端，但**不提供原始硬件控制权限**。

Askme 不替代机器人底盘、fleet app 或运动控制服务。它的架构职责是把客户现场需求转成可审计、可验收、可复用的任务和证据闭环，再把通过门禁的执行请求交给 Runtime / Safety / Hardware 层。

### 2.2 市场需求驱动的架构不变量

`docs/MARKET_RESEARCH.md` 把 Askme 的优先市场定义为机器人现场运营交付中台，而不是机器人聊天工具或底盘控制系统。`docs/SOLUTION_PROVIDER_ICP.md` 进一步把 P0 客户压到机器人方案商/集成商交付中台，并把 Demo-to-pilot 流程转成产品需求和架构约束。`docs/PRODUCT_REQUIREMENTS.md` 是 PRD 级产品需求主干，用来约束 P0、R1-R7、证据门禁、ROI/定价和 release gates。`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md` 定义 bounded contexts、包所有权、API 表面和架构变更门禁。`docs/DEMAND_EVIDENCE_LEDGER.md` 约束访谈、试点和材料证据状态，避免把 hypothesis 直接升级成架构不变量。`docs/PRODUCT_ARCHITECTURE_TRACE.md` 把这些需求映射到 Field Delivery Domain、API 表面、Runtime 边界和验证测试；`docs/PILOT_ACCEPTANCE_DOSSIER_PRODUCT_SURFACE.md` 则把试点验收包产品面限定为对象目录、场景清单、证据、缺口、责任边界和客户签收路径；`docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md` 约束 VMS、CMMS、IAM、地图、OEM fleet、通知系统和 SIEM/WORM 的最小字段、失败状态和审计边界；`docs/SITE_LAUNCH_READINESS_CHECKLIST.md` 约束 site_acceptance_checklist、launch_readiness、接管、回滚和现场硬件证据；`docs/SCENARIO_ROI_MODEL.md` 约束园区、厂区、仓储、景区场景 ROI 只能读取 baseline、target_delta、value metric、payback signal 和 usage evidence；`docs/PRICING_PACKAGING_HYPOTHESES.md` 约束项目费、站点费、机器人数量、技能包、交付包和 RaaS 运营报告只能绑定 usage evidence，不能改变 Field Delivery Domain 与 Runtime / Safety / Hardware 的所有权。由此推导出以下架构不变量：

| 市场需求 | 架构约束 |
| --- | --- |
| 方案商需要把 Demo 复制成可验收试点 | 客户项目、对象目录、行业模板、交付包和验收 dossier 必须是产品事实源，不能散落在一次性脚本里 |
| 现场主管需要事件闭环和证据 | Field Events、Evidence、Notification、Close/Review、Audit 必须形成同一条时间线 |
| 机器人必须能听懂人但不能乱动 | InteractionGate、RBAC、SkillGate、SafetyPreflight、Runtime Profile 必须串联，任何入口都不能绕过 |
| 知识回答必须可控 | KnowledgeCatalog 是信任事实源，RAG backend 只是检索实现 |
| 重复需求要沉淀成技能包 | Skill Growth 只能生成待审批草稿，启用必须经过 Skill Package、发布通道、灰度和审计 |
| 客户签收不等于生产上线 | acceptance dossier、onsite evidence、customer signoff 和 production readiness 必须保持不同状态机 |
| 外部系统是合同边界，不是事实所有者 | VMS、CMMS、IAM、地图、OEM fleet、通知系统和 SIEM/WORM 必须通过最小字段、失败状态和审计记录进入 Field Delivery Domain |
| 上线准入必须有现场硬件证据 | site_acceptance_checklist、launch_readiness、runtime roundtrip、takeover 和 rollback 必须独立于 customer signoff |
| ROI 是读模型，不是事实源 | baseline、target_delta、value metric、payback signal 和 pricing_signal 只能读取 Field Delivery Domain 的 usage evidence，不能由 Dashboard 自算通过 |
| 定价包装不能扩大产品承诺 | 项目费、站点费、机器人数量、技能包、交付包和 RaaS 运营报告只能读取 pricing_signal/usage evidence，不能把 Dashboard、Internal 或 Runtime 状态包装成生产上线 |
| 证据状态先于架构升级 | research_pending、validated、contradicted、sample_count 和 confidence 必须先进入需求证据台账；单个访谈不能生成架构不变量 |

### 2.3 MCP 集成点

Askme 通过 `askme.mcp.server` 暴露 MCP 工具服务，注册在 `pyproject.toml` 中：

```toml
[project.scripts]
askme-mcp = "askme.mcp.server:main"
```

MCP 服务器暴露的能力：

| 类别 | 工具/资源 | 安全控制 |
|------|-----------|----------|
| 知识检索 | memory_search | SkillGate + RBAC |
| 空间查询 | space_lookup_place, space_recommend_route | 只返回文本，不控制底盘 |
| 技能查询 | list_skills, skill_detail | 只读 |
| 现场事件 | field_event_trigger | SkillGate 审批 + SafetyPreflight |
| 机器人状态 | robot_state | 只读，来自运行时真相 |
| 文档资源 | docs/* | 静态 MCP 资源 |

### 2.4 ZeroClaw 决策流

```text
User intent
  │
  ▼
ZeroClaw (MCP Client)
  │  ── 分析意图
  │  ── 调用 askme MCP 工具获取上下文
  │  ── 规划多步骤任务
  │  ── 下达可执行的 TaskPlan
  │
  ▼
MCP 协议 ─── askme-mcp server
  │
  ▼
Askme TaskHandoff
  │  ── 验证任务合法性
  │  ── 检查权限与风险等级
  │
  ▼
SafetyPreflight
  │  ── 硬件安全预检
  │
  ▼
Runtime Arbiter ─── 执行
```

**关键约束：** ZeroClaw 不直接控制硬件。所有硬件相关操作必须经过 Askme 的 TaskHandoff → SafetyPreflight → Runtime Arbiter 链路。

---

## 3. 企业安全架构

### 3.1 四阶段安全门禁

```
┌─────────────────────────────────────────────────────┐
│ 1. ZeroClaw 决策                                    │
│    任务规划、子 Agent 编排                           │
│    约束：仅输出 TaskPlan，不可直接调用硬件            │
└──────────────────┬──────────────────────────────────┘
                   │ MCP / API
                   ▼
┌─────────────────────────────────────────────────────┐
│ 2. Askme 验证                                       │
│    ├─ InteractionGate —— 语音/文本输入验证           │
│    ├─ SkillGate     —— 技能白名单 + 风险等级         │
│    ├─ RBAC          —— 操作员权限校验                │
│    └─ RAG Policy    —— 知识证据校验                  │
└──────────────────┬──────────────────────────────────┘
                   │ 验证通过
                   ▼
┌─────────────────────────────────────────────────────┐
│ 3. SafetyPreflight                                  │
│    ├─ 运行时可用性检查                               │
│    ├─ 硬件安全状态检查                               │
│    ├─ 操作确认（高风险任务）                          │
│    └─ 审计记录写入                                   │
└──────────────────┬──────────────────────────────────┘
                   │ 预检通过
                   ▼
┌─────────────────────────────────────────────────────┐
│ 4. Runtime Arbiter 执行                              │
│    ├─ TaskRun 状态机管理                             │
│    ├─ RuntimeEvent 流                                │
│    ├─ 硬件控制接口（通过 dog-safety-service）         │
│    └─ TaskReport + Audit evidence                    │
└─────────────────────────────────────────────────────┘
```

### 3.2 安全不变量

| 不变量 | 说明 |
|--------|------|
| LLM 不直接控制硬件 | 所有硬件动作必须经过 SafetyPreflight + Runtime Arbiter |
| 高风险任务需确认 | 风险等级为 high 的任务在 planning 阶段必须获得操作员确认 |
| 知识过期不可用 | expired/conflict 状态的知识不进入 prompt |
| 审计不可篡改 | 所有 operator action 记录 actor、reason、risk acknowledgement |
| 默认安全 | lab/prod 运行时必须显式启用，默认禁用 |
| 技能不能绕过安全 | 现场技能通过 `field_event_trigger` 调用，不直接控制硬件 |
| 客户可见接口不暴露硬件 | product 表面不能注册运行时仲裁或硬件控制路由 |

### 3.3 权限分层

```
企业 IAM / OIDC 网关
  │  Token 校验
  │
  ├── x-askme-iam-operator-id
  ├── x-askme-iam-roles
  └── x-askme-iam-display-name
        │
        ▼
Askme RBAC
  ├── OperatorDirectory（操作员注册表）
  ├── 角色 → 权限映射
  ├── 项目范围（tenant / customer / project / site 层级）
  └── Operator Audit Log
        │
        ▼
SkillGate（技能级安全）
  ├── 技能白名单（built-in / generated）
  ├── 风险等级（normal / elevated / high / critical）
  ├── 执行前确认策略
  └── 包发布/灰度控制
```

---

## 4. 运行时安全升级

### 4.1 任务执行安全门禁

任务从创建到完成经过以下安全检查点：

```
TaskPlan（用户意图）
  │
  ├── InteractionGate ── 语音误识别拦截
  ├── SkillGate ──────── 技能白名单检查
  ├── RBAC ───────────── 操作员权限验证
  ├── RAG Policy ─────── 知识证据二次校验
  │
  ▼
TaskHandoff（结构化计划）
  │
  ├── SafetyPreflight ── 硬件安全预检
  ├── Risk Assessment ── 风险等级评估
  ├── Operator Confirm ─ 高风险操作确认
  │
  ▼
Runtime Arbiter
  │
  ├── Runtime Profile ── fake/sim/shadow/lab/prod
  ├── TaskRun ────────── 状态机（含 pause/resume/cancel）
  ├── RuntimeEvent ───── 实时事件流
  │
  ▼
Hardware Interface
  ├── dog-safety-service（急停/安全状态）
  ├── dog-control-service（运动控制）
  └── 审计记录
```

### 4.2 语音安全

语音入口同样遵守以上安全门禁：

```text
ASR 转录
  → Interaction Gate（唤醒检测/多人仲裁/声画一致性）
  → 确认阶段（高风险任务需语音确认）
  → 取消阶段（可取消已确认的计划）
  → 执行阶段（受 Runtime Arbiter 控制）
  → 停下命令（优先安全路径，不受 LLM 干预）
```

关键规则：
- 旁人的话不唤醒机器人
- 多人同时说话时澄清
- 声源和画面人物不一致时不猜
- "停下"、"急停"等安全相关命令走安全优先路径
- ASR 误识别不能绕过安全确认

---

## 5. MCP 安全边界

Askme 的 MCP 工具服务遵守以下边界原则：

| 原则 | 说明 |
|------|------|
| 只读优先 | 默认暴露只读工具（知识检索、空间查询、状态读取） |
| 写操作受控 | 现场事件触发等写操作需要 SkillGate + RBAC 双重校验 |
| 不暴露硬件控制 | MCP 工具不提供 cmd_vel、motor、gait 等原始控制接口 |
| 结果不替代安全 | MCP 返回的空间/状态信息不替代 SafetyPreflight 检查 |
| 审计完整 | 所有 MCP 工具调用写入 SkillAuditLog |

---

## 6. 四表面 API 安全

Askme 的 HTTP API 被分为四个表面，每个表面有独立的安全策略：

| 表面 | 路由数 | 客户可见 | 硬件权限 | 安全策略 |
|------|--------|----------|----------|----------|
| Platform | 系统健康/指标 | 否 | 否 | 无业务数据，只含运营指标 |
| Product | 对话/知识/能力 | 是 | 否 | 不能暴露硬件控制或运行时仲裁 |
| Admin | 治理/审计/审批 | 否 | 否 | 高风险操作受 RBAC 保护 |
| Internal | 运行时/视觉/设备 | 否 | 允许 | 可承接硬件回调，不能出现在客户 UI |

此分层确保：
- Dashboard 不是单一权限表面；客户说明页依赖 Product，治理/交付页显式依赖 Admin，系统状态页可依赖 Platform。
- Internal 只能用于运行时、设备、视觉、回调和低层接入，不能驱动客户说明页、销售口径或验收结论。
- 新客户业务路由必须归类到 Product 表面。
- 新审批、审计和交付治理路由必须归类到 Admin 表面。
- 新机器人运行时、设备接入和低层回调路由必须归类到 Internal 表面。

---

## 7. 可靠性与可观测性

### 7.1 健康监控

```
/healthz  (liveness,  <10ms)
/ready    (readiness, full check)
/health   (full health document with components)
/metrics  (Prometheus format)
/trace    (pipeline timing traces)
```

### 7.2 追踪

每个 HTTP 请求自动注入 `trace_id`，贯穿所有日志和外部调用：

```text
X-Trace-Id 请求头 → TraceContext → 结构化日志 → 响应头
```

### 7.3 审计流水线

```
Operator Action → AuditQueryService → AuditExportService
  │                                      │
  ├── SkillAuditLog (技能审计)            ├── 导出证据包 (JSONL + manifest)
  ├── Field Audit (现场事件审计)           ├── SHA-256 + 可选 HMAC
  ├── Runtime Audit (运行时审计)           └── SIEM/WORM webhook 投递
  └── Review Service (复核决策)
```

---

## 8. 文档索引

| 文档 | 路径 | 说明 |
|------|------|------|
| 架构 v1 | `docs/ARCHITECTURE.md` | 基础架构、模块、模块依赖方向 |
| 架构 v2 | 本文 | ZeroClaw 集成 + 企业安全架构 |
| 高级软件架构蓝图 | `docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md` | bounded contexts、包所有权、API 表面和架构变更门禁 |
| 市场调研 | `docs/MARKET_RESEARCH.md` | 市场边界、客户分群、需求假设和访谈计划 |
| 方案商 ICP | `docs/SOLUTION_PROVIDER_ICP.md` | P0 客户、Demo-to-pilot 流程、产品需求和架构约束 |
| 产品架构追踪 | `docs/PRODUCT_ARCHITECTURE_TRACE.md` | R1-R7 需求到代码所有权、表面边界和验证测试的映射 |
| 试点验收包产品面 | `docs/PILOT_ACCEPTANCE_DOSSIER_PRODUCT_SURFACE.md` | 对象目录、场景清单、证据、缺口、责任边界和客户签收路径 |
| 外部系统合同 | `docs/EXTERNAL_SYSTEM_INTEGRATION_CONTRACTS.md` | VMS、CMMS、IAM、地图、OEM fleet、通知系统和 SIEM/WORM 的最小字段、失败状态和审计边界 |
| 上线准入清单 | `docs/SITE_LAUNCH_READINESS_CHECKLIST.md` | site_acceptance_checklist、launch_readiness、runtime roundtrip、接管、回滚和现场硬件证据 |
| API 文档 | `docs/API.md` | 全部 HTTP 端点参考 |
| 产品手册 | `docs/PRODUCT.md` | 产品能力和路线图 |
| 运维手册 | `docs/OPERATIONS.md` | 部署和运维指南 |
| 操作边界 | `docs/ASKME_BOUNDARY.md` | 核心/感知/插件三层边界 |
| 主动智能 | `docs/PROACTIVE_INTELLIGENCE_PLAN.md` | 主动监测和 ReactionEngine |
