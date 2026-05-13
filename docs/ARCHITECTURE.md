# askme Architecture

更新时间：2026-05-13

本文是 askme 当前唯一的架构入口。

## 一句话架构

```text
User speech/text
  -> Interaction Gate
  -> ASR / text input
  -> LLM + Memory + RAG policy
  -> Cognition Planner
  -> TaskHandoff
  -> SafetyPreflight
  -> Runtime Arbiter
  -> fake/sim/shadow/lab runtime
  -> TaskReport + Audit evidence
```

核心原则：LLM 只负责理解、规划、解释和交互；硬件动作必须由 runtime、安全服务和机器人控制系统负责。

## 运行时模块

askme 使用 declarative runtime module 组合。主要模块：

| 模块 | 职责 |
| --- | --- |
| `LLMModule` | LLM client、模型健康、延迟指标 |
| `MemoryModule` | MemoryBridge、KnowledgeCatalog、RAG 检索与导入 |
| `PipelineModule` | 文本/语音 turn 执行链路 |
| `VoiceModule` | ASR、VAD、TTS、VoiceLoop、InteractionGate |
| `CognitionModule` | WorldState、WorkingMemory、CognitivePlanner、ActivePerceptionResolver |
| `RuntimeHandoffModule` | TaskHandoff、TaskRun、runtime profile、pause/resume/cancel/advance |
| `HealthModule` | Dashboard、HTTP API、health snapshot、readiness evidence |
| `SkillModule` | 工具/技能注册、SkillGate、安全边界 |

## Agent 与 Skill 增长机制

askme 借鉴 Claude Code 的可配置 agent 思路，但面向机器人现场产品做了更强的安全边界：

- Agent Profile 是可审计的角色配置，不是自由人格。内置角色包括现场任务总控、知识运营、园区问路、安全复核和在线技能增长。
- Profile 可从 `~/.askme/agents/*.md`、项目 `.askme/agents/*.md`、项目 `agents/*.md`、managed `.askme/managed/agents/*.md` 加载；项目覆盖用户，managed 覆盖项目。
- Profile frontmatter 支持 `tools`、`disallowedTools`、`spawnableProfiles`、`skills`、`mcpServers`、`hooks`、`memory`、`model`、`permissionMode`、`maxTurns`、`timeoutSeconds`、`effort`、`isolation`、`color`、`disabled`、`risk_level`。
- `skills` 表示启动时预加载的领域能力说明；工具 allow/deny 决定 agent 真实可用工具。
- `maxTurns`、`timeoutSeconds`、`model` 已进入 AgentShell 运行策略；子 agent 只能由父 profile 显式 allowlist 派生。
- `hooks` 当前支持声明式 `PreToolUse` / `PostToolUse` 规则，可按工具名和参数/结果内容阻断调用；不会执行 profile 中的任意 shell/HTTP hook。
- `mcpServers` 当前进入 profile catalog 和系统提示，用于产品审计与下一步 MCP scope 接入；真实安全边界仍由服务端 RBAC、SkillGate、SafetyPreflight 兜底。
- LLM 生成的新能力写入 `data/skills/<skill>/SKILL.md`，先进入 governance store。
- Skill Growth Backlog 从 `SkillAuditLog` 派生增长候选，聚合失败、阻断、未命中请求；候选只进入产品判断，不会自动创建或启用技能。
- `/api/skill-growth/backlog/{candidate_id}/draft` 可把候选转换为 generated skill 草稿，接口受 `skill:review` 保护，生成后仍保持待审批和禁用状态。
- 生成技能通过 validation 后，由具备 `skill:review` 权限的 operator 审批。
- 审批后还要进入客户/园区 `Skill Package`；未分配到启用包的技能不会注册语音触发。
- `Skill Package` 是客户项目的能力发布单元，包含 `release_version`、`release_channel`、`rollout_percent` 和历史快照。
- 能力包每次保存、分配技能、移除技能、灰度发布或回滚都会产生版本记录；`rollout_percent=0` 会暂停该包内技能生效。
- `/api/skill-packages/{package_id}/release` 用于 pilot/prod 发布或灰度比例调整，`/api/skill-packages/{package_id}/rollback` 会从历史快照恢复并生成新的版本。
- 所有生成、审批、包分配、调用结果写入 SkillAuditLog，供 Dashboard 和交付复盘查看。
- `field_event_trigger` 是技能层进入现场事件系统的受控工具。摔倒、卡住、电机故障、夜间陌生人、违停、烟火、垃圾桶、问路准入和访客带路等 built-in 技能通过它调用 `FieldOperationsService.trigger_payload()`，由现场事件系统统一完成证据校验、通知、归档、runtime handoff 和审计；技能本身不直接绕过 safety 控制硬件。
- `CapabilityCenter` 现在输出 `scenario_blueprints`，把 `FIELD_SCENARIOS` 的客户场景映射到 required skills、现场依赖、证据、通知、归档、审批和验收标准。Dashboard 可以直接显示“这个场景为什么 ready/partial/blocked”，而不是只展示散乱技能。
- `space_lookup_place` 和 `space_recommend_route` 把 `ParkSpaceService` 接入技能执行层；`lookup_place` / `recommend_route` skills 只能返回点位确认、语音路线或 escort handoff payload，不直接控制底盘。

## 身份与权限治理

现场产品不能依赖浏览器本地变量来决定谁能审批、启停能力或控制任务。当前实现把 demo operator directory 收敛到服务端 `OperatorDirectory`：

- `/api/governance/operator-directory` 返回操作员目录、角色说明、权限矩阵、SSO/IAM 配置状态和生产 readiness findings。
- `/api/governance/current-operator` 按请求头或 query 解析当前操作员，返回已知身份、认证来源和可用权限。
- `/api/governance/authorize` 用同一套 RBAC 判断单个权限，未知操作员不会继承默认 operator 权限。
- Dashboard 只把服务端目录中的人员显示为可操作账号；未知本地 operator 会进入“未登记/无权限”状态。
- 当前 `local_config` 只适合演示和试点。生产部署必须把同一接口背后的身份来源替换为企业 OIDC/IAM，并保持高风险动作的审计链路。
- 企业模式采用“网关验签 + 受信身份头”模式：OIDC/IAM 网关负责校验 token，askme 只消费 `x-askme-iam-operator-id`、`x-askme-iam-roles`、`x-askme-iam-display-name` 等已验证 claims。
- 当 `identity_provider` 是 `oidc/iam/sso` 时，HTTP body 里的 `operator_id` 不能覆盖受信身份头，避免前端或脚本冒充审批人。

## 统一审计

产品验收需要能回答“谁在什么时候做了什么、结果是什么、证据来自哪里”。当前新增 `AuditQueryService`，把分散的审计源标准化为统一 timeline：

- Skill 审计：读取 `SkillAuditLog`，覆盖技能调用、生成、审批、能力包发布和回滚。
- Field 审计：优先读取 `field-action-audit.jsonl` 的 append-only 记录；没有独立审计文件时回退到 field event archive 的 `action_audit`。
- Runtime 审计：读取 runtime handoff audit JSONL，覆盖 operator action、runtime event 和 terminal snapshot。
- `/api/audit/events` 支持按 source、operator、action、outcome 和关键词过滤，接口受 `audit:read` 权限保护。
- `/api/audit/export` 生成统一审计证据包：JSONL records + manifest，manifest 包含 SHA-256 和可选 HMAC 签名；接口受 `audit:export` 权限保护。
- `/api/audit/export/retry` 提供外部审计投递队列的状态查询和重试投递；审计包可选投递到企业 SIEM/WORM webhook，投递失败会写入 retry queue，避免导出证据丢失。
- Dashboard 交付检查页展示最近统一审计记录，作为客户验收和事后追溯入口。

## 语音链路

推荐国产低延迟链路：

```text
Realtime ASR
  -> MiniMax-M2.7-highspeed
  -> askme TaskHandoff / SafetyPreflight / runtime arbiter
  -> MiniMax Speech 2.8 TTS
```

语音入口必须遵守同一套状态机：

- “确认”在 planning 阶段确认计划。
- “取消”在 planning 阶段取消草案，在 executing 阶段取消 TaskRun。
- “停下”走安全优先路径。
- 语音误识别不能绕过安全确认。

## Interaction Gate

Interaction Gate 是真实场景的准入门。它不会把所有人声都送进 LLM。

输入：

- ASR final transcript 与 confidence。
- 是否被明确呼叫。
- 视觉注意力、距离、姿态、手势。
- 声源方向、声画一致性。
- 多人仲裁结果。
- 感知 freshness。

输出：

- `respond`：进入大脑回复或规划。
- `clarify`：先澄清说话对象或意图。
- `record_only`：只记录环境，不回复。
- `ignore`：忽略。
- `refuse`：安全/隐私拒绝。

关键规则：

- 旁观者说“这个机器狗好可爱”不等于唤醒。
- 多人且 speaker lock 不清楚时澄清。
- 声源和画面人物不一致时不猜。
- stop/emergency intent 优先安全路径。

## RAG 与知识生命周期

KnowledgeCatalog 是可信知识事实源。Memory backend 只是检索实现，不是最终信任来源。

知识状态：

- `draft`
- `pending`
- `approved`
- `published`
- `rejected`
- `deleted`
- `conflicted`

硬约束：

- expired 不进入 prompt。
- draft/pending/rejected/deleted 不进入 prompt。
- 同一 `entity_key + fact_key` 出现互斥 `value` 时进入 conflict。
- 检索命中后必须用 `record_id + evidence_version` 回 catalog 二次校验。
- `answer_policy` 会随回答返回，约束 LLM 不用无证据、过期、冲突知识编答案。

## 任务运行链路

任务对象分层：

- `TaskPlan`：用户想做什么。
- `TaskHandoff`：交给 runtime 的结构化计划。
- `TaskRun`：这一次实际执行发生了什么。
- `RuntimeEvent`：状态变化、step 事件、operator action。
- `TaskReport`：完成或失败后的结构化结果。

典型状态：

```text
draft
  -> awaiting_confirmation
  -> ready_for_arbiter
  -> submitted
  -> validating
  -> preflight
  -> queued
  -> executing
  -> paused / blocked / completed / failed / cancelled
```

Profile：

- `fake`：本地演示。
- `sim`：可手动 advance 的模拟运行。
- `shadow`：只做将要执行什么的验证，不发硬件动作。
- `lab`：受控实验室，默认禁用。
- `prod`：生产，默认禁用。

external/lab runtime 必须显式配置 endpoint 和 enable flag。默认不会联网或触碰硬件。

## 感知与 WorldState

WorldState 是 planner 和 safety 的事实来源。感知快照要带 `observed_at` 和 freshness。

当前支持的 interaction perception 字段：

- person detected/count/distance/angle。
- visual attention。
- person facing robot。
- posture。
- gesture。
- sound source angle。
- sound source matches person。

仍待接入真实 provider：

- 姿态/视线估计。
- 手势识别。
- 麦克风阵列 DOA。
- 声画关联。
- 接近/停留追踪。
- 多人仲裁。

## 安全不变量

- 不直接发 motor/gait/arm/serial/cmd_vel。
- 不绕过 runtime arbiter。
- 不绕过 SafetyPreflight。
- 不用 stale/conflict/unapproved knowledge 驱动高风险任务。
- operator action 必须记录 actor、reason、risk acknowledgement。
- lab/prod 必须显式启用，默认安全。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `askme/voice/interaction_gate.py` | 交互准入门 |
| `askme/voice/perception_context.py` | 感知快照归一化 |
| `askme/memory/catalog.py` | 知识生命周期事实源 |
| `askme/runtime/modules/memory_module.py` | 知识导入、检索、重建、批量更新 |
| `askme/cognition/active_perception.py` | 缺事实时主动刷新感知 |
| `askme/runtime/handoff.py` | TaskHandoff、TaskRun、runtime state machine |
| `askme/runtime/arbiter_client.py` | external/lab contract-only client |
| `askme/runtime/modules/health_module.py` | HTTP/Dashboard wiring 与 evidence report |
| `askme/static/dashboard.html` | Voice Mission Center |

## Agent Profile 管理入口

- `POST /api/agent-profiles` 可写入项目级 `.askme/agents/<profile>.md`，用于新增或更新一个可审计的 Agent Profile。
- `GET /api/agent-profiles/{profile_name}/preview` 返回解析后的权限策略和原始 Markdown，方便产品、交付和安全人员复核。
- Agent Profile 的工具清单由服务端从真实 `ToolRegistry` 构建，前端提交的 `known_tools` 不会扩权；未知工具会被拒绝且不会写入 profile 文件。
- `create_agent_profile` 工具允许受控增长代理提出新的 agent lane，但同样使用工具 allowlist 校验，只写项目配置和审计，不会自动授予硬件执行权限。
- 该机制借鉴 Claude Code 的项目级 subagent 文件、工具白名单和独立角色配置；askme 额外保留 RBAC、SkillGate、SafetyPreflight、runtime arbiter 作为机器人安全边界。
