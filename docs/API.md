# Askme API 文档

> 更新时间：2026-06-01
> 版本：4.1.0

本文档覆盖 Askme 全部 HTTP API 端点。所有端点通过 FastAPI 注册，按产品表面（Platform / Product / Admin / Internal）分层管理。

---

## 目录

- [认证方式](#认证方式)
- [通用约定](#通用约定)
- [错误码说明](#错误码说明)
- [Platform API](#platform-api)
- [Product API](#product-api)
- [Admin API](#admin-api)
- [Internal API](#internal-api)

---

## 认证方式

Askme 采用 **网关验签 + 受信身份头** 的企业认证模式。

### 身份头

| 请求头 | 说明 | 示例 |
|--------|------|------|
| `X-Askme-Operator-Id` | 操作员 ID（企业 IAM 验证后注入） | `operator-zhang` |
| `X-Askme-Api-Key` | API 密钥（服务间信任） | `askme-sk-...` |
| `X-Trace-Id` | 请求追踪 ID（跨系统关联） | `trac-abc123` |

### 权限模型

高风险端点通过 `authorize` 回调校验操作员权限。权限字符串格式：

| 权限 | 说明 |
|------|------|
| `knowledge:read` | 知识库读权限 |
| `knowledge:preview` | 知识预览权限 |
| `knowledge:import` | 知识导入权限 |
| `knowledge:approve` | 知识审批权限 |
| `knowledge:rollback` | 知识回滚权限 |
| `voice:profile:update` | 语音配置更新 |
| `skill:review` | 技能审核权限 |
| `runtime:submit` | 运行时任务提交 |
| `runtime:pause` | 运行时暂停 |
| `runtime:resume` | 运行时恢复 |
| `runtime:cancel` | 运行时取消 |
| `runtime:advance` | 运行时推进 |
| `audit:read` | 审计记录读取 |
| `audit:review` | 审计复核 |
| `audit:export` | 审计导出 |
| `field:event:create` | 现场事件创建 |
| `field:event:close` | 现场事件关闭 |
| `field:event:acknowledge` | 现场事件确认 |
| `field:event:request_close` | 请求关闭现场事件 |

### 企业 IAM 模式

生产部署时，OIDC/IAM 网关负责校验 token，Askme 消费已验证的受信身份头：

```
x-askme-iam-operator-id
x-askme-iam-roles
x-askme-iam-display-name
```

此模式下 HTTP body 中的 `operator_id` 不能覆盖受信身份头。

---

## 通用约定

### 响应格式

所有端点返回 JSON，包含以下公共字段（若适用）：

```json
{
  "ok": true,
  "error": "...",
  ...
}
```

### CORS

所有 API 端点自动处理 CORS OPTIONS 预检请求，返回 `Access-Control-Allow-Origin: *`。

### 缓存控制

动态数据端点设置 `Cache-Control: no-store`，静态资源设置 `private, max-age=30`。

---

## 错误码说明

| 状态码 | 含义 | 典型场景 |
|--------|------|----------|
| 200 | 成功 | 请求正常处理 |
| 400 | 请求参数错误 | JSON 格式错误、缺少必填字段 |
| 403 | 权限不足 | 操作员未授权指定操作 |
| 404 | 资源不存在 | 查询的 mission/event/profile 不存在 |
| 409 | 资源冲突 | 事件已关闭不可重复关闭 |
| 422 | 请求语义错误 | 设备载荷未使用正确端点 |
| 429 | 过载 | 聊天并发数超限 |
| 500 | 服务器内部错误 | 未捕获的服务端异常 |
| 503 | 服务不可用 | 依赖的服务未配置或不可用 |
| 504 | 网关超时 | 运行时语音轮次超时 |

---

## Platform API

系统健康与平台监控接口。属于 `askme.api.platform` 表面。

### GET /health

系统详细健康文档。运行全部注册组件检查。

**响应示例：**
```json
{
  "status": "healthy",
  "uptime_s": 86400,
  "components": {
    "llm": { "status": "healthy", "latency_ms": 320 },
    "memory": { "status": "healthy" }
  },
  "snapshot_at": "2026-06-01T12:00:00.000Z"
}
```

### GET /healthz

Kubernetes liveness 探针。不运行组件检查，响应应在 10ms 内完成。

**响应：** `{"alive": true, "status": "ok"}`

### GET /ready

Kubernetes readiness 探针。运行全部组件检查并聚合。

**响应示例：**
```json
{
  "ready": true,
  "status": "healthy",
  "uptime_s": 86400,
  "components": { "llm": { "status": "healthy" } }
}
```

### GET /metrics

运行时指标快照（JSON 格式）。

**响应：** 系统指标快照 JSON

### GET /metrics/prometheus

Prometheus 格式指标。

**响应：** `text/plain` Prometheus metrics

### GET /trace

返回最近管道计时追踪（用于诊断）。

**响应：**
```json
{
  "summary": { "total": 150, "avg_ms": 850 },
  "recent": [ ... ]
}
```

### GET /api/surfaces

返回 API 表面分层清单和边界状态。

**响应模型：** `ApiSurfacesResponse`

### GET /api/status

统一系统状态——所有关键指标在一个端点。

**响应模型：** `SystemStatusResponse`

### GET /api/live

返回内存中的对话历史（语音 + Web 聊天合并）。

**响应模型：** `ConversationHistoryResponse`

### GET /api/conversations

返回监控 UI 使用的对话历史。

**响应模型：** `ConversationHistoryResponse`

---

## Product API

客户可见的业务入口。属于 `askme.api.product` 表面。

### Conversation

#### POST /api/chat

向大脑管道发送文本并返回响应。

**请求体：**
```json
{
  "text": "今天天气怎么样",
  "session_id": "optional-session-id"
}
```

**响应模型：** `ChatResponse`

**错误码：** 400（空文本）、429（过载）、504（超时）、503（不可用）

#### GET /api/conversation/diagnostics

返回非敏感聊天执行诊断信息。

**响应模型：** `ConversationDiagnosticsResponse`

#### POST /api/runtime/voice-turn

将最终语音转录路由到运行时控制（不受 LLM 处理）。

**请求体：**
```json
{
  "text": "停下来",
  "speak": true,
  "transcript_id": "tran-abc",
  "confidence": 0.95,
  "is_final": true,
  "channel": "voice"
}
```

**响应模型：** `RuntimeVoiceTurnResponse`

### Memory / Knowledge

#### POST /api/memory/search

搜索配置的记忆/RAG 后端并返回可审计证据。

**请求体：** `{"query": "...", "filters": {...}}`

**权限：** `knowledge:read`

**响应模型：** `MemorySearchResponse`

#### GET /api/memory/health

返回产品层面的记忆后端就绪状态和数据位置。

**权限：** `knowledge:read`

**响应模型：** `MemoryHealthResponse`

#### POST /api/knowledge/preview

预览上传的知识记录而不索引。

**权限：** `knowledge:preview`

**响应模型：** `KnowledgePreviewResponse`

#### POST /api/knowledge/import

将上传的知识记录导入配置的记忆后端。

**权限：** `knowledge:import`

**响应模型：** `KnowledgeImportResponse`

#### POST /api/knowledge/list

列出本地索引的知识记录（知识控制台使用）。

**权限：** `knowledge:read`

**响应模型：** `KnowledgeListResponse`

#### POST /api/knowledge/update

更新知识元数据（审批状态、软删除等）。

**请求体：**
```json
{
  "action": "approve",
  "record_id": "rec-001"
}
```

**权限：** 根据 action 自动映射（如 approve→`knowledge:approve`）

**响应模型：** `KnowledgeUpdateResponse`

### Voice Profiles

#### GET /api/voice/profiles

列出客户可选语音配置。

**响应模型：** `VoiceProfileCatalogResponse`

#### POST /api/voice/profile

设置客户语音配置。

**权限：** `voice:profile:update`

**响应模型：** `VoiceProfileUpdateResponse`

### Space / Park Guidance

#### GET /api/space/health

空间认知服务健康状态。

**响应模型：** `SpaceHealthResponse`

#### GET /api/space/points

园区点位列表。

**响应模型：** `SpacePointsResponse`

#### GET /api/space/service-points

服务点列表。

**响应模型：** `SpaceServicePointsResponse`

#### GET /api/space/routes

园区路线列表。

**响应模型：** `SpaceRoutesResponse`

#### GET /api/space/history

空间变更历史。

**响应模型：** `SpaceHistoryResponse`

#### GET /api/space/proposals

空间数据提案列表。

**响应模型：** `SpaceProposalsResponse`

#### GET /api/space/interactions

空间交互记录。

**响应模型：** `SpaceInteractionsResponse`

#### POST /api/space/resolve-destination

解析目的地文本到点位。

**权限：** `knowledge:read`

**响应模型：** `SpaceResolveDestinationResponse`

#### POST /api/space/guide

生成引导任务。

**权限：** `field:event:create`

**响应模型：** `SpaceGuideResponse`

#### POST /api/space/service-point-trigger

触发服务点事件。

**权限：** `knowledge:read`

**响应模型：** `SpaceServicePointTriggerResponse`

#### POST /api/space/manage

管理空间数据（添加/编辑点位等）。

**权限：** `knowledge:approve`

**响应模型：** `SpaceManageResponse`

#### POST /api/space/proposals

创建空间数据提案。

**权限：** `knowledge:import`

**响应模型：** `SpaceProposalCreateResponse`

#### POST /api/space/proposals/review

审核空间数据提案。

**权限：** `knowledge:approve`

**响应模型：** `SpaceProposalReviewResponse`

#### POST /api/space/rollback

回滚空间数据变更。

**权限：** `knowledge:rollback`

**响应模型：** `SpaceRollbackResponse`

### Capabilities & Blueprints

#### GET /api/capabilities

返回运行时配置文件和组件合同。

**响应模型：** `RuntimeCapabilitiesResponse`

#### GET /api/capability-center

返回客户可见的分组机器人能力列表。

**响应模型：** `CapabilityCenterResponse`

#### GET /api/capability-packages

返回客户可见的能力包和场景包目录。

**响应模型：** `CapabilityPackageCatalogResponse`

#### POST /api/capability-packages/readiness

评估能力包或场景包是否可以启用。

**请求体：**
```json
{
  "kind": "capability_package",
  "manifest": { "skills": ["patrol", "greet"] }
}
```

**响应模型：** `CapabilityPackageReadinessResponse`

#### GET /api/scenario-intents

返回可审计的口语场景路由规则。

**响应模型：** `ScenarioIntentCatalogResponse`

#### POST /api/scenario-intents/preview

预览口语或文本输入在执行前的路由决策。

**请求体：** `{"text": "带我去东门"}`

**响应模型：** `ScenarioIntentPreviewResponse`

#### GET /api/blueprints

返回产品运行时蓝图目录。

**响应模型：** `BlueprintCatalogResponse`

#### GET /api/blueprints/{blueprint_name}

按名称或别名返回单个运行时蓝图。

**响应模型：** `BlueprintDetailResponse`

#### GET /api/blueprints/{blueprint_name}/delivery-package

返回单个运行时蓝图的客户交接包。

**响应模型：** `BlueprintDeliveryPackageResponse`

### Missions

#### POST /api/missions/draft

草拟高级任务而不调度硬件。

**请求体：**
```json
{
  "intent": "巡检A区",
  "parameters": { "zone": "A" }
}
```

**响应模型：** `MissionDraftResponse`

#### POST /api/missions

干运行或提交任务到配置的运行时仲裁器。

**请求体：**
```json
{
  "plan": { ... },
  "dry_run": true
}
```

**响应模型：** `MissionSubmitResponse`

#### GET /api/missions

返回本地草拟/提交的任务记录列表。

**响应模型：** `MissionListResponse`

#### GET /api/missions/{mission_id}

返回单个任务计划及其最新提交状态。

**响应模型：** `MissionDetailResponse`

#### GET /api/missions/{mission_id}/report

从任务证据构建检查报告外壳。

**响应模型：** `MissionReportResponse`

### Field Events (Customer-Facing)

#### GET /api/field/scenarios

返回客户可见的现场操作场景列表。

**响应模型：** `FieldScenarioCatalogResponse`

#### GET /api/field/scenario-acceptance

返回客户可读的场景验收覆盖范围和边界。

**响应模型：** `FieldScenarioAcceptanceResponse`

#### GET /api/field/events

返回最近的现场操作事件列表。

**查询参数：** `limit`, `status`, `notification_group`, `needs_attention`, `tenant_id`, `delivery_namespace`, `customer_id`, `project_id`, `site_id`, `managed_object_id`

**响应模型：** `FieldEventListApiResponse`

#### GET /api/field/events/{event_id}

返回单个现场事件详情（含工作流和证据）。

**响应模型：** `FieldEventDetailApiResponse`

#### GET /api/field/evidence

提供本地现场证据文件。

**查询参数：** `path`, `event_id`

**响应：** 文件（由 mimetype 决定 Content-Type）

#### POST /api/field/events

触发现场事件并评估通知规则。

**权限：** `field:event:create`

**响应模型：** `FieldEventTriggerResponse`

#### POST /api/field/events/{event_id}/close

关闭现场事件并附加操作员备注。

**权限：** `field:event:close`

**响应模型：** `FieldEventActionResponse`

#### POST /api/field/events/{event_id}/request-close

请求主管批准关闭高风险现场事件。

**权限：** `field:event:request_close`

**响应模型：** `FieldEventActionResponse`

#### POST /api/field/events/{event_id}/acknowledge

确认现场事件但不关闭。

**权限：** `field:event:acknowledge`

**响应模型：** `FieldEventActionResponse`

#### POST /api/field/events/{event_id}/resend-notification

重试未送达的现场事件通知。

**权限：** `field:event:acknowledge`

**响应模型：** `FieldEventActionResponse`

#### GET /api/field/events/{event_id}/report

返回可审计的客户可见现场事件报告。

**响应模型：** `FieldEventReportResponse`

### Dashboard

#### GET /dashboard

产品 Dashboard 外壳页面。

**响应：** `text/html`

#### GET /api/dashboard/pages

返回 Dashboard 外壳的产品页面映射。

**响应模型：** `DashboardPageRegistryResponse`

#### GET /dashboard/{asset_path}

提供 Dashboard 页面和静态资源。

---

## Admin API

主管、交付工程师和产品管理员使用。属于 `askme.api.admin` 表面。

### Governance

#### GET /api/governance/operator-directory

返回操作员目录、角色说明、权限矩阵和 IAM 配置状态。

**响应模型：** `OperatorDirectoryResponse`

#### GET /api/governance/identity-readiness

返回身份网关就绪状态。

**响应模型：** `IdentityGatewayReadinessResponse`

#### GET /api/governance/current-operator

按请求头或查询参数解析当前操作员。

**查询参数：** `operator_id`

**响应模型：** `CurrentOperatorResponse`

#### POST /api/governance/authorize

基于 RBAC 判断单个权限。

**请求体：**
```json
{
  "permission": "skill:review",
  "operator_id": "operator-zhang"
}
```

**响应模型：** `AuthorizationDecisionResponse`

### Audit

#### GET /api/skill-audit

返回最近的技能执行审计记录。

**查询参数：** `limit` (默认 50, 最大 200)

**响应模型：** `SkillAuditResponse`

#### GET /api/audit/events

返回跨 field/runtime/skill 记录的统一审计时间线。

**权限：** `audit:read`

**查询参数：** `limit`, `source`, `operator_id`, `action`, `outcome`, `q`, `since`, `until`, 项目范围参数

**响应模型：** `AuditEventsResponse`

#### GET /api/audit/reviews

返回追加式统一审计复核决策列表。

**权限：** `audit:read`

**响应模型：** `AuditReviewsResponse`

#### POST /api/audit/reviews

提交主管对一条审计记录的复核决策。

**权限：** `audit:review`

**请求体：**
```json
{
  "record_id": "rec-001",
  "decision": "approved",
  "note": "已验证"
}
```

**响应模型：** `AuditReviewSubmitResponse`

#### GET /api/audit/export/retry

返回待处理的审计导出投递重试队列。

**权限：** `audit:export`

**响应模型：** `AuditExportRetryStatusResponse`

#### GET /api/audit/exports

返回最近的审计导出清单。

**权限：** `audit:export`

**响应模型：** `AuditExportsResponse`

#### POST /api/audit/export/retry

重放待处理的审计导出投递。

**权限：** `audit:export`

**响应模型：** `AuditExportRetryResponse`

#### POST /api/audit/export

创建签名审计导出包并可选择投递。

**权限：** `audit:export`

**响应模型：** `AuditExportResponse`

### Skills & Growth

#### GET /api/skill-growth/backlog

返回可审核的在线技能成长候选列表。

**查询参数：** `min_occurrences` (默认 2), `limit` (默认 20)

**响应模型：** `SkillGrowthBacklogResponse`

#### POST /api/skill-growth/backlog/{candidate_id}

标记技能成长候选为已提升、已驳回或重新打开。

**权限：** `skill:review`

**响应模型：** `SkillGrowthMutationResponse`

#### POST /api/skill-growth/backlog/{candidate_id}/draft

从已审核的成长候选创建生成技能 SKILL.md 草稿。

**权限：** `skill:review`

**响应模型：** `SkillGrowthDraftResponse`

#### GET /api/skills/generated

返回生成技能的审核队列。

**响应模型：** `GeneratedSkillsResponse`

#### GET /api/skills/generated/{skill_name}/validation

返回单个生成技能的前置验证结果。

**响应模型：** `GeneratedSkillValidationResponse`

#### GET /api/skills/generated/{skill_name}/preview

返回单个生成技能的可审阅 SKILL.md 正文和解析策略。

**响应模型：** `GeneratedSkillPreviewResponse`

#### POST /api/skills/generated/{skill_name}/review

审核（批准/驳回/禁用/返回审核）一个生成技能。

**权限：** `skill:review`

**响应模型：** `GeneratedSkillReviewResponse`

#### GET /api/skill-packages

返回客户/园区的能力包目录。

**响应模型：** `SkillPackageCatalogResponse`

#### POST /api/skill-packages

创建或更新客户/园区能力包。

**权限：** `skill:review`

**响应模型：** `SkillPackageMutationResponse`

#### POST /api/skill-packages/{package_id}/skills/{skill_name}

分配或移除能力包中的生成技能。

**权限：** `skill:review`

**响应模型：** `SkillPackageMutationResponse`

#### GET /api/skill-packages/{package_id}/history

返回能力包的版本快照历史。

**响应模型：** `SkillPackageHistoryResponse`

#### POST /api/skill-packages/{package_id}/release

发布或灰度发布能力包。

**权限：** `skill:review`

**响应模型：** `SkillPackageMutationResponse`

#### POST /api/skill-packages/{package_id}/rollback

回滚能力包到历史快照。

**权限：** `skill:review`

**响应模型：** `SkillPackageMutationResponse`

### Agent Profiles

#### GET /api/agent-profiles

返回产品可审阅的 Agent Profile 目录。

**响应模型：** `AgentProfileCatalogResponse`

#### POST /api/agent-profiles

创建或更新项目级 Agent Profile Markdown 文件。

**权限：** `skill:review`

**响应模型：** `AgentProfileUpsertResponse`

#### GET /api/agent-profiles/{profile_name}/preview

返回解析后的 Profile 策略和原始 Markdown。

**响应模型：** `AgentProfilePreviewResponse`

### Field Admin & Delivery

#### POST /api/field/notification-test

发送测试通知。

**响应模型：** `FieldNotificationTestResponse`

#### GET /api/field/notification-preflight

通知渠道预检。

**响应模型：** `FieldNotificationPreflightResponse`

#### GET /api/field/readiness

现场交付就绪状态。

**响应模型：** `FieldReadinessResponse`

#### GET /api/field/audit/integrity

现场审计完整性检查。

**响应模型：** `FieldActionAuditIntegrityResponse`

#### POST /api/field/customer-projects/from-template

从模板创建客户项目。

**响应模型：** `FieldCustomerProjectFromTemplateResponse`

#### GET|POST /api/field/delivery-resource-registry

交付资源注册表（CRUD）。

#### GET /api/field/delivery-resource-registry/history

资源注册历史。

#### POST /api/field/delivery-resource-registry/{resource_type}/{resource_id}/disable

禁用交付资源。

#### POST /api/field/delivery-resource-registry/rollback

回滚资源变更。

#### GET|POST /api/field/delivery-resource-governance-requests

交付资源治理请求。

#### POST /api/field/delivery-resource-governance-requests/escalate-overdue

升级逾期治理请求。

#### POST /api/field/delivery-resource-governance-requests/{request_id}/review

审核治理请求。

### Field Customer Projects

#### GET|POST /api/field/customer-projects/{identifier}

客户项目管理。

#### GET|POST /api/field/customer-projects

客户项目列表/创建。

#### GET|POST /api/field/customer-projects/{identifier}/managed-objects/{object_id}

受管对象管理。

#### GET /api/field/customer-projects/{identifier}/history

项目变更历史。

#### POST /api/field/customer-projects/{identifier}/rollback

回滚项目变更。

#### POST /api/field/customer-projects/{identifier}/archive

归档项目。

#### GET /api/field/customer-projects/import

导入客户项目。

#### POST /api/field/customer-projects/package/verify

验证项目包。

#### POST /api/field/customer-projects/package/diff

项目包差异比较。

#### POST /api/field/customer-projects/proposal-bundle/verify

验证提案包。

#### POST /api/field/customer-projects/acceptance-dossier/verify

验证验收档案。

#### GET /api/field/customer-projects/{identifier}/export

导出项目。

#### GET|POST /api/field/customer-projects/{identifier}/acceptance-dossier

验收档案管理。

#### GET|POST /api/field/customer-projects/{identifier}/proposal-bundle

提案包管理。

#### GET /api/field/customer-projects/{identifier}/acceptance-report

验收报告。

#### POST /api/field/customer-projects/{identifier}/onsite-evidence

现场证据上传/获取。

#### POST /api/field/customer-projects/{identifier}/acceptance-closure

验收关闭。

#### POST /api/field/customer-projects/{identifier}/acceptance-review

验收审核。

#### POST /api/field/customer-projects/{identifier}/customer-signoff

客户签字确认。

#### GET|POST /api/field/customer-projects/{identifier}/execution-bindings

执行绑定管理。

#### POST /api/field/customer-projects/{identifier}/execution-bindings/{object_id}/rehearsal

执行预演。

### Field Templates

#### GET|POST /api/field/customer-project-templates

客户项目模板管理。

#### GET /api/field/customer-project-templates/{template_id}/history

模板版本历史。

#### GET|POST /api/field/customer-project-template-release-requests

模板发布请求。

#### GET|POST /api/field/customer-project-template-release-notes

模板发布说明。

#### GET /api/field/customer-project-template-release-notes/export

导出模板发布说明。

#### POST /api/field/customer-project-templates/{template_id}/release-requests

提交模板发布请求。

#### POST /api/field/customer-project-template-release-requests/{request_id}/review

审核模板发布请求。

#### POST /api/field/customer-project-templates/{template_id}/release

发布模板。

### Field Product Catalog

#### GET /api/field/site-profiles

现场配置目录。

#### GET /api/field/customer-projects

客户项目目录。

#### GET /api/field/customer-project-workbench

项目工作台。

#### GET /api/field/product-launch-readiness

产品发布就绪状态。

#### GET /api/field/customer-projects/managed-object-directory

受管对象目录。

#### GET /api/field/customer-project-acceptance-registry

验收注册表。

#### GET /api/field/customer-project-resource-catalog

资源目录。

#### GET /api/field/solution-delivery-readiness

解决方案交付就绪状态。

---

## Internal API

机器人运行时和设备集成接口。属于 `askme.api.internal` 表面。

### Cognition

#### GET /api/cognition/context

返回认知上下文（WorldState + 工作记忆）。

**查询参数：** `refresh_perception` (bool)

**响应模型：** `CognitionContextResponse`

#### POST /api/cognition/plan

从用户输入生成认知计划。

**请求体：**
```json
{
  "text": "检查A区温度",
  "context": { "urgency": "normal" }
}
```

**响应模型：** `CognitionPlanResponse`

### Runtime

#### GET /api/runtime/context

返回运行时上下文（当前 TaskRun、profile、模块状态）。

**响应模型：** `RuntimeContextResponse`

#### GET /api/runtime/events

SSE 事件流——实时运行时事件。

**查询参数：** `once` (bool), `after` (float cursor), `limit`

**响应：** `text/event-stream`

#### GET /api/runtime/profiles

返回运行时配置列表。

**响应模型：** `RuntimeProfilesResponse`

#### GET /api/runtime/runs

返回运行时执行记录列表。

**响应模型：** `RuntimeRunListResponse`

#### POST /api/runtime/handoff

提交任务计划给运行时执行。

**权限：** `runtime:submit`

**请求体：**
```json
{
  "plan": { ... }
}
```

**响应模型：** `RuntimeHandoffSubmitResponse`

#### GET /api/runtime/runs/{run_id}

返回单个运行时执行记录详情。

**响应模型：** `RuntimeRunDetailResponse`

#### GET /api/runtime/runs/{run_id}/report

返回运行时执行报告。

**响应模型：** `RuntimeRunReportResponse`

#### POST /api/runtime/runs/{run_id}/pause

暂停运行时执行。

**权限：** `runtime:pause`

**响应模型：** `RuntimeRunActionResponse`

#### POST /api/runtime/runs/{run_id}/resume

恢复运行时执行。

**权限：** `runtime:resume`

**响应模型：** `RuntimeRunActionResponse`

#### POST /api/runtime/runs/{run_id}/cancel

取消运行时执行。

**权限：** `runtime:cancel`

**响应模型：** `RuntimeRunActionResponse`

#### POST /api/runtime/runs/{run_id}/advance

推进运行时执行（模拟模式）。

**权限：** `runtime:advance`

**响应模型：** `RuntimeRunActionResponse`

### Vision

#### GET /api/vision/snapshot

从机器人相机捕获帧并返回 base64 JPEG。

**响应模型：** `VisionSnapshotResponse`

#### POST /api/vision/analyze

用 VLM 分析 base64 JPEG 图片并返回描述。

**请求体：**
```json
{
  "image_base64": "..."
}
```

**响应模型：** `VisionAnalyzeResponse`

#### GET /api/vision/captures

列出归档捕获元数据（不含 base64 图片）。

**查询参数：** `limit`, `label`

**响应模型：** `VisionCaptureListResponse`

#### GET /api/vision/captures/{capture_id}

返回完整元数据及 base64 图片。

**响应模型：** `VisionCaptureDetailResponse`

#### DELETE /api/vision/captures/{capture_id}

删除捕获记录和镜像文件。

**响应模型：** `VisionCaptureDeleteResponse`

### Field Internal (Device Ingest)

#### POST /api/field/events/{event_id}/runtime-delivery

运行时事件回调投递。

**响应模型：** `FieldRuntimeDeliveryResponse`

#### GET /api/field/devices

设备列表。

**响应模型：** `FieldDeviceStatusResponse`

#### GET /api/field/device-onboarding

设备上线证据查询。

**响应模型：** `FieldDeviceOnboardingResponse`

#### POST /api/field/ingest

设备传感器/相机/机器人数据接入。

**响应模型：** `FieldEventTriggerResponse`

---

## 字段表

### 端点到 API 表面映射

| 前缀模式 | 表面 | 客户可见 |
|----------|------|----------|
| `/health`, `/metrics`, `/trace`, `/api/status`, `/api/live`, `/api/conversations`, `/api/surfaces` | Platform | 否（运营用） |
| `/api/chat`, `/api/conversation/`, `/api/runtime/voice-turn` | Product | 是 |
| `/api/memory/`, `/api/knowledge/` | Product | 是 |
| `/api/voice/` | Product | 是 |
| `/api/space/` | Product | 是 |
| `/api/capabilities`, `/api/capability-center`, `/api/capability-packages`, `/api/scenario-intents` | Product | 是 |
| `/api/blueprints` | Product | 是 |
| `/api/missions` | Product | 是 |
| `/api/field/scenarios`, `/api/field/events` (read) | Product | 是 |
| `/dashboard` | Product | 是 |
| `/api/governance/` | Admin | 否 |
| `/api/audit/` | Admin | 否 |
| `/api/skill-*`, `/api/skill-packages`, `/api/skill-growth/` | Admin | 否 |
| `/api/agent-profiles` | Admin | 否 |
| `/api/field/notification-*`, `/api/field/readiness`, `/api/field/audit/` | Admin | 否 |
| `/api/field/customer-projects/` | Admin / Product | 按端点 |
| `/api/field/delivery-*` | Admin | 否 |
| `/api/field/customer-project-templates` | Admin | 否 |
| `/api/field/site-profiles` | Admin | 否 |
| `/api/cognition/` | Internal | 否 |
| `/api/runtime/` | Internal | 否 |
| `/api/vision/` | Internal | 否 |
| `/api/field/ingest`, `/api/field/devices`, `/api/field/device-onboarding`, `/api/field/events/*/runtime-delivery` | Internal | 否 |

---

> 本文档从 `askme/api/routes/` 模块自动提取端点信息。新增路由后应保持此文档同步。
