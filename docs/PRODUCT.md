# askme Product Brief

更新时间：2026-05-13

askme 是面向机器人现场任务的自然语言入口。它不是普通聊天框，也不是机器人底层控制器；它负责把人的语音或文字目标变成可解释、可确认、可审计、可评测的任务意图，再交给安全和 runtime 层处理。

当前只维护三个入口文档：

- `docs/PRODUCT.md`：产品定位、能力边界、路线图。
- `docs/ARCHITECTURE.md`：系统结构、模块边界、数据流。
- `docs/OPERATIONS.md`：配置、启动、验收、排障。

## 产品目标

现场用户应该可以直接说：

- “请问洗手间在哪里？”
- “开始 A 区巡检。”
- “暂停一下。”
- “刚才巡检结果怎么样？”
- “停下。”

系统要做到：

- 听得见：麦克风、ASR、VAD、打断链路可观测。
- 知道何时该回话：Interaction Gate 能区分问路、任务指令、旁观闲聊、多人不确定和噪声。
- 回答有依据：RAG evidence 能显示来源、状态、是否被采用。
- 不乱执行：机器人任务必须经过 TaskHandoff、SafetyPreflight、runtime arbiter。
- 可接管：任务运行中可以暂停、继续、取消、查询状态。
- 可审计：回答依据、任务计划、operator action、runtime event、报告都能留痕。

## 已有能力

### 语音交互

- Voice Mission Center 三栏 UI：语音状态、对话、当前任务/服务能力。
- MiniMax 文本和 TTS 基础链路。
- Voice Turn Trace：ASR、LLM、TTS、播放、打断延迟桶。
- Interaction Gate：判断 `respond`、`clarify`、`record_only`、`ignore`、`refuse`。
- 旁观者提到“机器狗”不会误唤醒；多人/声画不一致时优先澄清。

### 记忆与 RAG

- MemoryBridge 支持 `mem0`、`robotmem`、`vector fallback`。
- KnowledgeCatalog 是知识生命周期事实源。
- 支持 Markdown、JSON、JSONL、CSV 导入。
- 未发布、过期、删除、冲突知识不会进入 prompt。
- 回答会返回 `evidence` 和 `rag.answer_policy`。
- Dashboard 气泡展示回答依据。

### 任务运行

- TaskHandoff、SafetyPreflight、TaskRun、RuntimeEvent、TaskReport 已有 fake/sim/shadow 基础。
- RuntimeArbiterClient 是 contract-only，external/lab 默认禁用，不直接触碰硬件。
- Dashboard runtime 控制动作会记录 `operator_id`、`reason`、`risk_acknowledgement`。

### 评测证据

- RAG Trust 离线评测：游客问路、过期知识、冲突位置、删除知识、未知位置。
- Voice E2E 离线评测：游客问路、未知地点拒答、巡检 SOP、设备位置、过期路线拒答、旁观噪声、多人澄清、急停。
- Health snapshot 与 Dashboard 运营诊断显示 Knowledge Trust 和 Voice E2E 结果。

### 能力中心与在线技能增长

- Dashboard `能力中心` 展示客户可读的能力分组、场景能力蓝图、缺口、Agent Profile、生成技能审批队列、技能包和调用审计。
- `在线增长候选` 从真实调用审计里聚合失败、阻断、未命中请求，帮助产品经理判断哪些重复需求值得沉淀成技能。
- 产品经理可从增长候选一键生成 `SKILL.md` 草稿；草稿仍然是 `pending_approval`，不会自动启用。
- LLM 生成的新 `SKILL.md` 默认进入草稿/待审批，不会自动变成生产可用能力。
- 生成技能必须通过结构校验、触发词冲突检查、工具边界检查和人工审批。
- 审批通过后还必须分配到客户/园区 `Skill Package`，同一套产品可针对不同项目启用不同能力。
- `Skill Package` 已升级为客户项目发布单元，支持版本快照、pilot/prod 发布通道、灰度比例和回滚。
- 灰度比例为 `0%` 时，该能力包内技能不会进入可触发状态；回滚会生成新的版本记录，保留谁在何时回滚到哪个版本。
- 支持项目级、用户级和 managed Agent Profile Markdown 配置，字段包含工具 allow/deny、可派生子 agent、预加载 skills、MCP server、hooks、模型、最大轮次、超时、隔离方式、记忆范围和风险等级。
- Agent Profile 的 hooks 已支持产品级声明式拦截：`PreToolUse` 可在工具调用前拒绝，`PostToolUse` 可在结果返回前阻断敏感输出；系统不会执行任意 shell hook。
- `create_skill` 工具统一走 `SkillManager.create_generated_skill_draft`，所以语音/文本 Agent、Dashboard 候选生成和后端 API 都进入同一套待审批、禁用、校验、审计流程。
- 首批园区场景技能已从 planned 落成 built-in：`report_fall_unrecoverable`、`report_stuck`、`report_motor_fault`、`detect_night_intruder`、`detect_illegal_parking`、`detect_fire_smoke`、`inspect_trash_bin`、`offer_wayfinding_help`、`escort_visitor`。这些技能通过 `field_event_trigger` 进入 FieldOperationsService，生成事件、按策略通知、归档和审计，而不是只返回聊天文案。
- 操作员治理已从前端兜底推进到服务端目录：Dashboard 会读取 `/api/governance/current-operator`，未知操作员显示为未登记并且无权限；目录页面返回角色矩阵、SSO/IAM readiness 和生产阻塞原因。
- 企业账号接入采用网关验签模式：客户的 OIDC/IAM 网关验证登录 token 后注入受信身份头，askme 只根据已验证的 operator/roles 做权限和审计，不相信请求 body 里的 operator_id。
- 统一审计查询和导出已具备产品入口：`/api/audit/events` 汇总技能、现场事件和 runtime 审计；`/api/audit/export` 生成带 SHA-256/可选 HMAC 的 JSONL 证据包，并可投递到 SIEM/WORM webhook；`/api/audit/export/retry` 可查询和重放失败投递，Dashboard 交付页能看到待投递数量。

## 产品边界

askme 可以：

- 理解用户目标。
- 追问缺失信息。
- 生成高层任务计划。
- 检索知识并给出有依据的回答。
- 把确认后的计划交给 runtime arbiter。
- 展示任务状态、报告和审计证据。

askme 不可以：

- 直接控制电机、步态、机械臂、串口或 `cmd_vel`。
- 绕过 SafetyPreflight。
- 用过期或冲突知识驱动高风险任务。
- 在多人、声画不一致、感知过期时猜测说话对象。
- 在没有明确授权时执行真实硬件动作。

## 客户演示路径

1. 打开 Dashboard。
2. 用文本或语音输入“请问洗手间在哪里”。
3. 展示回答气泡中的依据。
4. 输入“开始 A 区巡检”。
5. 展示 Planning 卡片、确认动作、Runtime timeline。
6. 任务运行中输入“暂停”“继续”“取消”“现在执行到哪了”。
7. 打开运营诊断，看 Knowledge Trust、Voice E2E、Latency。

## 下一步路线

近期优先级：

1. 把 Voice E2E 从离线模拟升级为真实麦克风/录音回放评测。
2. 把 Knowledge Trust 与 Voice E2E 合并为统一 Readiness Evidence 页面。
3. Knowledge Console 增加审批、版本、冲突处理和异步重建索引 job。
4. TaskRunStore 持久化运行状态、runtime events、operator actions、reports。
5. Operator RBAC 下一步补企业登录页/会话 UI、审批流通知、审计导出重试任务和外部 SIEM/WORM 生产联调；当前已具备 OIDC/IAM 网关受信身份头适配、统一审计查询和签名导出。
6. Skill Package 增加客户项目验收状态、发布日历和字段级变更对比。
7. 接入真实感知 provider：pose/gaze、gesture、DOA、声画关联、接近/停留、多人仲裁。
8. external/lab runtime 只开放低风险 shadow/lab skill：status_report、capture_image、read_status_panel、generate_report、return_home。

本次新增的产品化增长能力：

- Agent Profile 可以通过 `POST /api/agent-profiles` 写入项目级配置，和 Claude Code 的项目 subagent 文件类似，但会进入 askme 审计链。
- 受控 agent 可调用 `create_agent_profile` 生成新的项目级 agent lane，用于沉淀“知识运营、问路、停车检测、垃圾桶巡检”等专职代理；工具权限由服务端 allowlist 校验，不能由前端或 LLM 自行扩权。
- Dashboard 能力中心已提供 Agent Profile 创建表单和预览按钮，产品经理可以在界面上填写角色边界、工具范围、可派生 agent 和预加载技能。
- 能力中心新增 `scenario_blueprints`：把机器人异常、夜间陌生人、违停、烟火、垃圾桶、突发巡检、人群聚集、问路和带路映射到 required skills、传感器/数据依赖、通知归档和验收标准。
- 园区问路已具备可调用技能入口：`lookup_place` 调用空间语义地图解析目的地，`recommend_route` 调用路线推荐服务生成语音指路或带路前 handoff 建议，`answer_wayfinding` 封装成游客可直接触发的语音指路能力；未知地点必须拒答或要求人工更新点位库。
- 人群聚集已具备可调用技能入口：`detect_crowd_gathering` 会在人数、停留时长或复巡证据满足策略时进入安保事件闭环，短暂停留不能被夸大成告警。
- 新 profile 只定义角色、工具边界、可派生 agent、预加载技能和风险等级；真实机器人动作仍必须经过 SkillGate、SafetyPreflight 和 runtime arbiter。

暂不做：

- 真实生产硬件动作默认开启。
- 机械臂抓取、靠近游客、开门、支付、删除数据等高风险动作。
- 让 LLM 直接输出底层控制命令。
