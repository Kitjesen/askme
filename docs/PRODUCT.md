# askme Product Brief

更新时间：2026-05-10

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
5. Operator RBAC：viewer/operator/supervisor/admin。
6. 接入真实感知 provider：pose/gaze、gesture、DOA、声画关联、接近/停留、多人仲裁。
7. external/lab runtime 只开放低风险 shadow/lab skill：status_report、capture_image、read_status_panel、generate_report、return_home。

暂不做：

- 真实生产硬件动作默认开启。
- 机械臂抓取、靠近游客、开门、支付、删除数据等高风险动作。
- 让 LLM 直接输出底层控制命令。
