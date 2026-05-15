# askme 机器人语音大脑产品化推进计划

更新日期：2026-05-11

## 当前目标

把 askme 从“能听、能聊、能规划”的机器人语音入口，推进成客户可演示、可审计、可评测、可接真实机器人 runtime 的语音大脑。

核心体验不是一个聊天框，而是：

- 用户能用语音或文字提问、问路、发起巡检、追问结果。
- 系统能说明“为什么这么回答”，并展示 RAG 证据。
- 过期、未审批、冲突知识不能进入回答或任务规划。
- 机器人任务必须经过 TaskHandoff、SafetyPreflight、runtime arbiter，不允许 LLM 直接控制硬件。
- 环境噪声、路人闲聊、多人同时说话时，Interaction Gate 能判断是否应该回应、澄清、记录或忽略。

## 已完成基线

- Voice Mission Center 已有三栏 UI：语音状态、对话、当前任务和服务能力。
- MiniMax 文本与 TTS 链路已有基础接入。
- MemoryBridge 支持 mem0、robotmem、mempalace、vector fallback 四类后端。
- RAG 导入链路支持 Markdown、JSON、JSONL、CSV。
- TaskHandoff、SafetyPreflight、TaskRun、RuntimeEvent、TaskReport 已有 fake/sim runtime 基础。
- InteractionGate 已接入 VoiceLoop，可以区分环境语音、弱交互、明确交互和危险请求。
- `pyproject.toml` 已修正 RobotMem optional dependency 为 `robotmem[cjk]>=0.1.3,<0.2`。
- `/api/chat` 会返回 `evidence` 与 `rag` 字段。
- `/api/live` 的 assistant 消息会保留 evidence/rag。
- Dashboard 的 bot 气泡支持展示“回答依据”。
- Dashboard 的 bot 气泡已区分“进入回答的证据”和“被系统拦截的证据”，并展示可信状态、策略、后端降级原因。
- Dashboard 服务能力区已新增 Memory Health strip，客户可直接看到配置后端、实际后端、证据数量、拦截数量和检索耗时。
- Knowledge Console 已新增知识生命周期视图：可回答、草稿待审批、待审批、需重建索引、即将过期、已过期、知识冲突、已删除。
- Knowledge Console 已支持批量审批选中知识和重建选中知识；每条知识可单独审批、发布、重建、删除、恢复。
- Knowledge Console 已支持冲突处理：对冲突知识可选择“保留此条”，系统会发布保留项、驳回同组冲突项，并重新同步 RAG backend。
- KnowledgeCatalog 已记录审批/驳回/索引事件，公开 approved_by、approved_at、rejected_by、rejected_at、review_note 和最近事件，形成基础审计链。
- KnowledgeCatalog health 已输出 prompt_eligible、needs_review、needs_reindex、expiring_soon、expired、conflicted、deleted 和 by_state，便于 Dashboard 和运营检查使用。
- MemoryBridge 已拦截未发布、过期知识，不让它们进入 prompt。
- MemoryBridge 的 MemPalace backend 在可用但空结果或查询异常时会透明降级到 vector；如果命中的是过期/冲突/未审批证据，则不会降级绕过安全策略。
- 新增 `/api/memory/search`，用于 Retrieval Test Bench。
- 新增 `/api/knowledge/preview` 与 `/api/knowledge/import`。
- 新增 `/api/knowledge/list` 与 `/api/knowledge/update`。
- Dashboard 已新增 Knowledge Console MVP：文件/文本导入、预览、导入、检索测试、知识目录、发布/删除/恢复。
- RAG 证据过滤已补冲突检测：同一 `entity_key + fact_key` 出现不同 `value` 时全部剔除，不进入 prompt。
- SkillGate 已阻止 disabled skill 被直接执行，避免绕过技能目录开关触发危险能力。
- 新增 backend-independent `KnowledgeCatalog` MVP：import/list/update/delete 先落 catalog，再同步 RAG backend。
- `KnowledgeCatalog` 已支持持久化、prompt eligibility、软删除、恢复、冲突检测、indexed_at 标记。
- `KnowledgeCatalog` 已支持 `source_version/evidence_version`；prompt 相关字段变化会递增 `evidence_version`。
- MemoryBridge 检索命中后会用 `record_id/evidence_version` 回 catalog 二次校验，stale backend evidence 会被拒绝进入 prompt。
- RAG 已新增结构化 `answer_policy`：grounded、conflict、stale、unapproved、no_evidence 等状态会随 `/api/chat`、`/api/memory/search` 和 Dashboard evidence 一起展示。
- `answer_policy` 已接入 TurnExecutor/PromptBuilder：无证据、冲突、过期、未审批知识会进入本轮 LLM prompt 约束；prompt seed 模式也会保留该策略。
- 新增 RAG 信任场景评测：`tests/scenario_tests/test_rag_trust_evaluation.py` 覆盖游客问路、过期知识、冲突位置、已删除知识和未知位置。
- 新增 `scripts/evaluate_rag_trust_scenarios.py`，可生成 `artifacts/rag_trust/scenario-evaluation.json` 作为可审计评测证据包。
- Health snapshot 与 Dashboard 运营诊断已接入 RAG trust report，产品侧可看到 Knowledge Trust 状态、通过数和场景列表。
- KnowledgeCatalog / MemoryModule 已新增重建索引与批量 metadata 更新入口，Dashboard Knowledge Console 已支持“刷新问答可用性、发布选中、删除选中”。
- KnowledgeIndexJobStore 已新增知识刷新任务历史：记录 job_id、操作者、开始/完成时间、耗时、扫描/写入/跳过/错误、后端和 fallback 原因。
- Dashboard Knowledge Console 已展示最近刷新任务，单次刷新会返回任务号和失败原因。
- RAG evidence 已新增 `record_id/source_record_id/evidence_version`，聊天证据和被拦截证据可回到知识目录定位处理。
- RAG answer_policy 已新增 `required_operator_action`，过期、冲突、未审批知识会给出可执行的运营处理动作。
- 新增 ActivePerceptionResolver：当规划缺少新鲜感知事实时，可请求本地感知刷新、记录 request、重跑规划。
- 新增 RuntimeArbiterClient contract：external/lab runtime 默认禁用，必须显式启用 endpoint 才能进入 contract-only handoff，不会直接触碰硬件。
- Dashboard runtime 控制动作已携带 operator_id、reason、risk_acknowledgement，并写入 TaskRun operator_actions。
- 新增离线 Voice E2E 场景评测：覆盖游客问路、未知地点拒答、巡检 SOP、设备位置、过期路线拒答、噪声旁观、多人成员澄清和急停。
- Health snapshot 与 Dashboard 运营诊断已接入 Voice E2E report，产品侧可看到 false respond、TTS first audio 和场景结果。
- VoiceTurnTrace 已新增 live 对话 SLO 判定，Dashboard 可展示“可对话门禁通过 / 证据不足 / 响应超时”和失败桶，避免只给客户看工程延迟数字。

## 仍然欠缺的关键能力

1. RobotMem 真实 SDK 尚未完成线上验证
   - 当前可安装版本已确认是 `0.1.3`。
   - 仍需用真实环境验证 `backend: robotmem` 是否稳定，不稳定时 UI 必须明确显示 fallback。

2. RAG 证据产品化还需要从“可看见”升级到“可运营”
   - 当前回答气泡已能区分引用证据和拦截证据。
   - 聊天证据已能携带 record_id，并可在 UI 中回到知识目录定位处理。
   - 下一步需要补版本 diff、冲突合并编辑、审批队列和更细的冲突解释。
   - 无证据、过期证据、冲突证据已经有结构化 answer_policy 和 required_operator_action；还需要补一套可配置的客户话术模板。
   - Mem0/RobotMem/MemPalace 返回裸文本或 metadata 不完整时，仍必须回到 KnowledgeCatalog 做可信事实校验。

3. Knowledge Console 还是 MVP
   - 已有预览、导入、检索测试、目录、软删除、恢复、刷新问答可用性和批量发布/删除入口。
   - 已接入 KnowledgeCatalog 作为知识生命周期事实源。
   - 已新增生命周期状态、批量审批、选中刷新和刷新任务提示。
   - 已新增审批人/审批时间、驳回人/驳回时间、审查备注和基础事件记录。
   - 已新增冲突“保留此条”处理能力。
   - 已新增可持久化的刷新任务历史；还缺真正后台异步队列、重试按钮和进度页。
   - 还缺版本 diff 和冲突合并编辑 UI。
   - 还缺独立知识列表页和每条知识的生命周期状态详情。

4. Freshness / expiry / conflict / version 仍需硬约束
   - metadata 不能只是展示字段。
   - expired、draft、pending、rejected、deleted 永不进入 prompt。
   - 同一 entity_key + fact_key 出现互斥 value 时，系统必须拒绝确定性回答或阻止 runtime handoff。

5. 语音端到端评测不足
   - RAG 信任场景已覆盖游客问路、过期知识、冲突位置、已删除知识和未知位置。
   - 离线 Voice E2E 已覆盖巡检 SOP、设备位置、错误知识拒答、噪声、多人环境和急停。
   - VoiceTurnTrace 已能把 ASR final、LLM 首 token、TTS 首音频、播放开始和打断停播纳入 SLO 判定。
   - 还需要把真实麦克风 ASR、真实 MiniMax TTS 播放和现场噪声录音接进同一个证据包。

6. 感知算法还只是接口和融合层
   - 现有代码承载视觉注意力、声源方向、距离、姿态、手势、freshness。
   - 尚未接入真实姿态/视线估计、手势识别、麦克风阵列 DOA、声画关联、接近/停留追踪、多人仲裁算法。

7. 真实机器人 runtime 尚未接入
   - 当前仍以 fake/sim/shadow 为边界。
   - external/lab handoff contract 已有 disabled-by-default 适配层。
   - 真正 lab/prod 仍必须通过 RuntimeCapabilityRegistry、SafetyPreflight、runtime profile、审计日志和显式硬件开关。

8. Mission Control 的 operator 模型还不够硬
   - Dashboard runtime 控制动作已记录 operator_id、reason、risk acknowledgement。
   - TaskRunStore 已可持久化恢复任务运行记录。
   - 后续需要 operator RBAC 和审计查询页。

## Phase 0：可信 RAG 回答闭环

目标：客户问一句话，系统回答时能展示依据；没有可靠依据时不编。

已完成：

- MemoryBridge 增加 approval / expiry 过滤。
- `/api/chat` 返回 `evidence` 和 `rag`。
- `/api/live` 历史 assistant 消息保留 evidence/rag。
- Dashboard bot 气泡支持展示回答依据。
- 增加 memory 过滤、chat evidence、runtime wiring 测试。

下一步：

- 给无证据、冲突、过期知识补可配置的统一拒答/澄清话术模板。
- 把 Memory Health strip 的状态纳入客户演示脚本和运营检查清单。
- 扩展 scenario tests 到完整语音端到端：ASR transcript、InteractionGate、RAG evidence、reply、TTS、latency。

验收标准：

- `approval_status=draft` 的知识不进入 prompt。
- `expires_at` 已过期的知识不进入 prompt。
- published 且未过期的知识进入 prompt，并显示在回答依据中。
- Dashboard 每条 bot 气泡可以展开查看依据。

## Phase 1：Knowledge Console

目标：让非工程用户能在浏览器维护知识库。

已完成 MVP：

- 支持粘贴或上传 Markdown / JSON / JSONL / CSV 内容。
- 支持 dry-run 预览解析结果。
- 支持导入到 MemoryBridge。
- 支持在 UI 内输入问题做检索测试。
- 后端已新增 `/api/knowledge/preview` 和 `/api/knowledge/import`。
- 后端已新增 `/api/knowledge/list` 和 `/api/knowledge/update`。
- UI 已支持知识目录、发布、删除、恢复。
- 后端已支持 `rebuild_index` 与 `bulk_update`。
- UI 已支持刷新问答可用性、发布选中、删除选中。
- UI 已支持审批选中、刷新选中、每条知识独立审批/发布/刷新/删除/恢复。
- 目录已展示每条知识的生命周期状态、证据版本、索引版本、过期倒计时、冲突 ID 和是否需要重建索引。
- 目录已展示审批/驳回审计字段和最近事件。
- 冲突记录已支持选择一个版本保留，其余同组冲突项自动驳回。
- Catalog health 已提供客户可读运营指标：可回答、待审批、需重建、冲突、快过期、已过期、已删除。
- 重建索引已升级为可持久化的 KnowledgeIndexJobStore 任务历史，UI 显示最近刷新任务、任务号、耗时和失败原因。

下一步：

- 审批发布：draft -> pending -> approved -> published 的完整权限控制、审批队列和审批历史查询。
- 刷新任务产品化：当前已记录 job history，下一步补后台异步队列、进度轮询、重试和取消。
- 批量编辑 metadata 产品化：category、source、owner、expires_at、approval_status 的表格编辑与批量提交。
- 冲突处理产品化：同一事实出现互斥版本时进入 conflict，不允许确定性回答；当前已支持保留单条，下一步补合并编辑和版本 diff。

验收标准：

- 浏览器内完成“上传 -> 预览 -> 导入 -> 检索 -> 问答引用”的闭环。
- 删除或撤回后，回答不再引用该知识。
- 重建索引失败时 UI 显示明确错误。

## Phase 2：Freshness / Conflict / Version 硬约束

目标：把知识从“能搜到”升级成“可被信任的事实”。

数据模型需要补齐：

- record_id
- content_hash
- entity_key
- fact_key
- value
- source
- source_version
- authority
- observed_at
- imported_at
- valid_from
- expires_at
- approval_status
- evidence_version
- supersedes
- conflict_set_id

规则：

- expired 永不进入 prompt。
- draft/pending/rejected/deleted 永不进入 prompt。
- superseded 默认不进入 prompt，除非用户查历史。
- 同 entity_key + fact_key 的互斥 value 进入 conflict。
- 高风险任务遇到 stale/conflict knowledge 必须阻止 handoff。
- 无 metadata 的裸文本 backend 结果不能长期作为“可审计事实”直接驱动高风险回答或任务。

验收标准：

- 旧版 SOP 不覆盖新版 SOP。
- 两条设备位置冲突时，系统拒绝确定性回答。
- 过期区域/路线知识不能驱动 runtime plan。

## Phase 3：语音端到端评测

目标：证明系统不是组件可用，而是语音产品可用。

P0 场景：

- 游客问路，有依据回答。
- 游客问未知地点，拒答并建议咨询人工。
- 巡检 SOP，引用 SOP 后回答。
- 设备位置，引用位置知识。
- 错误知识拒答。
- 噪声 / 路人闲聊不触发回答。
- 多人同时说话时要求澄清。
- 急停 / 停止请求绕过普通交互门控。

核心指标：

- evidence_top1_hit
- unsupported_claim_count
- stale_evidence_usage
- false_respond_rate
- missed_help_rate
- first_useful_response_latency
- asr_final_ms
- rag_retrieve_ms
- tts_first_audio_ms

## Phase 4：真实感知算法接入合同

目标：askme 不内置全部算法，但能消费真实算法结果并做安全门控。

外部 provider 输出：

- pose/gaze：person_facing_robot、head_pose、gaze_confidence、observed_at。
- gesture：wave、raise_hand、pointing、stop、confidence、observed_at。
- DOA：sound_source_angle_deg、doa_confidence、audio_observed_at。
- audio-visual association：matched_track_id、association_confidence、reason。
- approach/dwell：approach_state、dwell_s、distance_m。
- arbitration：active_person_track_id、speaker_track_id、ambiguity_reason。

规则：

- 每个字段有自己的 freshness。
- 声源和画面人物不一致时不能默认回复画面人物。
- 多人不确定时澄清，不猜。
- stop / emergency intent 优先安全路径。

## Phase 5：真实 runtime 接入路线

目标：安全地从 fake/sim/shadow 过渡到 lab/prod。

profiles：

- fake：本地演示，不接外部服务。
- sim：模拟运行事件。
- shadow：真实预检和能力解析，但不发硬件动作。
- lab：受控实验室，低风险 skill。
- prod：默认禁用，需要显式配置和完整审计。

第一批 lab skill：

- status_report
- capture_image
- read_status_panel
- generate_report
- return_home

暂不开放：

- 复杂巡逻
- 机械臂抓取
- 靠近游客
- 操作门、电源、支付、删除数据
- 任何绕过 SafetyPreflight 的直接硬件控制

## 立即执行顺序

1. 把离线 Voice E2E 升级为真实麦克风回放评测：输入现场噪声/游客问路录音，输出同一套 transcript、gate decision、RAG evidence、reply、TTS、latency 证据包。
2. 把 Knowledge Trust 与 Voice E2E 报告合并成统一 Readiness Evidence 页面，让客户看到“能不能上线”的证据。
3. 将 Knowledge Console 的重建索引从同步按钮升级为异步 job，并补 job history、失败原因、重试入口。
4. 补 Knowledge Console 审批/版本/冲突处理 UI：draft -> pending -> approved -> published、superseded、conflict_set_id。
5. 新增 TaskRunStore：持久化 run state、runtime events、operator_actions、safety assessments 和 report。
6. 新增 operator RBAC：viewer/operator/supervisor/admin，对高风险 task、resume/advance/cancel 做权限和确认约束。
7. 接入真实感知 provider adapter 的最小合同测试：pose/gaze、gesture、DOA、audio-visual association、approach/dwell、multi-person arbitration。
8. 继续推进 external/lab runtime，但只做 shadow/lab 低风险 skill：status_report、capture_image、read_status_panel、generate_report、return_home。

## 本次验证记录

- `python -m pytest tests/test_memory_importer.py tests/test_runtime_modules.py tests/test_health.py tests/test_memory_bridge.py -q`
- `python -m ruff check askme/memory/importer.py askme/memory/vector_store.py askme/memory/bridge.py askme/runtime/modules/memory_module.py askme/health_server.py tests/test_memory_importer.py tests/test_memory_bridge.py tests/test_runtime_modules.py tests/test_health.py`
- Dashboard 内嵌脚本已用 Node 做语法解析检查。
- 2026-05-10 追加：修复 RAG `expires_at` ISO 解析回归；新增冲突 evidence 过滤；SkillGate 禁止 disabled skill 执行；Dashboard knowledge API 静态契约补测。
- 2026-05-10 追加：新增 `KnowledgeCatalog` MVP；MemoryModule import/list/update 已切到 catalog-first；deleted/conflicted records 不再同步进 prompt eligible backend。
- 2026-05-10 追加：新增 `source_version/evidence_version`；MemoryBridge 检索命中后执行 catalog version gate；stale/deleted catalog evidence 会进入 dropped evidence。
- 2026-05-10 追加：新增 RAG `answer_policy` 并在 Dashboard evidence / Retrieval Test Bench 中展示；当前策略字段已可供后续 prompt 拒答使用。
- 2026-05-10 追加：新增 RAG trust scenario evaluation；`python scripts/evaluate_rag_trust_scenarios.py --output artifacts/rag_trust/scenario-evaluation.json` 已生成评测证据包。
- 2026-05-10 追加验证：`python -m pytest tests/scenario_tests/test_rag_trust_evaluation.py tests/test_memory_catalog.py tests/test_memory_bridge.py tests/test_runtime_modules.py tests/test_health.py tests/test_memory_importer.py tests/test_skill_gate.py -q`，177 passed。
- 2026-05-10 追加验证：`python -m ruff check scripts/evaluate_rag_trust_scenarios.py tests/scenario_tests/test_rag_trust_evaluation.py askme/memory/catalog.py askme/memory/bridge.py askme/runtime/modules/memory_module.py askme/runtime/modules/health_module.py`，通过。
- 2026-05-10 追加：`answer_policy` 已进入 PromptBuilder；seed 模式下也会以策略交换消息保留，避免 system prompt 被丢弃后失效。
- 2026-05-10 追加验证：`python -m pytest tests/test_prompt_builder.py tests/test_turn_executor.py tests/scenario_tests/test_rag_trust_evaluation.py tests/test_memory_bridge.py -q`，112 passed。
- 2026-05-10 追加验证：`python -m ruff check askme/pipeline/prompt_builder.py askme/pipeline/turn_executor.py tests/test_prompt_builder.py tests/test_turn_executor.py`，通过。
- 2026-05-10 追加：Health snapshot 新增 `rag_trust`，Dashboard 运营诊断新增 Knowledge Trust 卡片。
- 2026-05-10 追加验证：`python -m pytest tests/test_runtime_modules.py::TestHealthModule tests/test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests/test_prompt_builder.py tests/test_turn_executor.py tests/scenario_tests/test_rag_trust_evaluation.py tests/test_memory_bridge.py -q`，128 passed。
- 2026-05-10 追加验证：Dashboard 内嵌脚本抽取后通过 `new Function(...)` 语法检查。
- 2026-05-10 追加：KnowledgeCatalog / MemoryModule 新增重建索引和批量 metadata 更新；Dashboard Knowledge Console 新增重建索引、发布选中、删除选中。
- 2026-05-10 追加：ActivePerceptionResolver 接入 CognitionModule，规划缺少新鲜感知事实时可请求刷新并重跑规划。
- 2026-05-10 追加：RuntimeArbiterClient contract 接入 runtime handoff，external/lab profile 默认禁用且不直接触碰硬件。
- 2026-05-10 追加：Dashboard runtime 控制动作携带 operator_id、reason、risk_acknowledgement，并进入 TaskRun operator_actions。
- 2026-05-10 追加验证：`python -m pytest tests/test_memory_catalog.py tests/test_runtime_modules.py tests/test_active_perception_resolver.py tests/test_cognition.py tests/test_runtime_arbiter_client.py tests/test_runtime_handoff.py tests/test_runtime_handoff_module.py tests/test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests/test_health.py::TestHealthServer::test_runtime_endpoints_delegate_to_handler tests/test_health.py::TestHealthServer::test_runtime_control_endpoint_forwards_operator_context -q`，104 passed。
- 2026-05-10 追加验证：Dashboard 内嵌脚本抽取后通过 `new Function(...)` 语法检查；Python 修改文件通过 `py_compile`；ruff 已排除 HTML 后重新验证通过。
- 2026-05-10 追加：新增 `scripts/evaluate_voice_e2e_scenarios.py` 和 `tests/scenario_tests/test_voice_e2e_evaluation.py`，生成 `artifacts/voice_e2e/scenario-evaluation.json`。
- 2026-05-10 追加：InteractionGate 修复两类真实场景：旁观者提到“机器狗”不再误唤醒；多人声画不一致时优先澄清。
- 2026-05-10 追加：Health snapshot 新增 `voice_e2e`，Dashboard 运营诊断新增 Voice E2E 卡片。
- 2026-05-10 追加验证：`python -m pytest tests/test_interaction_gate.py tests/scenario_tests/test_voice_e2e_evaluation.py -q`，26 passed；`python scripts/evaluate_voice_e2e_scenarios.py --output artifacts/voice_e2e/scenario-evaluation.json`，passed。

## Phase 6：园区现场场景包

目标：把机器狗从“会对话/会巡检”推进到“能值守园区现场”。本阶段覆盖 9 类客户可感知场景：机器人异常、夜间陌生人、车辆违停、火灾烟雾、垃圾桶满溢、突发巡检、人群聚集、路人指路、路人带路。

已新增详细计划：`plans/field-operations-scenario-plan.md`。

已落地：
- 新增 `askme.pipeline.field_scenarios`，把 9 类场景注册成产品级场景矩阵。
- 扩展 `askme.pipeline.incident_alerts`，新增夜间陌生人、违停、火灾烟雾、垃圾桶满溢、人群聚集、突发巡检的固定播报、钉钉消息、处理动作和归档要求。
- 告警模板现在带 `notification_group`，可区分安保、保洁、运营等处理对象。

下一步：
- 给现场事件补模拟触发入口和事件看板。
- 钉钉通知支持安保群/保洁群分流。
- 建立 SiteMap 配置：停车区、主通道、窗户、角落、垃圾桶、路引点、禁行区。
- 把感知输入统一成带 timestamp/confidence/source/freshness 的事件。
- 路人指路只在固定帮助点触发，路人带路必须依赖地图数据库和安全路线。

验收标准：
- 模拟每个场景都能生成证据、通知对象、语音话术、归档记录。
- 垃圾桶满溢只通知保洁，不打扰安保。
- 夜间陌生人、违停、火灾烟雾、人群聚集都能通知安保并保留证据。
- 路人问路不会误触发机器人任务，带路不会绕过地图和安全边界。

## 2026-05-11 Productization checkpoint: TaskRun recoverability

本次把 runtime handoff 从“内存里的运行状态”推进到“产品可恢复的任务记录”：

- 新增 `TaskRunStore`，持久化 TaskRun、runtime_events、safety_assessments、skill_results、operator_actions、report、shadow/sim state。
- `RuntimeHandoffService` 支持 `store_config`，服务重启后可以恢复任务历史、报告和人工操作记录。
- `RuntimeHandoffModule` 支持 `runtime_handoff.store` 配置注入。
- `config.yaml` 默认启用本地 demo store：`artifacts/runtime_handoff/task_runs.json`。
- 新增测试覆盖：completed run 重启恢复、sim pause/operator action 重启恢复、模块 store 配置注入。

验证：

- `python -m pytest tests/test_runtime_handoff.py tests/test_runtime_handoff_module.py -q` -> 32 passed
- `python -m pytest tests/test_runtime_handoff.py tests/test_runtime_handoff_module.py tests/test_field_operations.py tests/test_voice_profiles.py tests/test_health.py -q` -> 79 passed
- `python -m ruff check askme/runtime/handoff.py askme/runtime/modules/runtime_handoff_module.py tests/test_runtime_handoff.py tests/test_runtime_handoff_module.py` -> passed

产品反思：

- 这一步解决的是“客户刷新或服务重启后还能不能看到任务发生了什么”。没有这个能力，Dashboard timeline、异常报告、人工暂停/取消都只是演示态，不是产品态。
- 仍未完成的真实能力：真实摄像头/传感器 provider、真实钉钉发送、真实机器人 lab runtime、语音端到端现场麦克风评测、知识库冲突处理 UI。

## 2026-05-11 Productization checkpoint: notification delivery evidence

本次把现场事件通知从“只知道发了哪些渠道”推进到“能审计每个渠道是否真正送达”：

- `AlertDispatcher` 新增 `last_delivery_report`，记录每个渠道的 `sent / not_sent / failed / skipped`。
- 事件归档 `incident-alerts.jsonl` 会写入 `delivery_report`，不再只存 `channels`。
- `FieldEventRecord` 新增 `delivery_report`，现场事件 API / Dashboard 都能看到钉钉、日志等渠道的送达状态。
- Dashboard 现场事件卡片显示 `channel:status`，并在有失败/未发送时标记“通知需复核”。
- 现场事件支持“确认收到”和“处理完成”两步处置，事件列表会返回待处理/未关闭/已确认/已关闭汇总，Dashboard 可直接操作。

验证：

- `python -m pytest tests/test_alert_dispatcher.py tests/test_field_operations.py -q` -> 46 passed
- `python -m ruff check askme/pipeline/alert_dispatcher.py askme/pipeline/field_operations.py tests/test_alert_dispatcher.py tests/test_field_operations.py` -> passed
- Dashboard script `node --check .tmp/dashboard-script.js` -> passed

产品反思：

- 对客户来说，“已触发告警”不等于“保安群收到了”。送达证据必须进入事件记录，否则现场异常闭环不可审计。
- 下一步应接真实钉钉 webhook 的现场 smoke test，并把 webhook 响应码/错误内容细分为更明确的 failure reason。

## 2026-05-11 Productization checkpoint: real perception provider adapter

本次把 InteractionGate 从“预留字段”推进到“可接真实算法输出的 adapter”：

- 新增 `askme.perception.interaction_provider.FileInteractionPerceptionProvider`。
- 支持外部算法通过 JSON 文件接入：pose/gaze、gesture、microphone-array DOA、audio-visual association、approach/dwell、multi-person arbitration。
- 每个传感器输入独立做 freshness 判断；过期输入不会合并进语音交互门控。
- `PerceptionModule` 在配置启用时优先使用 interaction provider，否则回退到 VisionBridge。
- `config.yaml` 新增 `perception.interaction_provider` 示例路径。

验证：

- `python -m pytest tests/test_interaction_perception_provider.py tests/test_interaction_gate.py tests/test_runtime_modules.py -q` -> 70 passed
- `python -m ruff check askme/perception/interaction_provider.py askme/runtime/modules/perception_module.py tests/test_interaction_perception_provider.py` -> passed

产品反思：

- 真实机器狗不应该只靠“听到人声”判断是否回答。这个 adapter 让声源方向、视觉注意力、手势、停留时间和多人仲裁进入同一个准入门。
- 仍未完成：具体算法进程本身仍需由摄像头/麦克风阵列/VLM/姿态模型提供；本次完成的是可接入、可校验、可回归的产品合同。

## 2026-05-11 Productization checkpoint: knowledge operations auditability

本次把知识/RAG 从“能导入、能检索”推进到“客户能运营、能追责、能处理问题证据”：

- 新增 `KnowledgeIndexJobStore`，持久化知识刷新任务历史。
- `MemoryModule.rebuild_knowledge_index_payload` 现在返回并记录 `job_id/status/operator_id/started_at/completed_at/duration_ms/scanned/eligible/indexed/skipped/errors/backend/fallback_reason`。
- `/api/knowledge/list` 返回最近 `index_jobs`，Dashboard 可以看到最近刷新任务。
- Dashboard 将“重建索引”产品化为“刷新问答可用性”，展示任务号、写入/跳过/错误和最近任务历史。
- `MemoryBridge` 的 accepted/dropped evidence 都携带 `record_id/source_record_id/evidence_version`，聊天气泡里的证据可以定位到知识目录。
- `answer_policy` 新增 `required_operator_action`，冲突、过期、未审批证据不再只是提示原因，还给出运营处理动作。

验证：

- `python -m pytest tests/test_memory_bridge.py tests/test_runtime_modules.py -q` -> 90 passed
- `python -m pytest tests/test_memory_bridge.py tests/test_runtime_modules.py tests/test_alert_dispatcher.py tests/test_field_operations.py tests/test_runtime_handoff.py tests/test_runtime_handoff_module.py tests/test_interaction_perception_provider.py tests/test_interaction_gate.py tests/test_voice_profiles.py tests/test_health.py -q` -> 232 passed
- `python -m ruff check askme/memory/bridge.py askme/memory/index_jobs.py askme/runtime/modules/memory_module.py ...` -> passed
- Dashboard script `node --check .tmp/dashboard-script.js` -> passed

产品反思：

- 客户真正关心的不是“向量索引是否调用成功”，而是“这条知识现在能不能用于回答，不能用时谁来处理，处理动作有没有记录”。
- 本次仍是同步执行后记录 job history；下一步应补后台异步刷新队列、进度轮询、重试/取消按钮和独立知识审计页。

## 2026-05-11 Productization checkpoint: controlled RAG reply language

本次把知识/RAG 的“安全策略”继续往产品可控话术推进：

- `PromptBuilder` 新增 `rag_policy_templates`，让 no_evidence / filtered / stale / conflict / unapproved 的对外回复话术可配置。
- `BrainPipeline` 和 `PipelineModule` 已把 `brain.rag_policy_templates` 接入运行时，不需要改代码就能调整客户话术。
- `config.yaml` 默认提供中文产品话术，避免把 `state/action/reason` 这类内部字段直接说给客户。
- `[知识回答策略]` 现在会携带 `required_operator_action`，模型知道下一步应该引导管理员刷新、审批、复核或解决冲突。
- prompt 规则明确：对外优先使用 `customer_reply_template` 的含义，保持口语、简短、可执行，不念内部状态字段。

验证：

- `python -m pytest tests/test_prompt_builder.py tests/test_stage3_runtime.py tests/test_brain_pipeline.py tests/test_runtime_modules.py -q` -> 110 passed
- `python -m ruff check askme/pipeline/prompt_builder.py askme/pipeline/brain_pipeline.py askme/runtime/modules/pipeline_module.py tests/test_prompt_builder.py tests/test_stage3_runtime.py` -> passed

产品反思：

- 客户听到的应该是“我没有可靠依据，不能直接回答，请补充位置或让管理员上传知识”，而不是“RAG stale/conflict”。
- 这一步仍然依赖 LLM 遵守 prompt；下一步应在 `TurnExecutor` 或回答后处理层加入强制兜底，当策略要求拒答时即使模型输出偏离，也能替换为模板话术。

## 2026-05-11 Productization checkpoint: deterministic RAG refusal guard

本次把“知识不可用时拒答”从 prompt 约束升级为运行时硬边界：

- `PromptBuilder.build_forced_rag_reply()` 根据知识策略生成确定性客户话术。
- `TurnExecutor` 在调用 LLM 前检查 RAG policy；当状态为 conflict / stale / unapproved / filtered，或 no_evidence 显式要求 refuse 时，直接返回模板话术。
- 语音模式下强制拒答会直接播报模板，不再等待 LLM 生成，也不会把错误自由回答提前播出去。
- 对 grounded 和普通 `no_evidence + clarify_or_refuse` 保持非强制，避免普通寒暄或可澄清问题被过度拦截。
- 强制拒答仍会写入 conversation 和 memory，后续审计能看到用户问了什么、系统为什么没有回答。

验证：

- `python -m pytest tests/test_prompt_builder.py tests/test_turn_executor.py tests/test_stage3_runtime.py tests/test_brain_pipeline.py tests/test_runtime_modules.py -q` -> 140 passed
- `python -m ruff check askme/pipeline/prompt_builder.py askme/pipeline/turn_executor.py askme/pipeline/brain_pipeline.py askme/runtime/modules/pipeline_module.py tests/test_prompt_builder.py tests/test_turn_executor.py tests/test_stage3_runtime.py` -> passed

产品反思：

- 产品级语音机器人不能只“建议模型别编造”，必须在运行时阻止不可靠知识进入回答。
- 仍需补完整 E2E 场景：上传过期/冲突知识后通过 `/api/chat` 提问，验证用户气泡、回答气泡、证据卡片和知识处理入口全部一致。

## 2026-05-11 Productization checkpoint: chat evidence visibility

本次把 `/api/chat` 从“只返回文本”推进到“回答气泡可显示证据”：

- `/api/chat` 会在 chat handler 返回普通字符串时，从 `memory_handler.health()` 补齐最近 RAG 证据。
- 返回 payload 现在可以包含 `evidence`、`rag.answer_policy`、`rag.dropped_evidence`、`last_backend`、`last_retrieve_ms` 等可展示字段。
- 如果 chat handler 自己已经返回 evidence/rag，接口不会覆盖，避免破坏更专业的上层 handler。
- 这让 Dashboard 的聊天气泡能看到“这句话依据哪里”，也能看到过期/冲突/未审批证据为什么被拒绝。

验证：

- `python -m pytest tests/test_health.py tests/test_prompt_builder.py tests/test_turn_executor.py tests/test_runtime_modules.py -q` -> 150 passed
- `python -m ruff check askme/health_server.py askme/pipeline/prompt_builder.py askme/pipeline/turn_executor.py tests/test_health.py tests/test_prompt_builder.py tests/test_turn_executor.py` -> passed

产品反思：

- 客户问路、问设备位置、问 SOP 时，前端必须能展示证据来源；否则系统看起来像“在编”。
- 下一步仍要补 UI 细节：每条气泡下方显示证据摘要、知识编号、状态和“处理知识”入口，而不是只在全局知识面板里查。

## 2026-05-11 Productization checkpoint: customer-readable evidence UI

本次继续把聊天气泡里的证据展示从工程视角改成产品视角：

- Dashboard 不再直接展示 `state/action/reason/score` 这类内部字段。
- 证据卡片改为“可信状态、回答策略、建议处理、原因、匹配度、知识编号”。
- drop reason 映射为中文业务含义，例如“知识过期、知识冲突、未审批、不可用于回答”。
- 证据记录按钮继续直达知识目录，运营可以从聊天气泡进入处理知识。

验证：

- Dashboard script `node --check .tmp/dashboard-script.js` -> passed
- `python -m pytest tests/test_health.py -q` -> 36 passed

产品反思：

- 对客户展示证据时，要让人知道“能不能信、为什么不能信、下一步谁处理”，而不是暴露内部实现名。
- 仍需做视觉回归截图，确认移动端和桌面端气泡证据卡不会挤压或遮挡。

## 2026-05-11 Productization checkpoint: DingTalk production webhook signing

本次把现场告警通知从“裸 webhook”补到可接生产钉钉机器人的安全配置：

- `AlertDispatcher` 支持 `dingtalk_secret`，按钉钉机器人加签规则追加 `timestamp/sign` 参数。
- 未配置 secret 时保持原有裸 webhook 行为，兼容本地和测试环境。
- `config.yaml` 在 `proactive.alerts` 下新增 `dingtalk_secret` 配置位，可在部署配置中用 `${DINGTALK_SECRET}` 注入。
- 现场异常场景仍通过同一条 dispatch/archive 链路写入 delivery_report。

验证：

- `python -m pytest tests/test_alert_dispatcher.py tests/test_field_operations.py -q` -> 48 passed
- `python -m ruff check askme/pipeline/alert_dispatcher.py tests/test_alert_dispatcher.py` -> passed

产品反思：

- 客户现场常会开启钉钉机器人加签；如果只支持裸 webhook，演示能过、生产会失败。
- 下一步应给不同通知组 security / cleaning / operations 分别支持独立 secret，并提供配置校验/试发接口。

## 2026-05-11 Productization checkpoint: grouped DingTalk secrets

本次继续把现场通知推进到真实部署形态：

- `FieldOperationsService` 支持 `dingtalk_secrets.security / cleaning / operations`。
- 现场事件分发时会按通知组同时选择 webhook 和 secret。
- 未配置某个组 secret 时可回退到 security secret，保证关键告警仍有默认安全配置。
- `config.yaml` 新增 `ASKME_DINGTALK_SECURITY_SECRET / CLEANING_SECRET / OPERATIONS_SECRET` 配置入口。

验证：

- `python -m pytest tests/test_field_operations.py tests/test_alert_dispatcher.py -q` -> 50 passed
- `python -m ruff check askme/pipeline/field_operations.py askme/pipeline/alert_dispatcher.py tests/test_field_operations.py tests/test_alert_dispatcher.py` -> passed

产品反思：

- 保安、保洁、运维是不同响应队伍，不能共用一个群和一套密钥。分组 webhook + secret 是现场交付的基本要求。
- 仍缺：Dashboard 上的“试发通知”按钮、配置健康检查、真实 webhook smoke test 的响应码归档。

## 2026-05-11 Productization checkpoint: notification smoke test API

本次把“通知配置是否可用”做成可测试产品能力：

- `FieldOperationsService.test_notification_payload()` 支持向 security / cleaning / operations 发送低风险测试通知。
- 新增 `/api/field/notification-test`，Dashboard 或交付脚本可直接试发通知。
- 试发返回 `sent/status/webhook_configured/secret_configured/sent_channels/delivery_report`。
- 非法通知组返回 422，不会误发。

验证：

- `python -m pytest tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py -q` -> 88 passed
- `python -m ruff check askme/pipeline/field_operations.py askme/health_server.py tests/test_field_operations.py` -> passed

产品反思：

- 现场交付时不能等异常真的发生才知道钉钉群没配好；必须有试发和送达报告。
- 仍缺 Dashboard 按钮和真实 webhook 响应细节归档，例如 HTTP 状态码、钉钉错误码、错误消息。

## 2026-05-11 Productization checkpoint: notification smoke test UI

本次把通知试发从 API 补到客户可操作的 Dashboard：

- 现场事件面板新增“试发保安群”“试发保洁群”。
- 点击后调用 `/api/field/notification-test`，展示 webhook/secret 是否配置、每个渠道的送达状态和失败原因。
- 试发结果显示在现场事件区域，不需要客户看日志或懂接口。

验证：

- Dashboard script `node --check .tmp/dashboard-script.js` -> passed
- `python -m pytest tests/test_field_operations.py tests/test_health.py -q` -> 50 passed

产品反思：

- 现场交付的第一件事应该是试发通知，而不是等真实火警/违停/卡住时才发现群没配。
- 仍需真实钉钉响应码和错误码归档，以及一次带真实 webhook 的人工 smoke test。

## 2026-05-11 Productization checkpoint: notification delivery diagnostics

本次把通知送达从“成功/失败”推进到“可诊断失败原因”：

- `_post_json(..., return_result=True)` 现在可返回 `http_status / response_excerpt / error_type / reason`。
- DingTalk / WeCom / Feishu / generic webhook 的真实 HTTP 发送路径会把响应状态和响应摘要带入 `delivery_report`。
- 兼容旧调用：普通 `_post_json(url, body)` 仍返回 bool；现有 mock 和测试不需要整体重写。
- 未配置 webhook 的失败原因从模糊的 `not_configured_or_failed` 细化为 `not_configured`。

验证：

- `python -m pytest tests/test_alert_dispatcher.py tests/test_field_operations.py -q` -> 55 passed
- `python -m ruff check askme/pipeline/alert_dispatcher.py tests/test_alert_dispatcher.py` -> passed

产品反思：

- 现场通知失败时，客户需要看到“未配置、HTTP 500、钉钉关键词不匹配、签名错误”等可行动原因。
- 仍缺：现场事件卡片要把这些细节展示出来，而不只是 `dingtalk:not_sent`。

## 2026-05-11 Productization checkpoint: notification diagnostics visible in UI

本次把通知送达诊断显示到 Dashboard：

- 现场事件卡片会显示 `已送达/未送达/失败/跳过` 的中文状态。
- 如果后端提供 `http_status / reason / response_excerpt`，卡片会显示 HTTP 状态码、失败原因和响应摘要。
- 试发通知和真实现场事件共用同一套送达报告展示。

验证：

- Dashboard script `node --check .tmp/dashboard-script.js` -> passed
- `python -m pytest tests/test_field_operations.py tests/test_health.py -q` -> 50 passed

产品反思：

- 交付人员不应该打开日志才能知道“钉钉为什么没收到”。失败原因必须出现在产品界面上。
- 仍需真实 webhook 人工 smoke test；没有真实密钥时只能验证代码路径和 mock HTTP 响应。

## 2026-05-11 Productization checkpoint: field scenario evaluation suite

本次把现场运营从单点测试推进到产品场景评测：

- 新增 `scripts/eval/evaluate_field_operations_scenarios.py`。
- 覆盖 10 个客户场景：机器人卡住、夜间陌生人拍照、车辆违停、烟火传感器、垃圾桶满溢、管理员突发巡检、人群聚集、固定路引点问路、游客带路、通知试发。
- 评测明确 `external_services=false`、`hardware_dispatch=false`，可在无真实机器人/无真实钉钉密钥时验证产品规则。
- 新增 `tests/scenario_tests/test_field_operations_evaluation.py`，把评测纳入回归。
- 评测暴露并修复了一个业务路由问题：`patrol.urgent_dispatch` 现在通知 operations，而不是默认 security。

验证：

- `python -m pytest tests/scenario_tests/test_field_operations_evaluation.py tests/test_field_operations.py tests/test_alert_dispatcher.py -q` -> 56 passed
- `python -m ruff check scripts/eval/evaluate_field_operations_scenarios.py tests/scenario_tests/test_field_operations_evaluation.py askme/pipeline/incident_alerts.py tests/test_alert_dispatcher.py` -> passed

产品反思：

- 这些场景正是客户会问“机器狗到底能干什么”的答案，必须能一键评测，不应该散落在单元测试里。
- 仍缺真实输入源：摄像头检测、烟感/温度传感器、机器人故障事件、地图区域服务要实际接到 `/api/field/ingest`。

## 2026-05-11 Productization checkpoint: real field ingest adapters and readiness visibility

本次把“现场运营场景”从框架继续推进到可接真实输入源的产品能力：

- 新增 `askme/pipeline/field_ingest_adapters.py`，把真实摄像头、传感器、机器人诊断、地图区域 payload 统一归一化为 `/api/field/ingest` 可消费的事件。
- 摄像头输入支持 `class_id / class_name / label` 映射，能把车、人、烟火、垃圾桶等检测结果转成业务标签；例如 `class_id=2` 可触发车辆违停判断。
- 传感器输入支持温度、烟雾、垃圾桶满溢等字段，能触发火灾烟雾、垃圾桶满溢等场景。
- 机器人诊断输入支持 `fault_type / fault_code / joint_id` 等字段，明确来自诊断事件，不再靠自然语言或电机负载猜测故障。
- 地图区域输入支持 `zone / map_zone` 展开为 `zone_id / zone_name / location / zone_type / parking_allowed / help_point_id`，让违停、路引点、带路等场景有区域约束。
- `RuntimeHealthProvider` 新增 `field_operations` 健康快照，读取 `artifacts/field_operations/scenario-evaluation.json`，Dashboard 能看到现场运营场景是否通过、是否接了真实硬件、是否依赖外部服务。
- `askme runtime field-eval` 可生成现场运营场景评测报告；`askme runtime field-ingest-file` 可用 JSON/JSONL 离线验证真实设备样例，支持 dry-run 和 POST 到运行中的服务。
- Dashboard 修复了多处历史中文乱码/坏标签，恢复知识审批、现场事件、RAG 证据、准入门、语音识别等客户可见入口。

验证：

- `python -m pytest tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 164 passed
- `python -m ruff check askme/pipeline/field_ingest_adapters.py askme/pipeline/field_operations.py askme/runtime/modules/health_module.py askme/cli.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_runtime_modules.py tests/test_cli.py` -> passed
- `node --check .tmp/dashboard-script.js` -> passed
- `python -m askme runtime field-eval --output artifacts/field_operations/scenario-evaluation.json --json` -> 10/10 scenarios passed

产品反思：

- 现在真正补上的是“真实设备数据进系统”的薄适配层，不是直接接硬件控制。这样摄像头、烟感、机器人诊断、地图区域任一接入后，都会走同一套通知、归档、UI、评测链路。
- 下一步不能再停在 mock 场景：要拿一份真实相机检测 JSON、一份真实烟感/温度 payload、一份真实机器人诊断 payload，跑 `field-ingest-file --dry-run`，然后在本地服务上 POST 到 `/api/field/ingest` 做端到端冒烟。
- 仍未完成：真实 DingTalk key 人工 smoke test、真实机器人 runtime/lab 接入、真实摄像头/传感器常驻进程、现场麦克风语音场景评测。

## 2026-05-11 Productization checkpoint: field device bridge process

本次继续补齐“真实输入源接入”的最后一段交付路径：

- 新增 `askme/pipeline/field_ingest_bridge.py`，作为可测试的桥接逻辑：读取真实设备产生的 JSON/JSONL，增量处理新增事件，归一化后 POST 到 `/api/field/ingest`。
- 新增 `scripts/runtime/bridges/field_ingest_bridge.py`，现场可以作为相机、烟感、机器人诊断旁路进程运行。
- 新增 `askme runtime field-ingest-bridge`，让交付人员不用记脚本路径，可以直接用 CLI 做 dry-run 或 watch：
  - `python -m askme runtime field-ingest-bridge camera-events.jsonl --dry-run --json`
  - `python -m askme runtime field-ingest-bridge camera-events.jsonl --watch --server http://127.0.0.1:8765`
- `/api/field/ingest` 的帮助 payload 现在返回 bridge 命令、JSONL offset 策略、JSON snapshot fingerprint 策略，以及带 `class_id=2` 的相机违停样例。
- JSONL 使用 offset state，避免同一条设备事件被重复上报；JSON snapshot 使用 fingerprint，文件未变化时不重复处理。
- 支持 dry-run，本地即可验证真实相机/传感器/机器人诊断样例是否会触发正确场景。

验证：

- `python -m pytest tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 168 passed
- `python -m ruff check askme/pipeline/field_ingest_bridge.py askme/pipeline/field_ingest_adapters.py askme/pipeline/field_operations.py askme/runtime/modules/health_module.py askme/cli.py scripts/runtime/bridges/field_ingest_bridge.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_runtime_modules.py tests/test_cli.py` -> passed
- `python -m askme runtime field-ingest-bridge .tmp/field-ingest-sample.jsonl --dry-run --json` -> `status=ok,count=1`，相机 `class_id=2` 被归一化为 `vehicle`，并保留 `zone_id / zone_type / parking_allowed`。

产品反思：

- 这一层解决的是“真实设备程序不需要懂 Askme 内部场景规则”。相机只要写检测 JSONL，烟感只要写温度/烟雾 JSON，机器人诊断只要写 fault JSON，桥接器负责送入统一现场运营链路。
- 下一步要做真实现场冒烟：让一个真实相机检测进程、一个真实烟感/温度采集进程、一个真实机器人诊断进程分别写入 JSONL，启动 bridge `--watch`，确认 Dashboard 出现事件、通知送达、归档落盘。

## 2026-05-11 Productization checkpoint: field ingest HTTP smoke

本次把“设备 JSONL -> bridge -> `/api/field/ingest` -> 事件查询/归档”做成可一键执行的交付冒烟：

- 新增 `askme runtime field-ingest-smoke`。
- 不传 `--server` 时，会自动启动一个临时本地 FastAPI 服务，配置独立事件归档文件。
- 命令会生成三类设备事件：
  - 摄像头车辆检测：`class_id=2` + 主通道禁停区域 -> `illegal_parking`
  - 烟感/温度：`temperature_c=68` + `smoke_level=0.82` -> `fire_or_smoke`
  - 机器人诊断：`joint_motor_fault` -> `robot_abnormal_incident`
- 三类事件都通过 bridge 真实 POST 到 `/api/field/ingest`，再查询 `/api/field/events` 验证归档。
- 冒烟报告写入 `artifacts/field_operations/smoke/field-ingest-smoke.json`，可作为现场交付证据。

验证：

- `python -m askme runtime field-ingest-smoke --output-dir artifacts/field_operations/smoke --json` -> `status=passed,event_count=3`
- `python -m pytest tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 169 passed
- `python -m ruff check askme/pipeline/field_ingest_bridge.py askme/pipeline/field_ingest_adapters.py askme/pipeline/field_operations.py askme/runtime/modules/health_module.py askme/cli.py scripts/runtime/bridges/field_ingest_bridge.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_runtime_modules.py tests/test_cli.py` -> passed

产品反思：

- 这比 dry-run 更接近真实交付：它验证了 HTTP endpoint、bridge、归一化、业务规则、事件归档和查询，而不是只验证单个函数。
- 仍缺真实设备：目前 smoke 使用生成样例。下一步要把同一命令的 `--server` 指向实际运行服务，再把真实相机/烟感/机器人诊断进程写出的 JSONL 交给 `field-ingest-bridge --watch`。

## 2026-05-11 Productization checkpoint: field deployment readiness gate

本次把“还差什么才能交付”做成可运行的 readiness gate：

- 新增 `askme/pipeline/field_deployment_readiness.py`。
- `FieldOperationsService.readiness_payload()` 汇总以下证据：
  - 现场运营场景评测是否通过。
  - field ingest HTTP smoke 是否通过。
  - 事件归档是否已有事件。
  - security / cleaning / operations 三组钉钉 webhook 和 secret 是否配置。
  - 最近一次 smoke 是否打到真实部署服务，而不是临时本地服务。
  - scenario report 是否仍是 `hardware_dispatch=false` / `external_services=false`。
- 新增 `/api/field/readiness`。
- 新增 `python -m askme runtime field-readiness`。
- readiness 状态分三档：
  - `blocked`：缺场景评测、HTTP smoke 或事件归档。
  - `ready_for_lab`：链路可演示，但缺真实部署/真实硬件/真实外部服务。
  - `production_ready`：场景、smoke、归档、通知、真实硬件和外部服务证据都齐。

当前实测结果：

- `python -m askme runtime field-readiness --json` -> `status=ready_for_lab`
- 当前 blockers 为空。
- 当前 warnings 明确指出：
  - security / cleaning / operations 钉钉 webhook 未配置。
  - 最近 HTTP smoke 使用临时本地服务，不是运行中的部署服务。
  - `hardware_dispatch=false`，还没接真实硬件。
  - `external_services=false`，还没做真实钉钉送达。

验证：

- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 174 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_ingest_bridge.py askme/pipeline/field_ingest_adapters.py askme/pipeline/field_operations.py askme/runtime/modules/health_module.py askme/health_server.py askme/cli.py scripts/runtime/bridges/field_ingest_bridge.py tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_runtime_modules.py tests/test_cli.py` -> passed

产品反思：

- 这一步解决的是“不要靠感觉判断完成度”。现在产品能明确告诉客户：当前达到哪一级交付状态、下一步还需要补齐哪些现场上线项。
- 下一步应按 readiness 的 `next_actions` 顺序推进：配置真实钉钉 webhook/secret，启动真实 askme 服务，用 `field-ingest-smoke --server` 打真实服务，再把真实设备 JSONL 交给 `field-ingest-bridge --watch`。

## 2026-05-11 Productization checkpoint: Dashboard field readiness gate

本次把后端已有的现场交付 readiness gate 显示到客户可见的 Dashboard：

- Dashboard 新增 `Field Readiness` 诊断卡片，直接请求 `/api/field/readiness`。
- 卡片会显示 `production_ready / ready_for_lab / blocked`，并展示 blockers、warnings、gates 和 next actions。
- 这让客户演示时不用看 CLI 或日志，就能知道当前系统是“可实验室演示”还是“可生产交付”，以及真实钉钉、真实硬件、真实部署 smoke 还缺什么。
- `tests/test_health.py` 新增 Dashboard 契约断言，避免后续 UI 回退成只有后端接口。

验证：

- Dashboard script `node --check .tmp/dashboard-script.js` -> passed
- `python -m pytest tests/test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests/test_field_deployment_readiness.py -q` -> 4 passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 174 passed

产品反思：

- 现场运营不是“写了接口就算产品化”。客户真正需要看到的是：系统现在能不能演示、能不能上线、不能上线的证据是什么。
- 下一步继续从 `ready_for_lab` 往 `production_ready` 推：真实钉钉 webhook smoke、真实 askme 部署 smoke、真实设备 JSONL 常驻 bridge、真实机器人 runtime/lab profile。

## 2026-05-11 Productization checkpoint: incident response playbooks

本次把现场事件从“消息通知”推进成“可执行处置流程”：

- `incident_alerts.py` 新增 `IncidentPlaybook`，每类事件都带上客户状态、机器人动作策略、播报音色、响应组、操作员清单、证据要求、升级等待时间。
- `FieldEventRecord` 新增 `playbook`，所有归档事件、HTTP 返回、Dashboard 展示都能看到同一份处置策略。
- 机器人异常、夜间陌生人、违停、烟火、垃圾桶满、人群聚集、突发巡检都具备固定 playbook。
- 路人问路和带路补了服务类 playbook：只在固定路引点主动询问，只回答地图库已有地点，未知地点拒绝编造；带路要求低速、路线安全、目的地存在。
- Dashboard 现场事件卡片新增 playbook 行：显示 motion policy、TTS profile、升级时间和前三条操作员清单。

验证：

- Dashboard script `node --check .tmp/dashboard-script.js` -> passed
- `python -m pytest tests/test_field_operations.py tests/test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 15 passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 174 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_ingest_bridge.py askme/pipeline/field_ingest_adapters.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/runtime/modules/health_module.py askme/health_server.py askme/cli.py scripts/runtime/bridges/field_ingest_bridge.py tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_runtime_modules.py tests/test_cli.py tests/test_health.py` -> passed

产品反思：

- 客户不会只问“有没有发钉钉”，而会问“机器人现场应该停下、后退、观察、继续，还是等人接管”。Playbook 把这类决策显性化。
- 下一步要让 runtime 真正消费 `playbook.robot_motion_policy` 和 `playbook.tts_profile`，而不是只在事件里展示。

## 2026-05-11 Productization checkpoint: playbook voice profile resolution

本次把 playbook 的播报策略接到真实 TTS profile：

- `voice_profiles.py` 新增 `VOICE_PROFILE_ALIASES` 和 `resolve_voice_profile_id()`。
- 现场 playbook 里的产品化策略名可以映射到真实 TTS profile：
  - `emergency_alert` -> `emergency_short`
  - `security_alert` -> `security_clear`
  - `service_notice` / `visitor_service` -> `visitor_friendly`
  - `patrol_notice` / `mission_control` -> `patrol_default`
  - `night_security` -> `night_quiet`
- `TTSEngine.set_voice_profile_payload()` 现在接受别名，并返回 `requested_profile` 与 `resolved_profile`，便于 Dashboard/日志解释为什么选择了某个音色。

验证：

- `python -m pytest tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/test_field_operations.py -q` -> 79 passed
- `python -m ruff check askme/voice/voice_profiles.py askme/voice/tts.py tests/test_voice_profiles.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py tests/test_field_operations.py` -> passed

产品反思：

- 播报策略必须用产品语言表达，例如“紧急告警”“游客服务”，但底层 TTS 需要稳定 profile id。别名层让产品策略和技术配置解耦。
- 下一步要让现场事件触发时自动调用 voice profile 切换和播报，而不是只把 profile 放在事件返回里。

## 2026-05-11 Productization checkpoint: field event voice directives

本次继续把现场 playbook 接近真实语音执行：

- `FieldEventRecord` 新增 `voice_directive`。
- 每个有现场播报的事件都会生成：
  - `text`：最终要播报的文本。
  - `requested_profile`：playbook 中的产品化播报策略。
  - `resolved_profile`：真实 TTS profile id。
  - `interrupt_current_speech`：P0/紧急事件会打断当前播报。
  - `playback_mode`：紧急事件 `immediate`，普通事件 `queued`。
- LLM 改写低风险播报后，`voice_directive.text` 会同步更新，避免播报文本和事件文本不一致。
- Dashboard 事件卡片显示 `requested_profile -> resolved_profile` 和 playback mode。

验证：

- Dashboard script `node --check .tmp/dashboard-script.js` -> passed
- `python -m pytest tests/test_field_operations.py tests/test_voice_profiles.py tests/test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 18 passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 239 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_ingest_bridge.py askme/pipeline/field_ingest_adapters.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/voice/voice_profiles.py askme/voice/tts.py askme/runtime/modules/health_module.py askme/health_server.py askme/cli.py scripts/runtime/bridges/field_ingest_bridge.py tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_runtime_modules.py tests/test_cli.py tests/test_health.py tests/test_voice_profiles.py` -> passed

产品反思：

- 现在系统已经能给出“这一事件该怎么说、用什么音色、是否打断当前播报”的机器可执行指令。
- 仍未完成的是把 `voice_directive` 自动交给正在运行的 `TTSEngine`，这需要 FieldOperationsService 与 VoiceModule/AudioAgent 之间建立受控依赖，不能直接硬耦合。

## 2026-05-11 Productization checkpoint: field events dispatch to voice handler

本次把现场事件播报从“可执行指令”继续推进到真实 HTTP 链路：

- `/api/field/events` 和 `/api/field/ingest` 在事件触发成功、且服务已配置 `voice_handler` 时，会自动消费 `event.voice_directive`。
- 执行顺序：
  1. 根据 `voice_directive.resolved_profile` 调用 `set_voice_profile_payload()`。
  2. 调用 `speak()` 排队播报现场话术。
  3. 如果 voice handler 支持 `start_playback()`，立即启动播放。
- API 响应新增 `voice_delivery`，说明语音是否已排队、使用了哪个 profile、文本长度，或说明跳过/失败原因。
- 未配置 voice handler 时不会报错，会返回 `voice_delivery.status=skipped`，方便实验室/纯后端测试。

验证：

- `python -m pytest tests/test_field_operations.py tests/test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests/test_voice_profiles.py -q` -> 19 passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 240 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_ingest_bridge.py askme/pipeline/field_ingest_adapters.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/voice/voice_profiles.py askme/voice/tts.py askme/runtime/modules/health_module.py askme/health_server.py askme/cli.py scripts/runtime/bridges/field_ingest_bridge.py tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_runtime_modules.py tests/test_cli.py tests/test_health.py tests/test_voice_profiles.py` -> passed

产品反思：

- 这一步让“异常情况语音播报”从需求描述变成真实接口行为：烟火、摔倒、卡住、恶意挡路、关节故障等事件可以直接排队播放对应话术。
- 还差真实设备端 smoke：需要启动真实服务并配置真实 MiniMax/本地 TTS，再用相机/传感器/机器人 JSONL 触发 `/api/field/ingest`，确认现场真的出声。

## 2026-05-11 Productization checkpoint: field voice smoke command

本次新增可交付的现场语音冒烟命令：

- 新增 `python -m askme runtime field-voice-smoke`。
- 默认启动临时本地 FastAPI 服务和录音式 `VoiceHandler`，验证完整链路：
  `/api/field/events -> event.playbook -> event.voice_directive -> set_voice_profile_payload -> speak -> start_playback`。
- 支持 `--scenario fire|joint_fault|illegal_parking`，可以分别验证烟火、关节电机故障、违停播报。
- 支持 `--live-tts`，在临时本地服务中加载真实 `TTSEngine`，用于现场确认真实扬声器/ MiniMax / 本地 TTS 是否出声。
- 命令会写入 `field-voice-smoke.json`，包含请求、事件、voice_delivery、voice_directive、录音式 handler 记录。

实测：

- `python -m askme runtime field-voice-smoke --output-dir artifacts/field_operations/smoke --scenario fire --json` -> `status=passed`
- 实测结果中 `voice_delivery.status=queued`，`emergency_alert -> emergency_short`，`playback_started=true`。
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 242 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_ingest_bridge.py askme/pipeline/field_ingest_adapters.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/voice/voice_profiles.py askme/voice/tts.py askme/runtime/modules/health_module.py askme/health_server.py askme/cli.py scripts/runtime/bridges/field_ingest_bridge.py tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_runtime_modules.py tests/test_cli.py tests/test_health.py tests/test_voice_profiles.py` -> passed

产品反思：

- 这条命令是给交付和售前用的，不是单元测试。它回答“异常事件能不能真的触发播报链路”。
- 真实出声仍需要人工在目标机器上运行：`python -m askme runtime field-voice-smoke --scenario fire --live-tts`，并确认音频设备、MiniMax key 或本地 TTS 可用。

## 2026-05-11 Productization checkpoint: voice smoke as deployment gate

本次把“异常事件能否触发语音播报”纳入现场部署 readiness，而不是只停留在独立 smoke 命令：

- `field-readiness` 新增 `voice_smoke_report_path`，默认读取 `artifacts/field_operations/smoke/field-voice-smoke.json`。
- readiness 新增 gates：
  - `voice_smoke_passed`
  - `voice_smoke_uses_live_tts`
  - `voice_smoke_against_existing_server`
- 如果语音 smoke 没跑过，系统会进入 blocker：`field voice smoke has not passed`。
- 如果只跑了录音式 handler 或临时本地服务，系统不会谎称 production ready，而是给出 warnings 和 next actions：
  - `Run field-voice-smoke against the deployed service with --server`
  - `Run field-voice-smoke with --live-tts on the target audio device`
- Dashboard readiness card 修复 gate 布尔值渲染，之前后端返回 boolean 时前端会当成对象读取，导致 gate 容易误显示失败；现在会正确显示前 8 个部署 gate。
- Dashboard readiness card 的 gate 展示改成产品优先排序，先显示“场景评测、传感器入口、异常播报链路、真实 TTS 出声、是否连接部署服务”，再显示通知、硬件和外部服务。

实测：

- `python -m askme runtime field-voice-smoke --output-dir artifacts\field_operations\smoke --scenario fire --json` -> `status=passed`
- `python -m askme runtime field-readiness --json` -> `status=ready_for_lab`
- readiness 当前明确显示：
  - `voice_smoke_passed=true`
  - `voice_smoke_uses_live_tts=false`
  - `voice_smoke_against_existing_server=false`
  - 钉钉 webhook、真实硬件、真实外部服务仍未配置
- `node --check .tmp/dashboard-readiness.js` -> passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_cli.py::test_cli_runtime_field_readiness_reads_local_files tests/test_cli.py::test_cli_runtime_field_voice_smoke_queues_recorded_voice -q` -> 5 passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 242 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/health_server.py askme/cli.py askme/voice/voice_profiles.py askme/voice/tts.py tests/test_field_deployment_readiness.py tests/test_field_operations.py tests/test_health.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py` -> passed

产品反思：

- 这一步解决的是客户验收里的“说得清楚”问题：不是笼统说异常播报已做，而是区分“事件到播报链路已通”“是否真实 TTS 出声”“是否连接部署服务”“是否连接钉钉和真实硬件”。
- 仍未完成的是现场真机验证：需要在目标机器上用真实 MiniMax key/本地 TTS、真实扬声器、真实部署服务跑 `field-voice-smoke --live-tts --server ...`，并把钉钉 webhook、摄像头/烟感/机器人诊断 JSONL 接入同一 readiness。

## 2026-05-11 Productization checkpoint: customer-facing evidence language

本次继续把 Dashboard 从工程控制台收敛成客户能理解的产品界面：

- `Knowledge Trust` 改为 `可信知识检查`。
- `Field Readiness` 改为 `现场交付就绪`。
- 聊天气泡里的证据状态从 `No evidence / Stale knowledge / Evidence filtered / RAG backend` 改成：
  - `没有可靠依据`
  - `知识已过期`
  - `知识有冲突`
  - `知识未审批`
  - `证据已拦截`
  - `知识库来源`
  - `降级原因`
- 证据元信息从 `record / score / used / blocked` 改成 `知识编号 / 匹配 / 已引用 / 已拦截`，保留一键查看或审核知识记录的按钮。
- `operatorActionLabel()` 和 `dropReasonLabel()` 改成中文业务动作，避免客户看到英文策略名。

验证：

- `node --check .tmp/dashboard-script.js` -> passed
- `python -m pytest tests/test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests/test_health.py::TestHealthServer::test_chat_endpoint_attaches_memory_evidence_for_plain_text_handler -q` -> 2 passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 242 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/health_server.py askme/cli.py askme/voice/voice_profiles.py askme/voice/tts.py tests/test_field_deployment_readiness.py tests/test_field_operations.py tests/test_health.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py` -> passed

产品反思：

- 用户前面明确指出不能再说 `fake/sim runtime` 这类专业词。界面和回答都要把工程状态翻译成客户关心的问题：能不能交付、依据是否可靠、为什么拒答、下一步谁处理。
- 仍需要继续处理的语言面：runtime/voice/program cards 里还有 `ASR/TTS/runtime/arbiter/backend` 等工程词；这些应逐步移到“技术详情”，主界面保留“正在听、正在理解、等待确认、已通知安保、需要人工接管”等产品状态。

## 2026-05-11 Productization checkpoint: DingTalk notification smoke

本次把“通知钉钉群处理”从配置项继续推进成可运行 smoke：

- 新增 `python -m askme runtime field-notification-smoke`。
- 本地模式会同时启动：
  - 临时 Askme HTTP 服务；
  - 本地 HTTP webhook collector；
  - 将 security / cleaning / operations 三个响应组的 DingTalk webhook 指向 collector；
  - 通过 `/api/field/notification-test` 走真实 `AlertDispatcher -> dingtalk -> HTTP POST` 路径。
- 部署模式支持 `--server`，可直接打到已有 Askme 服务，由真实服务配置决定是否发到真实钉钉。
- smoke 报告写入 `artifacts/field_operations/smoke/field-notification-smoke.json`，包含每个组的 delivery_report、collector 请求、HTTP 状态、发送组列表。
- `field-readiness` 新增读取 `notification_smoke_report_path`，并新增 gates：
  - `notification_smoke_passed`
  - `notification_smoke_uses_external_services`
  - `notification_smoke_against_existing_server`
- Dashboard readiness card 新增“钉钉通知链路通过 / 真实钉钉通知验证 / 通知验证连接部署服务”。
- 修复 Windows GBK 控制台 JSON 输出：非 UTF-8 stdout 下 `_json()` 自动转义 Unicode，避免钉钉 markdown 里的告警图标或中文导致 CLI 崩溃。

实测：

- `python -m askme runtime field-notification-smoke --output-dir artifacts\field_operations\smoke --groups security,cleaning,operations --json` -> `status=passed`，`collector_request_count=3`
- `python -m askme runtime field-readiness --json` -> `status=ready_for_lab`，`notification_smoke_passed=true`
- `python -m pytest tests/test_cli.py::test_cli_json_output_escapes_unicode_on_non_utf8_stdout tests/test_cli.py::test_cli_runtime_field_notification_smoke_uses_local_collector -q` -> 2 passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 245 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/health_server.py askme/cli.py askme/voice/voice_profiles.py askme/voice/tts.py tests/test_field_deployment_readiness.py tests/test_field_operations.py tests/test_health.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py` -> passed

产品反思：

- “通知钉钉群”不能只靠 webhook 字符串是否存在判断。现在至少能证明 Askme 会按组发出真实 HTTP POST，并把每次 delivery_report 写进 smoke 报告和 readiness。
- 仍未完成的是生产钉钉验证：需要在目标部署上配置真实 `ASKME_DINGTALK_*_WEBHOOK` 和 secret，然后运行 `field-notification-smoke --server <deployment>`，让 `notification_smoke_uses_external_services=true`。

## 2026-05-11 Productization checkpoint: field smoke suite

本次把分散的现场验收命令收敛成一条一键 smoke suite：

- 新增 `python -m askme runtime field-smoke-suite`。
- suite 会顺序运行：
  1. `field-eval` 场景评测；
  2. `field-ingest-smoke` 传感器/相机/机器人 JSONL 到 `/api/field/ingest`；
  3. `field-voice-smoke` 异常事件到语音播报；
  4. `field-notification-smoke` 钉钉通知 HTTP 链路；
  5. `field-readiness` 汇总部署门禁。
- 输出 `field-smoke-suite.json`，包含每个子报告和总 `checks`：
  - `scenario_eval`
  - `field_ingest_smoke`
  - `field_voice_smoke`
  - `field_notification_smoke`
  - `readiness_unblocked`
- 支持 `--voice-scenario`、`--groups`、`--live-tts`，用于现场按需切换异常播报和通知组。

实测：

- `python -m askme runtime field-smoke-suite --output-dir artifacts\field_operations\smoke --json` -> `status=passed`
- suite 实测 `checks` 全部为 true。
- readiness 仍为 `ready_for_lab`，因为真实硬件、真实部署服务、真实钉钉和 live TTS 还没接入。
- `python -m pytest tests/test_cli.py::test_cli_runtime_field_smoke_suite_aggregates_reports tests/test_cli.py::test_cli_runtime_field_smoke_suite_command_forwards_args -q` -> 2 passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 247 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/health_server.py askme/cli.py askme/voice/voice_profiles.py askme/voice/tts.py tests/test_field_deployment_readiness.py tests/test_field_operations.py tests/test_health.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py` -> passed

产品反思：

- 现场交付不能要求客户记住五六条工程命令。一键 suite 是销售演示、实验室验收、部署前自检的共同入口。
- suite 不把 `ready_for_lab` 伪装成 production ready：它能证明本地闭环已通，同时把真实部署服务、真实 TTS、真实钉钉、真实硬件作为下一步门禁暴露出来。

## 2026-05-11 Productization checkpoint: customer-facing field evidence report

本次继续把现场交付从“工程命令能跑”推进到“客户能看懂验收结果”：

- `field-smoke-suite` 新增 `field-smoke-suite.html` 输出。
- HTML 报告面向演示、实验室验收和部署前自检，展示：
  - 总体 suite 状态；
  - readiness 状态；
  - 场景评测、传感器入口、语音播报、通知链路、部署门禁是否通过；
  - blockers、warnings、next actions；
  - 原始 JSON 证据路径。
- suite JSON 新增 `customer_summary`，把工程结果整理成产品验收摘要：
  - 本地链路是否通过；
  - 语音播报是否验证；
  - 是否已经 live TTS 出声；
  - 通知链路是否验证；
  - 是否连接真实外部服务；
  - 下一步要补哪些现场条件。
- `tests/test_cli.py` 新增 HTML 报告存在性和关键文案断言，避免后续把客户可读报告改坏。

实测：

- `python -m askme runtime field-smoke-suite --output-dir artifacts\field_operations\smoke --json` -> `status=passed`
- 生成：
  - `artifacts\field_operations\smoke\field-smoke-suite.json`
  - `artifacts\field_operations\smoke\field-smoke-suite.html`
- `python -m pytest tests/test_cli.py::test_cli_runtime_field_smoke_suite_aggregates_reports tests/test_cli.py::test_cli_runtime_field_smoke_suite_command_forwards_args -q` -> 2 passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 247 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/health_server.py askme/cli.py askme/voice/voice_profiles.py askme/voice/tts.py tests/test_field_deployment_readiness.py tests/test_field_operations.py tests/test_health.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py` -> passed

产品反思：

- 客户不应该被要求理解 `fake/sim/runtime`、单个 smoke 命令或 JSON 结构。现在最少有一个可打开的 HTML 验收入口，能直接说明“已通过什么、还差什么、下一步去哪补”。
- 仍未完成的是把 warnings 里的真实条件消掉：真实部署服务、真实 MiniMax/live TTS、真实钉钉 webhook、真实摄像头/烟感/机器人诊断流、真实硬件 dispatch。

## 2026-05-11 Productization checkpoint: real field payload adapters

本次继续把“真实传感器接入”从框架推进到可运行适配：

- `askme/pipeline/field_ingest_adapters.py` 扩展真实设备格式：
  - 相机/视觉：支持 `detections`、`predictions`、`objects`、`boxes`、`frame.boxes`、`result/results` 等常见输出。
  - YOLO/Ultralytics：支持 `cls/conf/xyxy`，把 class id 转成 `vehicle/person/...`。
  - Roboflow/通用检测：支持 `class/name/label/score/probability/confidence`。
  - 传感器：支持 `telemetry/values/data/properties`，并把 `temperature/temp/smoke/fill_percent/fullness` 等别名归一化。
  - 机器人诊断：支持 ROS diagnostics 风格 `status[]`、`level/message/values`，可把 motor overcurrent、joint stall 等映射成 `joint_motor_fault`。
- `FieldOperationsService` 修复人群聚集计数：以前多个 person 检测会被标签集合去重成 1，现在按检测条数统计。
- `field-ingest-smoke` 从 3 类样例升级为 5 类更真实格式：
  - YOLO boxes -> 车辆违停；
  - sensor telemetry -> 火灾/烟雾；
  - ROS diagnostics -> 关节电机故障；
  - trash bin telemetry -> 垃圾桶满；
  - person predictions -> 人群聚集。
- 一键 `field-smoke-suite` 现在归档 5 个现场事件，readiness 的 archive 也能看到 5 类场景。

实测：

- `python -m pytest tests/test_field_ingest_adapters.py tests/test_field_ingest_bridge.py tests/test_cli.py::test_cli_runtime_field_ingest_smoke_runs_local_http tests/test_cli.py::test_cli_runtime_field_smoke_suite_aggregates_reports -q` -> 13 passed
- `python -m askme runtime field-smoke-suite --output-dir artifacts\field_operations\smoke --json` -> `status=passed`，`field_ingest_smoke.event_count=5`
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 251 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_ingest_adapters.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/health_server.py askme/cli.py askme/voice/voice_profiles.py askme/voice/tts.py tests/test_field_deployment_readiness.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_health.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py` -> passed

产品反思：

- 这一步补的是“客户现场已有算法/设备怎么接进来”的实际问题。客户不一定按 askme 的理想 schema 输出数据，相机算法、烟感网关、机器人诊断程序各有格式；适配层越能吞真实格式，产品越不像 demo。
- 仍未完成的是直接连接真实设备进程：现在 bridge 已能读 JSON/JSONL 和常见格式，但还需要把摄像头检测程序、烟感/温度网关、机器人诊断服务配置成持续写入或推送到这条入口。

## 2026-05-11 Productization checkpoint: governed LLM field narration

本次把“有些情况需要调用大模型更形象回复”从简单开关推进成可控能力：

- `FieldEventRecord` 新增：
  - `llm_narrative_status`
  - `llm_narrative_reason`
- 大模型播报现在有清晰状态：
  - `used`：低风险文案被采纳；
  - `skipped`：高风险/P0/固定话术场景不允许模型改写；
  - `unavailable`：没有配置 LLM client；
  - `failed`：调用失败或超时；
  - `rejected`：模型输出不安全或不合规。
- 修复 playbook 允许 LLM 的逻辑：垃圾桶满、人群聚集、违停这类低风险事件，如果 playbook 标记 `allow_llm_narrative=true`，在总开关打开且有 LLM client 时可以润色。
- 高风险事件仍然走固定话术：火灾/烟雾、摔倒、卡住、关节电机故障等不允许 LLM 自由改写。
- 增加模型输出安全校验：
  - 拒绝 Markdown/链接；
  - 拒绝把普通事件夸大成“火灾、爆炸、死亡、报警、撤离、危险、警察、罚款、强制”等新事实；
  - 拒绝空输出。

实测：

- `python -m pytest tests/test_field_operations.py::test_low_risk_service_can_use_llm_narrative tests/test_field_operations.py::test_playbook_allowed_incident_can_use_llm_narrative tests/test_field_operations.py::test_unsafe_llm_narrative_is_rejected_and_fixed_voice_remains tests/test_field_operations.py::test_high_risk_event_skips_llm_narrative_even_when_enabled -q` -> 4 passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 254 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_ingest_adapters.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/health_server.py askme/cli.py askme/voice/voice_profiles.py askme/voice/tts.py tests/test_field_deployment_readiness.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_health.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py` -> passed

产品反思：

- 机器人现场播报不能把“更自然”放在“可控”前面。客户真正需要的是：紧急场景固定、低风险场景可润色、每次为什么用了/没用都能审计。
- 下一步需要把 `llm_narrative_status/reason` 展示到 Dashboard 的事件详情里，并把生产模型调用接到真实 MiniMax-M2.7-highspeed 或本地模型配置上。

## 2026-05-11 Productization checkpoint: LLM narration audit in Dashboard

本次把大模型播报治理结果从后端 JSON 推到产品界面：

- `dashboard.html` 的现场事件卡片会显示大模型播报状态：
  - `大模型播报已采用`
  - `固定话术`
  - `未接入大模型`
  - `大模型播报失败`
  - `大模型播报已拦截`
  - `未请求大模型`
- 状态会带上 `llm_narrative_reason`，让客户/运维能知道为什么某次播报用了模型、跳过模型或拦截模型。
- `tests/test_health.py` 增加 Dashboard 文案和 `narrativeLabel` 断言。

实测：

- `node --check .tmp/dashboard-script.js` -> passed
- `python -m pytest tests/test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests/test_field_operations.py::test_unsafe_llm_narrative_is_rejected_and_fixed_voice_remains -q` -> 2 passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 254 passed
- `python -m ruff check askme/pipeline/field_deployment_readiness.py askme/pipeline/field_ingest_adapters.py askme/pipeline/field_operations.py askme/pipeline/incident_alerts.py askme/health_server.py askme/cli.py askme/voice/voice_profiles.py askme/voice/tts.py tests/test_field_deployment_readiness.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_health.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py` -> passed

产品反思：

- “智能”不能只体现在模型有没有生成一句更顺的话，还要体现在用户能不能知道系统为什么这么说。现场事件卡片现在能暴露模型采用/拒绝/跳过的证据，降低客户对黑盒播报的不信任。
- 仍未完成的是把 Dashboard 现场事件列表从“工程记录”进一步整理成“事件处置工作台”：筛选、按响应组分组、处理 SLA、照片预览、通知重发、事件关闭审批还需要继续补。

## 2026-05-11 Productization checkpoint: field event handling workflow

本次把现场事件列表从“只展示告警”推进到“能被值班人员处理”：

- `FieldOperationsService` 新增 `acknowledge_payload()`：
  - 事件可从 `triggered/needs_evidence` 进入 `acknowledged`；
  - 记录 `acknowledged_at / acknowledged_by / acknowledge_note`；
  - 已关闭事件不能再确认，避免审计状态倒退。
- `/api/field/events/{event_id}/acknowledge` 新增 HTTP 入口。
- `/api/field/events/{event_id}/resend-notification` 新增通知重发入口：会重新走 `AlertDispatcher`，刷新 `delivery_report`，并写入 `notification_resends` 审计记录。
- 现场事件新增 `evidence_media`：从 `image_path/image_url/photo/snapshot/frame/video` 以及 detection 子项中提取照片/视频证据。
- 新增 `/api/field/evidence?path=...` 受限证据文件服务，只允许读取 `artifacts/`、`output/`、`data/` 下的本地证据文件，源码路径和路径穿越返回 404。
- `/api/field/events` 新增 `summary`：
  - `needs_attention`；
  - `open`；
  - `acknowledged`；
  - `closed`；
  - `by_status`；
  - `by_notification_group`。
- `/api/field/events` 新增筛选参数：`status`、`notification_group`、`needs_attention`，并返回 `filtered_total/filter`。
- 现场事件视图新增 `sla`：根据 playbook `escalation_after_s` 或优先级默认时限计算 `active/due_soon/overdue/closed`、`due_at`、`remaining_s`、`target_s`。
- 新增 `/api/field/events/{event_id}/report`：生成客户可读处置报告，包含事件、SLA、通知送达、证据、确认和关闭信息，并提供 Markdown。
- Dashboard 现场事件区新增客户可读汇总：待处理、未关闭、已确认、已关闭。
- Dashboard 现场事件区新增快捷筛选：全部事件、只看待处理、安保事件、保洁事件。
- Dashboard 现场事件汇总新增 `超时 / 即将超时`，单条事件 evidence line 会展示 SLA 状态和剩余秒数。
- Dashboard 单条事件新增 `查看报告`，可直接展示事件处置 Markdown 报告。
- Dashboard 现场事件卡片新增证据预览区：远程图片 URL 可直接显示缩略图，本地 `artifacts/output/data` 证据会通过受限 HTTP 路由显示缩略图。
- 每条未关闭事件支持：
  - `确认收到`；
  - `重发通知`；
  - `处理完成`。

验证：

- `python -m pytest tests\test_field_operations.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 24 passed
- Dashboard script `node --check .tmp/dashboard-script.js` -> passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/test_voice_turn_trace.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 271 passed
- `python -m ruff check askme\pipeline\field_deployment_readiness.py askme\pipeline\field_ingest_adapters.py askme\pipeline\field_operations.py askme\pipeline\incident_alerts.py askme\health_server.py askme\cli.py askme\voice\voice_profiles.py askme\voice\tts.py askme\voice\turn_trace.py tests\test_field_deployment_readiness.py tests\test_field_ingest_adapters.py tests\test_field_operations.py tests\test_health.py tests\test_cli.py tests\test_voice_profiles.py tests\test_tts.py tests\test_voice_turn_trace.py` -> passed
- `git diff --check -- ...` -> passed，仅有 CRLF 工作区提示

产品反思：

- 客户现场不是看见告警就结束，而是必须知道“谁已接单、谁处理完、还有几个待处理”。这一步把事件从工程日志推进到最小处置闭环。
- 通知失败或未读不能要求工程师重新触发事件；值班人员需要在同一事件上重发通知，并保留重发原因和送达证据。
- 现场事件如果没有照片/视频证据，客户很难相信系统判断。证据预览把“检测到了”推进到“我能看到证据”。这次没有开放任意文件读取，而是把证据限定在交付产物目录。
- 值班人员不应该在一长串事件里找安保/保洁/待处理事项；快捷筛选是现场可用性的底线。
- 现场事件还要能回答“哪些快超时、哪些已超时”。SLA 状态让工作台开始具备调度价值，而不是静态日志。
- 处理报告把“发生了什么、谁确认、是否通知、证据在哪、是否超时、谁关闭”收敛成一份可交付文本，是客户验收和事后复盘的基本材料。
- 仍未完成：更完整的按响应组看板、对象存储签名 URL、关闭审批和 PDF/HTML 报告导出。

## 2026-05-11 Productization checkpoint: voice conversation SLO gate

本次把“语音为什么慢、什么时候能对话”从主观体验推进成可审计门禁：

- `VoiceTurnTraceRecorder` 新增默认实时对话 SLO：
  - ASR 最终结果；
  - LLM 首 token；
  - TTS 首音频；
  - 播放开始；
  - 打断停播。
- `snapshot()` 会输出 `slo.status` 和 `ready_to_converse`：
  - `passed`：证据完整且在阈值内；
  - `insufficient_evidence`：缺少关键桶，语音体验仍处于待补证状态；
  - `failed`：某个关键桶超时；
  - `no_turn`：还没有语音轮次。
- Dashboard Voice Turn 卡片展示客户可读状态：`可对话门禁通过`、`证据不足`、`响应超时`，并显示缺失或超时的具体环节。

验证：

- `python -m pytest tests\test_voice_turn_trace.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 13 passed
- Dashboard script `node --check .tmp/dashboard-script.js` -> passed
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/test_voice_turn_trace.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 266 passed
- `python -m ruff check askme\pipeline\field_deployment_readiness.py askme\pipeline\field_ingest_adapters.py askme\pipeline\field_operations.py askme\pipeline\incident_alerts.py askme\health_server.py askme\cli.py askme\voice\voice_profiles.py askme\voice\tts.py askme\voice\turn_trace.py tests\test_field_deployment_readiness.py tests\test_field_ingest_adapters.py tests\test_field_operations.py tests\test_health.py tests\test_cli.py tests\test_voice_profiles.py tests\test_tts.py tests\test_voice_turn_trace.py` -> passed

产品反思：

- 用户说“断断续续、慢、不知道什么时候 OK 能说话”，本质不是 UI 小问题，而是系统没有把实时对话链路拆成可解释的产品状态。现在 SLO gate 能告诉客户：是 ASR 慢、模型慢、TTS 慢、播放慢，还是证据不足。
- 仍未完成的是把这套 SLO 接到真实麦克风/真实 MiniMax TTS 的现场录音评测报告里，并在主 Voice Console 用更少工程词展示“现在可以说 / 正在想 / 正在播 / 可以打断 / 需要重试”。
## 2026-05-11 Productization checkpoint: field event close governance and audit timeline

This pass tightened the customer-facing field event lifecycle from "operator can close an event" to "high-risk events require auditable closure governance".

- P0/error field events now return HTTP 409 when the dashboard or API tries to close them without `supervisor_approved` and `supervisor_id`.
- Events in `needs_review`, `needs_evidence`, or `duplicate` can no longer be closed even with supervisor fields; the operator must first supply evidence or resolve the review condition.
- `FieldEventRecord` now persists `notification_resends` and `close_approval` as first-class event fields instead of ad hoc runtime-only dict keys.
- Field event reports now include:
  - `created_at`, `acknowledged_at`, `closed_at`;
  - `ack_latency_s`, `resolution_latency_s`;
  - `sla_target_s`, `sla_due_at`, `sla_met`;
  - `timeline` covering created, notification sent/resent, acknowledged, close approved, closed;
  - `notification_attempts` covering initial delivery plus each resend.
- Markdown reports now include "处置时间线" and "通知尝试", so customer review is not limited to raw status fields.

Validation:

- `python -m pytest tests\test_field_operations.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 26 passed
- `python -m ruff check askme\pipeline\field_operations.py askme\health_server.py tests\test_field_operations.py tests\test_health.py` -> passed

Product reflection:

- Security/fire/robot-fault events should not be closable by the same one-click path as low-risk events. The important product question is not only "did someone click handled", but whether evidence was complete, who approved closure, when it happened, and whether the SLA was met.
- This is still not a full enterprise approval workflow. The next production step is an explicit `pending_close_approval` state and real operator role/permission checks instead of trusting a dashboard-supplied supervisor id.
## 2026-05-11 Productization checkpoint: two-step high-risk close approval

This pass moved high-risk field event closure closer to a real operations workflow.

- Added `request_close_payload()` and `/api/field/events/{event_id}/request-close`.
- High-risk P0/error events can now enter `pending_close_approval` before a supervisor closes them.
- Dashboard now separates `申请关闭` from `主管审批关闭` instead of treating a supervisor id prompt as the only workflow.
- `acknowledge_payload()` no longer hides `needs_review`, `needs_evidence`, or `pending_close_approval` events by converting them to plain `acknowledged`; it records acknowledgement while preserving the safety-relevant status.
- Event reports include close request actor/time/note and a `close_requested` timeline item.

Validation:

- `python -m pytest tests\test_field_operations.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 27 passed
- `python -m ruff check askme\pipeline\field_operations.py askme\health_server.py tests\test_field_operations.py tests\test_health.py` -> passed
- `node --check .tmp\dashboard-script.js` -> passed

Product reflection:

- This prevents a common field-ops failure mode: an operator acknowledges a weak/unsafe event and it disappears from the active queue, or closes a P0 event without a visible approval state. The workflow still needs real RBAC, but the product state model is now closer to how security teams actually operate.

## 2026-05-11 Productization checkpoint: production readiness gates for field closure evidence

本次把“能不能给客户说已经可上线”从静态配置检查推进到现场处置证据检查。

- `build_field_deployment_readiness()` 新增两道产品门禁：
  - `close_approval_workflow_verified`：归档里必须出现高风险事件的申请关闭和主管审批记录。
  - `event_report_timeline_verified`：归档里必须出现已关闭事件、关闭人、通知记录和完整处置时间线证据。
- `/api/field/readiness` 和 Dashboard readiness 卡片会展示这两道门禁；没有这些证据时只能进入 `ready_for_lab`，不能误报为 `production_ready`。
- 缺少关闭审批链路时，next action 会明确提示：创建 P0 事件、申请关闭、主管审批关闭。
- 缺少处置报告时间线时，next action 会明确提示：生成一条已关闭事件报告并验证 timeline。

验证：

- `python -m pytest tests\test_field_deployment_readiness.py -q` -> 3 passed
- `python -m pytest tests\test_field_deployment_readiness.py tests\test_field_operations.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 30 passed
- `python -m ruff check askme\pipeline\field_deployment_readiness.py askme\pipeline\field_operations.py askme\health_server.py tests\test_field_deployment_readiness.py tests\test_field_operations.py tests\test_health.py` -> passed
- Dashboard inline script parse check -> checked 1 inline script

产品反思：

- 客户真正关心的不是“系统有没有按钮”，而是高风险现场事件是否留下了可追责证据：谁申请关闭、谁审批、通知有没有发、什么时候处理完、SLA 是否满足。
- 这一步仍然不是完整 RBAC。下一步应继续补 operator/supervisor/admin 角色、登录态绑定、审批队列和不可抵赖审计，而不是继续依赖前端传入的 operator id。

## 2026-05-11 Productization checkpoint: backend RBAC for field-event handling

本次把现场事件处置从“前端传一个 operator_id 就相信”推进到“后端根据操作员角色判定权限”。

- `FieldOperationsService` 新增最小操作员目录：
  - `operator` 可以确认、重发通知、申请关闭、执行低风险关闭。
  - `supervisor/admin` 可以审批高风险事件关闭。
  - 未登记人员不能确认、重发、申请关闭或关闭事件。
- 高风险事件关闭现在同时要求：
  - 执行动作的人是 `operator/supervisor/admin`。
  - `supervisor_id` 属于 `supervisor/admin`。
  - 仍然保留两步流程：先申请关闭，再主管审批关闭。
- HTTP 层把 `operator_not_authorized` 和 `supervisor_not_authorized` 映射为 403，避免客户界面把权限问题误显示成“找不到事件”或普通冲突。
- `config.yaml` 新增 demo operator directory，便于演示和后续接企业身份系统。

验证：

- `python -m pytest tests\test_field_operations.py -q` -> 27 passed
- `python -m pytest tests\test_field_deployment_readiness.py tests\test_field_operations.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 31 passed
- `python -m ruff check askme\pipeline\field_deployment_readiness.py askme\pipeline\field_operations.py askme\health_server.py tests\test_field_deployment_readiness.py tests\test_field_operations.py tests\test_health.py` -> passed
- Dashboard inline script parse check -> checked 1 inline script
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/test_voice_turn_trace.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 275 passed, 2 warnings

产品反思：

- 真正的客户现场不能让“主管审批”成为前端按钮上的自填字段。后端至少要知道谁是值班员、谁是主管、谁是管理员，并把权限拒绝写成审计事件。
- 这仍然只是产品级最小闭环，不是企业级身份系统。下一步应接入登录态、角色来源、班次、审批队列、审批意见不可篡改签名，以及“谁看过/谁导出过报告”的审计。

## 2026-05-11 Productization checkpoint: field-event action audit

本次把现场事件处置从“状态变了”推进到“每个关键操作都有事件内审计证据”。

- `FieldEventRecord` 新增 `action_audit`。
- 以下动作会写入审计：
  - `acknowledge`
  - `resend_notification`
  - `request_close`
  - `close`
- 审计记录包含：
  - `at`
  - `action`
  - `outcome`
  - `operator_id`
  - `operator_roles`
  - `required_roles`
  - `reason`
  - `note`
  - supervisor 相关角色信息
- 未授权操作、状态不允许、证据不足、重复关闭、缺少主管审批等拒绝不再只是返回错误；如果事件存在，会把被拒绝的尝试写回该事件，后续报告能看到谁试图操作、被什么规则拒绝。
- 处置报告新增 `action_audit`，Markdown 报告新增“操作审计”段落。
- Dashboard 现场事件卡片会显示 `audit N` 和 `blocked operation N`，客户不用打开原始 JSON 也能看到有权限拒绝或操作记录。

验证：

- `python -m pytest tests\test_field_operations.py -q` -> 27 passed
- `python -m pytest tests\test_field_operations.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 28 passed
- `python -m ruff check askme\pipeline\field_operations.py askme\health_server.py tests\test_field_operations.py tests\test_health.py` -> passed
- Dashboard inline script parse check -> checked 1 inline script
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/test_voice_turn_trace.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 275 passed, 2 warnings

产品反思：

- 审计不是工程日志。客户要的是某一条火灾、违停、机器人故障事件的完整处置证据：谁确认、谁重发、谁申请关闭、谁审批、谁被系统拒绝。
- 当前审计已经落在事件内，但仍缺企业级不可篡改能力。下一步需要把这些记录复制到 append-only audit store，并绑定登录态、班次、设备编号和导出记录。

## 2026-05-11 Productization checkpoint: append-only field action audit store

本次把现场事件审计从“跟随事件 JSON 的可变字段”推进到“独立追加写入的 JSONL 审计流”。

- `field_operations.action_audit` 新增配置：
  - `enabled`
  - `path`
  - `swallow_errors`
- 默认审计文件：`artifacts/field_ops/field-action-audit.jsonl`。
- 每次 `action_audit` 追加到事件内时，同步写入一条独立 JSONL：
  - `kind=field_event_action`
  - `robot_id`
  - `event_id`
  - `scenario_id`
  - `status`
  - `priority`
  - `severity`
  - `location`
  - `audit`
- 覆盖成功和拒绝两类动作：确认、重发通知、申请关闭、关闭，以及权限拒绝、状态拒绝、缺主管审批等拒绝原因。

验证：

- `python -m pytest tests\test_field_operations.py -q` -> 28 passed
- `python -m pytest tests\test_field_operations.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 29 passed
- `python -m ruff check askme\pipeline\field_operations.py askme\health_server.py tests\test_field_operations.py tests\test_health.py` -> passed
- Dashboard inline script parse check -> checked 1 inline script
- `python -m pytest tests/test_field_deployment_readiness.py tests/test_field_ingest_bridge.py tests/test_field_ingest_adapters.py tests/test_field_operations.py tests/test_alert_dispatcher.py tests/test_health.py tests/test_runtime_modules.py tests/test_cli.py tests/test_voice_profiles.py tests/test_tts.py tests/test_tts_minimax.py tests/test_voice_turn_trace.py tests/scenario_tests/test_field_operations_evaluation.py -q` -> 276 passed, 2 warnings

产品反思：

- 这一步解决“事件记录被改写后，操作证据是否还在”的问题。客户验收、安保复盘和事故追责需要 append-only 证据流，而不是只看当前事件状态。
- 仍未完成不可抵赖审计。下一步需要给 JSONL 增加 hash chain / signature、导出校验器、登录态绑定和 report export audit，才能接近企业级合规要求。

## 2026-05-11 Productization checkpoint: field action audit hash chain

Target result: field event handling now has a tamper-evident append-only action audit chain, not just best-effort per-event JSON fields.

- `FieldOperationsService` writes `hash_alg`, `prev_hash`, and `record_hash` into every `field_event_action` JSONL record.
- `record_hash` is computed from canonical JSON excluding `record_hash`, so edits to operator, action, event metadata, status, or notes are detectable.
- `prev_hash` links each new audit record to the previous line, starting from `GENESIS`.
- `/api/field/audit/integrity` verifies the JSONL file and returns `valid`, `checked_count`, `latest_hash`, and concrete failure reasons such as `record_hash_mismatch` or `prev_hash_mismatch`.
- HTTP returns `409` when audit integrity is enabled but the chain is missing or invalid, so deployment checks can fail closed instead of silently trusting a damaged audit file.

Validation evidence:

- `python -m pytest tests\test_field_operations.py -q` -> 29 passed
- `python -m pytest tests\test_field_operations.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 30 passed
- `python -m ruff check askme\pipeline\field_operations.py askme\health_server.py tests\test_field_operations.py` -> passed
- `python -m py_compile askme\pipeline\field_operations.py askme\health_server.py` -> passed

Remaining risk:

- This is a hash chain, not a cryptographic signature or remote immutable ledger. A privileged local attacker who can rewrite the whole audit file can still recompute hashes. Next production hardening step is signing or external append-only storage.

## 2026-05-11 Productization checkpoint: stricter field audit gate

Review hardening applied after verifier feedback:

- Added monotonic `sequence` to each field action audit JSONL record.
- Integrity verification now cross-checks audit JSONL count against `events.jsonl` embedded `action_audit` counts per event.
- Truncated audit chains now fail with `audit_count_mismatch` / `event_audit_count_mismatch`.
- Missing operator identity no longer defaults to `askme.operator`; action endpoints return unauthorized actor `anonymous` with `authorization_reason=operator_identity_required`.
- Field action audit defaults to fail-closed: `swallow_errors` default is false and `config.yaml` sets `field_operations.action_audit.swallow_errors: false`.
- Tests now cover hash fields, sequence, record tampering, chain truncation, missing operator identity, HTTP 409 on broken integrity, and audit-write failure blocking state mutation.

Validation evidence:

- `python -m pytest tests\test_field_operations.py -q` -> 32 passed
- `python -m pytest tests\test_field_operations.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 33 passed
- `python -m ruff check askme\pipeline\field_operations.py askme\health_server.py tests\test_field_operations.py` -> passed
- `python -m py_compile askme\pipeline\field_operations.py askme\health_server.py` -> passed

Remaining risk:

- The chain is tamper-evident against local edits/truncation when the event archive remains authoritative. It is still not an external immutable ledger or signed audit trail; production should add HMAC/signature and remote/WORM checkpoint anchoring.

## 2026-05-11 Productization checkpoint: signed field action audit

Review hardening continued: the local audit chain can now be signed when deployment provides a secret.

- `field_operations.action_audit.hmac_secret` reads `${ASKME_FIELD_ACTION_AUDIT_HMAC_SECRET}` or direct config.
- When a secret is present, every JSONL action audit record includes `signature_alg=hmac-sha256`, `signature_key_id`, and `record_signature`.
- The signature covers canonical record content including `record_hash`, `sequence`, `prev_hash`, event metadata, and actor/action details.
- Integrity verification now returns `signed=true` when a verifier secret is configured and fails on `record_signature_mismatch`.
- This specifically closes the earlier “attacker edits a line and recomputes plain hash” gap unless the attacker also has the HMAC secret.

Validation evidence:

- `python -m pytest tests\test_field_operations.py -q` -> 33 passed
- `python -m pytest tests\test_field_operations.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 34 passed
- `python -m ruff check askme\pipeline\field_operations.py tests\test_field_operations.py` -> passed
- `python -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('config.yaml').read_text(encoding='utf-8')); print('yaml ok')"` -> yaml ok
- `python -m py_compile askme\pipeline\field_operations.py askme\health_server.py` -> passed

Remaining risk:

- HMAC protects against offline recompute only if the secret is managed outside the writable app directory. Production should still anchor latest signed checkpoint to WORM/object storage or enterprise SIEM.

## 2026-05-11 Productization checkpoint: audit readiness gate

Signed audit is now part of field deployment readiness rather than a hidden backend-only capability.

- `FieldOperationsService.readiness_payload()` now passes `/api/field/audit/integrity` evidence into `build_field_deployment_readiness()`.
- Readiness gates include `action_audit_integrity_verified` and `action_audit_signed`.
- If an event contains operator actions such as acknowledge, close request, close, or notification resend, readiness blocks when the action audit chain is missing or invalid.
- If the action audit chain exists but is not HMAC-signed, readiness warns and avoids `production_ready`.
- Production-ready readiness tests now create a real P0 event, request close approval, supervisor-close it, and verify the signed audit gate.

Validation evidence:

- `python -m pytest tests\test_field_deployment_readiness.py -q` -> 4 passed
- `python -m pytest tests\test_field_deployment_readiness.py tests\test_field_operations.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 38 passed
- `python -m ruff check askme\pipeline\field_deployment_readiness.py askme\pipeline\field_operations.py tests\test_field_deployment_readiness.py tests\test_field_operations.py` -> passed

Remaining risk:

- Dashboard readiness rendering still uses fallback labels for new audit gates until the UI label table is refreshed cleanly. Backend readiness contract and tests already expose the fields.

## 2026-05-11 Productization checkpoint: operator-facing audit CLI

The field audit chain is no longer UI-only or backend-only. Operators and delivery engineers can now verify it from the command line before lab/prod handoff.

- Added `askme runtime field-audit-integrity`.
- Local mode verifies an event archive plus `field-action-audit.jsonl`, including sequence, hash chain, expected action count, and optional HMAC signature.
- Server mode reads the running runtime endpoint `/api/field/audit/integrity`.
- Human output prints status, path, checked/expected counts, latest hash, signature state, and concrete failure reasons.
- JSON output is available for CI or deployment scripts.
- Invalid enabled audit exits with code `2`, so deployment scripts can fail closed.

Validation evidence:

- `python -m pytest tests\test_cli.py::test_cli_runtime_field_audit_integrity_reads_server tests\test_cli.py::test_cli_runtime_field_audit_integrity_verifies_local_signed_chain tests\test_cli.py::test_cli_runtime_field_audit_integrity_exits_nonzero_when_invalid -q` -> 3 passed
- `python -m ruff check askme\cli.py tests\test_cli.py` -> passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 74 passed

Remaining risk:

- The CLI proves local/runtime audit integrity, but production still needs secret management and external/WORM checkpoint anchoring. This is the next hardening line before claiming production-grade non-repudiation.

## 2026-05-11 Productization checkpoint: dashboard audit gate clarity

The deployment readiness UI now exposes the new audit gates as operator-readable product language instead of leaking backend field names.

- Dashboard readiness rendering now orders `action_audit_integrity_verified` and `action_audit_signed` next to voice/notification gates.
- The UI label layer renders these as `处置审计链完整` and `处置审计链已签名`.
- The health-server dashboard contract test asserts that the label helper and both audit gate keys are shipped in the served HTML.

Validation evidence:

- `python -m pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed
- `python -m ruff check askme\cli.py tests\test_cli.py tests\test_health.py` -> passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 74 passed

Remaining risk:

- This is a contract-level UI check, not a browser screenshot regression. Before a customer demo, run a local server and capture the dashboard to verify visual spacing around the readiness timeline.

## 2026-05-11 Productization checkpoint: audit checkpoint anchoring

The field action audit chain can now produce a deployment checkpoint artifact and optionally deliver it to an external SIEM/WORM-style webhook.

- Added `askme runtime field-audit-anchor`.
- The command reuses `field-audit-integrity` and writes a compact checkpoint with latest hash, count, signature state, source path, and full integrity payload.
- Optional `--webhook-url` posts the checkpoint to an external endpoint for immutable or enterprise audit storage.
- Default behavior is fail-closed: if integrity is invalid, the command exits with code `2` unless `--allow-invalid` is explicitly set.
- `--output` writes `artifacts/field_ops/audit-checkpoint.json` by default, so delivery teams can attach it to handoff evidence.

Validation evidence:

- `python -m pytest tests\test_cli.py::test_cli_runtime_field_audit_anchor_writes_checkpoint tests\test_cli.py::test_cli_runtime_field_audit_anchor_posts_webhook tests\test_cli.py::test_cli_runtime_field_audit_anchor_exits_nonzero_when_invalid -q` -> 3 passed
- `python -m ruff check askme\cli.py tests\test_cli.py tests\test_health.py` -> passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 77 passed
- `python -m askme.cli runtime field-audit-anchor --help` -> command and options listed

Remaining risk:

- The webhook is generic and tested through a mocked sender, not a real SIEM/WORM provider. A production deployment still needs provider-specific authentication, retry/backoff, and immutable-retention verification.

## 2026-05-11 Productization checkpoint: smoke suite includes audit checkpoint

The one-command field smoke suite now packages audit checkpoint evidence with the rest of the customer handoff artifacts.

- `askme runtime field-smoke-suite` now creates `audit-checkpoint.json` by default.
- Added `--audit-hmac-secret` so the suite can verify signed audit chains during delivery.
- Added `--audit-webhook-url` so the suite can deliver the checkpoint to an external SIEM/WORM endpoint.
- Added `--skip-audit-anchor` for offline demos or early development runs.
- Suite status now includes `audit_checkpoint_created` in its checks and embeds `audit_anchor` in `field-smoke-suite.json`.

Validation evidence:

- `python -m pytest tests\test_cli.py::test_cli_runtime_field_smoke_suite_aggregates_reports tests\test_cli.py::test_cli_runtime_field_smoke_suite_command_forwards_args -q` -> 2 passed
- `python -m ruff check askme\cli.py tests\test_cli.py tests\test_health.py` -> passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 77 passed
- `python -m askme.cli runtime field-smoke-suite --help` -> audit options listed

Remaining risk:

- The suite currently creates a checkpoint even if the local smoke run has no operator action audit entries, because early smoke runs are mostly ingest/voice/notification. The stricter production readiness gate still blocks real operator-action deployments when the audit chain is missing or invalid.

## 2026-05-11 Productization checkpoint: smoke suite creates real operator audit evidence

The ingest smoke path now performs a real operator acknowledgement after ingesting field events, so audit checkpoint evidence is based on an actual operator action.

- `_run_field_ingest_smoke()` now acknowledges the first generated field event through the HTTP API with operator `security-1`.
- The smoke report includes `operator_action` with the acknowledgement response.
- The smoke pass condition requires the acknowledgement to succeed.
- The CLI test now asserts the acknowledged event contains an `action_audit` entry with `action=acknowledge`.

Validation evidence:

- `python -m pytest tests\test_cli.py::test_cli_runtime_field_ingest_smoke_runs_local_http tests\test_cli.py::test_cli_runtime_field_smoke_suite_aggregates_reports -q` -> 2 passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 77 passed
- `python -m ruff check askme\cli.py tests\test_cli.py tests\test_health.py` -> passed

Remaining risk:

- The smoke acknowledgement is still against the local temporary runtime unless a real `--server` is passed. It proves the product control loop and audit write path, not physical现场人员处置.

## 2026-05-11 Productization checkpoint: smoke suite fails closed on audit checkpoint

The smoke suite now treats audit checkpoint validity as a real gate.

- `field-smoke-suite` calls `_run_field_audit_anchor(..., require_valid=True)`.
- If the ingest smoke did not produce a valid operator action audit chain, the audit checkpoint is blocked and `audit_checkpoint_created` fails.
- Added a regression test that runs real ingest smoke, then creates a strict audit anchor and verifies `checked_count == expected_count == 1`.

Validation evidence:

- `python -m pytest tests\test_cli.py::test_cli_runtime_field_ingest_smoke_produces_strict_audit_anchor tests\test_cli.py::test_cli_runtime_field_smoke_suite_aggregates_reports -q` -> 2 passed
- `python -m ruff check askme\cli.py tests\test_cli.py tests\test_health.py` -> passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 78 passed

Remaining risk:

- The audit checkpoint is now strict for local smoke evidence, but the webhook delivery path still uses a generic test double. Real SIEM/WORM delivery must be validated with provider credentials in a staging environment.

## 2026-05-11 Productization checkpoint: audit webhook delivery is inspectable

Audit checkpoint webhook delivery now records delivery attempts and fails with an explicit product state instead of crashing as an unstructured HTTP exception.

- `field-audit-anchor` now accepts `--webhook-retries`.
- `field-smoke-suite` now accepts `--audit-webhook-retries`.
- Webhook delivery returns `webhook_delivery.status=sent|failed`, `attempts`, and either `response` or `error`.
- If checkpoint integrity is valid but external webhook delivery fails, the command returns `status=delivery_failed` and exits with code `3`.
- The output checkpoint file is rewritten after webhook delivery so the persisted artifact includes delivery evidence.

Validation evidence:

- `python -m pytest tests\test_cli.py::test_cli_runtime_field_audit_anchor_posts_webhook tests\test_cli.py::test_cli_runtime_field_audit_anchor_reports_webhook_failure tests\test_cli.py::test_cli_runtime_field_smoke_suite_command_forwards_args -q` -> 3 passed
- `python -m ruff check askme\cli.py tests\test_cli.py tests\test_health.py` -> passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 79 passed

Remaining risk:

- Retries are bounded and inspectable, but there is no persistent retry queue yet. If a real external SIEM/WORM provider is down during handoff, an operator must rerun the command after fixing delivery.

## 2026-05-11 Productization checkpoint: audit delivery retry queue

Failed audit checkpoint webhook deliveries now persist into a JSONL retry queue and can be replayed explicitly.

- `field-audit-anchor` now accepts `--retry-queue`.
- Failed webhook deliveries are appended to `artifacts/field_ops/audit-delivery-retry.jsonl` by default.
- Added `askme runtime field-audit-retry-delivery`.
- Retry delivery removes the queue after all items are sent, or rewrites it with only remaining failed/invalid items.
- Retry command exits with code `3` while queued items remain.

Validation evidence:

- `python -m pytest tests\test_cli.py::test_cli_runtime_field_audit_anchor_reports_webhook_failure tests\test_cli.py::test_cli_runtime_field_audit_retry_delivery_replays_queue tests\test_cli.py::test_cli_runtime_field_audit_retry_delivery_keeps_failed_items -q` -> 3 passed
- `python -m ruff check askme\cli.py tests\test_cli.py tests\test_health.py` -> passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 81 passed
- `python -m askme.cli runtime field-audit-retry-delivery --help` -> command and options listed

Remaining risk:

- The retry queue is a local JSONL file. Production still needs retention policy, file permissions, and/or migration to a durable job queue when multiple operators or machines can process delivery.

## 2026-05-11 Productization checkpoint: audit retry queue status gate

Audit delivery backlog is now inspectable without sending webhooks, so staging/prod deployment scripts can block release when external audit delivery is still pending.

- Added `askme runtime field-audit-retry-status`.
- The command reports `pending`, `invalid`, queue path, and per-line checkpoint evidence such as `latest_hash`.
- `--fail-on-pending` exits with code `3` when queued deliveries remain.
- Invalid JSONL entries are counted separately so operators can distinguish delivery backlog from corrupted queue data.

Validation evidence:

- `python -m pytest tests\test_cli.py::test_cli_runtime_field_audit_retry_status_reports_empty_missing_queue tests\test_cli.py::test_cli_runtime_field_audit_retry_status_reports_pending_and_invalid_queue tests\test_cli.py::test_cli_runtime_field_audit_retry_status_fail_on_pending_exits_nonzero -q` -> 3 passed
- `python -m askme.cli runtime field-audit-retry-status --help` -> command and options listed
- `python -m ruff check askme\cli.py tests\test_cli.py tests\test_health.py` -> passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 84 passed

Remaining risk:

- This is still local-file queue visibility. A multi-machine deployment should move the queue to a shared durable job store, but the current CLI now gives operators a concrete, testable preflight gate instead of guessing whether audit delivery has caught up.

## 2026-05-11 Productization checkpoint: audit delivery backlog blocks readiness

The field deployment readiness gate now treats undelivered audit checkpoint webhooks as a release blocker.

- `build_field_deployment_readiness()` inspects the audit delivery retry queue.
- `gates.audit_delivery_retry_queue_empty` is false when the queue has pending or invalid JSONL entries.
- Readiness returns a blocker and next actions pointing to `field-audit-retry-delivery` and `field-audit-retry-status`.
- `config.yaml` now exposes `field_operations.action_audit.retry_queue_path`.
- Dashboard readiness rendering includes the retry queue gate so operators can see why a deployment is blocked.

Validation evidence:

- `python -m pytest tests\test_field_deployment_readiness.py::test_field_deployment_readiness_can_be_production_ready tests\test_field_deployment_readiness.py::test_field_deployment_readiness_blocks_pending_audit_delivery_retry_queue tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 3 passed
- `python -m ruff check askme\pipeline\field_deployment_readiness.py tests\test_field_deployment_readiness.py tests\test_health.py` -> passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 85 passed

Remaining risk:

- The readiness gate blocks local backlog, but it still relies on a local queue file. For multi-robot or multi-operator deployments this should become a shared durable delivery store with ownership/lease semantics.

## 2026-05-11 Productization checkpoint: audit delivery retry lock

Audit delivery replay now has a local concurrency lock, reducing duplicate SIEM/WORM webhook delivery risk when deployment scripts or operators run retry jobs at the same time.

- `field-audit-retry-delivery` now creates a queue-adjacent `.lock` file before replaying delivery.
- Active locks return `status=locked` and the CLI exits with code `4` without sending webhooks.
- Stale locks are detected through `expires_at` and can be taken over automatically.
- Successful or failed retry runs release the lock in a `finally` block.
- `--lock-timeout` is exposed for deployment tuning.

Validation evidence:

- `python -m pytest tests\test_cli.py::test_cli_runtime_field_audit_retry_delivery_replays_queue tests\test_cli.py::test_cli_runtime_field_audit_retry_delivery_exits_when_locked tests\test_cli.py::test_cli_runtime_field_audit_retry_delivery_takes_stale_lock tests\test_cli.py::test_cli_runtime_field_audit_retry_delivery_keeps_failed_items -q` -> 4 passed
- `python -m ruff check askme\cli.py tests\test_cli.py tests\test_health.py askme\pipeline\field_deployment_readiness.py tests\test_field_deployment_readiness.py` -> passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 87 passed
- `python -m askme.cli runtime field-audit-retry-delivery --help` -> `--lock-timeout` listed

Remaining risk:

- This is still a single-filesystem lock. It is good enough for one deployed host, but multi-host robot fleets still need a shared queue with lease ownership.

## 2026-05-11 Productization checkpoint: RAG-blocked chat refuses at API boundary

The chat API now enforces RAG blocking policy for plain text handlers, so expired/conflicting/unapproved knowledge cannot silently coexist with a fabricated reply.

- `/api/chat` still preserves handler-provided evidence when the handler returns a structured evidence payload.
- When the handler returns plain text and memory health reports no usable evidence plus a blocking `last_answer_policy`, the API replaces the reply with a deterministic refusal.
- The response includes `rag_blocked=true`, `rag.answer_blocked=true`, `rag.forced_reply=true`, and `rag.block_reason`.
- Dropped evidence remains visible in `rag.dropped_evidence`, so the chat bubble can show exactly what was blocked and why.

Validation evidence:

- `python -m pytest tests\test_health.py::TestHealthServer::test_chat_endpoint_forces_refusal_when_rag_policy_blocks_plain_text_reply tests\test_health.py::TestHealthServer::test_chat_endpoint_attaches_memory_evidence_for_plain_text_handler tests\test_health.py::TestHealthServer::test_chat_endpoint_does_not_overwrite_handler_evidence_with_memory_context -q` -> 3 passed
- `python -m ruff check askme\health_server.py tests\test_health.py` -> passed
- `python -m pytest tests\test_memory_bridge.py tests\test_prompt_builder.py tests\test_health.py::TestHealthServer -q` -> 118 passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 87 passed

Remaining risk:

- The refusal templates are local to the API boundary. They should later be centralized with `PromptBuilder` RAG policy templates so voice, text, and HTTP surfaces share one copy deck.

## 2026-05-11 Productization checkpoint: voice style and customer-facing operations UI

This checkpoint moves two visible product surfaces beyond framework status:

- Voice profiles now expose the actually applied MiniMax/Speech settings instead of only listing candidate profiles.
- Switching a voice profile returns `applied_settings` and `persistence_status=session_only`, so operators can see the current limitation instead of assuming permanent configuration.
- When `voice_profile_state_path` is configured, the selected voice profile is now persisted to disk and restored after `TTSEngine` restart.
- `config.yaml` enables this product behavior with `voice.tts.voice_profile_state_path: data/voice/active_voice_profile.json`.
- Built-in profiles now have product-facing Chinese names, use cases, and sample lines for patrol, visitor service, security, emergency, and night mode.
- Dashboard first-viewport title, knowledge console, diagnostics title, voice profile panel, and field event copy were cleaned for customer demos.
- Field event cards now emphasize scenario, human-readable status, robot broadcast line, notification target, delivery evidence, evidence freshness, SLA, audit, and close actions instead of exposing raw engineering labels first.
- Dashboard regression tests now lock the visible product terms: knowledge management, RAG lifecycle copy, robot broadcast, product status labels, and field event empty state.

Validation evidence:

- `python -m pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_voice_profiles.py tests\test_rag_policy.py -q` -> 8 passed
- `python -m pytest tests\test_voice_profiles.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 6 passed
- `python -m pytest tests\test_voice_profiles.py tests\test_tts_minimax.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 38 passed
- `python -m pytest tests\test_memory_bridge.py tests\test_prompt_builder.py tests\test_health.py::TestHealthServer tests\test_rag_policy.py -q` -> 121 passed
- `python -m pytest tests\test_cli.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_voice_profiles.py -q` -> 91 passed
- `python -m ruff check tests\test_health.py askme\voice\tts.py askme\voice\voice_profiles.py askme\pipeline\rag_policy.py askme\pipeline\prompt_builder.py askme\health_server.py` -> passed
- `python -m ruff check askme\voice\tts.py tests\test_voice_profiles.py tests\test_health.py` -> passed
- `python -m ruff check askme\voice\tts.py askme\voice\voice_profiles.py askme\pipeline\rag_policy.py askme\pipeline\prompt_builder.py askme\health_server.py tests\test_voice_profiles.py tests\test_rag_policy.py tests\test_prompt_builder.py tests\test_health.py` -> passed
- `node --check output\askme-dashboard-script-check.js` -> passed after extracting the dashboard script
- Dashboard text scan: no `????` or Unicode replacement characters in `askme/static/dashboard.html`

Remaining risk:

- This is still not a full browser screenshot review. The next UI pass should run the service and capture desktop/mobile screenshots.
- Voice profile persistence is durable for a single deployed host when `voice_profile_state_path` is configured. Fleet-wide settings still need a shared operator settings store.
- Field operations are real API flows with archive/audit/notification hooks, but external DingTalk and hardware sensor evidence still must be proven in a deployed environment.

## 2026-05-11 Productization checkpoint: browser visual smoke evidence

Dashboard QA is now a repeatable browser-level check instead of only static DOM assertions.

- Added `scripts/eval/check_dashboard_visual.py`.
- The script starts a local FastAPI app with visual-smoke cognition/runtime handlers, a real `FieldOperationsService`, and a no-network TTS handler.
- It injects a real field event through `/api/field/events`, including an approved local evidence image path.
- It captures desktop and mobile screenshots under `output/playwright`.
- It fails on missing product text, `????`/replacement characters, page errors, actionable HTTP 4xx/5xx responses, and horizontal overflow.

Validation evidence:

- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed
- `python -m ruff check scripts\eval\check_dashboard_visual.py` -> passed
- Evidence JSON: `output/playwright/dashboard-visual-smoke.json`
- Desktop screenshot: `output/playwright/askme-dashboard-desktop.png`
- Mobile screenshot: `output/playwright/askme-dashboard-mobile.png`

Remaining risk:

- This is a deterministic smoke scene, not a full design review. It proves the page loads cleanly with a seeded field event on desktop/mobile; it does not replace manual product critique or real operator walkthroughs.

## 2026-05-11 Productization checkpoint: real field-device ingest adapters

The field runtime now handles more realistic external device payloads instead of only hand-authored demo JSON.

- `normalize_field_ingest_payload()` now flattens common webhook/MQTT envelopes such as `payload`, `event`, `message`, and `params` while preserving `raw_device_payload` for audit.
- Camera ingest now recognizes ANPR/vehicle/parking/smoke/person style vendor fields, including `eventType`, `cameraIndexCode`, `dateTime`, `pictureUrl`, nested `ANPR.plateNo`, and common image aliases.
- Sensor ingest now accepts reported telemetry envelopes and smoke/fire boolean alarms such as `smokeAlarm` / `fireAlarm`, converting them into the stable smoke/temperature contract.
- Robot ingest now maps operational state packages such as `nav_state=stuck`, `motion_state`, `is_fallen`, `blocked_by_human`, joint/motor/stall text into the existing abnormal incident topics.
- `FieldOperationsService.trigger_payload()` now preserves normalized ingest facts when a raw device body also contains a nested `payload` field. This fixed a real bug where MQTT envelopes could infer a scenario but fail evidence checks because normalized fields were dropped.
- `/api/field/ingest` help now includes product-facing examples for Hikvision-style ANPR, MQTT smoke alarm, and robot status events.
- Incident templates now have a regression test that fails on replacement characters and common mojibake markers, protecting customer-facing voice/DingTalk/operator copy.

Validation evidence:

- `python -m ruff check askme\pipeline\field_ingest_adapters.py askme\pipeline\field_operations.py tests\test_field_ingest_adapters.py tests\test_alert_dispatcher.py` -> passed
- `python -m pytest tests\test_field_ingest_adapters.py tests\test_alert_dispatcher.py -q` -> 53 passed
- `python -m pytest tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_field_scenarios.py -q` -> 42 passed
- `python -m pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_field_operations.py::test_field_event_endpoint_dispatches_voice_directive -q` -> 2 passed
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed

Follow-up in the same checkpoint:

- `field-ingest-smoke` now writes 8 JSONL device events instead of 5, adding Hikvision-style ANPR, MQTT smoke alarm, and robot status `stuck` packets to the bridge smoke.
- The smoke report now exposes `expected_bridge_count=8`, so CI/operator scripts can catch accidental loss of a real-format sample.
- This moves the delivery check from "adapter unit tests pass" to "bridge -> HTTP endpoint -> normalization -> field rules -> archive -> action audit" for realistic vendor payloads.

Additional validation evidence:

- `python -m ruff check askme\cli.py tests\test_cli.py askme\pipeline\field_ingest_adapters.py` -> passed
- `python -m pytest tests\test_cli.py::test_cli_runtime_field_ingest_smoke_runs_local_http tests\test_cli.py::test_cli_runtime_field_ingest_smoke_produces_strict_audit_anchor tests\test_field_ingest_adapters.py -q` -> 13 passed
- `python -m askme runtime field-ingest-smoke --output-dir artifacts\field_operations\smoke --json` -> `status=passed`, `bridge.count=8`, `event_count=8`
- Artifact report: `artifacts/field_operations/smoke/field-ingest-smoke.json`
- `python -m askme runtime field-readiness --json` -> `status=ready_for_lab`, `smoke_report.event_count=8`, `archive.sources=[camera, robot, sensor]`

Remaining risk:

- These are adapter-level integrations, not live vendor-certified connectors. Real deployments still need device-by-device sample captures from the actual camera platform, smoke/temperature gateway, robot diagnostic bus, and DingTalk production group.
- External DingTalk delivery is tested through dispatcher mocks/local delivery reports here. A production key/webhook smoke test still must be run in the target network.
- Vision recognition itself is still assumed to arrive as detector output; model selection, camera calibration, and false-positive tuning remain separate work.

## 2026-05-11 Productization checkpoint: high-risk incident disposition smoke

High-risk现场事件 now have an executable local acceptance path instead of stopping at “事件已创建”.

- Added `python -m askme runtime field-disposition-smoke`.
- The smoke creates a P0 smoke/fire event through the real HTTP API, acknowledges it as a security operator, requests closure, closes it with supervisor approval, reads the event report, and verifies audit integrity.
- The scenario uses a unique generated location per run so duplicate detection does not hide a real workflow failure.
- `field-ingest-smoke` now clears the field action audit file before a fresh local smoke run. This prevents an old audit chain from invalidating a new operator handoff run.
- `field-smoke-suite` now includes `field_disposition_smoke` as a first-class check, so the suite verifies creation, voice/notification hooks, operator acknowledgement, supervisor close approval, report generation, readiness, and audit checkpointing together.

Validation evidence:

- `python -m ruff check askme\cli.py tests\test_cli.py` -> passed
- `python -m pytest tests\test_cli.py::test_cli_runtime_field_disposition_smoke_closes_p0_with_report tests\test_cli.py::test_cli_runtime_field_smoke_suite_aggregates_reports -q` -> 2 passed
- `python -m pytest tests\test_cli.py::test_cli_runtime_field_ingest_smoke_runs_local_http tests\test_cli.py::test_cli_runtime_field_disposition_smoke_closes_p0_with_report tests\test_cli.py::test_cli_runtime_field_smoke_suite_aggregates_reports tests\test_cli.py::test_cli_runtime_field_smoke_suite_command_forwards_args -q` -> 4 passed
- `python -m askme runtime field-disposition-smoke --output-dir artifacts\field_operations\smoke --json` -> `status=passed`
- `python -m askme runtime field-readiness --json` -> `status=ready_for_lab`, with `close_approval_workflow_verified=true` and `event_report_timeline_verified=true`
- `python -m askme runtime field-smoke-suite --output-dir artifacts\field_operations\smoke --skip-audit-anchor --json` -> `status=passed`
- Artifact reports:
  - `artifacts/field_operations/smoke/field-disposition-smoke.json`
  - `artifacts/field_operations/smoke/field-smoke-suite.json`

Product meaning:

- We can now demo a believable closed-loop security workflow: real event intake, operator action, supervisor approval, timeline/report, and audit verification.
- This is still a lab-grade product workflow, not a production deployment. Production readiness still requires HMAC-signed audit storage, real DingTalk delivery, a deployed always-on askme service, live TTS playback, and live device streams.

Follow-up in the same checkpoint:

- The field smoke suite now passes the same audit HMAC secret through the local event writer, readiness verifier, and audit anchor.
- `field-ingest-smoke`, `field-disposition-smoke`, and `field-readiness` now accept audit HMAC settings for local verification.
- With a secret configured, action audit records are written as an HMAC-signed hash chain, and readiness reports `action_audit_signed=true` instead of warning that the audit chain is unsigned.

Additional validation evidence:

- `python -m ruff check askme\cli.py tests\test_cli.py` -> passed
- `python -m pytest tests\test_cli.py::test_cli_runtime_field_ingest_smoke_produces_strict_audit_anchor tests\test_cli.py::test_cli_runtime_field_disposition_smoke_closes_p0_with_report tests\test_cli.py::test_cli_runtime_field_smoke_suite_aggregates_reports tests\test_cli.py::test_cli_runtime_field_readiness_reads_local_files -q` -> 4 passed
- `python -m askme runtime field-smoke-suite --output-dir artifacts\field_operations\smoke --audit-hmac-secret local-smoke-audit-secret --json` -> `status=passed`, `action_audit_signed=true`
- `python -m askme runtime field-readiness --audit-hmac-secret local-smoke-audit-secret --json` -> `status=ready_for_lab`, `action_audit_integrity.valid=true`, `action_audit_integrity.signed=true`

Operational hardening:

- CLI audit HMAC now defaults to `ASKME_FIELD_ACTION_AUDIT_HMAC_SECRET` when the explicit flag is omitted. This keeps the normal deployment path out of shell history and operator command logs.
- The env-sourced secret is resolved once inside `field-smoke-suite` and reused for ingest smoke, disposition smoke, readiness, and audit anchoring, preventing mixed signed/unsigned evidence in one run.

Additional validation evidence:

- `python -m pytest tests\test_cli.py::test_cli_runtime_field_smoke_suite_uses_env_audit_hmac_secret tests\test_cli.py::test_cli_runtime_field_ingest_smoke_produces_strict_audit_anchor tests\test_cli.py::test_cli_runtime_field_disposition_smoke_closes_p0_with_report tests\test_cli.py::test_cli_runtime_field_readiness_reads_local_files -q` -> 4 passed
- `ASKME_FIELD_ACTION_AUDIT_HMAC_SECRET=env-smoke-audit-secret python -m askme runtime field-smoke-suite --output-dir artifacts\field_operations\smoke --json` -> `status=passed`, `action_audit_integrity.signed=true`, `signature_alg=hmac-sha256`

Updated remaining risk:

- Local audit integrity is now signed when a secret is supplied. Production still needs a real secret-management policy, key rotation, and external WORM/SIEM anchoring.
- Real DingTalk webhooks, real MiniMax/live audio playback, deployed server smoke, and live device streams remain the production blockers.

## 2026-05-11 Productization checkpoint: DingTalk notification preflight

The external responder notification path now has a real preflight instead of relying on local collector smoke.

- Added `FieldOperationsService.notification_preflight_payload()`.
- Added HTTP endpoint `GET /api/field/notification-preflight`.
- Added CLI command `python -m askme runtime field-notification-preflight`.
- The preflight checks each responder group (`security`, `cleaning`, `operations`) for webhook and signing secret readiness, reports missing environment variable names, and exits non-zero when production notification configuration is incomplete.
- Fixed a production config bug: `${ASKME_DINGTALK_*}` placeholders are now resolved from environment variables instead of being treated as configured literal webhook/secret values.

Validation evidence:

- `python -m ruff check askme\pipeline\field_operations.py askme\health_server.py askme\cli.py tests\test_field_operations.py tests\test_cli.py tests\test_health.py` -> passed
- `python -m pytest tests\test_field_operations.py::test_notification_preflight_blocks_placeholder_config_without_env tests\test_field_operations.py::test_notification_preflight_resolves_env_placeholders tests\test_cli.py::test_cli_runtime_field_notification_preflight_reads_local_config tests\test_cli.py::test_cli_runtime_field_notification_preflight_exits_when_blocked -q` -> 4 passed
- `python -m pytest tests\test_health.py::TestHealthServer::test_field_notification_preflight_endpoint_reports_blocked -q` -> 1 passed
- With env configured for the security group: `python -m askme runtime field-notification-preflight --groups security --json` -> `status=ready`

Product meaning:

- Local notification smoke remains useful for demo wiring, but production delivery now has a separate gate that cannot pass just because a local webhook collector exists.
- The next production step is a real deployed `field-notification-smoke --server <url>` against DingTalk credentials in the target network.

## 2026-05-11 Productization checkpoint: deployed service smoke gate

The field-operations acceptance path now has a deployment-facing smoke command instead of only temporary local-server checks.

- Added `python -m askme runtime field-deployed-smoke`.
- The command targets an already-running Askme service and checks:
  - `/health`
  - `/api/field/notification-preflight`
  - device ingest smoke against `/api/field/ingest`
  - voice event smoke against `/api/field/events`
  - notification smoke against `/api/field/notification-test`
  - `/api/field/readiness`
- Real notification smoke is gated by notification preflight by default. If DingTalk credentials are not ready, the command marks notification smoke as skipped/failed instead of sending a misleading local-collector success.
- `--allow-notification-not-ready` exists for partial deployment diagnosis, but the default path is strict enough for customer acceptance.

Validation evidence:

- `python -m ruff check askme\cli.py tests\test_cli.py` -> passed
- `python -m pytest tests\test_cli.py::test_cli_runtime_field_deployed_smoke_runs_against_existing_server tests\test_cli.py::test_cli_runtime_field_deployed_smoke_blocks_when_notification_preflight_fails tests\test_cli.py::test_cli_runtime_field_deployed_smoke_command_forwards_args -q` -> 3 passed
- `python -m askme runtime field-deployed-smoke --help` -> command and options listed

Product meaning:

- We now have a clean boundary between lab smoke (`field-smoke-suite`) and deployed service acceptance (`field-deployed-smoke`).
- The next missing proof is running this command against a real always-on Askme service with real DingTalk credentials and target-network access.

## 2026-05-11 Productization checkpoint: real input contracts and production gates

This checkpoint turns two previous gaps into executable product controls.

1. Deployed service smoke was exercised against a real temporary FastAPI app, not just monkeypatched functions.
2. Real device input shapes are now stored as fixtures instead of only inline test dictionaries.
3. Production readiness now has a hard mode that blocks lab-only evidence.

Implemented:

- `field-voice-smoke` now adds a unique `smoke_run_id` and unique dedupe fields to each event. This fixes a real acceptance bug where `field-deployed-smoke` could fail after `field-ingest-smoke` because the voice smoke event was correctly deduplicated as a repeated fire event.
- Added `tests/fixtures/field_devices/site-a-device-events.jsonl` with site-style payloads:
  - Hikvision-style ANPR vehicle event.
  - MQTT smoke alarm packet.
  - ROS diagnostics joint motor fault.
  - Trash-bin fullness telemetry plus detection.
  - Crowd detection with multiple person boxes.
  - Night stranger near a window/corner.
  - Help-point visitor interaction.
- Added fixture tests proving the bridge normalizes these packets and `FieldOperationsService` turns them into the expected business scenarios.
- Added `field_operations.deployment_mode` in `config.yaml`.
- Added production-mode readiness blockers for:
  - missing responder webhooks/secrets,
  - local-only smoke evidence,
  - non-live TTS,
  - local webhook collector notification smoke,
  - `external_services=false`,
  - `hardware_dispatch=false`,
  - missing close approval/report timeline,
  - missing HMAC-signed audit chain.

Validation evidence:

- `python -m ruff check askme\cli.py tests\test_cli.py tests\test_field_ingest_bridge.py` -> passed
- `python -m pytest -p no:cacheprovider tests\test_cli.py::test_cli_field_voice_smoke_event_gets_unique_dedupe_fields tests\test_cli.py::test_cli_runtime_field_voice_smoke_queues_recorded_voice tests\test_cli.py::test_cli_runtime_field_deployed_smoke_runs_against_existing_server tests\test_field_ingest_bridge.py -q` -> 8 passed
- Temporary deployed service smoke with notification collector and voice handler -> `status=passed`, all checks true, `voice_delivery.status=queued`, responder notification requests captured.
- `python -m ruff check askme\pipeline\field_deployment_readiness.py tests\test_field_deployment_readiness.py` -> passed
- `python -m pytest -p no:cacheprovider tests\test_field_deployment_readiness.py -q` -> 6 passed

Product meaning:

- Local lab success can no longer be confused with production readiness when `deployment_mode=production`.
- The product now has a real acceptance seam for field devices: vendors can be asked to produce payloads compatible with `tests/fixtures/field_devices/site-a-device-events.jsonl`, and regressions will be caught.
- The next missing proof is not more framework code. It is real target-network verification: real DingTalk credentials, live TTS on the robot audio device, live camera/sensor/ROS streams, and non-fake runtime/hardware dispatch.

## 2026-05-11 Productization checkpoint: trusted device ingest

The field-event path now has a real trust boundary before business rules run.

Implemented:

- Added field-device HMAC signing helper: `sign_field_device_payload(body, secret=...)`.
- Added device registry support to `FieldOperationsService`.
- Added `require_trusted_devices` and `device_signature_max_age_s` config.
- When trusted-device mode is enabled, `/api/field/ingest` rejects:
  - missing device id,
  - unregistered device,
  - missing device signature,
  - signature mismatch,
  - unsupported signature algorithm,
  - missing/stale/future signature timestamp,
  - source type not allowed for that device.
- The normalized ingest result now includes `device_trust` evidence, so UI/readiness/audit can show whether the event came from a trusted device.
- Production readiness now requires both a device registry and signed field-device ingest.

Validation evidence:

- `python -m ruff check askme\pipeline\field_operations.py askme\pipeline\field_deployment_readiness.py tests\test_field_operations.py tests\test_field_deployment_readiness.py` -> passed
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py::test_signed_registered_device_ingest_is_trusted tests\test_field_operations.py::test_trusted_device_ingest_rejects_missing_signature tests\test_field_operations.py::test_trusted_device_ingest_rejects_wrong_source tests\test_field_deployment_readiness.py -q` -> 9 passed

Product meaning:

- Field ingest is no longer only “JSON shape compatible”. It can now be made “device-authenticated” for production.
- This is still not a full IoT platform. The next layer should add long-lived device heartbeat/online status, key rotation, per-site device ownership, and webhook/MQTT/ROS listener services that apply the same signing contract automatically.

## 2026-05-11 Productization checkpoint: device bridge, voice profiles, and entry boundaries

This checkpoint moves several customer-visible and production-safety items out of framework status.

Implemented:

- `/api/field/devices` is now consumed by the Dashboard diagnostics area as "现场设备", showing online/stale/never-seen/unregistered device state.
- Production readiness now includes trusted-device event evidence, not only a configured registry.
- Field ingest bridge can sign normalized JSON/JSONL device events before posting to `/api/field/ingest` via `--device-secret DEVICE_ID=SECRET`.
- Bridged signed payloads now include `device_signature_timestamp`, so production-mode trusted ingest can verify both HMAC integrity and freshness.
- `/api/field/events` is explicitly treated as a manual/operator trigger path:
  - manual events receive `admission_path=field_events_manual`;
  - an operator id is attached from body/header/default;
  - raw camera/sensor/robot payloads without `scenario_id` are rejected and directed to `/api/field/ingest`.
- Voice profile catalog was expanded beyond one generic voice:
  - cleaning notice,
  - operations dispatch,
  - crowd guidance,
  - escort guidance,
  - robot fault alarm.
- Incident playbooks now route trash, crowd, operations, and robot-fault cases to the new profile aliases instead of one generic emergency/patrol voice.

Validation evidence:

- `python -m ruff check askme\voice\voice_profiles.py askme\pipeline\incident_alerts.py askme\pipeline\field_ingest_bridge.py askme\cli.py tests\test_voice_profiles.py tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_health.py` -> passed
- `python -m pytest -p no:cacheprovider tests\test_voice_profiles.py tests\test_field_ingest_bridge.py tests\test_field_operations.py::test_trash_event_routes_to_cleaning_webhook tests\test_field_operations.py::test_sensor_ingest_triggers_fire_or_smoke tests\test_field_operations.py::test_field_operations_http_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 15 passed
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py::test_field_operations_http_endpoints tests\test_health.py::TestHealthServer::test_control_api_key_protects_non_probe_routes -q` -> 2 passed

Product meaning:

- A vendor device file bridge can now participate in the same signed ingest contract as direct HTTP device payloads.
- Operators and customers can see whether field devices are online/trusted instead of guessing from event logs.
- The product can speak differently for cleaning, security, crowd guidance, escort, operations, and robot-fault cases.
- Manual event creation and device event ingest now have separate admission paths, reducing the risk that raw device payloads bypass trust checks.

Still missing before production:

- Real DingTalk group smoke using customer credentials and target network.
- Live MiniMax TTS playback on the robot audio device, not recorded handler proof.
- Live camera/smoke/temperature/ROS diagnostic listeners running as deployment services, not only JSONL bridge fixtures.
- Hardware runtime dispatch evidence for pause/retreat/escort/urgent patrol.
- FieldIncident lifecycle that writes alert/voice/runtime/memory delivery state back to one event timeline.

## 2026-05-11 Productization checkpoint: FieldIncident workflow state

This checkpoint closes part of the "framework only" gap by giving every field event a product-facing lifecycle, not just a raw status string.

Implemented:

- `FieldEventRecord` now carries:
  - `incident_state`
  - `incident_stage`
  - `incident_workflow`
- Each event now computes workflow stages for:
  - admission: manual trigger, trusted device, blocked device, or not required
  - assessment: accepted, blocked, or duplicate
  - notification: sent, partial, failed, pending, or not required
  - voice: queued, delivered status when present, or not required
  - robot motion: runtime policy readiness or not required
  - operator: pending, acknowledged, pending approval, or closed
  - archive: written
  - memory: written/pending/not connected
- Append and rewrite paths refresh the workflow before writing JSONL, so acknowledge/request-close/close updates move the same event through the lifecycle.
- Reports now expose `incident_state`, `incident_stage`, and `incident_workflow`, so the UI or exported report can explain what is still missing.
- Dashboard field-event cards now render the workflow stages directly, so operators can see whether an incident is waiting on notification, voice, runtime motion, operator action, archive, or memory.
- Closed incidents can now write a structured summary into site memory when `incident_memory_enabled=True` or `ASKME_FIELD_INCIDENT_MEMORY=1`.
- Memory writeback records `memory_delivery` on the event, then the same incident workflow moves the memory stage to `written`; failures are captured as `memory_delivery.status=failed`.
- Field voice delivery is now persisted back to the event archive through `record_voice_delivery_payload`, so voice playback status survives after the HTTP response and appears in the same workflow.

Validation evidence:

- `python -m ruff check askme\pipeline\field_operations.py tests\test_field_operations.py` -> passed
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py::test_fire_event_dispatches_and_archives tests\test_field_operations.py::test_p0_event_close_requires_supervisor_approval -q` -> 2 passed
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py -q` -> 51 passed
- `python -m py_compile askme\pipeline\field_operations.py` -> passed
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 52 passed
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py::test_closed_incident_writes_memory_when_enabled tests\test_field_operations.py::test_fire_event_dispatches_and_archives tests\test_field_operations.py::test_p0_event_close_requires_supervisor_approval -q` -> 3 passed
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 53 passed
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py::test_field_event_endpoint_dispatches_voice_directive tests\test_field_operations.py::test_closed_incident_writes_memory_when_enabled -q` -> 2 passed
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_health.py::TestHealthServer::test_control_api_key_protects_non_probe_routes -q` -> 54 passed

Product meaning:

- Customers no longer have to infer incident progress from scattered fields. A field event can now say whether it is waiting for operator handling, notification delivery, voice playback, runtime action, or memory closure.
- The current workflow marks memory as `pending` until closure. When incident memory is enabled, closure writes an anomaly/observation into site memory and records the delivery result on the event.

Still missing before production:

- Replace queued/mock voice delivery with target-device playback callbacks when the MiniMax/S100P audio chain is available.
- Persist actual runtime arbiter/hardware dispatch callbacks into `runtime_delivery`.
- Add a curated RAG/catalog write path for selected closed incident summaries. Current writeback goes to site memory, not yet the approval-gated RAG catalog.
- Replace the current compact workflow row with a richer customer timeline once visual QA is available.
## 2026-05-11 Productization checkpoint: Field runtime delivery

Implemented in this pass:

- Field incidents now carry `runtime_delivery` as a first-class archived field, alongside `voice_delivery` and `memory_delivery`.
- `/api/field/events` and `/api/field/ingest` now evaluate `playbook.robot_motion_policy` after voice dispatch.
- If no runtime handler is configured, the archive records a safe `policy_ready` delivery with `hardware_dispatch=false` and reason `runtime_handler_not_configured`.
- If a runtime handler is configured, the event is converted into a confirmed high-level TaskPlan and submitted through `submit_plan_payload`; the archive records run id, handoff id, profile, current state, dispatch mode, and safety boundary.
- The Dashboard field-event card now shows runtime delivery state so customers can see whether the robot-side action is merely policy-ready or has entered the runtime arbiter.
- `/api/field/events/{event_id}/runtime-delivery` now accepts runtime-arbiter/robot callbacks and updates the same archived FieldIncident workflow.
- Runtime delivery callbacks now support HMAC verification through `field_runtime_callback_secret` or `ASKME_FIELD_RUNTIME_CALLBACK_HMAC_SECRET`; unsigned callbacks are allowed only when the secret is not configured and are marked as unsigned in the archive.
- Runtime delivery callbacks now use a controlled status vocabulary. Unknown robot/runtime status values return `422` and do not mutate the archived incident.
- Runtime callbacks are idempotent through `runtime_callback_id`; when the runtime omits an id, the HTTP layer derives one from the unsigned callback payload. Duplicate callbacks return `runtime_callback_already_recorded` and do not append duplicate receipts.
- Each accepted runtime callback writes a `runtime_delivery_receipts` entry, so a customer audit can prove which runtime callback moved the incident state.
- Runtime-side code now has a reusable `askme.runtime.field_callbacks` helper plus `scripts/runtime/post_field_runtime_callback.py`, so shadow/lab/robot processes can produce signed callbacks without reimplementing HMAC signing.
- The helper can now consume a `RuntimeHandoffService.submit_plan_payload()` result JSON and produce a signed status sequence such as `submitted -> validating -> preflight -> shadowed/completed`, preserving the field event id and robot motion policy.
- `/api/field/events` now returns `runtime_handoff_result` when a runtime handler is configured. Shadow/lab callback producers can consume the response directly instead of reconstructing handoff state from logs.
- A roundtrip regression now proves: create FieldIncident -> submit to shadow runtime -> build signed callback sequence -> POST callbacks to `/runtime-delivery` -> archive workflow reaches `shadowed` with trusted receipts.
- `scripts/eval/smoke_field_runtime_roundtrip.py` is now the product smoke gate for this loop. It can run in-process by default, or against an existing askme HTTP service with `--base-url`.
- The smoke gate can now also start a temporary local uvicorn askme service with `--start-local-server`, run the same roundtrip over real HTTP, then stop the service.
- Field deployment readiness now checks `runtime_callback_signature_configured`; production mode blocks when runtime delivery callbacks are not signed.
- Field deployment readiness now also consumes `runtime_roundtrip_report_path` and blocks when the FieldIncident -> runtime handoff -> signed callback -> archive roundtrip smoke has not passed.
- Production readiness now requires the runtime roundtrip smoke to run against an existing deployed askme service, include trusted callback receipts, and finish with `shadowed` or `completed` runtime delivery.

Validation evidence:

- `python -m ruff check askme\health_server.py askme\pipeline\field_operations.py tests\test_field_operations.py` -> passed.
- `python -m py_compile askme\health_server.py askme\pipeline\field_operations.py` -> passed.
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py::test_field_event_endpoint_dispatches_voice_directive tests\test_field_operations.py::test_field_event_endpoint_submits_runtime_policy_when_handler_is_configured tests\test_field_operations.py::test_field_operations_http_endpoints -q` -> 3 passed.
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py::test_field_event_endpoint_submits_runtime_policy_when_handler_is_configured tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 2 passed.
- `python -m pytest -p no:cacheprovider tests\test_runtime_handoff.py tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 84 passed.
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py::test_field_event_runtime_delivery_callback_requires_valid_signature tests\test_field_operations.py::test_field_event_runtime_delivery_callback_updates_archive tests\test_field_deployment_readiness.py -q` -> 8 passed.
- `python -m ruff check askme\pipeline\field_operations.py askme\health_server.py tests\test_field_operations.py` -> passed.
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py::test_field_event_runtime_delivery_callback_requires_valid_signature tests\test_field_operations.py::test_field_event_runtime_delivery_callback_rejects_invalid_status tests\test_field_operations.py::test_field_event_runtime_delivery_callback_is_idempotent tests\test_field_operations.py::test_field_event_runtime_delivery_callback_updates_archive -q` -> 4 passed.
- `python -m pytest -p no:cacheprovider tests\test_runtime_handoff.py tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 87 passed.
- `git diff --check -- askme\health_server.py askme\pipeline\field_deployment_readiness.py askme\runtime\handoff.py askme\pipeline\field_operations.py askme\static\dashboard.html tests\test_runtime_handoff.py tests\test_field_operations.py tests\test_field_deployment_readiness.py tests\test_health.py plans\plan.md docs\OPERATIONS.md .env.example` -> passed; Git only reported line-ending warnings.
- `python -m pytest -p no:cacheprovider tests\test_field_runtime_callbacks.py tests\test_field_operations.py::test_field_event_runtime_delivery_callback_requires_valid_signature tests\test_field_operations.py::test_field_event_runtime_delivery_callback_rejects_invalid_status tests\test_field_operations.py::test_field_event_runtime_delivery_callback_is_idempotent -q` -> 6 passed.
- `python -m pytest -p no:cacheprovider tests\test_field_runtime_callbacks.py tests\test_runtime_handoff.py tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 90 passed.
- `python -m pytest -p no:cacheprovider tests\test_field_runtime_callbacks.py tests\test_runtime_handoff.py tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 92 passed after runtime-result sequence support.
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py::test_field_event_shadow_runtime_callback_roundtrip_updates_archive -q` -> 1 passed.
- `python -m pytest -p no:cacheprovider tests\test_field_runtime_callbacks.py tests\test_runtime_handoff.py tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 93 passed after shadow roundtrip coverage.
- `python scripts\eval\smoke_field_runtime_roundtrip.py --secret runtime-secret --output artifacts\runtime_handoff\field-runtime-roundtrip-smoke.json` -> `ok: true`, final runtime delivery `shadowed`, 5 trusted callback receipts.
- `python scripts\eval\smoke_field_runtime_roundtrip.py --start-local-server --secret runtime-secret --output artifacts\runtime_handoff\field-runtime-roundtrip-live-smoke.json` -> `ok: true`, mode `local_server`, real HTTP roundtrip, final runtime delivery `shadowed`, 5 trusted callback receipts.
- `python scripts\runtime\post_field_runtime_callback.py --event-id evt-demo --status executing --run-id run-demo --secret runtime-secret --dry-run` -> printed a signed callback payload.
- `python scripts\runtime\post_field_runtime_callback.py --result-json output\askme-runtime-result.json --secret runtime-secret --dry-run` -> printed a signed four-status shadow callback sequence; the temporary JSON fixture was removed after the smoke.
- `python -m ruff check askme\pipeline\field_deployment_readiness.py tests\test_field_deployment_readiness.py` -> passed after runtime roundtrip readiness gates.
- `python -m pytest -p no:cacheprovider tests\test_field_deployment_readiness.py -q` -> 6 passed after runtime roundtrip readiness gates.

Still not production-complete:

- Real hardware/runtime submission still depends on an explicitly configured runtime handler/profile. The default remains no hardware dispatch.
- The runtime skill registry now models dedicated field-response skills such as `stop_and_hold`, `safe_pause`, `retreat_to_safe_distance`, `keep_distance_observe`, `observe_then_recheck`, `record_then_continue`, and `low_speed_escort`.
- Remaining gap: these are still high-level arbiter skills with fake/sim execution evidence unless an external/lab runtime profile is explicitly wired.
- Runtime delivery callbacks now have an HTTP entry point and HMAC verification, but a real robot/lab runtime still needs a subscribed callback producer and key provisioning on the robot/runtime side.
- The next product-grade step is a real runtime callback producer in the lab runtime process: submit a field incident, execute the high-level policy in shadow/lab, then POST signed `queued/executing/completed/failed` callbacks into this endpoint.

## 2026-05-12 Productization checkpoint: Customer-facing field copy

Implemented in this pass:

- Replaced corrupted product text in `field_scenarios.py` with readable Chinese scenario names, trigger rules, evidence requirements, robot behaviors, notification groups, and acceptance criteria.
- Replaced corrupted `incident_alerts.py` voice announcements, DingTalk messages, operator actions, and playbook checklists with customer-demo-ready Chinese copy.
- Preserved the existing incident topics and runtime policies, so routing behavior remains stable while the operator/customer language is now understandable.
- Added clean visitor-service wording for wayfinding and escort events in `field_operations.py`; visitor questions remain service interactions and do not become security alarms or robot missions.
- Added readable high-risk sensor rejection reasons for stale evidence and low detection confidence.

Validation evidence:

- `python -m ruff check askme\pipeline\incident_alerts.py askme\pipeline\field_scenarios.py askme\pipeline\field_operations.py askme\pipeline\field_deployment_readiness.py tests\test_alert_dispatcher.py tests\test_field_scenarios.py tests\test_field_operations.py tests\test_field_deployment_readiness.py` -> passed.
- `python -m pytest -p no:cacheprovider tests\test_alert_dispatcher.py tests\test_field_scenarios.py tests\test_field_operations.py -q` -> 92 passed.
- `python -m pytest -p no:cacheprovider tests\test_field_runtime_callbacks.py tests\test_runtime_handoff.py tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_alert_dispatcher.py tests\test_field_scenarios.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 139 passed.

Still not production-complete:

- Some older docs and tests still contain mojibake text; the field runtime/customer-facing scenario surface is fixed, but a repo-wide encoding cleanup is still needed.
- The wayfinding/escort wording is fixed, but full map-route grounding still depends on approved map knowledge and route database quality.

## 2026-05-12 Productization checkpoint: Runtime readiness surfaced in Dashboard

Implemented in this pass:

- Dashboard field readiness now exposes the runtime roundtrip gates that already exist in backend readiness:
  `runtime_roundtrip_smoke_passed`, `runtime_roundtrip_against_existing_server`,
  `runtime_roundtrip_trusted_callbacks`, and `runtime_roundtrip_final_status_verified`.
- The readiness card now uses readable customer-facing Chinese labels for the delivery gates instead of only raw engineering keys.
- When `runtime_roundtrip_report` is present, the card shows runtime evidence directly: final runtime status, smoke mode, and trusted callback receipt count.
- The UI success/failure markers and readiness summary now render readable text such as `现场运行门禁已通过`, making the deployability state inspectable during customer demos.

Validation evidence:

- `python -m ruff check tests\test_health.py askme\pipeline\field_deployment_readiness.py` -> passed.
- `python -m pytest -p no:cacheprovider tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_field_deployment_readiness.py -q` -> 7 passed.

Still not production-complete:

- This surfaces existing roundtrip evidence; it does not yet start a real robot/lab runtime callback producer by itself.
- Other Dashboard sections still contain older mojibake strings and need a dedicated product UI cleanup pass.
- A browser visual QA pass should follow after the broader Dashboard copy/layout cleanup.

## 2026-05-12 Productization checkpoint: Domestic device event ingestion

Implemented in this pass:

- Field ingest adapters now understand common domestic camera/model labels and event names:
  `车辆违停`, `夜间陌生人拍照`, `垃圾桶满溢`, `人员聚集`, plus Chinese labels for person, vehicle, fire, smoke, trash bin, phone, and camera.
- Night stranger handling now preserves photo evidence. When a night/window/restricted-zone camera event includes phone/camera/photo evidence, the system can trigger `night_stranger_photo` after a shorter configurable dwell gate (`night_photo_dwell_s`, default 3s) while still rejecting known/authorized persons.
- Domestic trash-bin and crowd alarms no longer need upstream code to fabricate internal askme detections. The adapter can infer `trash_bin_full` and `crowd_gathering` from vendor-style event names plus count/fill fields.
- Robot fault normalization now recognizes Chinese states such as `卡住`, `无法运动`, `人为挡路`, `关节`, `电机`, and `过流`.
- `/api/field/ingest` help now returns clean product examples for domestic camera parking, night photo, trash-bin full, crowd gathering, smoke sensor, MQTT smoke alarm, and robot fault payloads.
- High-risk stale/low-confidence sensor rejections now produce readable Chinese operator actions instead of corrupted text.

Validation evidence:

- `python -m py_compile askme\pipeline\field_operations.py askme\pipeline\field_ingest_adapters.py` -> passed.
- `python -m ruff check askme\pipeline\field_ingest_adapters.py askme\pipeline\field_operations.py tests\test_field_operations.py` -> passed.
- `python -m pytest -p no:cacheprovider tests\test_field_operations.py::test_domestic_camera_event_name_triggers_illegal_parking tests\test_field_operations.py::test_night_stranger_photo_uses_photo_evidence tests\test_field_operations.py::test_domestic_trash_bin_alarm_triggers_cleaning_workflow tests\test_field_operations.py::test_domestic_crowd_alarm_triggers_security_workflow tests\test_field_operations.py::test_stale_sensor_ingest_is_archived_without_dispatch tests\test_field_operations.py::test_low_confidence_camera_ingest_requires_review -q` -> 6 passed.
- `python -m pytest -p no:cacheprovider tests\test_field_runtime_callbacks.py tests\test_runtime_handoff.py tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_alert_dispatcher.py tests\test_field_scenarios.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 143 passed.

Still not production-complete:

- This makes the HTTP/file ingest path product-real; live camera/VMS/MQTT adapters still need site-specific credentials and long-running bridge deployment.
- Detection quality is still inherited from upstream camera/VLM models. Askme now gates freshness/confidence and archives evidence, but it is not yet training or running the visual detector itself.
- The old unreachable ingest-help block and other legacy mojibake text should be removed in a dedicated cleanup pass after current behavior is fully locked.

## 2026-05-12 Productization checkpoint: Voice profile catalog and sound cues

Implemented in this pass:

- Rebuilt `askme.voice.voice_profiles` with clean product-facing voice styles:
  巡检播报, 访客服务, 安保提醒, 紧急短句, 夜间低声, 保洁通知, 运维沉稳, 人群疏导, 带路引导, 故障急报, and 确认提示.
- Added profile metadata for `category` and `cue`, so the UI can distinguish visitor, security, emergency, cleaning, operations, and interaction styles, plus cue intents such as `welcome_chime`, `notice_beep`, `emergency_tone`, `fault_tone`, and `confirm_chime`.
- Added alias routing for product playbook terms such as `fire_alarm`, `night_photo`, `trash_notice`, `wayfinding_prompt`, and `confirm_prompt`.
- Existing config overrides can still change `voice_id`, speed, volume, pitch, and emotion, but corrupted/mojibake labels, use cases, and sample text now fall back to clean built-in copy.
- TTS status and `/api/voice/profiles` now expose the cue/category metadata through active profile settings.
- Dashboard voice profile UI now shows readable current voice, persistence state, category, cue, and sample wording instead of corrupted status text.

Validation evidence:

- `python -m ruff check askme\voice\voice_profiles.py askme\voice\tts.py tests\test_voice_profiles.py tests\test_health.py` -> passed.
- `python -m pytest -p no:cacheprovider tests\test_voice_profiles.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_field_operations.py::test_sensor_ingest_triggers_fire_or_smoke tests\test_field_operations.py::test_trash_event_routes_to_cleaning_webhook tests\test_field_operations.py::test_field_operations_http_endpoints -q` -> 10 passed.
- `python -m pytest -p no:cacheprovider tests\test_voice_profiles.py tests\test_field_runtime_callbacks.py tests\test_runtime_handoff.py tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_alert_dispatcher.py tests\test_field_scenarios.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 149 passed.
- `python -m pytest -p no:cacheprovider tests\test_voice_profiles.py tests\test_tts_minimax.py tests\test_field_runtime_callbacks.py tests\test_runtime_handoff.py tests\test_field_operations.py tests\test_field_ingest_bridge.py tests\test_field_deployment_readiness.py tests\test_alert_dispatcher.py tests\test_field_scenarios.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 183 passed after local sound cue queueing.

Follow-up implemented after initial checkpoint:

- `cue` is now real runtime behavior, not only metadata. `TTSEngine.queue_sound_cue()` generates short local PCM chimes/tones and appends them to the normal playback buffer.
- `set_voice_profile_payload(..., speak_sample=true)` queues the selected profile cue before speaking the sample sentence.
- `/api/voice/profiles` exposes `sound_cues_enabled` and `available_sound_cues`, so UI/client code can know which cue intents are actually supported.
- Cues can be disabled per deployment with `voice_profile_cues_enabled=false`.

Still not production-complete:

- Generated PCM cues are now real and test-covered, but a branded asset bank can still replace them if the customer wants custom production audio.
- Real MiniMax voice IDs for cloned/customer voices must be configured per deployment. The built-ins use the current default provider voice unless overridden.
- Dashboard still needs browser visual QA after the broader copy cleanup.

External reference notes:

- GitHub `sancliffe/ollama-STT-TTS` (https://github.com/sancliffe/ollama-STT-TTS) reinforces that a usable voice assistant needs explicit wake/listening readiness, VAD stop conditions, local/fast TTS, and device listing for audio troubleshooting.
- GitHub `m15-ai/TrooperAI` (https://github.com/m15-ai/TrooperAI) uses a bidirectional audio message loop and an explicit playback completion marker, which maps to our remaining need for a stronger client/runtime playback lifecycle signal.
- GitHub `pipecat-ai/pipecat` (https://github.com/pipecat-ai/pipecat) validates the pipeline approach: speech, LLM, TTS, transport, and multimodal context should stay pluggable rather than becoming one monolithic voice loop.
- GitHub `m15-ai/Local-Voice` (https://github.com/m15-ai/Local-Voice) shows practical edge-device concerns: local STT/TTS fallback, ALSA/device controls, and optional audio effects matter as much as model quality for perceived responsiveness.

## 2026-05-12 Productization checkpoint: Dashboard customer-language pass and visual QA

Implemented in this pass:

- Dashboard runtime/voice status no longer exposes raw words like `unknown`, `Executing`, `Live sync`, or `No active runtime task` in the primary customer view. These now render as `未连接`, `执行中`, `实时同步`, and `暂无运行任务`.
- Voice console labels now answer the user’s real question directly: whether they can speak, whether the microphone is connected, whether speech recognition is local/cloud, whether TTS is idle/playing, whether interruption is available, and whether safety preflight is configured.
- Service capability cards now show customer-facing Chinese names and status details for voice, model brain, memory/RAG, task handoff, task runtime, and safety preflight.
- Field event cards now translate notification delivery, priority/severity, workflow state, runtime delivery, robot motion policy, and notification test results into product language. Examples: `钉钉:未发送 未配置`, `P0 紧急`, `处置流程 进行中`, `运动策略 观察后继续`, `运行交接 提交失败`.
- The visual QA script is now part of the evidence loop: it launches a local FastAPI/uvicorn server, seeds a real field event through `/api/field/events`, opens `/dashboard` in Chromium, captures desktop/mobile screenshots, and checks required text, frontend errors, HTTP errors, bad text markers, and horizontal overflow.

Validation evidence:

- `node --check .tmp\dashboard-script.js` -> passed.
- `python -m pytest -p no:cacheprovider tests\test_health.py -q` -> 39 passed.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed.
  - Desktop screenshot: `output\playwright\askme-dashboard-desktop.png`
  - Mobile screenshot: `output\playwright\askme-dashboard-mobile.png`
  - No missing required text, no frontend console errors, no page errors, no HTTP response errors, no bad text marker, no horizontal overflow.

Product reflection:

- This is still not the same as a production deployment. It proves the customer-facing surface can show the current product capabilities cleanly and that the field-event path is exercised through HTTP, but the machine dog hardware, site camera streams, MiniMax live keys, DingTalk real webhook, and long-running sensor bridges still require environment-specific credentials and on-site smoke tests.
- The next highest-value product step is a live “园区演示包”: one command starts the service, loads sample map/knowledge/device registry, starts a mock camera/MQTT/robot-fault bridge, and opens a scenario checklist for patrol, visitor wayfinding, illegal parking, smoke, trash-bin full, crowd, and robot fault.

## 2026-05-12 Productization checkpoint: Field operations customer demo evidence

Implemented in this pass:

- Added `scripts/demo/field_operations_demo.py`, a one-command customer demo package builder.
- Added `deploy/site-profiles/park-demo.yaml`, a concrete customer-style site profile for map zones, help points, parking policy, responder groups, devices, and thresholds.
- Added `askme.pipeline.field_site_profile` plus `scripts/demo/validate_field_site_profile.py` so site profiles can be validated and converted into a `field_operations` runtime config fragment before deployment.
- `FieldOperationsService` now accepts `site_profile_path` (or `ASKME_FIELD_SITE_PROFILE`) and loads the site profile into the real runtime config path before ingesting camera/sensor/robot events. Explicit runtime config still overrides the profile.
- `config.yaml` now points field operations at `deploy/site-profiles/park-demo.yaml` for the local product demo baseline.
- Field deployment readiness now includes site-profile gates: configured, valid, map configured, parking policy configured, wayfinding configured, responder groups configured, and device registry configured.
- `askme runtime field-readiness` now accepts `--site-profile` and prints a site-profile readiness summary in the operator console.
- The demo builder writes:
  - `artifacts/field_operations/demo/scenario-evaluation.json`
  - `artifacts/field_operations/demo/customer-demo-guide.md`
  - `artifacts/field_operations/demo/field-demo-package.json`
  - `artifacts/field_operations/demo/site-profile-readiness.json`
  - optional Dashboard screenshots under `artifacts/field_operations/demo/dashboard-visual/`
- Upgraded `scripts/eval/evaluate_field_operations_scenarios.py` from a technical pass/fail artifact into a customer-facing field demo package.
- The generated `artifacts/field_operations/scenario-evaluation.json` now includes `product_demo` with ten customer scenarios:
  - robot fall/stuck recovery failure
  - night stranger photo capture
  - illegal parking
  - fire/smoke monitoring
  - trash-bin full monitoring
  - urgent patrol dispatch
  - crowd gathering
  - visitor wayfinding
  - visitor escort
  - DingTalk notification smoke test
- Each product row records trigger source, expected robot action, expected notification group, archive expectation, actual voice text, notification delivery status, runtime submission status, workflow gaps, event id, incident topic, priority, severity, media count, and delivery report.
- Runtime health now exposes the product demo readiness summary so the health/dashboard layer can distinguish demo-ready scenario coverage from real site integration readiness.
- `askme runtime field-eval` now prints the same product demo checklist by default, so an operator can run one command and read the customer scenes without opening the raw JSON.
- The report explicitly states what is still not real: camera/VMS streams, smoke/temperature/MQTT sensors, production DingTalk webhook, physical robot runtime callbacks, MiniMax live voice devices, and customer-specific map/parking/help-point configuration.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\scenario_tests\test_field_operations_evaluation.py -q` -> 1 passed.
- `python -m pytest -p no:cacheprovider tests\scenario_tests\test_field_operations_demo.py -q` -> 1 passed.
- `python -m pytest -p no:cacheprovider tests\test_field_site_profile.py -q` -> 5 passed.
- `python -m pytest -p no:cacheprovider tests\test_field_site_profile.py tests\test_field_operations.py::test_domestic_camera_event_name_triggers_illegal_parking -q` -> 6 passed.
- `python -m pytest -p no:cacheprovider tests\test_field_deployment_readiness.py tests\test_cli.py::test_cli_runtime_field_readiness_reads_local_files tests\test_cli.py::test_cli_runtime_field_readiness_exits_nonzero_when_blocked -q` -> 8 passed.
- `python -m ruff check askme\pipeline\field_deployment_readiness.py askme\pipeline\field_operations.py askme\cli.py tests\test_field_deployment_readiness.py tests\test_cli.py` -> passed.
- `python -m askme.cli runtime field-readiness --archive-path artifacts\field_operations\smoke\field-events.jsonl --scenario-report artifacts\field_operations\demo\scenario-evaluation.json --smoke-report artifacts\field_operations\smoke\field-ingest-smoke.json --voice-smoke-report artifacts\field_operations\smoke\field-voice-smoke.json --notification-smoke-report artifacts\field_operations\smoke\field-notification-smoke.json --site-profile deploy\site-profiles\park-demo.yaml` -> ready_for_lab, site profile configured/valid, remaining warnings are real integration gaps.
- `python -m pytest -p no:cacheprovider tests\test_runtime_modules.py -q` -> 46 passed.
- `python -m pytest -p no:cacheprovider tests\test_cli.py::test_cli_runtime_field_eval_writes_report tests\test_cli.py::test_cli_runtime_field_eval_prints_product_demo -q` -> 2 passed.
- `python -m ruff check askme\pipeline\field_site_profile.py scripts\demo\validate_field_site_profile.py scripts\demo\field_operations_demo.py tests\test_field_site_profile.py tests\scenario_tests\test_field_operations_demo.py` -> passed.
- `python -m ruff check scripts\demo\field_operations_demo.py tests\scenario_tests\test_field_operations_demo.py` -> passed.
- `python -m ruff check askme\cli.py tests\test_cli.py` -> passed.
- `python scripts\eval\evaluate_field_operations_scenarios.py --output artifacts\field_operations\scenario-evaluation.json` -> passed.
- `python -m askme.cli runtime field-eval --output artifacts\field_operations\scenario-evaluation.json` -> passed and printed 10/10 product scenes plus real integration gaps.
- `python scripts\demo\validate_field_site_profile.py --profile deploy\site-profiles\park-demo.yaml --output artifacts\field_operations\demo\site-profile-readiness.json` -> passed, 6 zones, 4 devices, 3 responder groups.
- `python scripts\demo\field_operations_demo.py --output-dir artifacts\field_operations\demo --site-profile deploy\site-profiles\park-demo.yaml --with-dashboard-visual` -> passed, generated customer guide, package JSON, scenario report, site profile readiness, desktop screenshot, and mobile screenshot.
- Generated report summary: `园区机器狗场景演示包`, `demo_ready=True`, `real_integration_ready=False`, `passed=10/10`.

Product reflection:

- This closes a product evidence gap, not the full production gap. We can now show a customer which scenes work in the software path and what evidence each scene produces.
- The next step should be a one-command live demo runner that starts the service, seeds the site map/device registry/knowledge, opens the dashboard, and lets an operator trigger these scenarios through HTTP/device-style inputs instead of only reading a JSON report.

## 2026-05-12 Productization checkpoint: Site profile environment preflight

Implemented in this pass:

- `FieldOperationsService` can now validate whether the site profile's referenced DingTalk, device, and responder environment variables are actually present before a lab/prod run.
- `askme runtime field-readiness` now accepts `--check-site-env`, so an operator can run the same readiness command and see exactly which real deployment secrets are missing.
- Deployment readiness surfaces site profile environment warnings in the normal warning list instead of hiding them inside profile metadata.
- Readiness next actions now include a concrete instruction to configure site profile environment variables for DingTalk responders and field devices.
- Site profile env checking is explicit via config/CLI/env (`site_profile_check_env`, `check_site_profile_env`, `ASKME_FIELD_SITE_PROFILE_CHECK_ENV`) so tests and local demos can still use explicit runtime overrides without false production failures.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\test_field_site_profile.py tests\test_field_deployment_readiness.py tests\test_cli.py::test_cli_runtime_field_readiness_reads_local_files tests\test_cli.py::test_cli_runtime_field_readiness_exits_nonzero_when_blocked -q` -> 15 passed.
- `python -m ruff check askme\pipeline\field_operations.py askme\pipeline\field_deployment_readiness.py askme\cli.py tests\test_field_site_profile.py tests\test_field_deployment_readiness.py tests\test_cli.py` -> passed.
- `python -m askme.cli runtime field-readiness --archive-path artifacts\field_operations\smoke\field-events.jsonl --scenario-report artifacts\field_operations\demo\scenario-evaluation.json --smoke-report artifacts\field_operations\smoke\field-ingest-smoke.json --voice-smoke-report artifacts\field_operations\smoke\field-voice-smoke.json --notification-smoke-report artifacts\field_operations\smoke\field-notification-smoke.json --site-profile deploy\site-profiles\park-demo.yaml --check-site-env` -> `ready_for_lab`, site profile valid, and missing DingTalk/device/robot env vars listed.

Product reflection:

- This turns "site profile exists" into an auditable deployment preflight. A customer success engineer can now see whether map/zones/devices are configured and whether the real credentials behind them are missing.
- This still does not make the external services real. The next product step remains a live demo runner plus real bridge smokes for camera/VMS, MQTT/smoke sensors, robot fault callbacks, MiniMax live TTS, and DingTalk delivery.

## 2026-05-12 Productization checkpoint: Live field demo runner

Implemented in this pass:

- Added `scripts/demo/live_field_operations_demo.py`, a customer demo runner that drives scenarios through the real HTTP API instead of only generating static evaluation JSON.
- Default mode uses an in-process FastAPI app and TestClient, so it exercises `/api/field/ingest`, `/api/field/events`, `/api/field/events/{id}/report`, `/api/field/devices`, and `/api/field/readiness` without requiring external services.
- `--server` mode can target an already running Askme deployment, letting the same scenario set run against a lab or customer site service.
- Demo scenarios now include signed device-style inputs for:
  - fire/smoke sensor alert,
  - illegal parking camera alert,
  - robot joint motor fault,
  - crowd gathering,
  - visitor wayfinding at a help point.
- The runner writes:
  - `live-field-demo.json` with HTTP status, accepted state, event ids, incident topic, voice text, notification group, runtime status, devices, readiness, and reports.
  - `live-field-demo.md` as a customer-readable demo guide.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\scenario_tests\test_live_field_operations_demo.py -q` -> 1 passed.
- `python scripts\demo\live_field_operations_demo.py --output-dir artifacts\field_operations\live-demo --site-profile deploy\site-profiles\park-demo.yaml` -> `passed accepted=5/5 mode=inprocess_http`.
- `python -m ruff check scripts\demo\live_field_operations_demo.py tests\scenario_tests\test_live_field_operations_demo.py` -> passed.
- `python -m pytest -p no:cacheprovider tests\scenario_tests\test_live_field_operations_demo.py tests\test_field_site_profile.py tests\test_field_deployment_readiness.py tests\test_cli.py::test_cli_runtime_field_readiness_reads_local_files tests\test_cli.py::test_cli_runtime_field_readiness_exits_nonzero_when_blocked -q` -> 16 passed.

Product reflection:

- This is a better customer demo primitive than static reports because it proves actual HTTP contracts, event archive, device trust, voice directive generation, and report generation are wired together.
- It is still not a final site acceptance test. Site acceptance must run the same script with `--server` against a deployed service and real camera/VMS, MQTT/smoke sensor, robot diagnostics, DingTalk, MiniMax voice, and runtime callback credentials.

Follow-up implemented after the live-demo smoke:

- Rebuilt `askme.pipeline.incident_alerts` with clean product Chinese for voice announcements, DingTalk messages, operator actions, and playbook status/checklists.
- Covered the key customer scenes: fall unrecoverable, immobilized/stuck robot, malicious blocking, joint motor fault, night stranger photo, illegal parking, fire/smoke, trash-bin full, crowd gathering, and urgent patrol dispatch.
- The live demo test now asserts the generated voice text contains real Chinese customer wording for smoke, parking, and wayfinding, so broken/mojibake event speech is no longer silently accepted.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\scenario_tests\test_live_field_operations_demo.py tests\test_field_operations.py::test_field_operations_http_endpoints tests\test_field_operations.py::test_sensor_ingest_triggers_fire_or_smoke tests\test_field_operations.py::test_domestic_camera_event_name_triggers_illegal_parking tests\test_field_operations.py::test_domestic_crowd_alarm_triggers_security_workflow -q` -> 5 passed.
- `python -m pytest -p no:cacheprovider tests\test_alert_dispatcher.py tests\scenario_tests\test_live_field_operations_demo.py tests\test_field_operations.py::test_field_operations_http_endpoints -q` -> 44 passed.
- `python -m pytest -p no:cacheprovider tests\scenario_tests\test_live_field_operations_demo.py tests\test_field_site_profile.py tests\test_field_deployment_readiness.py tests\test_cli.py::test_cli_runtime_field_readiness_reads_local_files tests\test_cli.py::test_cli_runtime_field_readiness_exits_nonzero_when_blocked -q` -> 16 passed.
- `python -m ruff check askme\pipeline\incident_alerts.py askme\pipeline\field_operations.py askme\pipeline\field_deployment_readiness.py askme\cli.py scripts\demo\live_field_operations_demo.py tests\scenario_tests\test_live_field_operations_demo.py tests\test_field_site_profile.py tests\test_field_deployment_readiness.py tests\test_cli.py tests\test_alert_dispatcher.py` -> passed.

## 2026-05-12 Productization checkpoint: Live field demo CLI entry

Implemented in this pass:

- Promoted the live customer scenario runner from a standalone script into the main product CLI as `askme runtime field-live-demo`.
- The command supports:
  - local in-process HTTP demo mode,
  - `--server` mode against an already deployed Askme service,
  - `--site-profile` for customer map/device/responder configuration,
  - `--output-dir` for demo artifacts,
  - `--json` for machine-readable CI/deployment use.
- CLI output now gives a human-readable acceptance summary: pass/fail, accepted scenario count, mode, readiness status, report paths, and per-scenario HTTP/event ids.
- Failed demo runs now exit nonzero, so this can be used as a deployment gate instead of a manually inspected script.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\test_cli.py::test_cli_runtime_field_live_demo_forwards_args tests\test_cli.py::test_cli_runtime_field_live_demo_exits_nonzero_when_failed tests\scenario_tests\test_live_field_operations_demo.py -q` -> 3 passed.
- `python -m ruff check askme\cli.py tests\test_cli.py scripts\demo\live_field_operations_demo.py tests\scenario_tests\test_live_field_operations_demo.py` -> passed.
- `python -m askme.cli runtime field-live-demo --output-dir artifacts\field_operations\live-demo-cli --site-profile deploy\site-profiles\park-demo.yaml` -> passed, accepted 5/5 scenarios, generated `artifacts\field_operations\live-demo-cli\live-field-demo.json` and `artifacts\field_operations\live-demo-cli\live-field-demo.md`.

Product reflection:

- This is now an operator-facing acceptance command, not just an engineering helper script.
- It still proves the software HTTP contract and workflow only. True site acceptance still requires running the same command with `--server` against a deployed service, real camera/sensor/robot bridges, real DingTalk credentials, real MiniMax voice playback, and signed runtime callbacks.

## 2026-05-12 Productization checkpoint: Live field demo customer HTML report

Implemented in this pass:

- The live field demo now writes `live-field-demo.html` alongside JSON and Markdown.
- The HTML report is written in customer-facing Chinese and summarizes:
  - pass/fail status,
  - accepted scenario count,
  - local software-loop vs deployed-service mode,
  - readiness status,
  - each customer scene,
  - event id,
  - notification target,
  - robot voice line,
  - real integration caveats.
- `askme runtime field-live-demo` now prints the HTML path, so a product/demo operator does not need to inspect JSON to find the customer report.
- The report explicitly says local mode is not hardware acceptance, preventing us from overstating what has been proven.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\scenario_tests\test_live_field_operations_demo.py tests\test_cli.py::test_cli_runtime_field_live_demo_forwards_args tests\test_cli.py::test_cli_runtime_field_live_demo_exits_nonzero_when_failed -q` -> 3 passed.
- `python -m ruff check scripts\demo\live_field_operations_demo.py askme\cli.py tests\scenario_tests\test_live_field_operations_demo.py tests\test_cli.py` -> passed.
- `python -m askme.cli runtime field-live-demo --output-dir artifacts\field_operations\live-demo-html --site-profile deploy\site-profiles\park-demo.yaml` -> passed, accepted 5/5, generated `live-field-demo.json`, `live-field-demo.md`, and `live-field-demo.html`.
- Manual UTF-8 read of `artifacts\field_operations\live-demo-html\live-field-demo.html` confirmed Chinese content renders correctly: `Askme 现场场景验收报告`, `火灾/烟雾异常`, `车辆违停`, and real integration caveats.

Product reflection:

- This closes a presentation gap: the same real HTTP demo now produces a customer-readable artifact instead of forcing the user to read command output or raw JSON.
- The next product gap is still external reality: run `field-live-demo --server` against an actual deployed service and attach real device bridge evidence, DingTalk delivery evidence, MiniMax voice playback evidence, and runtime callback evidence.

## 2026-05-12 Productization checkpoint: Customer scenario file replay

Implemented in this pass:

- `scripts/demo/live_field_operations_demo.py` and `askme runtime field-live-demo` now support `--scenario-file`.
- Scenario files can be either a JSON array or an object with a `scenarios` array.
- Each scenario can define:
  - `scenario_id`,
  - `customer_scene`,
  - HTTP `path` (`/api/field/ingest` or `/api/field/events`),
  - optional `device_secret` for signed device-style payloads,
  - `payload`.
- Scenario file loading accepts UTF-8 BOM via `utf-8-sig`, because Windows/customer-exported JSON often includes BOM.
- Added explicit `--refresh-scenario-timestamps` for demo replay. This refreshes `observed_at` only when requested, so stale captured samples can be replayed for demos without weakening the normal freshness gate.
- HTML reports now prioritize customer scene names for unknown custom scenario ids instead of exposing internal ids.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\scenario_tests\test_live_field_operations_demo.py tests\test_cli.py::test_cli_runtime_field_live_demo_forwards_args tests\test_cli.py::test_cli_runtime_field_live_demo_exits_nonzero_when_failed -q` -> 6 passed.
- `python -m ruff check scripts\demo\live_field_operations_demo.py askme\cli.py tests\scenario_tests\test_live_field_operations_demo.py tests\test_cli.py` -> passed.
- `python -m askme.cli runtime field-live-demo --output-dir artifacts\field_operations\live-demo-scenario-file --site-profile deploy\site-profiles\park-demo.yaml --scenario-file artifacts\field_operations\customer-scenarios-smoke.json --refresh-scenario-timestamps` -> passed, accepted 1/1, generated JSON/Markdown/HTML.
- Manual UTF-8 read of `artifacts\field_operations\live-demo-scenario-file\live-field-demo.html` confirmed it shows `客户烟感真实样本`, event id, 安保 notification target, and the smoke/high-temperature voice line.

Product reflection:

- This moves field acceptance closer to real deployment: customer/device samples can now be replayed through the same product command instead of requiring code changes or bespoke scripts.
- The safety posture remains explicit: without `--refresh-scenario-timestamps`, stale sensor evidence still fails freshness gates; with the flag, the report marks `refresh_scenario_timestamps: true` in JSON evidence.

## 2026-05-12 Productization checkpoint: Field ingest bridge product summary

Implemented in this pass:

- `run_field_ingest_bridge_once()` now returns a `summary` object for real JSON/JSONL device bridge runs.
- The summary includes:
  - processed event count,
  - posted count,
  - accepted count,
  - failed count,
  - signed count,
  - created event count,
  - business scenario counts when the server returns scenario ids,
  - source counts (`camera`, `sensor`, `robot`, etc.),
  - device counts,
  - source format,
  - offset/fingerprint evidence.
- `askme runtime field-ingest-bridge` now prints this summary in non-JSON output, including source/device distribution, so a现场交付人员 can verify a live JSONL bridge without opening raw JSON.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\test_field_ingest_bridge.py tests\test_cli.py::test_cli_runtime_field_ingest_bridge_forwards_args -q` -> 7 passed.
- `python -m ruff check askme\pipeline\field_ingest_bridge.py askme\cli.py tests\test_field_ingest_bridge.py tests\test_cli.py` -> passed.
- `python -m askme.cli runtime field-ingest-bridge tests\fixtures\field_devices\site-a-device-events.jsonl --state-path artifacts\field_operations\bridge-fixture-print.state.json --dry-run` -> ok, count 7, sources `camera:5, robot:1, sensor:1`, devices `bin-17:1, cam-main-road-01:1`.

Product reflection:

- This makes the常驻接入 bridge observable. The next step is to run the same bridge without `--dry-run` against a deployed service and require signed device secrets for production.

## 2026-05-12 Productization checkpoint: Bridge-to-HTTP smoke requires accepted events

Implemented in this pass:

- `field-ingest-smoke` now treats bridge summary as a hard gate.
- A smoke run only passes when the JSONL bridge:
  - processes 8 events,
  - posts 8 events,
  - receives 8 accepted responses,
  - creates 8 event ids,
  - covers required field scenarios,
  - records operator acknowledgement evidence.
- Human-readable smoke output now prints the bridge handoff summary: `posted`, `accepted`, and `events_created`.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\test_cli.py::test_cli_runtime_field_ingest_smoke_runs_local_http tests\test_field_ingest_bridge.py -q` -> 7 passed.
- `python -m ruff check askme\cli.py askme\pipeline\field_ingest_bridge.py tests\test_cli.py tests\test_field_ingest_bridge.py` -> passed.
- `python -m askme.cli runtime field-ingest-smoke --output-dir artifacts\field_operations\bridge-http-smoke-print` -> passed, events 8, bridge `posted=8 accepted=8 events_created=8`, scenarios include crowd gathering, smoke/fire, illegal parking, robot abnormal, trash bin.

Product reflection:

- This moves the bridge from "normalization utility" to "verifiable integration gate". It proves non-dry-run JSONL device data can create real field events through the HTTP API.
- The remaining production gap is signed device secrets and an external deployed service, not the local product workflow.

## 2026-05-12 Productization checkpoint: Signed device ingest smoke

Implemented in this pass:

- `askme runtime field-ingest-smoke` now supports `--require-device-signatures`.
- In signed mode, the temporary local server enables trusted-device admission and a device registry for sample camera, smoke sensor, robot, and trash-bin devices.
- The bridge signs all 8 sample events with configured device secrets before POSTing to `/api/field/ingest`.
- The smoke pass condition now requires `signed=8` when signed mode is enabled.
- Sample smoke events now carry stable device ids where needed, so signature and device-trust evidence can be audited per source.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\test_cli.py::test_cli_runtime_field_ingest_smoke_runs_local_http tests\test_cli.py::test_cli_runtime_field_ingest_smoke_can_require_device_signatures -q` -> 2 passed.
- `python -m ruff check askme\cli.py tests\test_cli.py` -> passed.
- `python -m askme.cli runtime field-ingest-smoke --output-dir artifacts\field_operations\signed-bridge-smoke --require-device-signatures` -> passed, events 8, bridge `posted=8 accepted=8 events_created=8 signed=8`.
- Manual JSON read of `artifacts\field_operations\signed-bridge-smoke\field-ingest-smoke.json` confirmed `require_device_signatures: true`, `signed: 8`, and every bridge result has `device_signing.reason == signed`.

Product reflection:

- This closes an important production-readiness gap inside the local product workflow: trusted-device signing is now part of the smoke gate, not just an optional bridge feature.
- Remaining production work is to configure real device secrets and run the same signed smoke against an external deployed service.

## 2026-05-12 Productization checkpoint: Deployed smoke can require signed device ingest

Implemented in this pass:

- `askme runtime field-deployed-smoke` now supports `--require-device-signatures`.
- The deployed smoke command passes signed-device requirements through to `field-ingest-smoke`.
- Deployed smoke checks now include `signed_device_ingest_smoke`, so a deployed validation can fail specifically when signed bridge evidence is missing.
- The deployed smoke report records `require_device_signatures`.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\test_cli.py::test_cli_runtime_field_deployed_smoke_runs_against_existing_server tests\test_cli.py::test_cli_runtime_field_deployed_smoke_blocks_when_notification_preflight_fails tests\test_cli.py::test_cli_runtime_field_deployed_smoke_command_forwards_args -q` -> 3 passed.
- `python -m ruff check askme\cli.py tests\test_cli.py` -> passed.

Product reflection:

- This gives customer/site acceptance one coherent command: deployed health, signed ingest, voice smoke, notification readiness/smoke, and readiness status.
- The next missing piece is real environment configuration: external service must have matching device registry secrets before `--require-device-signatures` can pass in production.

## 2026-05-12 Productization checkpoint: Every registered device must be signature-ready

Implemented in this pass:

- `build_field_deployment_readiness()` now distinguishes "some signed devices exist" from "all registered devices are production ready".
- Readiness payload now reports `unsigned_device_count`, `unsigned_device_ids`, `missing_secret_device_ids`, `signature_disabled_device_ids`, `signature_ready_device_ids`, and `all_registered_devices_signature_ready`.
- Device secrets that are still unresolved `${ENV_VAR}` placeholders now count as missing secrets, so demo profiles cannot be mistaken for production credentials.
- Production readiness now blocks when any registered field device has signatures disabled or lacks a signing secret.
- `askme runtime field-readiness` now prints a `device-trust` summary for delivery teams, so they can see registered/signed/unsigned counts without opening JSON.
- Dashboard field readiness now shows the customer-facing gate `全部注册设备签名就绪`.

Validation evidence:

- `python -m pytest -p no:cacheprovider tests\test_field_deployment_readiness.py tests\test_cli.py::test_cli_runtime_field_readiness_reads_local_files -q` -> 8 passed.
- `python -m ruff check askme\pipeline\field_deployment_readiness.py askme\cli.py tests\test_field_deployment_readiness.py tests\test_cli.py` -> passed.
- `python -m pytest -p no:cacheprovider tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_field_deployment_readiness.py tests\test_cli.py::test_cli_runtime_field_readiness_reads_local_files -q` -> 9 passed.
- `python -m ruff check askme\pipeline\field_deployment_readiness.py askme\cli.py tests\test_field_deployment_readiness.py tests\test_cli.py tests\test_health.py` -> passed.
- `python -m pytest -p no:cacheprovider tests\test_field_deployment_readiness.py tests\test_cli.py::test_cli_runtime_field_readiness_reads_local_files tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 10 passed.
- `python -m askme.cli runtime field-readiness --archive-path artifacts\field_operations\smoke\field-events.jsonl --scenario-report artifacts\field_operations\demo\scenario-evaluation.json --smoke-report artifacts\field_operations\smoke\field-ingest-smoke.json --voice-smoke-report artifacts\field_operations\smoke\field-voice-smoke.json --notification-smoke-report artifacts\field_operations\smoke\field-notification-smoke.json --site-profile deploy\site-profiles\park-demo.yaml` -> `ready_for_lab`, `device-trust: registered=4 signed=0 unsigned=4 all_ready=False` because site-profile device secrets are unresolved env placeholders on this machine.

Product reflection:

- This moves the gate from a developer-friendly smoke check to a deployable product rule: a site cannot be called production-ready if one legacy camera, smoke sensor, or robot diagnostic source is still unsigned.
- The remaining readiness blockers are real deployment evidence: DingTalk credentials, deployed service smoke, live TTS on target audio, signed runtime callback secret, real hardware dispatch, and external-service evidence.
# 2026-05-14 Update - Solution Provider Customer Project Layer

This update moves AskMe from a single demo-site robot assistant toward a repeatable solution-provider product for many customers, sites, and managed objects.

Delivered in this slice:

- Customer project and managed object catalog APIs now expose customer_id, project_id, site_id, industry, delivery stage, object categories, and object bindings.
- Industry starter templates were added for factory inspection, park visitor service, warehouse logistics, and scenic-area service.
- Customer projects can be created from a template, updated as a profile, imported from a customer project package, exported as a handoff package, and archived.
- Managed objects can be updated through API/UI and now carry declarative bindings for vision models, sensor protocols, skill packages, and acceptance tests.
- Dashboard delivery view now shows industry templates, customer project scope, multi-site rollout, export actions, and a managed-object quick edit form.
- Write operations require the `field:project:write` permission; default supervisor permissions include it.
- Product documentation records the current boundary: this is project/profile-level delivery separation, not full SaaS tenant isolation.

Still not complete:

- Full enterprise tenant isolation, SSO/IAM, project-scoped audit filters, and data-store namespaces.
- Runtime execution from managed-object bindings; current bindings are declarative references only.
- Signed customer project package manifest, package diff, and rollback workflow.
- Real sensor/vision/robot runtime matching against managed_object_id.
- Customer-facing template marketplace governance, publishing workflow, and version lifecycle.

Next product step:

1. Bind field ingest and event records to `customer_id`, `project_id`, `site_id`, and `managed_object_id`.
2. Add package manifest checksum/signature and import dry-run diff.
3. Add project-scoped audit and permission checks before runtime actions.
4. Promote industry templates into versioned delivery packages with acceptance-case coverage.
## 2026-05-14 Customer Project Delivery Update

- Managed object acceptance is no longer just a string list in YAML.
- `acceptance_status.acceptance_checks` now checks whether each configured acceptance reference points to local repository evidence.
- Missing acceptance files block customer acceptance.
- Existing files with unresolved nodes require manual review instead of being marked ready.
- Known scenario aliases such as `illegal_parking`, `fire_or_smoke`, `trash_bin_full`, `visitor_wayfinding`, and `visitor_escort` resolve to deterministic scenario or pytest evidence.
- `/dashboard/projects` renders per-object acceptance gates so delivery can see linked, manual, and blocked evidence from the product UI.
- Customer project export packages now carry `acceptance_summary`, and their manifest records overall/ready/manual/blocked acceptance counts.
- Customer project import dry-run now reports incoming/current acceptance summaries so delivery can judge whether a copied project package is ready for customer signoff.
- Customer project packages now have standalone `package/verify` and `package/diff` APIs so delivery tooling can validate a handoff package before import.
- `/dashboard/projects` now exposes a customer-readable acceptance report action for each project.
- Acceptance reports split the status into site-profile validity, managed-object acceptance evidence, deployment credentials, and onsite acceptance boundary.
- Acceptance reports now also attach compact field-readiness evidence: scenario evaluation, ingest smoke, voice smoke, notification smoke, runtime roundtrip, audit, and device-trust gates.
- Current demo project status is intentionally `blocked`: local managed-object evidence is ready, but field readiness still has an unresolved high-risk audit review plus missing real DingTalk/device/live-service evidence.
- Customer projects can now export an acceptance dossier JSON handoff file and a printable HTML dossier containing the acceptance report, evidence inventory, per-file SHA-256 hashes, and optional HMAC signature metadata.
- Customer projects now have an acceptance registry API and Dashboard card. It scans project profiles and industry templates, groups every managed-object acceptance reference, and shows linked/manual/blocked evidence counts before customer signoff.
- Customer project write operations are now guarded by operator project scope in addition to `field:project:write`. A scoped supervisor cannot create, import, archive, or mutate managed objects outside their assigned customer/project/site.
- Field event read/write actions now share the same project-scope boundary: out-of-scope operators cannot read event detail/report or acknowledge, request close, close, or resend notifications.
- Device ingest now ignores client-supplied project ownership fields and applies the server/site-profile scope, preventing camera/sensor/robot payloads from reassigning themselves across customer projects.
- Remaining gap: this proves local delivery evidence is wired; it does not prove physical cameras, sensors, DingTalk production robots, MiniMax audio, or robot runtime have passed onsite acceptance.
- Next product step: add browser/PDF export from the printable dossier and replace local smoke paths with live onsite smoke outputs.

## 2026-05-14 Customer Project Namespace Guardrail

This pass closes a product-delivery risk for solution providers: two customer
spaces may use the same `customer_id`, `project_id`, or `site_id`, but importing
one handoff package must not overwrite the other.

Delivered:

- Customer project profiles now normalize `customer.tenant_id` and
  `customer.delivery_namespace`, defaulting legacy projects to `default`.
- Non-default customer project profiles are written under a scoped path:
  `{tenant_id}/{delivery_namespace}/{customer_id}/{project_id}.yaml`.
- Customer project catalogs, field-operation config, package manifests, and
  acceptance dossier manifests now expose `tenant_id` and
  `delivery_namespace`.
- Package verification rejects manifest scope tampering for tenant,
  namespace, customer, project, or site fields.
- Import/diff now matches an existing target only when tenant, delivery
  namespace, customer, and project/site identity match.
- Import/diff reports same customer/project collisions in other namespaces
  instead of overwriting them.
- API project-scope enforcement now reads `tenant_ids` and
  `delivery_namespaces` from demo operators or trusted IAM headers.
- The default dashboard operator is explicitly scoped to the demo
  `default/default` customer delivery space.
- `/dashboard/projects` now makes the delivery scope visible in project cards,
  export results, and import dry-run results.
- Import dry-run now renders package collision candidates so delivery teams can
  see same-name projects in other namespaces before writing anything.
- The industry-template create form now exposes `tenant_id` and
  `delivery_namespace`, and sends both into the project-create API. Delivery can
  create pilot/lab/prod customer spaces from the UI instead of editing scoped
  YAML paths by hand.
- `/dashboard/projects` now includes a Project Metadata Editor. It loads a full
  customer project profile, edits customer-facing metadata, and writes the full
  profile back through the existing scoped upsert API without dropping devices,
  zones, responder groups, or managed objects.
- `/dashboard/projects` now includes a Managed Object Directory. It shows each
  customer-visible object with project scope, responder group, model/protocol/
  skill/test bindings, and a `Load into editor` action that pre-fills the
  managed-object editor.
- Managed-object removal is now an explicit offline lifecycle action. The API
  rejects removals without a customer-visible reason, returns the deleted object
  snapshot and `offline_reason`, and the Dashboard shows impact before removal.
- The managed-object editor is now grouped into basic object identity,
  detection scope, runtime bindings, and acceptance evidence. The field ids stay
  stable, so existing save logic and tests continue to apply.
- Managed-object create/update/offline actions now append `object_change_log`
  records to the customer project profile and surface recent changes in
  `/dashboard/projects`.
- Dashboard GET requests now send the selected `X-Askme-Operator-Id` header, so
  read paths and write paths use the same operator/project-scope identity.
- Industry templates are now closer to a product template market. The template
  API returns `delivery_summary` and `delivery_checklist` fields with default
  managed objects, scenario scope, runtime bindings, acceptance references, and
  rollout steps. `/dashboard/projects` shows these as customer-readable cards
  and lets delivery select a template into the project creation form.
- Customer project catalog/detail payloads now include `delivery_workflow`.
  Each project gets explicit handoff steps for customer scope, managed objects,
  runtime bindings, site map/devices, responder credentials, acceptance
  evidence, and package export. `/dashboard/projects` renders this workflow in
  each project card so the product shows blockers and next actions directly.
- Acceptance reports and acceptance dossiers now include the same
  `delivery_workflow`; the printable HTML dossier renders a Delivery Workflow
  table, so customer handoff artifacts match the Dashboard state.
- Unified audit is now project scoped. `AuditQueryService` indexes and filters
  tenant, delivery namespace, customer, project, site, and managed object ids.
  `/api/audit/events` and `/api/audit/export` apply the operator's project
  scope before returning records or building evidence packages.
- The delivery Dashboard audit panel now lets operators filter by customer
  project and managed object. Audit export uses the same selected scope, so a
  customer acceptance package does not accidentally include another customer's
  field or runtime records.
- Customer project packages are now reusable handoff packages, not just YAML
  copies. Export adds `package_schema`, `reuse_assessment`, and
  `deployment_dependencies`; package manifest records reuse status and
  dependency counts; verify rejects reuse-status tamper; import dry-run shows
  incoming/current reuse readiness so delivery can see whether a project can be
  copied to another customer, needs onsite manual checks, or is blocked.
- `/dashboard/projects` now renders reuse readiness cards for export and
  import dry-run results. The UI calls out dependencies such as device sources,
  vision models, sensor protocols, skill packages, and missing live environment
  bindings instead of forcing delivery to inspect raw JSON.
- Customer project changes now have a rollback lane. Before template overwrite,
  profile upsert, package import, managed-object edit/delete, or rollback, the
  system writes a JSON revision snapshot under `_revisions` so snapshots are not
  mistaken for active site profiles. `/history` lists saved revisions and
  `/rollback` supports dry-run/apply with the same project-scope authorization
  as other write endpoints.
- `/dashboard/projects` now exposes View Revisions, Rollback Dry-run, and
  Rollback actions in the project lifecycle panel, turning mistaken onsite
  configuration edits into a recoverable product operation.
- Industry templates are now versioned product packages. The four starter
  templates declare semantic version, pilot publish status, release channel,
  owner, upgrade policy, and minimum runtime metadata. The template API returns
  `template_package` with product status, blockers/manual checks, dependency
  counts, and a source hash; `/dashboard/projects` renders that readiness in
  the template market before delivery creates a customer project.
- Industry templates now have release governance operations. Product/delivery
  owners can use `/api/field/customer-project-templates/{template_id}/release`
  to mark a template as pilot, published, deprecated, or blocked, and
  `/history` to review prior release snapshots. The Dashboard template market
  exposes Publish, Mark pilot, Deprecate, and Release history actions.
- Template release routes now use product-owner permissions instead of generic
  project write permission. Non-published changes require
  `template:release:write`; published promotion requires
  `template:release:approve`.
- Published template promotion now requires a release request and second
  approver. Product owners create requests through
  `/api/field/customer-project-templates/{template_id}/release-requests`; a
  different product owner reviews them through
  `/api/field/customer-project-template-release-requests/{request_id}/review`.
  Direct `published` writes return `published_release_requires_approval_request`.
  The Dashboard template market now exposes Request publish, Release requests,
  Approve, and Reject controls.
- Approved published template requests now feed
  `/api/field/customer-project-template-release-notes` and the Dashboard
  Template Release Notes card. This is the customer-facing allowlist for what
  sales and delivery can claim as an approved reusable template package.
- Release notes can now be exported as a proposal/handoff bundle through
  `/api/field/customer-project-template-release-notes/export`. The bundle
  includes customer/project context, a manifest hash, JSON payload, and printable
  HTML so sales and delivery can attach only approved published template
  packages to customer-facing material.
- `/dashboard/projects` now turns that bundle into browser downloads: clicking
  Export proposal bundle downloads both the JSON bundle and printable HTML, so
  the feature is usable by sales or delivery without inspecting raw API output.
- The release-note bundle now includes a controlled `proposal_insert` section:
  customer message, approved template ids, safe claims, and delivery boundaries.
  This gives sales a reusable proposal block without allowing unsupported
  production claims.
- Customer projects now have a proposal bundle export:
  `/api/field/customer-projects/{identifier}/proposal-bundle` combines the
  customer project handoff package, acceptance dossier, approved template release
  notes, `proposal_insert`, and delivery boundaries into JSON + printable HTML.
  `/dashboard/projects` exposes this as Proposal / Proposal Bundle and downloads
  both files in the browser.
- Proposal bundles now have a verification gate:
  `/api/field/customer-projects/proposal-bundle/verify` recomputes the proposal
  manifest hash, checks tenant and delivery namespace scope, verifies the package,
  dossier, and release-note hashes, and rejects changed delivery boundaries.
  `/dashboard/projects` can paste or auto-load a proposal JSON and show a
  customer-readable verification result before delivery uses it.
- Customer projects now have an onsite acceptance evidence registry:
  `/api/field/customer-projects/{identifier}/onsite-evidence` lists and registers
  receipts for device ingest, voice playback, notification delivery, runtime
  roundtrip, and customer review. Registration snapshots the profile revision
  before writing. Acceptance reports now turn the onsite boundary into a real gate
  (`4/4 required onsite evidence receipts passed`), and acceptance dossiers carry
  onsite status/count fields plus per-file evidence hashes in the inventory.
- Customer projects now also have a closure layer:
  `/api/field/customer-projects/{identifier}/acceptance-closure` summarizes the
  customer-facing status across acceptance report, onsite evidence, and manual
  delivery-owner review. `/api/field/customer-projects/{identifier}/acceptance-review`
  writes a revisioned review decision and requires risk acknowledgement before
  `accepted` is allowed.
- `/dashboard/projects` now exposes a small onsite evidence form in the lifecycle
  panel plus an acceptance-closure review form, so delivery can register
  customer-visible proof and submit a controlled review without editing YAML.
- Acceptance closure now verifies the customer acceptance dossier in memory and
  looks for the latest matching proposal bundle and scoped audit export. The
  closure payload exposes `artifact_verification.acceptance_dossier`,
  `artifact_verification.proposal_bundle`, and `artifact_verification.audit_export`
  so delivery can see whether the customer handoff is complete from one screen.
- Acceptance dossiers now have a write-free verification endpoint:
  `/api/field/customer-projects/acceptance-dossier/verify`. `/dashboard/projects`
  adds a "验签验收包" action next to proposal verification, so exported or pasted
  customer handoff material can be validated without using curl or editing files.
- Negative acceptance coverage now locks the critical failure cases: unsupported
  onsite evidence type/status does not write to the profile, the latest failed
  required onsite receipt blocks acceptance, a later recovered receipt can restore
  readiness, tampered dossier payloads are rejected, signed dossiers require the
  configured HMAC secret, and out-of-scope operators cannot verify dossiers.

Next product step:

- Automate onsite evidence ingestion from real camera/sensor/voice/runtime
  callbacks instead of relying on manual delivery receipts.
- Turn proposal and acceptance HTML into formal PDF/browser export with customer
  branding, signature blocks, and versioned cover pages.

Validation evidence:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py -q` -> 29 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_field_customer_project_write_endpoints_enforce_project_scope tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 4 passed, 1 existing httpx deprecation warning.
- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py -q` -> 26 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_field_customer_project_write_endpoints_enforce_project_scope tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 3 passed, 1 existing httpx deprecation warning.
- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py -q` -> 25 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 2 passed, 1 existing httpx deprecation warning.
- `pytest tests\test_governance.py tests\test_health.py::TestHealthServer::test_governance_operator_directory_exposes_demo_boundary tests\test_health.py::TestHealthServer::test_governance_current_operator_resolves_permissions tests\test_health.py::TestHealthServer::test_field_customer_project_read_endpoints_require_known_operator tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 10 passed, 1 existing httpx deprecation warning.
- `pytest tests\test_field_operations.py::test_field_event_write_endpoints_enforce_project_scope tests\test_field_operations.py::test_device_ingest_ignores_client_supplied_project_scope -q` -> 2 passed.
- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py::test_customer_project_proposal_bundle_binds_package_dossier_and_release_notes tests\test_field_site_profile.py::test_customer_project_template_release_request_requires_second_approver -q` -> 2 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 2 passed, 1 existing httpx deprecation warning.
- `pytest tests\test_field_site_profile.py -q` -> 25 passed.
- `pytest tests\test_governance.py tests\test_health.py::TestHealthServer::test_governance_operator_directory_exposes_demo_boundary tests\test_health.py::TestHealthServer::test_governance_current_operator_resolves_permissions tests\test_health.py::TestHealthServer::test_field_customer_project_read_endpoints_require_known_operator tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 10 passed, 1 existing httpx deprecation warning.
- `git diff --check -- askme\governance.py askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\static\dashboard\app.js config.yaml tests\test_field_site_profile.py tests\test_health.py docs\PRODUCT.md plans\plan.md` -> no whitespace errors; Git reported LF/CRLF normalization warnings.
- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py::test_customer_project_template_release_request_requires_second_approver -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 2 passed, 1 existing httpx deprecation warning.
- `pytest tests\test_field_site_profile.py -q` -> 24 passed.
- `pytest tests\test_governance.py tests\test_health.py::TestHealthServer::test_governance_operator_directory_exposes_demo_boundary tests\test_health.py::TestHealthServer::test_governance_current_operator_resolves_permissions tests\test_health.py::TestHealthServer::test_field_customer_project_read_endpoints_require_known_operator tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 10 passed, 1 existing httpx deprecation warning.
- `git diff --check -- askme\governance.py askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\static\dashboard\app.js config.yaml tests\test_field_site_profile.py tests\test_health.py docs\PRODUCT.md plans\plan.md` -> no whitespace errors; Git reported LF/CRLF normalization warnings.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py` -> passed.
- `pytest tests\test_field_site_profile.py::test_customer_project_template_release_request_requires_second_approver -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 2 passed, 1 existing httpx deprecation warning.
- `pytest tests\test_field_site_profile.py -q` -> 24 passed.
- `pytest tests\test_governance.py tests\test_health.py::TestHealthServer::test_governance_operator_directory_exposes_demo_boundary tests\test_health.py::TestHealthServer::test_governance_current_operator_resolves_permissions tests\test_health.py::TestHealthServer::test_field_customer_project_read_endpoints_require_known_operator tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 10 passed, 1 existing httpx deprecation warning.
- `git diff --check -- askme\governance.py askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\static\dashboard\app.js config.yaml tests\test_field_site_profile.py tests\test_health.py docs\PRODUCT.md plans\plan.md` -> no whitespace errors; Git reported LF/CRLF normalization warnings.
- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py::test_customer_project_template_release_request_requires_second_approver -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 2 passed, 1 existing httpx deprecation warning.
- `pytest tests\test_field_site_profile.py -q` -> 24 passed.
- `pytest tests\test_governance.py tests\test_health.py::TestHealthServer::test_governance_operator_directory_exposes_demo_boundary tests\test_health.py::TestHealthServer::test_governance_current_operator_resolves_permissions tests\test_health.py::TestHealthServer::test_field_customer_project_read_endpoints_require_known_operator tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 10 passed, 1 existing httpx deprecation warning.
- `git diff --check -- askme\governance.py askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\static\dashboard\app.js config.yaml tests\test_field_site_profile.py tests\test_health.py docs\PRODUCT.md plans\plan.md` -> no whitespace errors; Git reported LF/CRLF normalization warnings.
- `python -m py_compile askme\pipeline\field_site_profile.py tests\test_field_site_profile.py` -> passed.
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py` -> passed.
- `pytest tests\test_field_site_profile.py -q` -> 21 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_field_customer_project_write_endpoints_enforce_project_scope tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints -q` -> 3 passed.
- `pytest tests\test_governance.py tests\test_health.py::TestHealthServer::test_governance_operator_directory_exposes_demo_boundary tests\test_health.py::TestHealthServer::test_governance_current_operator_resolves_permissions tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace -q` -> 7 passed.
- `pytest tests\test_field_operations.py tests\test_field_site_profile.py tests\test_health.py -q` -> 149 passed, 1 warning.
- `pytest tests\test_field_deployment_readiness.py -q` -> 11 passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `python -m py_compile askme\governance.py askme\api\routes\field.py askme\pipeline\field_site_profile.py` -> passed.
- `ruff check askme\governance.py askme\api\routes\field.py askme\pipeline\field_site_profile.py tests\test_health.py tests\test_field_site_profile.py` -> passed.
- `pytest tests\test_field_site_profile.py -q` -> 23 passed.
- `pytest tests\test_governance.py tests\test_health.py::TestHealthServer::test_governance_operator_directory_exposes_demo_boundary tests\test_health.py::TestHealthServer::test_governance_current_operator_resolves_permissions tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints -q` -> 7 passed, 1 existing httpx deprecation warning.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_read_endpoints_require_known_operator tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 4 passed, 1 existing httpx deprecation warning.
- `pytest tests\test_field_site_profile.py -q` -> 24 passed.
- `pytest tests\test_governance.py tests\test_health.py::TestHealthServer::test_governance_operator_directory_exposes_demo_boundary tests\test_health.py::TestHealthServer::test_governance_current_operator_resolves_permissions tests\test_health.py::TestHealthServer::test_field_customer_project_read_endpoints_require_known_operator tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 10 passed, 1 existing httpx deprecation warning.
- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py -q` -> 24 passed.
- `pytest tests\test_governance.py tests\test_health.py::TestHealthServer::test_governance_operator_directory_exposes_demo_boundary tests\test_health.py::TestHealthServer::test_governance_current_operator_resolves_permissions tests\test_health.py::TestHealthServer::test_field_customer_project_read_endpoints_require_known_operator tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 10 passed, 1 existing httpx deprecation warning.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_projects_endpoint_returns_solution_scope tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 2 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace -q` -> 2 passed, 1 warning.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_health.py::TestHealthServer::test_field_customer_projects_endpoint_returns_solution_scope tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace -q` -> 3 passed.
- `ruff check tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py::test_customer_project_profile_upsert_updates_metadata_without_losing_objects tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 2 passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_health.py::TestHealthServer::test_field_customer_project_read_endpoints_require_known_operator tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace -q` -> 3 passed.
- `pytest tests\test_field_site_profile.py::test_customer_project_template_create_update_export_import_and_archive tests\test_field_site_profile.py::test_customer_project_profile_upsert_updates_metadata_without_losing_objects -q` -> 2 passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `python -m py_compile askme\audit\query.py askme\audit\export.py askme\api\routes\audit.py` -> passed.
- `ruff check askme\audit\query.py askme\audit\export.py askme\api\routes\audit.py tests\test_audit_query.py` -> passed.
- `pytest tests\test_audit_query.py -q` -> 33 passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `pytest tests\test_field_site_profile.py::test_customer_project_template_create_update_export_import_and_archive tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 2 passed.
- `pytest tests\test_field_site_profile.py::test_customer_project_template_create_update_export_import_and_archive -q` -> 1 passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `python -m py_compile askme\pipeline\field_site_profile.py askme\audit\query.py askme\audit\export.py askme\api\routes\audit.py` -> passed.
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py askme\audit\query.py askme\audit\export.py askme\api\routes\audit.py tests\test_audit_query.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `pytest tests\test_field_site_profile.py tests\test_audit_query.py tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints -q` -> 57 passed, 1 warning.
- `git diff --check -- askme\pipeline\field_site_profile.py tests\test_field_site_profile.py askme\static\dashboard\app.js askme\static\dashboard\app.css docs\PRODUCT.md plans\plan.md askme\audit\query.py askme\audit\export.py askme\api\routes\audit.py tests\test_audit_query.py` -> no whitespace errors; Git reported existing LF/CRLF normalization warnings.
- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py tests\test_health.py::TestHealthServer::test_field_customer_project_read_endpoints_require_known_operator tests\test_health.py::TestHealthServer::test_field_customer_project_write_endpoints_enforce_project_scope tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 26 passed, 1 warning.
- `git diff --check -- askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py askme\static\dashboard\app.js docs\PRODUCT.md plans\plan.md` -> no whitespace errors; Git reported LF/CRLF normalization warnings.
- `python -m py_compile askme\pipeline\field_site_profile.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 24 passed, 1 warning.
- `git diff --check -- askme\pipeline\field_site_profile.py askme\static\dashboard\app.js tests\test_field_site_profile.py tests\test_health.py deploy\customer-project-templates\factory-inspection.yaml deploy\customer-project-templates\park-visitor-service.yaml deploy\customer-project-templates\scenic-area-service.yaml deploy\customer-project-templates\warehouse-logistics.yaml docs\PRODUCT.md plans\plan.md` -> no whitespace errors; Git reported LF/CRLF normalization warnings.

Remaining gap:

- This is still product/package-level isolation. Full enterprise tenant
  isolation still requires SSO/IAM-backed claims, data-store namespace
  enforcement, row-level storage policies, and customer-managed key strategy.

## 2026-05-15 Productization checkpoint: Readiness evidence can satisfy onsite acceptance when it is real

This checkpoint moves customer project acceptance from manual receipt-only
tracking toward evidence-driven delivery closure.

Implemented:

- `customer_project_acceptance_report()` now combines manually registered onsite
  receipts with deterministic read-only receipts auto-surfaced from field
  readiness reports.
- Auto receipts are created only for real-link evidence:
  - trusted device events plus real hardware flags and non-local ingest smoke;
  - live TTS plus non-local voice smoke;
  - external notification services plus non-local notification smoke;
  - trusted non-local runtime callbacks with a verified final status.
- Local server, mock, recorded, missing, or unverifiable evidence does not
  satisfy the onsite acceptance boundary.
- Auto receipts are read-only: they appear in reports and exported acceptance
  dossiers, but they do not write back to the customer project YAML or replace
  the manual onsite evidence registry.
- Dashboard receipt rows now label auto-surfaced readiness evidence separately
  from manually registered delivery receipts, including evidence type, SHA-256
  prefix, and external reference/path.
- The onsite-evidence endpoint now shares the same default evidence view as the
  acceptance report. `include_readiness_auto=false` remains available when a
  caller needs the raw manual registry only.
- Acceptance reports and exported dossiers now include `site_acceptance_checklist`,
  a customer-delivery checklist across site profile, managed-object bindings,
  deployment credentials, device ingest, live voice, external notification,
  runtime roundtrip, and audit/operator review. Dashboard renders the checklist
  with customer-readable owner/status/next-step cards.
- Acceptance closure now includes the checklist as a gate, and acceptance dossier
  manifests expose checklist status plus ready/manual/blocked counts.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py`
- `node --check askme\static\dashboard\app.js`
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py`
- `pytest tests\test_field_site_profile.py -q` -> 31 passed
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_field_customer_project_write_endpoints_enforce_project_scope tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 4 passed, 1 existing httpx deprecation warning

Remaining:

- Run the same path against a real deployed AskMe service, MiniMax voice,
  DingTalk, signed device samples, and runtime callbacks before claiming final
  customer-site acceptance.

## 2026-05-15 Productization checkpoint: Managed objects now have a delivery resource catalog

This checkpoint strengthens the solution-provider layer for many customers and
many customer object types. A managed object is no longer just a YAML block with
free-form strings; the project can now show whether each object is bound to
known product resources.

Implemented:

- Added a delivery resource catalog for managed-object bindings:
  `vision_models`, `sensor_protocols`, `skill_packages`, and
  `acceptance_tests`.
- Each managed object now carries `resource_binding_status`, including linked,
  manual-check, blocked, and unregistered resource counts.
- Customer project catalogs now include `resource_catalog_summary` and
  `binding_readiness_summary` so delivery can see whether all objects have
  usable product resources.
- Added `GET /api/field/customer-project-resource-catalog`, scoped by the same
  customer/project/tenant rules as the rest of the customer project API.
- Dashboard `/dashboard/projects` now renders a delivery resource catalog card
  and shows per-object resource readiness beside acceptance readiness.
- Customer project export/import/diff/verify now carries the same resource
  binding readiness in the package manifest. Dashboard import and export
  results show incoming/current resource readiness, unregistered resources, and
  catalog resource counts before a delivery team imports a customer package.
  Unregistered resource details include object id, resource type, and resource
  id so the delivery owner can decide whether to register a project extension
  or block reuse.
- Customer project import/export/verify result panels now use readable product
  copy for the paths a customer or delivery engineer sees during package
  preflight, proposal verification, dossier verification, and export.
- Acceptance report, onsite evidence, acceptance closure, acceptance dossier,
  and onsite checklist panels now also render readable customer-facing copy on
  the reachable Dashboard path.
- Dashboard source was scanned for common legacy mojibake markers across the
  customer-facing static app and the targeted health tests. The remaining
  customer-facing project, field event, package, dossier, and acceptance paths
  now have readable copy covered by static assertions.
- Tests cover backend resource registry output, per-object resource readiness,
  package tamper detection, read-permission enforcement, the API endpoint, and
  Dashboard static wiring.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py -q` -> 32 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_field_customer_project_write_endpoints_enforce_project_scope tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 4 passed, 1 existing httpx deprecation warning.

Remaining:

- Resource catalog is still local/product-config based. Production should bind
  it to the real model registry, sensor protocol registry, skill package
  registry, and acceptance execution service.
- Before customer demonstration, still run browser visual QA against the live
  Dashboard and capture screenshots for `/dashboard`, `/dashboard/field`,
  `/dashboard/projects`, `/dashboard/knowledge`, and `/dashboard/conversation`.

## 2026-05-15 Productization checkpoint: Customer project CRUD now edits the full object contract

This checkpoint closes two product-facing gaps in the solution-provider project
console. The goal is to let delivery teams configure a customer's real managed
objects from the Dashboard without falling back to YAML edits or hidden defaults.

Implemented:

- `/dashboard/projects` managed-object quick edit now exposes object labels,
  responder group, required evidence, vision models, sensor protocols, skill
  packages, and acceptance-test references.
- Loading an existing managed object now hydrates those full bindings back into
  the editor, so delivery can review and update the same contract that export,
  import, resource catalog, and acceptance checks use.
- Saving an object now submits explicit `vision_models` and `sensor_protocols`.
  The UI no longer saves empty vision bindings or derives sensor protocols from
  `device_sources`.
- Acceptance review now accepts comma-separated `evidence_refs`, so a delivery
  lead can bind the review decision to onsite receipts, reports, dossiers, or
  audit/export evidence instead of sending an empty review trail.
- Dashboard static regression tests now lock these fields and request payloads.

Validation:

- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check tests\test_health.py` -> passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `pytest tests\test_field_site_profile.py -q` -> 33 passed.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed. Screenshots:
  `askme-dashboard-desktop.png`, `askme-dashboard-mobile.png`,
  `askme-dashboard-field-interaction.png`,
  `askme-dashboard-capability-readiness.png`, and
  `askme-dashboard-audit-dossier.png`.
- `git diff --check -- askme\static\dashboard\app.js tests\test_health.py docs\PRODUCT.md plans\plan.md` -> no whitespace errors; Git reported LF/CRLF normalization warnings.
- `scripts/check_text_encoding.py` and the dashboard/static string checks -> no new mojibake markers.

Remaining:

- Evidence references are entered manually today. A better customer-facing flow
  should let the user select receipts/reports from the onsite evidence list.
- The project page still needs browser visual QA and likely information
  architecture cleanup before it is shown as a polished customer demo.

## 2026-05-15 Productization checkpoint: Customer project workspace navigation and artifact separation

This checkpoint improves the solution-provider handoff console from a long mixed
control page into a more readable customer-project workspace.

Implemented:

- `/dashboard/projects` now has a sticky workspace navigation with anchors for
  project catalog, template market, managed-object directory, package import,
  acceptance evidence, resource bindings, template release governance, event
  scope, and multi-site rollout.
- `/dashboard/projects` now has a template release governance board. It shows
  pending, approved, rejected, and total release requests, separates pending
  second-approver reviews from recently reviewed requests, and wires approve or
  reject actions from the central board.
- Project package import is now separated from proposal bundle verification and
  acceptance dossier verification. Each artifact has its own text area and
  purpose copy, so delivery users do not paste customer-facing proposal material
  into the same box used for project imports.
- Proposal bundle export now fills the proposal verifier input; acceptance
  dossier export now fills the dossier verifier input. Project package export
  still fills only the project handoff package input.
- Acceptance review now has an onsite-evidence picker. Loading an acceptance
  report or onsite evidence receipts populates selectable `onsite:<receipt_id>`
  references, and the UI appends the selected references into the review payload
  without requiring manual ID copying.
- Error copy for parsing and verifier failures now uses one reachable Chinese
  message per path instead of duplicated unreachable branches.
- Dashboard static tests now lock the new navigation anchors and separate
  artifact verifier inputs, the evidence-reference picker, and the template
  release governance board.

Validation:

- `node --check askme\static\dashboard\app.js` -> passed.
- `python -m py_compile scripts\eval\check_dashboard_visual.py` -> passed.
- `ruff check scripts\eval\check_dashboard_visual.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed. The visual smoke now includes `/dashboard/projects` and writes
  `askme-dashboard-projects-workspace.png`; it also verifies no horizontal
  overflow after long resource/test-path wrapping.

## 2026-05-15 Productization checkpoint: Delivery resources can be registered from the project workspace

This checkpoint moves customer-project resource binding into a reusable
solution-provider resource layer. The resource catalog was already computed and
shown, but delivery users still had to edit YAML or duplicate resource metadata
inside one customer project. The project workspace now writes a shared Delivery
Resource Registry that can be reused across customer projects when scope allows.

Implemented:

- `/dashboard/projects` resource section now includes a Delivery Resource
  Registry form.
- The registry lets a delivery user choose a customer project, resource type,
  resource ID, display name, version, owner, source, and description.
- `GET /api/field/delivery-resource-registry` exposes the shared registry under
  `deploy/delivery-resources`.
- `POST /api/field/delivery-resource-registry` registers one shared resource
  with owner/version/scope metadata and snapshots the previous registry before
  overwrite.
- The project resource catalog now merges built-in resources, the shared
  registry, and project-level `delivery_resources`; project-level resources
  remain available as customer-specific overrides.
- Registered shared resources are immediately available to the existing resource
  catalog and managed-object binding checks.
- Scoped operators cannot read with missing permissions or register resources
  outside their tenant/customer/project/site scope.
- Managed Object Quick Edit now includes a registered-resource picker. Selecting
  a resource appends its ID to the correct binding field (`vision_models`,
  `sensor_protocols`, `skill_packages`, or `acceptance_tests`) before the normal
  managed-object save action persists the object.
- The resource section now renders a resource binding action plan. It either
  lists unresolved string-only resources that must be registered, or tells the
  delivery team that the visible scope is ready to continue to package export,
  onsite evidence, and acceptance review.
- Dashboard static and visual smoke checks now assert that the registry is
  present on the customer-project workspace and that the object binding picker
  is visible.

Validation:

- `node --check askme\static\dashboard\app.js` -> passed.
- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py` -> passed.
- `python -m py_compile scripts\eval\check_dashboard_visual.py` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py scripts\eval\check_dashboard_visual.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `pytest tests\test_field_site_profile.py::test_profile_delivery_resources_register_project_specific_bindings tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 2 passed.
- `pytest tests\test_field_site_profile.py::test_shared_delivery_resource_registry_resolves_object_bindings tests\test_field_site_profile.py::test_profile_delivery_resources_register_project_specific_bindings -q` -> 2 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_read_endpoints_require_known_operator tests\test_health.py::TestHealthServer::test_field_customer_project_write_endpoints_enforce_project_scope -q` -> 2 passed.
- `pytest tests\test_field_site_profile.py -q` -> 35 passed.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed after adding both registry and object binding picker. The project workspace visual smoke now asserts
  `has_delivery_resource_registry: true`, `has_resource_action_plan: true`,
  `has_object_resource_picker: true`, `object_resource_binding_added: true`,
  and no horizontal overflow.

Remaining:

- The shared registry is still YAML-backed and file-scoped. Production should
  eventually connect it to a database-backed resource service with ownership
  workflow and stricter SaaS tenant isolation.
- The Dashboard registration smoke still uses mocked browser data; the next
  step is to exercise a real POST registration in the visual smoke without
  mutating production delivery resources.

## 2026-05-15 Productization checkpoint: Delivery resource governance now blocks unsafe customer bindings

This checkpoint upgrades the shared Delivery Resource Registry from simple
registration into a product governance surface. Solution teams can now audit
registry history, disable a resource, and dry-run or apply rollback before a
customer project exports a package.

Implemented:

- `publish_status` is now validated for shared resources. Supported values are
  `draft`, `pilot`, `published`, `deprecated`, `disabled`, and `blocked`.
- `GET /api/field/delivery-resource-registry/history` exposes registry revision
  history for unrestricted delivery operators.
- `POST /api/field/delivery-resource-registry/{resource_type}/{resource_id}/disable`
  marks a shared resource as disabled and snapshots the previous registry.
- `POST /api/field/delivery-resource-registry/rollback` supports dry-run and
  apply rollback to a previous revision. Scoped project operators are rejected
  because rollback can affect multiple customers.
- Managed-object binding readiness now treats disabled or blocked shared
  resources as blocking failures. Draft, pilot, and deprecated resources require
  delivery review. This applies to vision models, sensor protocols, skill
  packages, and acceptance tests.
- `/dashboard/projects` now renders a resource governance panel with history,
  disable, and rollback actions. The visible resource list also shows publish
  status and a disable control for shared resources.
- Customer-visible Chinese copy in the resource section was corrected so the
  page reads as a product workspace instead of a corrupted engineering panel.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py` -> passed.
- `python -m py_compile scripts\eval\check_dashboard_visual.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py scripts\eval\check_dashboard_visual.py tests\test_field_site_profile.py tests\test_health.py` -> passed.
- `pytest tests\test_field_site_profile.py -q` -> 37 passed.
- `pytest tests\test_health.py -q` -> 76 passed, 1 existing httpx deprecation warning.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed; the project workspace now asserts
  `has_resource_governance: true`,
  `resource_governance_history_loaded: true`,
  `resource_governance_rollback_previewed: true`, and
  `resource_governance_disable_handled: true`.

Remaining:

- Governance is still local YAML-backed. The production version should move to
  a database service with tenant isolation, release-owner approval, and
  searchable audit trails.
- The browser visual smoke now verifies the non-mutating mocked click flow for
  history, disable, and rollback controls. The remaining gap is a database-backed
  production service and real enterprise approval workflow.

## 2026-05-15 Productization checkpoint: Delivery resource governance now requires two-person review

This checkpoint closes the main product-risk gap in the shared Delivery
Resource Registry. A solution-provider resource can affect many customer
projects, so disable and rollback actions should not be one-click mutations
from the Dashboard.

Implemented:

- Added resource-governance request lifecycle functions:
  `create_delivery_resource_governance_request`,
  `list_delivery_resource_governance_requests`, and
  `review_delivery_resource_governance_request`.
- Supported two high-risk operations: `disable_resource` and
  `rollback_registry`. Both create a pending request with a dry-run preview,
  requester, reason, operation payload, and current registry hash.
- Enforced second-person review: the same operator that requested the resource
  change cannot approve it. Rejecting a request leaves the registry unchanged;
  approving a request applies the disable or rollback.
- Added API routes:
  `GET/POST /api/field/delivery-resource-governance-requests` and
  `POST /api/field/delivery-resource-governance-requests/{request_id}/review`.
- Added explicit RBAC permissions:
  `resource:governance:write` for creating requests and
  `resource:governance:approve` for review/apply paths. Direct rollback and
  direct disable are now approval-gated administrative paths.
- Updated `/dashboard/projects`: shared resource rows now submit disable
  requests, rollback apply now submits a rollback request, and the Resource
  Governance card includes an Approval Queue that displays request id, status,
  target, requester, reason, and approve/reject controls.
- Updated visual smoke mocks so Dashboard click coverage remains non-mutating
  while exercising history, rollback dry-run, disable request, and queue load.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py` -> passed.
- `pytest tests\test_field_site_profile.py tests\test_health.py -q` -> 116 passed, 1 existing httpx deprecation warning.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed; `resource_governance_history_loaded`, `resource_governance_rollback_previewed`, `resource_governance_disable_handled`, and `resource_governance_requests_loaded` were all true.

Remaining:

- The queue is still file-backed under the delivery-resource root. Production
  should move it to a tenant-aware database workflow with searchable audit,
  notification, SLA/escalation, and enterprise identity binding.
- Current approval is single-step second-person review. Multi-level approval,
  delegation, scheduled windows, and rollback blast-radius analysis remain
  future product work.

## 2026-05-15 Productization checkpoint: Resource governance requests now show customer impact

This checkpoint upgrades resource governance from "approve a technical action"
to "approve with visible customer-project impact". A delivery owner can now see
which customer projects, objects, templates, and consumers are connected to a
shared resource before approving a disable or rollback request.

Implemented:

- Disable-request preview now includes an `impact` object generated from
  `build_customer_project_resource_catalog`.
- Impact fields include `analysis_status`, `generated_at`, `resource_type`,
  `resource_id`, `affected_consumer_count`,
  `affected_customer_project_count`, `affected_object_count`,
  `affected_template_count`, `affected_projects`, `affected_objects`,
  `affected_templates`, `affected_consumers`, `truncated`, and a reviewer
  message.
- Rollback preview now includes a registry-wide manual-review impact summary
  with current and target resource counts.
- Dashboard resource governance request cards render a `Resource governance
  impact` block with project/object/template/consumer counts and a short sample
  of affected consumers.
- The approval queue now shows impact counts inline, so the approver can judge
  blast radius before clicking approve/reject.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py askme\governance.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py` -> passed.
- `pytest tests\test_field_site_profile.py tests\test_health.py -q` -> 116 passed, 1 existing httpx deprecation warning.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed with no console errors, no page errors, no response errors, and no horizontal overflow.

Remaining:

- Impact analysis only covers structured in-product resource bindings. It does
  not detect free-text notes, external scripts, customer-side integrations, or
  offline operational dependencies.
- Rollback impact is still registry-wide and manual-review oriented; field-level
  diff and automatic remediation recommendations remain future work.

## 2026-05-15 Productization checkpoint: Resource governance queue now has review SLA and overdue operation

This checkpoint upgrades the resource governance queue from a passive approval
list into an operations queue. Delivery owners can now see which shared-resource
change requests are active, due soon, overdue, or closed before they approve a
customer-facing mutation.

Implemented:

- Resource governance requests now store `sla_target_s`, `due_at`,
  `escalation_policy`, and computed `review_sla`.
- `review_sla` exposes `state`, `remaining_s`, `age_s`, `overdue_s`,
  `due_soon_threshold_s`, `escalation_required`, and reviewer-facing message.
- `list_delivery_resource_governance_requests` now supports
  `overdue_only=True` and summary counts for active, due-soon, and overdue
  requests.
- `GET /api/field/delivery-resource-governance-requests` accepts
  `overdue_only=true`; `POST /api/field/delivery-resource-governance-requests`
  can accept `sla_target_s` for customer-specific review windows.
- `/dashboard/projects` now shows Review SLA on each request, due time,
  remaining/overdue duration, escalation policy, and an "Overdue only" queue
  filter.
- Visual smoke now verifies the resource governance SLA is actually visible in
  the project workspace and that desktop/mobile layouts have no horizontal
  overflow.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py` -> passed.
- `pytest tests\test_field_site_profile.py tests\test_health.py -q` -> 116 passed, 1 existing httpx deprecation warning.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed; `resource_governance_sla_visible` was true and no console, page, response, or overflow failures were reported.

Remaining:

- SLA is visible and queryable, but notification/escalation delivery is not yet
  automatic. Production should bind overdue requests to DingTalk/enterprise
  identity and an on-call delivery owner.
- The governance queue is still local-file backed; production needs a tenant
  isolated workflow store with searchable audit and retention policy.

## 2026-05-15 Productization checkpoint: Overdue resource governance can be escalated

This checkpoint turns resource-governance SLA from a visible status into an
operable delivery action. Delivery owners can now scan overdue shared-resource
requests, create escalation records, and see the escalation result in the
Dashboard.

Implemented:

- Added `escalate_overdue_delivery_resource_governance_requests`.
- The escalation scan filters pending requests through the existing
  `review_sla.state == overdue` contract and records one escalation per
  request/due-time window.
- Escalation records include `escalation_id`, `escalated_at`, `escalated_by`,
  request target, overdue duration, escalation policy, local delivery owner
  queue notification payload, and delivery report.
- Public request payloads now include `escalation_count`, `last_escalation`,
  and recent `escalations`.
- Added
  `POST /api/field/delivery-resource-governance-requests/escalate-overdue`.
  It requires unrestricted `resource:governance:approve` authority because it
  can expose multi-customer resource impact to delivery owners.
- `/dashboard/projects` now includes an `Escalate overdue` action and renders
  escalation result cards plus per-request escalation status.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py` -> passed.
- `node --check askme\static\dashboard\app.js` -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py` -> passed.
- `pytest tests\test_field_site_profile.py tests\test_health.py -q` -> 116 passed, 1 existing httpx deprecation warning.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright` -> passed; `resource_governance_escalated` was true and there were no console, page, response, or overflow failures.

Remaining:

- Escalation now has a configurable delivery-owner notification outlet, but it
  still needs an environment-backed smoke test with a real customer DingTalk,
  WeCom, Feishu, or webhook target before a deployment can claim external
  delivery.
- Production should connect notification routing to tenant-specific delivery
  owner contacts, retry policy, and immutable audit export.

## 2026-05-15 Productization checkpoint: Resource governance overdue escalation now supports configured delivery-owner notifications

This checkpoint upgrades overdue resource-governance escalation from a local
queue-only artifact to a deployable notification contract. The product behavior
is now: local demo mode records a queue item; production mode can route the same
payload through configured delivery-owner channels and persist the channel
delivery report on the governance request.

Implemented:

- `escalate_overdue_delivery_resource_governance_requests` accepts an injected
  `notification_delivery` function, so pipeline logic stays testable and API
  deployment logic can attach real delivery channels.
- `register_field_routes` now receives `config_provider` and builds a
  resource-governance delivery outlet from
  `field_operations.delivery_resource_governance.delivery_owner_notifications`.
- Configurable channels use the existing `AlertDispatcher`: webhook, DingTalk,
  WeCom, Feishu, and log. Disabled config falls back to the local
  `delivery_owner_queue`.
- Dashboard escalation cards show `delivery_mode`, sent channels, and
  per-channel delivery report instead of hiding whether notification actually
  went anywhere.
- `config.yaml` documents the production notification keys:
  `ASKME_DELIVERY_OWNER_WEBHOOK_URL`,
  `ASKME_DELIVERY_OWNER_DINGTALK_WEBHOOK`,
  `ASKME_DELIVERY_OWNER_DINGTALK_SECRET`,
  `ASKME_DELIVERY_OWNER_WECOM_WEBHOOK`, and
  `ASKME_DELIVERY_OWNER_FEISHU_WEBHOOK`.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py`
  -> passed.
- `node --check askme\static\dashboard\app.js`
  -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py`
  -> passed.
- `pytest tests\test_field_site_profile.py tests\test_health.py -q`
  -> 116 passed, 1 existing httpx deprecation warning.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright`
  -> passed; resource-governance escalation, SLA visibility, project workspace,
  capability readiness, and audit dossier checks passed with no console, page,
  response, or overflow failures.

Remaining:

- Need one environment-backed smoke test against a real customer notification
  group before claiming external delivery in a customer acceptance report.

## 2026-05-15 Productization checkpoint: Customer project catalog now exposes solution-delivery acceptance gates

This checkpoint turns the方案商交付 P0 checklist into a customer-project catalog
contract. Delivery users no longer need to infer signoff readiness from raw YAML,
separate object cards, and separate acceptance reports; the catalog itself now
returns a product acceptance gate for every visible customer project and an
aggregate gate for the whole filtered customer-project set.

Implemented:

- `build_customer_project_catalog` now attaches `product_acceptance_gate` to
  each project. The gate checks customer/delivery scope, site profile validity,
  managed-object catalog, vision/sensor/skill/acceptance resource bindings,
  acceptance references, object-change audit policy, and handoff artifacts.
- The same catalog response now includes `delivery_acceptance_gate` plus summary
  fields: `delivery_acceptance_gate_status`,
  `delivery_acceptance_blocked_count`,
  `delivery_acceptance_manual_check_count`, and
  `delivery_acceptance_ready_count`.
- `/dashboard/projects` renders the aggregate Product acceptance gate and each
  project's Product gate, so a delivery lead can see blocked/manual/ready status
  before opening the deeper acceptance report, dossier, proposal, or export
  tools.
- Tests now assert the new gate shape through both pipeline and HTTP API paths.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py`
  -> passed.
- `node --check askme\static\dashboard\app.js`
  -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py`
  -> passed.
- `pytest tests\test_field_site_profile.py tests\test_health.py -q`
  -> 116 passed, 1 existing httpx deprecation warning.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright`
  -> passed; no console, page, response, or overflow failures.

Remaining:

- The catalog gate is a signoff readiness summary. Final customer acceptance
  still requires project-level onsite evidence, manual review, dossier verify,
  proposal verify, and scoped audit export.

## 2026-05-15 Productization checkpoint: Customer project directory now supports delivery filters

This checkpoint makes the multi-customer project catalog usable as a delivery
directory instead of a static list. A solution-provider delivery lead can now
filter projects by customer, tenant, delivery namespace, industry, site,
project id, product gate status, or deployment stage and still get correct
summary counts and aggregate acceptance gates for the filtered result set.

Implemented:

- `build_customer_project_catalog` accepts `tenant_id`, `delivery_namespace`,
  `customer_id`, `project_id`, `site_id`, `industry`, `gate_status`, and
  `deployment_stage` filters.
- `/api/field/customer-projects` exposes the same filters as query parameters.
- Permission scope filtering now recomputes `delivery_acceptance_gate`, summary
  counts, and managed-object category counts instead of returning stale
  aggregate numbers from the unfiltered catalog.
- `/dashboard/projects` now includes project filter controls backed by
  localStorage and API-side filtering.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py`
  -> passed.
- `node --check askme\static\dashboard\app.js`
  -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py`
  -> passed.
- `pytest tests\test_field_site_profile.py tests\test_health.py -q`
  -> 116 passed, 1 existing httpx deprecation warning.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright`
  -> passed; project workspace, resource governance, capability readiness, and
  audit dossier checks passed with no console, page, response, or overflow
  failures.

Remaining:

- Filters are directory filters, not enterprise tenant isolation. Production
  still needs IAM/SSO enforced claims and storage isolation before SaaS-style
  tenant guarantees can be claimed.

## 2026-05-15 Productization checkpoint: Template market now supports delivery filters

This checkpoint makes the reusable industry template market usable for a
solution-provider team with many customers and many scenario packages. Delivery
users can now narrow the template catalog before creating a customer project,
and the API returns counts for the visible template set instead of forcing the
UI to guess from the full folder.

Implemented:

- `list_customer_project_templates` now accepts `tenant_id`,
  `delivery_namespace`, `industry`, `publish_status`, `product_status`,
  `template_id`, `release_channel`, and `owner` filters.
- Template items now expose `tenant_id`, `delivery_namespace`,
  `release_channel`, `owner`, and `product_status` as first-class catalog
  fields.
- `/api/field/customer-project-templates` exposes the same filters as query
  parameters and recomputes template, tenant, namespace, industry, publish
  status, product status, and managed-object counts for the filtered set.
- The same endpoint now applies operator tenant/delivery namespace scope:
  shared default templates remain visible as common product starters, and
  tenant-specific templates can be hidden from unrelated delivery spaces.
- `/dashboard/projects` now includes template filter controls backed by
  localStorage and API-side filtering, so the template market behaves like a
  managed product catalog rather than a raw YAML list.
- Tests now assert pipeline filtering, HTTP filtering, and Dashboard static
  hooks for template filters.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py`
  -> passed.
- `node --check askme\static\dashboard\app.js`
  -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py`
  -> passed.
- `pytest tests\test_field_site_profile.py tests\test_health.py -q`
  -> 116 passed, 1 existing httpx deprecation warning.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright`
  -> passed; no console, page, response, or overflow failures.

Remaining:

- Template filters help delivery users find the right product package. They do
  not replace the release-request approval flow or onsite binding and
  acceptance evidence for each real customer project.

## 2026-05-15 Productization checkpoint: Solution delivery readiness now has a single product gate

This checkpoint reduces delivery ambiguity. Instead of asking a product owner or
delivery lead to inspect separate project cards, template cards, resource
bindings, and governance queues, the product now exposes one rollup gate that
states the current customer-facing claim boundary.

Implemented:

- Added `build_solution_delivery_readiness`, a deterministic rollup over:
  customer project acceptance, template-market readiness, model/protocol/skill
  resource bindings, and shared-resource governance requests.
- Added `GET /api/field/solution-delivery-readiness`, protected by
  `field:project:read` and project/tenant scope filtering. Scoped operators see
  their visible projects/templates/resources; shared governance queue access is
  reported as a manual-check gate unless the operator is unrestricted.
- `/dashboard/projects` and the delivery overview now render "客户交付总门禁"
  with gate cards and a customer-facing release-claim sentence.
- Tests now assert the pure pipeline rollup, HTTP endpoint, and Dashboard
  static hooks.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py`
  -> passed.
- `node --check askme\static\dashboard\app.js`
  -> passed.
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py scripts\eval\check_dashboard_visual.py`
  -> passed.
- `pytest tests\test_field_site_profile.py tests\test_health.py -q`
  -> 118 passed, 1 existing httpx deprecation warning.
- `python scripts\eval\check_dashboard_visual.py --output-dir output\playwright`
  -> passed; no console, page, response, or overflow failures.

Remaining:

- This is a product readiness rollup, not a replacement for signed customer
  acceptance. A real customer handoff still needs onsite evidence, verified
  dossier/proposal packages, scoped audit export, and customer signoff.

## 2026-05-15 Productization checkpoint: Customer signoff is now part of the acceptance lifecycle

This checkpoint closes the gap between internal delivery readiness and actual
customer acceptance. The product no longer stops at "ready for signoff"; it can
record customer decisions, preserve rejected/needs-fix feedback, and only claim
acceptance after a risk-acknowledged customer signoff is archived.

Implemented:

- Added customer signoff records to customer project profiles, with decision,
  signatory, organization, reason, risk acknowledgement, evidence references,
  operator, timestamp, and profile revision.
- Hardened signoff evidence with credential reference, credential SHA-256, a
  snapshot of the acceptance gates at signing time, handoff-material references,
  and a signoff payload hash for later integrity checks.
- Added `GET/POST /api/field/customer-projects/{identifier}/customer-signoff`,
  protected by project read/write permissions and project/tenant scope checks.
- Acceptance closure now separates internal readiness from customer acceptance:
  `ready_for_customer_signoff` means the evidence package is ready for customer
  review; `accepted_by_customer` means the customer has signed off.
- Accepted signoff is rejected until the internal closure gates are ready.
  Accepted signoff also requires credential reference/hash. `needs_fix` and
  `rejected` are still allowed so customer feedback is not lost.
- Acceptance dossiers and manifests now carry customer signoff history, latest
  decision, and signoff count.
- `/dashboard/projects` now has a customer signoff form and history renderer in
  the project lifecycle workspace, so delivery teams can register and show the
  customer decision without editing YAML.

Product impact:

- A solution-provider delivery lead can now answer: "Is this internally ready,
  waiting for customer signoff, or already accepted by the customer?"
- Customer-facing claims are stricter: internal readiness cannot be presented as
  customer acceptance.

Remaining:

- This is still a software signoff registry, not an enterprise e-signature or
  SSO/IAM system. Production deployments still need customer identity binding,
  legal signature workflow, and database/storage-level tenant isolation.

## 2026-05-15 Productization checkpoint: Managed-object bindings now produce executable field-ingest plans

This checkpoint starts closing the P0 gap between a configured object directory
and real field execution. Managed objects no longer only show model/protocol/
skill/test strings; the project can now explain how each object should enter
`/api/field/ingest`, which devices and protocols cover it, which skill package
will shape the field action, and where runtime callbacks must be recorded.

Implemented:

- Added `build_customer_project_execution_bindings` for one customer project.
  It returns per-object source coverage, matched devices, ingest adapter,
  vision model references, skill routes, acceptance tests, sample ingest
  payload, runtime callback boundary, blockers, and manual-review notes.
- Added `GET /api/field/customer-projects/{identifier}/execution-bindings`,
  protected by the same project read permission and tenant/project scope checks
  as the rest of the customer project API.
- `/dashboard/projects` now has an "执行接入计划" action in the project
  lifecycle workspace. It renders ready/manual/blocked object plans in customer
  language instead of exposing raw binding strings only.
- Field events created by `FieldOperationsService.ingest_payload` now archive a
  `resource_execution_context` with the matched managed object, payload source,
  bound vision models, sensor protocols, selected skill package, acceptance
  tests, and runtime callback endpoint.

Product impact:

- A delivery lead can now show a customer not only that an object is configured,
  but exactly how its device payload enters the product and which skill/runtime
  boundary handles the action.
- This still does not execute a real vision model or hardware command by
  itself. It creates the auditable adapter contract that real device/model
  integrations must satisfy before customer signoff.

Remaining:

- Connect the execution plan to real vendor adapters and live device payload
  samples per customer site.

## 2026-05-15 Productization checkpoint: Skill packages now resolve to auditable field capability contracts

This checkpoint upgrades managed-object execution plans from "the object has a
skill package string" to "the object has a named capability with an installed
contract, safety level, approval policy, required inputs, output contract, tool
route, and hardware boundary." It is still not a hardware driver; it is the
product contract that prevents delivery from claiming an unbound package is
ready for a real customer site.

Implemented:

- Added `askme.skills.field_capability_contracts`, which normalizes package ids
  such as `capability.detect_illegal_parking` to executable capability names and
  resolves built-in skill contracts from the code-defined skill registry.
- Customer-project execution bindings now show per-skill `capability`,
  `installed_contract`, `safety_level`, `confirm_before_execute`,
  `approval_policy`, `required_inputs`, `tool`, `output_contract`, and
  `hardware_boundary`.
- Field events created from `/api/field/ingest` now archive
  `capability_routes`, `selected_capability`, `approval_required`,
  `output_contract`, and `action_boundary` inside
  `resource_execution_context`.
- Customer acceptance reports now include an `execution_bindings` summary and a
  `managed_object_execution_bindings` gate, so signoff review can see whether
  objects have executable ingest plans instead of only checking static bindings.
- `/dashboard/projects` renders those capability contracts in the "执行接入计划"
  card so the customer-facing page can explain which ability will run, what it
  is allowed to output, and where hardware authority stops.

Validation:

- `python -m py_compile askme\skills\field_capability_contracts.py askme\pipeline\field_site_profile.py askme\pipeline\field_operations.py tests\test_field_site_profile.py tests\test_health.py`
- `node --check askme\static\dashboard\app.js`
- `ruff check askme\skills\field_capability_contracts.py askme\pipeline\field_site_profile.py askme\pipeline\field_operations.py tests\test_field_site_profile.py tests\test_health.py`
- `pytest tests\test_field_site_profile.py tests\test_health.py -q` -> 119 passed.

Remaining:

- Connect these capability contracts to actual vendor vision/sensor adapters and
  live device payload samples per customer site.
- Connect the `tool` routes to a controlled execution adapter when the runtime
  is explicitly enabled, while keeping SafetyPreflight and runtime arbiter as
  the hardware authority.

## 2026-05-15 Productization checkpoint: Real device adapter bridge contracts are visible in customer projects

This checkpoint turns the already implemented JSON/JSONL device bridge into a
customer-project handoff contract. The product can now tell a delivery engineer
how each managed object should receive real camera, sensor, or robot payloads,
how to dry-run parsing, how to live-post into `/api/field/ingest`, and what
signature evidence is needed before customer signoff.

Implemented:

- Execution bindings now attach an `adapter_contract` to every sensor protocol:
  normalizer module, `field-ingest-bridge`, supported formats, dry-run command,
  live watch command, signing command, sample fixture, verification outputs, and
  device secret env vars.
- Object plans now include a `bridge_contract` summarizing the required ingest
  endpoint, bridge state file, live-post requirement, production signature
  boundary, and bridge summary fields.
- Acceptance reports now include compact `execution_bindings.object_contracts`
  so reviewers can audit object-level adapter/bridge/signing readiness without
  opening the full execution plan.
- `/dashboard/projects` renders device adapter contracts, matched device counts,
  bridge commands, and live-post requirements in the execution-access plan.

Validation:

- `python -m py_compile askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `node --check askme\static\dashboard\app.js`
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `pytest tests\test_field_site_profile.py tests\test_health.py tests\test_field_ingest_bridge.py tests\test_field_ingest_adapters.py -q` -> 136 passed, 1 warning.

Remaining:

- Run the same `field-ingest-bridge --watch` contract against actual customer
  camera/VMS, MQTT smoke/temperature, and robot diagnostics feeds.
- Promote observed live-post, trusted-device, DingTalk, voice playback, and
  runtime callback evidence into onsite acceptance only when the deployment is
  non-local and signed/trusted.

## 2026-05-15 Productization checkpoint: Managed objects can run lab ingest rehearsals

This checkpoint closes the first customer-project execution loop after showing
adapter contracts. A delivery engineer can now select a managed object in the
project execution plan and run a lab-only接入演练, seeing the exact raw payload,
normalized `/api/field/ingest` payload, bound capability route, and production
claim boundary.

Implemented:

- Added `POST /api/field/customer-projects/{identifier}/execution-bindings/{object_id}/rehearsal`.
- `dry_run` mode merges the object sample payload, project scope, matched device
  identity, and bridge contract, then normalizes it through
  `normalize_field_ingest_payload` without creating a field event.
- `shadow_post` mode is available but requires explicit confirmation, because it
  can create a lab field event and may exercise configured external routes.
- `/dashboard/projects` now renders per-object "接入演练" and "实验室投递"
  actions in the execution binding card and shows the normalized payload plus
  release boundary.

Validation:

- `python -m py_compile askme\api\routes\field.py tests\test_health.py`
- `node --check askme\static\dashboard\app.js`
- `ruff check askme\api\routes\field.py tests\test_health.py`
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_field_customer_project_write_endpoints_enforce_project_scope -q` -> 2 passed, 1 warning.

Remaining:

- Add automated browser interaction tests for the Dashboard rehearsal buttons.

## 2026-05-15 Productization checkpoint: Rehearsal results have acceptance-evidence boundaries

This checkpoint prevents a solution-delivery demo from being mistaken for
customer production signoff. Object execution rehearsal remains useful for
delivery, but the system now separates lab rehearsal, acceptance candidate, and
site acceptance evidence in data, API, and Dashboard output.

Implemented:

- Onsite evidence receipts now carry `evidence_tier` and
  `production_eligible`. Manual receipts default to
  `acceptance_candidate` / `false`; auto-collected real-link readiness receipts
  are marked `site_acceptance` / `false`.
- Object rehearsal responses expose `evidence_tier=lab_rehearsal` and
  `production_eligible=false`.
- `dry_run` requests with `register_onsite_evidence=true` are rejected as
  `dry_run_rehearsal_not_onsite_evidence` without writing a receipt.
- Confirmed `shadow_post` can request onsite evidence registration, but it only
  writes a `device_ingest` acceptance-candidate receipt. It stays
  `manual_check` unless trusted device signature and runtime completion callback
  evidence are present.
- Dashboard rehearsal and onsite evidence rows now show boundary tags so
  delivery teams and customers can distinguish lab checks from production
  go-live proof.

Validation target:

- Run targeted health, site-profile, and dashboard syntax tests.

## 2026-05-15 Productization checkpoint: Customer-readable managed object directory

This checkpoint improves the solution-provider product surface for many
customers and many site object types. The backend already builds object,
resource, and acceptance status; the Dashboard now makes that status readable
and exportable for delivery managers.

Implemented:

- `/dashboard/projects` shows a Managed Object Directory summary: total
  customer objects, deliverable objects, manual-check objects, blocked objects,
  acceptance-test count, and tenant/scope-guard count.
- Each object card now displays the combined delivery status derived from
  resource binding plus acceptance status.
- Object cards show compact resource and acceptance check details so a delivery
  manager can see which model, sensor protocol, skill package, or test is
  blocking delivery.
- Added object directory exports: full JSON for audit/reuse and deliverable-only
  CSV for customer handoff preparation.

Validation target:

- Run Dashboard syntax/static assertions plus field health tests.

## 2026-05-15 Productization checkpoint: Scoped managed-object directory API

This checkpoint moves the managed-object directory from a Dashboard-only
presentation into a reusable product API for delivery scripts, customer handoff
automation, and acceptance checks.

Implemented:

- Added `GET /api/field/customer-projects/managed-object-directory`.
- The API reuses the same customer-project filters and operator project-scope
  rules as `/api/field/customer-projects`.
- Each row includes tenant, delivery namespace, customer, project, site,
  object, device source, scenario, resource binding, acceptance check, and a
  derived `delivery_status`.
- The response includes summary counts for ready, manual-check, blocked,
  customer-visible, acceptance-test, and scoped objects.
- Each non-ready object now exposes an `action_plan` that tells delivery and QA
  which resource, binding, or acceptance-test reference needs to be registered,
  replaced, reviewed, or fixed.
- Dashboard project loading now fetches this API and shows the scoped object
  count plus object repair actions in the Managed Object Directory section.

Validation target:

- Run field health tests plus Dashboard JS syntax checks.

## 2026-05-15 Productization checkpoint: Managed-object matching is project-scope aware

This checkpoint tightens the multi-customer product boundary for solution
provider deployments. A real device event can still infer its managed object
from scenario, detection labels, zone type, and source, but object binding now
honors optional tenant/customer/project/site constraints.

Implemented:

- Managed-object catalog payloads now expose optional `tenant_ids`,
  `delivery_namespaces`, `customer_ids`, `project_ids`, and `site_ids`.
- Execution binding plans show those constraints in `scope_constraints`.
- `FieldOperationsService` includes tenant and delivery namespace in project
  scope and filters candidate managed objects by scope before scoring
  `scenario/source/zone/label`.
- A device payload cannot bypass the constraint by sending a mismatched
  `managed_object_id`; the event is kept unbound and its execution context says
  `no_managed_object_matched`.
- `/dashboard/projects` managed-object editor now has a "Customer scope guard"
  section for tenant, namespace, customer, project, and site constraints, and
  execution plans render the active scope constraints per object.

Validation:

- `python -m py_compile askme\pipeline\field_operations.py askme\pipeline\field_site_profile.py tests\test_field_operations.py`
- `ruff check askme\pipeline\field_operations.py askme\pipeline\field_site_profile.py tests\test_field_operations.py`

Remaining:

- Add browser interaction tests for configuring scope constraints from the UI.

## 2026-05-15 Productization checkpoint: Customer package delivery gate

This checkpoint prevents solution-provider handoff packages from being
mistaken for customer-ready deliverables when managed objects still have
blocked bindings, missing acceptance evidence, or unresolved manual checks.

Implemented:

- Customer project package export now writes `managed_object_action_plan`.
  The plan lists per-object repair actions such as registering resources,
  replacing blocked resources, binding missing requirements, and fixing missing
  acceptance-test references.
- Customer project package export now writes `package_delivery_gate` with
  `delivery_gate_status`, `delivery_gate_reasons`, `export_allowed`,
  `import_allowed`, `customer_handoff_ready`, source version, and customer
  next step.
- Package manifests now mirror delivery-gate status and action counts so
  package verification can detect tampered handoff readiness metadata.
- Package diff now returns incoming/current delivery gates and action plans.
- Package import dry-run still shows blocked packages for diagnosis, but real
  import rejects a blocked gate with `package_delivery_gate_blocked`.
- Manual-check packages remain importable into controlled pilot work, but they
  are explicitly not `customer_handoff_ready`.

Validation target:

- `pytest tests\test_field_site_profile.py::test_customer_project_template_create_update_export_import_and_archive tests\test_field_site_profile.py::test_customer_project_package_blocks_import_when_delivery_gate_is_blocked -q`
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_projects_endpoint_returns_solution_scope -q`
- `ruff check askme\pipeline\field_site_profile.py askme\api\routes\field.py tests\test_field_site_profile.py tests\test_health.py`

## 2026-05-15 Productization checkpoint: Solution-provider project workbench copy

This checkpoint fixes the customer-project page as a product surface for a
solution provider. The underlying panels already expose projects, templates,
objects, resources, and packages; the page now adds a customer-readable
workbench layer so delivery and sales can explain what each surface does.

Implemented:

- `/dashboard/projects` now injects a solution-provider workbench strip with
  five delivery surfaces: customer project catalog, industry template market,
  managed-object directory, delivery resources, and package delivery gate.
- `/dashboard/projects` now consumes `/api/field/customer-project-workbench`
  directly and renders a golden-path strip: industry template -> customer
  project catalog -> managed-object directory -> delivery resources -> package
  delivery gate.
- The workbench badges reuse backend readiness states instead of hardcoded
  marketing status.
- `/api/field/customer-project-workbench` now exposes the same five surfaces in
  one API payload for delivery tools, customer demos, and future multi-page UI.
- The same polish pass rewrites the most visible project-page labels and
  placeholders into readable Chinese while preserving the existing controls and
  API calls.

Validation target:

- `node --check askme\static\dashboard\app.js`
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q`

## 2026-05-15 Productization checkpoint: Enterprise identity readiness gate

This checkpoint turns the remaining "demo operator directory vs enterprise
IAM/SSO" risk into a product contract that delivery teams can show and test.
It does not implement a customer SSO provider; it makes the deployment boundary
explicit so a local demo cannot be presented as production tenant isolation.

Implemented:

- `OperatorDirectory.identity_gateway_readiness()` returns a versioned gate with
  identity mode, trusted gateway header contract, required operator/role
  claims, tenant/project/site scope headers, blockers, warnings, customer
  status, release claim, and next step.
- `GET /api/governance/identity-readiness` exposes the same gate independently
  of the full operator-directory payload.
- `/api/governance/operator-directory` and `/api/governance/current-operator`
  now include the gate so existing clients do not need another call just to
  understand the production boundary.
- `/dashboard` now shows an "企业身份准入" card with production/demo status and
  the trusted header contract.

Validation target:

- `python -m py_compile askme\governance.py askme\api\routes\governance.py askme\health_server.py tests\test_governance.py tests\test_health.py`
- `node --check askme\static\dashboard\app.js`
- `pytest tests\test_governance.py tests\test_health.py::TestHealthServer::test_governance_operator_directory_exposes_demo_boundary tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q`

## 2026-05-15 Productization checkpoint: Customer launch readiness summary

This checkpoint gives the product a single customer-facing launch decision
instead of scattered engineering panels. It is meant for a solution provider:
one screen must say whether a customer project is demo/integration only,
ready for pilot/site trial, or ready for production acceptance.

Implemented:

- Added `askme.pipeline.product_launch_readiness.build_product_launch_readiness()`.
  It rolls up enterprise identity readiness, field runtime readiness,
  solution-delivery readiness, and the customer-project workbench.
- Added `GET /api/field/product-launch-readiness`. The endpoint exposes
  `readiness_type=askme.product_launch_readiness.v1`, `overall_status`,
  `launch_stage`, `production_ready`, customer-readable status, release claim,
  next step, gates, evidence-source API refs, and a compact source snapshot.
- Demo operator identity is treated as a production-claim blocker, not as a
  demo/trial blocker. This keeps the product honest without making customer
  pilot work look unusable.
- `/dashboard/delivery` now renders "客户上线准入总览" before lower-level
  delivery cards, with four gates: identity, field operations, solution
  delivery, and customer project workbench.
- Added pure rollup tests plus endpoint and Dashboard assertions.

Validation target:

- `python -m py_compile askme\pipeline\product_launch_readiness.py askme\api\routes\field.py askme\health_server.py tests\test_product_launch_readiness.py tests\test_health.py`
- `node --check askme\static\dashboard\app.js`
- `ruff check askme\pipeline\product_launch_readiness.py askme\api\routes\field.py askme\health_server.py tests\test_product_launch_readiness.py tests\test_health.py`
- `pytest tests\test_product_launch_readiness.py tests\test_health.py::TestHealthServer::test_field_solution_delivery_readiness_endpoint_returns_product_gate tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q`

## 2026-05-15 Productization checkpoint: Customer project launch readiness in handoff artifacts

This checkpoint moves launch readiness from a Dashboard-only rollup into the
actual customer handoff chain. The delivery team now exports the same
"can we claim this is ready?" answer inside the project acceptance report,
acceptance dossier, and proposal bundle.

Implemented:

- `customer_project_acceptance_report()` now returns
  `launch_readiness=askme.customer_project_launch_readiness.v1`. It rolls up the
  project acceptance status, managed-object execution bindings, deployment
  credentials, required onsite evidence, real field link, and site acceptance
  checklist into `demo_or_integration_only`, `pilot_or_site_trial`, or
  `production_acceptance_ready`.
- Acceptance dossier JSON and printable HTML now include "上线准入" with launch
  stage, readiness status, production-ready boolean, customer status, release
  claim, next step, and per-gate evidence.
- Acceptance dossier manifests now expose `launch_readiness_status`,
  `launch_stage`, and `production_ready`, and the existing payload hash catches
  any tampering with launch-readiness claims.
- Proposal bundles now carry the same launch-readiness section at top level and
  inside the embedded acceptance dossier summary. Their manifest records the
  launch status/stage, and proposal verification rejects launch-readiness
  tampering.

Validation target:

- `python -m py_compile askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `pytest tests\test_field_site_profile.py::test_customer_project_acceptance_report_summarizes_delivery_gates tests\test_field_site_profile.py::test_acceptance_report_auto_backfills_required_onsite_evidence_from_real_link_reports tests\test_field_site_profile.py::test_customer_project_acceptance_dossier_exports_hash_manifest tests\test_field_site_profile.py::test_customer_project_acceptance_dossier_verify_rejects_tamper_and_bad_signature tests\test_field_site_profile.py::test_customer_project_proposal_bundle_binds_package_dossier_and_release_notes -q`
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_projects_endpoint_returns_solution_scope -q`

## 2026-05-15 Productization checkpoint: Shared delivery resource seed registry

This checkpoint turns the resource registry from a UI/API capability into a
real solution-provider starting point. Customer projects can still override
resources, but the normal delivery path now starts from a shared registry file
instead of only relying on Python defaults.

Implemented:

- Added `deploy/delivery-resources/resources.yaml` as the shared delivery
  resource registry seed. It registers the reusable vision models, sensor
  protocols, skill packages, and acceptance test references used by the demo
  project and the factory, park, warehouse, and scenic-area templates.
- Each seeded resource has product metadata: `display_name`, `category`,
  `version`, `source=shared_registry`, `owner`, `publish_status`,
  `release_channel`, and a short description.
- Published resources remain customer-deliverable; pilot resources such as
  customer-specific gate inspection stay visible but require delivery review.
- Added a regression test proving the default registry is found, contains the
  seed resources, and leaves current project/template consumers without
  unregistered resource gaps.

Validation target:

- `python -m py_compile tests\test_field_site_profile.py`
- `ruff check tests\test_field_site_profile.py`
- `pytest tests\test_field_site_profile.py::test_default_delivery_resource_registry_seed_covers_customer_projects -q`

## 2026-05-15 Productization checkpoint: Customer-readable reusable delivery scope

This checkpoint turns industry templates and exported customer-project packages
into handoff material a solution provider can actually sell, deploy, test, and
review with a customer. The fields are deliberately customer-readable and avoid
claiming production readiness before onsite evidence exists.

Implemented:

- Industry template payloads now expose `applicability_scope`, `out_of_scope`,
  `customer_prerequisites`, `scenario_acceptance_criteria`, and
  `dependency_matrix`.
- Exported customer project packages carry the same structures, and their
  manifest records counts for prerequisites, scenario acceptance criteria,
  dependency matrix rows, and delivery-boundary items. Package verification
  rejects tampered manifest counts.
- Approved template release notes and proposal inserts now include scenario
  coverage and dependency summaries, so proposals can explain reusable scope
  without hiding behind internal runtime terms.
- Proposal bundles now include a `customer_readable_delivery` section and render
  prerequisites plus scenario acceptance criteria in the printable HTML.
- `/dashboard/projects` template cards now show applicability, customer
  prerequisites, scenario acceptance evidence, and out-of-scope boundaries.

Validation target:

- `python -m py_compile askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `pytest tests\test_field_site_profile.py::test_customer_project_templates_are_valid_solution_starters tests\test_field_site_profile.py::test_customer_project_template_release_request_requires_second_approver tests\test_field_site_profile.py::test_customer_project_template_create_update_export_import_and_archive tests\test_field_site_profile.py::test_customer_project_package_blocks_import_when_delivery_gate_is_blocked tests\test_field_site_profile.py::test_customer_project_package_rejects_manifest_scope_tamper tests\test_field_site_profile.py::test_customer_project_proposal_bundle_binds_package_dossier_and_release_notes -q`
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_templates_and_export_endpoints tests\test_health.py::TestHealthServer::test_field_customer_projects_endpoint_returns_solution_scope -q`

## 2026-05-15 Productization checkpoint: Solution-provider delivery workbench language

This checkpoint addresses the customer-facing product gap in the multi-customer
delivery layer. The platform already had project catalogs, template packages,
object directories, scoped permissions, and import/export APIs, but the
Dashboard still exposed too many implementation words. The customer should see a
delivery product, not a developer console.

Implemented:

- `/api/field/customer-project-workbench` now returns `customer_vocabulary`,
  `customer_acceptance_flow`, and `customer_readable_contract`.
- Delivery surfaces now include customer labels, descriptions, action guidance,
  and count labels for customer projects, industry templates, object directory,
  delivery resources, and package preflight.
- `/dashboard/projects` now uses customer-facing terms such as customer space,
  delivery space, site object, capability configuration, package preflight, and
  acceptance evidence in the primary project flow.
- Object directory cards now explain vision, device, skill, and acceptance
  bindings with customer-readable labels while keeping the underlying IDs for
  audit and export.
- Follow-up polish removed customer-visible engineering labels such as runtime,
  managed object, tenant, acceptance dossier, delivery workflow, package
  collision, and review SLA from the project handoff, audit dossier, delivery
  resource, and onsite acceptance panels.
- Second-pass polish localized the customer enablement package panel, template
  market actions, template release governance results, delivery-resource
  impact/review flows, project rollback messages, onsite-evidence empty states,
  proposal-bundle generation status, and project/template filter placeholders.
  The underlying API field names are intentionally preserved for imports,
  exports, audit, and tests.
- Customer project package verify, diff preview, and import dry-run responses now
  return both the incoming package scope and the operator project scope. This
  makes tenant/delivery-space separation auditable before a solution provider
  copies a package into a new customer project.
- `/dashboard/projects` now exposes package verification, difference preview,
  import rehearsal, and final import as separate actions. The result card shows
  customer-readable package ownership, current project ownership, current
  operator authorization scope, integrity, change type, write intent, and
  resource binding status before anything is written.
- Managed-object action plans now carry customer-readable action labels, reason
  labels, owner labels, and customer next steps. The object directory card uses
  those fields first, so delivery sees "what to fix, why, who owns it, and the
  next step" instead of raw action codes.
- The customer project creation form now includes a template creation-readiness
  preview. Selecting a template shows applicability, starter objects, customer
  prerequisites, scenario acceptance criteria, and out-of-scope boundaries before
  a delivery user creates a new customer project.
- After a customer project is created, the Dashboard now renders a Chinese
  implementation handoff card instead of raw Created/Failed text. It shows the
  selected template, customer/project, site, starter-object count, next steps,
  and buttons to load the new project or jump directly to object resource
  binding.
- `create_customer_project_from_template` now returns a reusable
  `implementation_handoff` contract. The contract includes customer/project/site
  scope, object binding readiness, customer-readable implementation steps, and
  per-object missing binding labels so the UI, package flow, and external
  delivery automation do not need to guess next steps.
- The `/api/field/customer-projects/from-template` HTTP route now has a direct
  contract test proving it returns the same implementation handoff when creating
  a customer project through the product API.
- Customer project package import now returns the same `implementation_handoff`
  for dry-run previews, blocked imports, and successful imports. This means a
  solution provider can copy a package into another customer project and still
  receive the next implementation steps, object-binding todos, and acceptance
  evidence guidance from the API response.
- `/dashboard/projects` now renders package-import implementation handoffs.
  Import rehearsal and confirmed import results show the next implementation
  steps, object readiness summary, and the first objects that still need binding
  work.
- `/api/field/customer-projects/import` now has a direct product-route test:
  it creates a package from a template-backed customer project, imports it
  through HTTP in dry-run and write modes, and verifies both responses include
  the implementation handoff.
- Explicit customer project profile upsert now also returns
  `implementation_handoff`. The `/api/field/customer-projects` write route has a
  direct test proving saved customer/project/site profiles receive the same
  implementation contract as template creation and package import.
- The object resource picker now uses customer-readable Chinese guidance for
  binding vision models, sensor protocols, capability packages, and acceptance
  tests to a site object.
- Managed-object save and offline routes now return the same
  `implementation_handoff`. The Dashboard renders an object-change result card
  with the changed object, operator, saved path, and post-change implementation
  steps, so object CRUD is no longer a raw Saved/Deleted control.
- Customer project detail now also returns `implementation_handoff`, and the
  Dashboard project loader renders a project-detail handoff card immediately
  after loading a customer project. Delivery can see "what is ready and what is
  missing" before editing or saving anything.
- The shared delivery-resource registry now treats the gate-inspection skill as
  a published stable resource, so customer-specific gate objects can become
  ready after their project-specific vision model is registered.

Validation target:

- `python -m py_compile askme\api\routes\field.py tests\test_health.py`
- `node --check askme\static\dashboard\app.js`
- `ruff check askme\api\routes\field.py tests\test_health.py`
- `pytest tests\test_health.py::TestHealthServer::test_field_solution_delivery_readiness_endpoint_returns_product_gate tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q`

Validation completed:

- `node --check askme\static\dashboard\app.js`
- `python -m py_compile askme\api\routes\field.py tests\test_health.py`
- `python scripts\check_text_encoding.py askme\api\routes\field.py askme\static\dashboard\app.js askme\static\dashboard\app.css tests\test_health.py docs\PRODUCT.md plans\plan.md`
- `ruff check askme\api\routes\field.py tests\test_health.py`
- `pytest tests\test_health.py -q` -> 79 passed, 1 existing httpx deprecation warning.
- `node --check askme\static\dashboard\app.js`
- `python -m py_compile tests\test_health.py`
- `python scripts\check_text_encoding.py askme\static\dashboard\app.js tests\test_health.py plans\plan.md`
- `ruff check tests\test_health.py`
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q`
- `pytest tests\test_health.py -q` -> 79 passed, 1 existing httpx deprecation warning.
- `python -m py_compile askme\api\routes\field.py tests\test_health.py`
- `python scripts\check_text_encoding.py askme\api\routes\field.py tests\test_health.py plans\plan.md`
- `ruff check askme\api\routes\field.py tests\test_health.py`
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace -q` -> 1 passed.
- `pytest tests\test_health.py -q` -> 79 passed, 1 existing httpx deprecation warning.
- `node --check askme\static\dashboard\app.js`
- `python -m py_compile tests\test_health.py`
- `python scripts\check_text_encoding.py askme\static\dashboard\app.js askme\static\dashboard\app.css tests\test_health.py`
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `ruff check tests\test_health.py`
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls tests\test_health.py::TestHealthServer::test_field_customer_project_scope_enforces_tenant_namespace -q` -> 2 passed.
- `pytest tests\test_health.py -q` -> 79 passed, 1 existing httpx deprecation warning.
- `python -m py_compile askme\api\routes\field.py tests\test_health.py`
- `node --check askme\static\dashboard\app.js`
- `python scripts\check_text_encoding.py askme\api\routes\field.py askme\static\dashboard\app.js tests\test_health.py`
- `ruff check askme\api\routes\field.py tests\test_health.py`
- `pytest tests\test_health.py::TestHealthServer::test_managed_object_directory_generates_action_plan_for_blocked_bindings -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `pytest tests\test_health.py -q` -> 79 passed, 1 existing httpx deprecation warning.
- `node --check askme\static\dashboard\app.js`
- `python -m py_compile tests\test_health.py`
- `python scripts\check_text_encoding.py askme\static\dashboard\app.js askme\static\dashboard\app.css tests\test_health.py plans\plan.md`
- `ruff check tests\test_health.py`
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `pytest tests\test_health.py -q` -> 79 passed, 1 existing httpx deprecation warning.
- `python -m py_compile askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `node --check askme\static\dashboard\app.js`
- `python scripts\check_text_encoding.py askme\pipeline\field_site_profile.py askme\static\dashboard\app.js askme\static\dashboard\app.css tests\test_field_site_profile.py tests\test_health.py plans\plan.md`
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `pytest tests\test_field_site_profile.py::test_customer_project_template_create_update_export_import_and_archive -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `python scripts\check_text_encoding.py deploy\delivery-resources\resources.yaml askme\pipeline\field_site_profile.py tests\test_field_site_profile.py`
- `pytest tests\test_field_site_profile.py -q` -> 43 passed.
- `pytest tests\test_health.py -q` -> 79 passed, 1 existing httpx deprecation warning.
- `python -m py_compile tests\test_health.py`
- `python scripts\check_text_encoding.py tests\test_health.py`
- `ruff check tests\test_health.py`
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_from_template_returns_implementation_handoff -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `pytest tests\test_health.py -q` -> 80 passed, 1 existing httpx deprecation warning.
- `pytest tests\test_field_site_profile.py -q` -> 43 passed.
- `python -m py_compile askme\pipeline\field_site_profile.py tests\test_field_site_profile.py`
- `python scripts\check_text_encoding.py askme\pipeline\field_site_profile.py tests\test_field_site_profile.py`
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py`
- `pytest tests\test_field_site_profile.py::test_customer_project_template_create_update_export_import_and_archive -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_from_template_returns_implementation_handoff -q` -> 1 passed.
- `pytest tests\test_field_site_profile.py -q` -> 43 passed.
- `pytest tests\test_health.py -q` -> 80 passed, 1 existing httpx deprecation warning.
- `node --check askme\static\dashboard\app.js`
- `python -m py_compile tests\test_health.py`
- `python scripts\check_text_encoding.py askme\static\dashboard\app.js tests\test_health.py`
- `ruff check tests\test_health.py`
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_from_template_returns_implementation_handoff -q` -> 1 passed.
- `pytest tests\test_health.py -q` -> 80 passed, 1 existing httpx deprecation warning.
- `python -m py_compile tests\test_health.py`
- `python scripts\check_text_encoding.py tests\test_health.py`
- `ruff check tests\test_health.py`
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_import_route_returns_implementation_handoff -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_from_template_returns_implementation_handoff -q` -> 1 passed.
- `pytest tests\test_health.py -q` -> 81 passed, 1 existing httpx deprecation warning.
- `python -m py_compile askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `python scripts\check_text_encoding.py askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `pytest tests\test_field_site_profile.py::test_customer_project_profile_upsert_updates_metadata_without_losing_objects -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_upsert_route_returns_implementation_handoff -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_import_route_returns_implementation_handoff -q` -> 1 passed.
- `pytest tests\test_field_site_profile.py -q` -> 43 passed.
- `pytest tests\test_health.py -q` -> 82 passed, 1 existing httpx deprecation warning.
- `python -m py_compile askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `node --check askme\static\dashboard\app.js`
- `python scripts\check_text_encoding.py askme\pipeline\field_site_profile.py askme\static\dashboard\app.js tests\test_field_site_profile.py tests\test_health.py`
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `pytest tests\test_field_site_profile.py::test_customer_project_template_create_update_export_import_and_archive -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_managed_object_routes_return_implementation_handoff -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
- `python -m py_compile askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `node --check askme\static\dashboard\app.js`
- `python scripts\check_text_encoding.py askme\pipeline\field_site_profile.py askme\static\dashboard\app.js tests\test_field_site_profile.py tests\test_health.py`
- `ruff check askme\pipeline\field_site_profile.py tests\test_field_site_profile.py tests\test_health.py`
- `pytest tests\test_field_site_profile.py::test_customer_project_template_create_update_export_import_and_archive -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_field_customer_project_detail_route_returns_implementation_handoff -q` -> 1 passed.
- `pytest tests\test_health.py::TestHealthServer::test_dashboard_contains_cognition_planning_controls -q` -> 1 passed.
