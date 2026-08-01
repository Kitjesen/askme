# Askme

[![CI](https://github.com/inovxio/askme/actions/workflows/ci.yml/badge.svg)](https://github.com/inovxio/askme/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-4.1.0-green.svg)](pyproject.toml)

[English README](README.en.md)

> 版本：4.1.0 | 更新时间：2026-07-19

Askme 是面向机器人方案商/集成商的现场运营交付中台。它把语音、文本、客户知识、现场事件、运行交接、acceptance dossier 和审计证据组合成可复制、可验收的 Demo-to-pilot 交付入口。

## 产品和架构入口

- 产品需求主干：`docs/PRODUCT_REQUIREMENTS.md`
- 高级软件架构蓝图：`docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md`
- R1-R7 需求到架构追踪：`docs/PRODUCT_ARCHITECTURE_TRACE.md`
- 需求证据台账：`docs/DEMAND_EVIDENCE_LEDGER.md`
- Warm Session 运维：`docs/WARM_SESSIONS.md`

当前 P0 是机器人方案商/集成商交付中台，不是通用聊天机器人，也不替代底盘控制。Field Delivery Domain 是客户项目、现场事件、证据、客户签收和 readiness gaps 的产品事实源；Runtime / Safety / Hardware 仍拥有真实执行。customer signoff != production readiness，不承诺无人值守生产上线。

## 快速开始

### Docker 两阶段启动

首次部署必须先启动 LiteLLM 并生成 AskMe scoped virtual key，再启动默认产品栈：

~~~powershell
Copy-Item docker/.env.example docker/.env
Copy-Item docker/litellm.env.example docker/.env.litellm
# 填写 docker/.env.litellm 后先启动控制面
docker compose --env-file docker/.env.litellm -f docker/docker-compose.litellm.yml up -d --wait litellm
# 按 docs/LITELLM_GATEWAY.md 生成 AskMe virtual key，
# 将 LITELLM_VIRTUAL_KEY 与 ASKME_CONTROL_API_KEY 写入 docker/.env。
# Linux edge 主机还必须把 audio 组 GID 写入 ASKME_AUDIO_GID。
# 第一阶段先校验 master/salt/DB；默认栈再校验 AskMe virtual key 与全部角色隔离。
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml -f docker/docker-compose.edge-linux.yml up -d

# 仅调试 ZeroClaw 模型路由时，另签发 robot-action key 后显式启用实验 profile：
docker compose --env-file docker/.env --env-file docker/.env.litellm -f docker/docker-compose.yml -f docker/docker-compose.edge-linux.yml --profile experimental-zeroclaw up -d
~~~

默认镜像仍运行 `edge_robot`，不是 headless 演示服务。入口会在启动前校验
配置所需模型和真实输入/输出声卡；缺少 `models/`、`/dev/snd` 或正确
`ASKME_AUDIO_GID` 时以 78 退出且不会伪报 ready。Docker Desktop/Windows
不属于这条 Linux edge 硬件部署路径；Windows 开发请使用下方本地蓝图。

启动后访问：
- Dashboard: `http://localhost:8765/dashboard`
- Readiness: `http://localhost:8765/ready`
- Liveness: `http://localhost:8765/healthz`

runtime 容器当前未挂载 FastMCP，因此不对外发布 `/mcp` 路由；ZeroClaw/MCP 链路必须按实验 profile 单独验收，不能把容器进程存活当作 MCP ready。

### 本地开发

```powershell
pip install -e ".[dev]"
python -m askme.blueprints.presets.text     # 文本运行时
python -m askme.blueprints.presets.voice    # 语音任务中心
python -m askme.blueprints.presets.edge_robot  # 园区巡检机器人
```

详细启动见[运行蓝图](#运行蓝图)章节。

## 当前版本能力

| 能力 | 状态 | 说明 |
|------|------|------|
| 语音任务中心 | 可用 | ASR→LLM→TTS 全链路，含打断和文本兜底 |
| 园区巡检机器人 | 试点可用 | 语音+感知+现场事件+运行交接+控制适配 |
| 客户知识库 | 已有闭环 | 上传→预览→审批→索引→检索证据→过期治理 |
| 现场事件 | 产品链路 | 摔倒、卡住、电机故障、违停、烟火等场景 |
| 空间认知 | 已有模型 | 点位+别名+服务点+路线+问路+带路 |
| 能力中心 | 已有目录 | 技能→能力包→场景蓝图→风险等级→审批依赖 |
| MCP 工具服务 | 已有入口 | 受控工具和资源暴露给 MCP 客户端 |
| Agent Profile | 已有管理 | 可审计的角色配置、工具白名单、MCP 连接 |
| 企业审计 | 已有 | SkillAuditLog + 统一审计时间线 + 导出证据包 |
| ZeroClaw 集成 | 阻断：v0.1.7 容器未接通 MCP | 默认产品栈不启动；仅显式实验 profile 可启动受 LiteLLM 约束的 gateway，且不能作为 AskMe MCP 集成验收证据 |
| Conversation Core | Phase 1 已接入（迁移期） | 统一 Thread / Turn / Generation、持久化轮次账本和旧会话 ID 兼容，逐步收敛多套历史写入 |
| LiteLLM 模型控制面 | 默认启用（需部署凭据） | 固定版本的本地 Proxy；已贯通安全 `LLMCallContext`、W3C trace、独立 call ID、逐请求时限、能力别名和隐私默认值。Askme 继续独占会话、轮次、记忆、工具、安全和打断；真实 A/B 与故障演练完成前不宣称提速 |

## 对话与语音实时架构（Conversation Core Phase 1）

Askme 将“产品中的一次长期对话”与“云端语音供应商的一条实时连接”分开管理。稳定的产品链路是：

```text
Person（人） -> Thread（跨连接的逻辑对话） -> Turn（一次可审计交互）
                                              -> Generation（一次生成尝试） -> Provider Session（可替换的实时连接）
```

| 对象 | 产品含义 | 生命周期 |
| --- | --- | --- |
| Person | 领域目标中的持续交互主体；Phase 1 仅保存可选 `person_id`，不负责说话人识别 | 需上游稳定身份才能跨日期、设备关联 |
| Thread | 用户感知到的逻辑对话，承载连续上下文 | 复用同一 `thread_id` 时可跨断线、重连和供应商切换 |
| Turn | 从用户最终输入到机器人实际向用户交付内容的一次业务/审计单元 | 每轮独立提交、取消或失败 |
| Generation | 同一 Turn 下的一次生成尝试，用于处理重试、抢答抑制和供应商恢复 | 可以被丢弃、截断或替换 |
| Provider Session | ASR/LLM/TTS 或端到端语音供应商的临时连接状态 | 断线后可重建，不创建新 Thread |

Phase 1 已把确认后的用户文本、成功交付的回答、取消/失败以及多次 Generation 写入同一轮次账本。火山实时链路在打断时会保存物理播放时长；当文字与音频无法可靠对齐时不会猜测“已听前缀”。本地级联成功完成播放后按完整回答提交，旧历史迁移数据标记为兼容导入，仍属于“假定已交付”。

| 实施层级 | 当前状态 |
| --- | --- |
| 已实现 | 本地单进程 JSONL Thread/Turn/Generation 账本、append+fsync、重放、终态幂等、匿名会话隔离、旧 ID/Turn ID 冲突检测、同 Thread 单一活动 Turn、网关/API alias 归一、provider session/generation 关联、火山实时失败回退到同一 Turn、任务级 RAG/认知/trace 隔离、取消/擦除拒绝迟到结算、写入降级计数并联动 `/ready` |
| 迁移中 | `ConversationManager` 与 Voice Gateway 仍保留 prompt-context/summary 兼容投影；Conversation Core 是新 Thread/Turn/Generation 的规范事实源，但还不是所有历史读写的唯一物理存储 |
| 尚未实现 | Conversation Summary 投影、Memory/Vision/Task 的 committed-event consumer 和证据 ID 回链、所有 skill/tool 执行回合的统一结算、Person 聚合/说话人识别、跨日期 Session Window、跨入口冲突 Turn 的自动排队/共享 lease、音频到文字的精确截断对齐、多进程/分布式 writer、协调停止所有在途 worker/播放、原始 JSONL 与旧历史的物理擦除/加密销钥 |

视觉能力目前是“按需相机/VLM 支路”，不是始终开启的统一多模态大模型：视觉问句可触发当前帧采集，普通轮也可在显式 `auto_capture` 时把场景描述加入 prompt。主配置 `config.yaml` 默认关闭本地视觉和云 VLM；板卡 profile 只启用本地摄像头/YOLO 感知，云端 `vlm_enabled` 在 `vision-scene` 别名和独立 scoped key 验收前继续保持关闭。原始图片/快照 ID、采集时间与 Turn/Generation 的证据回链尚未闭合，因此不能宣称连续视觉记忆或产品级可审计多模态已经完成。

迁移期接受 `thread_id`、`conversation_thread_id`、`conversation_session_id`、`conversation_id`、`chat_session_id` 和 `session_id`；多个非空值必须一致，否则请求被拒绝。Thread 的 `channel` 在 Phase 1 固定为稳定的 `voice` 业务通道，实际入口记录在 `Turn.source`。没有显式 ID 时，VoiceLoop/TextLoop 会在自身生命周期内生成稳定本地 ID；`/api/chat` 与 Voice Gateway 会为匿名请求 fail-new，并把新 ID 返回给客户端，后续轮次必须回传；裸 ledger 调用同样 fail-new，避免不同匿名用户串线。

同一 Thread 在任何入口同时最多只有一个非终态 Turn：同一 `turn_id` 的重试仍是幂等的，也允许实时供应商失败后在该 Turn 内创建新的 Generation；不同 `turn_id` 若撞上活动 Turn，会在账本内 fail-closed，HTTP 返回 `409` 和 `blocking_turn_id`，不会把第二轮写进旧历史。Phase 1 的跨入口策略是明确拒绝并由客户端短退避/重试；语音打断必须先取消旧 Turn，再开始新 Turn。本地 TextLoop/BrainPipeline 的同路径轮次会按 Thread 排队，不影响其他 Thread 并行。

CLOSED、EXPIRED、ERASED Thread、冲突 alias 和冲突 Turn payload 都属于领域拒绝，不会被当成存储故障绕过到旧历史投影。ERASED 会拒绝新写与迟到结算，BrainPipeline 发现擦除竞争时会清空该 Thread 的兼容投影；这不等于已撤回扬声器已经播放的声音，也不等于原始 JSONL 已物理擦除，完整的协调停播/等待 worker/加密销钥仍在后续范围。

默认语音仍是本地门控的 ASR→LLM→TTS 级联。可选火山豆包端到端语音位于 `voice.realtime`，默认 `enabled: false`。中央策略与 `prepare → durable Turn/Generation → release PCM` 两阶段安全补丁已完成离线回归，旧的一步放音入口也已 fail-closed；机器人动作、急停、工具/审批和视觉查询始终走本地级联。当前环境仍缺少线上凭据、shadow 隐私/稳定性证据和真机声学证据，因此不能直接启用 `general_chat`，发布仍必须按 `split → shadow → general_chat 小流量`。无音频、断线或审批失败会回退 cascade。普通麦克风/音箱一体设备只有通过硬件/AEC 门禁才开启全双工，否则自动退回半双工，且不需要 ROS2。详见 [火山实时语音](docs/VOLCENGINE_REALTIME_VOICE.md) 和 [全双工验收](docs/FULL_DUPLEX_VOICE.md)。

本阶段解决的是会话正确性与可审计性，不代表已经达到延迟目标。2026-07-19 在切换前的 DeepSeek 直连路径上，用完整 runtime persona 做 20 样本校正后，DeepSeek V4 Flash 首个非空文本为 P50 961.8ms / P95 1140.7ms，首个有效语义子句为 P50 1168.4ms / P95 1378.8ms；短首句合规率 95%，直接问题错误 `[SILENT]` 为 0/20。旧的约 110ms 只是首个任意 stream chunk，可能为空，不能再当作 TTFT 证据。这仍不含 ASR 和 TTS，因此产品级 P95 ≤1.2s 尚未达成。上线前仍需测量 ASR-final→physical-first-semantic-audio、barge-in→physical-stop、Turn commit 的 P50/P95。详见[语音延迟执行计划](docs/VOICE_LATENCY_EXECUTION_PLAN_2026-07-19.md)。领域词汇见 [CONTEXT.md](CONTEXT.md)，目标架构决策见 [ADR-0001](docs/adr/0001-conversation-core-single-write-owner.md)。

Gate A 代码优化现已落地：独立 health-probe 能力别名的非阻塞 LLM 保温、只接受稳定运行时缓存键的隔离短语 prime、包含声学参数的 v2 缓存签名、MiniMax SSE/WS 共用的首续片状态机，以及优先于静音/审批/对话门控的精确急停端点。长尾轮在 1.5 秒可播放独立 thinking feedback；空 delta 不会提前熄灭保险丝，取消/首个真实 payload 会原子阻止或停止反馈，ACK 也不会误触发 thinking 限频。该反馈只改善等待感，绝不计作首个语义音频。配置中的 36/54ms TTS 首续片阈值仍是实验值；普通意图继续使用 300ms，精确急停实际最早约 160ms。统一报告脚本会在样本少于 20、证据损坏或缺少目标硬件字段时拒绝给出“通过”。这些都是代码就绪结论，不是物理提速结论。

MiniMax 在线 TTS 采集器已经补齐，固定 20 句语料、禁用短语缓存、只合成不播放。2026-07-19 当前最新可审计 MiniMax speech-2.8-turbo WebSocket 实测：warm 复用 provider 首 PCM P50 270.71ms / P95 376.08ms、buffer commit P50 277.01ms / P95 379.32ms；cold 每条新连接且带 4.5s case 间隔 provider 首 PCM P50 631.75ms / P95 2294.78ms、buffer commit P50 652.09ms / P95 2314.36ms。Cold 的长尾明显，且一次无间隔 cold 重测出现 13/20 passed、7/20 provider failure，因此产品主路径必须保持 warm WebSocket 复用与后台预热，而不是每轮新建连接。启动和运行时 immediate/pending provider 切换现在都会非阻塞预热；替换/shutdown 会取消旧预热，并以 0.5 秒总预算收割不合作的 daemon worker。这仍不是物理首音：统一报告仍因缺少 `physical_first_nonzero_ms` 与 `barge_in_to_speaker_stop_ms` 保持 `insufficient_evidence`。采集器现在默认生成唯一文件名并拒绝覆盖既有证据，也支持 `--case-delay-ms` 避免 cold/new-connection 重测撞上供应商 RPM 限流。

火山 TTS V3 已完成离线接线：协议 codec、并发安全 WebSocket client、`TTSEngine` backend、配置项、health、dashboard、online smoke 和固定 20 句 collector 都已接入。默认 TTS backend 仍是 MiniMax；火山的 `resource_id` 与 `speaker` 是账号专属资源，不在代码中硬编码，项目里的 `model` 字段对火山 TTS 代表 `X-Api-Resource-Id`。当前环境没有 `VOLCENGINE_TTS_*` 凭据，所以还没有火山同语料 online measured 数据，也不能选择 TTS 主通道。切换 provider 时还要注意 `data/voice/system_control.json` 可覆盖 YAML，生产切换应走控制 API 或明确更新持久状态。

真机报告已升级到 `askme.full_duplex_hardware.v2` 并默认 fail-closed。`--latency-mode entry` 与 stopwatch 只产生 `manual` 诊断数据，WASAPI/loopback 只产生独立 `render_chain` 指标；两者都不能填充物理首音或物理停播门槛。产品通过必须各有至少 20 条严格 `physical_acoustic` trial，使用自动 capture/reference、同一 monotonic clock、有效校准和零丢帧；重叠说话停播还必须使用与人声通道隔离的 `isolated_speaker_monitor`。当前仓库尚未实现自动真机采集器，因此不会宣称 20+20+20 已完成。

## ZeroClaw 集成状态

AskMe 已提供独立 MCP 服务端能力，但 **ZeroClaw v0.1.7 容器尚未接通 AskMe MCP**。该版本已核对的配置 schema 没有 MCP connector 字段。默认 Compose 与 quickstart 不启动 ZeroClaw；只有显式 `experimental-zeroclaw` profile 或 `docker-zeroclaw` / `local-zeroclaw` 命令可启动 gateway，并把它的模型请求限制到 LiteLLM `robot-action` 别名。

因此，启动 ZeroClaw 进程不等于 MCP 集成可用，也不能据此宣称知识检索、现场工具或 runtime handoff 已贯通。生产验收必须保持阻断，直到选定支持的 ZeroClaw 版本或实现明确的 HTTP/MCP adapter，并补齐端到端工具调用、认证、取消和审计测试。

`scripts/dev/setup_zeroclaw.py` 目前只配置模型控制面与清理直连 fallback，不负责建立 MCP 连接。

## 企业级特性

| 特性 | 说明 |
|------|------|
| 四表面 API 分层 | Platform / Product / Admin / Internal，每层独立访问策略 |
| 网关验签 IAM | 支持 OIDC/IAM 网关注入受信身份头，覆盖本地 operator_id |
| RBAC 权限模型 | 细粒度权限（knowledge:read, skill:review, runtime:pause 等） |
| 项目范围隔离 | tenant/customer/project/site 多层 scope 过滤 |
| SafetyPreflight | 硬件动作执行前的安全预检门禁 |
| 审计证据链 | SHA-256 + 可选 HMAC 签名的审计证据包 |
| SIEM 投递 | 审计包可投递到企业 SIEM/WORM webhook |
| 统一审计查询 | 跨 skill/field/runtime 的审计时间线查询和复核 |
| 技能治理流水线 | 成长候选→草稿→审核→能力包→灰度发布→回滚 |
| 灰度发布 | 能力包 rollout_percent 控制灰度比例 |
| 操作确认门禁 | 高风险任务在执行前必须获得操作员确认 |
| 客户项目验收 | acceptance-report / onsite-evidence / customer-signoff 闭环 |
| 交付资源治理 | 资源注册表 + 治理请求 + 升级机制 |
| K8s 就绪探针 | /healthz (liveness) + /ready (readiness) + /health (detail) |
| Prometheus 指标 | /metrics/prometheus 提供标准 Prometheus 格式 |
| 分布式追踪 | X-Trace-Id 贯穿所有请求和日志 |

## 更多文档

| 文档 | 说明 |
|------|------|
| [产品需求主干](docs/PRODUCT_REQUIREMENTS.md) | P0、R1-R7、证据门禁、ROI/定价和 release gates |
| [高级软件架构蓝图](docs/SOFTWARE_ARCHITECTURE_BLUEPRINT.md) | bounded contexts、包所有权、API 表面和架构变更门禁 |
| [需求到架构追踪](docs/PRODUCT_ARCHITECTURE_TRACE.md) | R1-R7 到 Field Delivery Domain、API 表面和验证测试 |
| [需求证据台账](docs/DEMAND_EVIDENCE_LEDGER.md) | evidence_id、hypothesis_status 和 validated/research_pending 边界 |
| [Conversation Core 领域词汇](CONTEXT.md) | Person、Thread、Turn、Generation、Memory、Vision 与 Task 的统一含义 |
| [ADR-0001：会话事实单一写入者](docs/adr/0001-conversation-core-single-write-owner.md) | Conversation Core 与网关、供应商、记忆、感知之间的所有权边界 |
| [语音延迟执行计划](docs/VOICE_LATENCY_EXECUTION_PLAN_2026-07-19.md) | 语义首音、预热、缓存、动态端点、TTS A/B、S2S 与真机验收闸门 |
| [LiteLLM 网关运行手册](docs/LITELLM_GATEWAY.md) | sidecar 部署、scoped virtual key、打断、路由所有权、A/B 延迟门禁与回滚 |
| [架构说明](docs/ARCHITECTURE.md) | v1 架构——模块依赖和运行时模块 |
| [架构 v2](docs/ARCHITECTURE_V2.md) | ZeroClaw 集成 + 企业安全架构 |
| [API 文档](docs/API.md) | 全部 HTTP 端点参考 |
| [产品手册](docs/PRODUCT.md) | 产品能力和路线图 |
| [运维手册](docs/OPERATIONS.md) | 部署和运维指南 |
| [操作边界](docs/ASKME_BOUNDARY.md) | 核心/感知/插件三层边界 |

---

## Contributor Start Here

For parallel development, start with the ownership and workflow contracts before
editing code:

- `docs/MODULE_OWNERSHIP.md` maps each collaboration lane to its package scope,
  exclusions, and required verification command.
- `docs/MULTI_AGENT_WORKFLOW.md` explains how the lead agent assigns independent
  work, reserves shared files, and integrates worker results.
- Boundary-sensitive changes should always run
  `pytest tests/test_six_layer_package_boundaries.py tests/test_package_migration_compat.py -q`.

Askme 是面向园区、厂区、仓储和景区机器人项目的现场运营交付中台。它把语音、文本、知识库、技能、现场事件、运行交接和审计记录组合成一个可验收、可复用、可审计的方案商交付中台。

Askme 的产品边界很明确：

- 用户可以用语音或文本发起问路、巡检、异常处置和知识问答。
- 系统会把自然语言转成可确认、可审计、可中断的任务或回答。
- 大模型和语音层不直接控制硬件；机器人动作必须经过 TaskHandoff、SafetyPreflight 和 runtime arbiter。
- 客户知识必须经过上传、治理、检索和证据展示；没有依据时应要求确认或拒答。

## 当前产品能力

| 能力 | 当前状态 | 说明 |
| --- | --- | --- |
| 语音任务中心 | 可用 | 麦克风输入、ASR、LLM、TTS、打断和文本兜底由 `voice` 运行时承载。 |
| 园区巡检机器人运行时 | 可用作试点验证 | 覆盖语音、感知、现场事件、运行交接、控制适配、灯光状态和主动监测。 |
| 客户知识库 | 已有基础闭环 | 支持上传、预览、审批、重建索引、检索证据和过期/冲突治理。 |
| 现场事件 | 已有产品链路 | 支持摔倒无法恢复、卡住、电机故障、违停、烟火、垃圾桶、人群聚集、陌生人和问询点触发。 |
| 空间认知 | 已有园区点位模型 | 支持点位、别名、服务点、路线说明、问路和带路任务基础能力。 |
| 能力中心 | 已有目录和准入 | 把底层 skills 映射成客户可见能力包、场景蓝图、风险等级和审批依赖。 |
| MCP 工具服务 | 已有入口 | 可向 MCP 客户端暴露受控工具和资源，不提供原始硬件控制权限。 |

## 快速启动

安装依赖：

```powershell
cd <repo-root>
pip install -e ".[dev]"
```

查看可交付运行蓝图：

```powershell
python -m askme.cli runtime blueprints --customer-visible
```

启动文本运行时：

```powershell
python -m askme.blueprints.presets.text
```

启动语音任务中心：

```powershell
python -m askme.blueprints.presets.voice
```

启动园区巡检机器人运行时：

```powershell
python -m askme.blueprints.presets.edge_robot
```

启动 Web Dashboard：

```powershell
python scripts/dev/run_dashboard_only.py --host 127.0.0.1 --port 8766
```

然后打开：

```text
http://127.0.0.1:8766/dashboard
```

## 运行蓝图

蓝图是 Askme 的产品运行组合。代码位置：

- `askme/blueprints/catalog/`：蓝图目录、客户可见描述、readiness、交付包。
- `askme/blueprints/presets/`：具体运行时组合。
- `askme/blueprints/runner/`：统一启动辅助。

客户可见蓝图：

| 蓝图 | 启动命令 | 用途 |
| --- | --- | --- |
| 语音任务中心 | `python -m askme.blueprints.presets.voice` | 客户演示、语音问答、语音任务确认，不直接接硬件。 |
| 语音感知运行时 | `python -m askme.blueprints.presets.voice_perception` | 在语音基础上接入感知 freshness、交互准入和安全状态。 |
| 园区巡检机器人运行时 | `python -m askme.blueprints.presets.edge_robot` | 面向园区/厂区试点，接入现场事件、控制适配和主动监测。 |
| 灵途语音导航适配器 | `python -m askme.blueprints.presets.lingtu_voice` | 面向灵途导航项目的站点定制入口。 |

内部蓝图：

| 蓝图 | 启动命令 | 用途 |
| --- | --- | --- |
| 文本运营控制台 | `python -m askme.blueprints.presets.text` | 研发、交付、CI 和无音频环境调试。 |
| MCP 工具服务 | `python -m askme.mcp.server` | 向 MCP 客户端提供受控工具能力。 |

导出某个蓝图交付包：

```powershell
python -m askme.cli runtime blueprints --name park --delivery-package --json
```

## Dashboard 页面

Dashboard 是客户和交付人员查看产品能力的入口。当前重点页面：

- `/dashboard`：总览、对话、现场事件、运行状态。
- `/dashboard/knowledge`：客户知识库，查看已有知识、上传、预览、审批、重建索引和证据。
- `/dashboard/capabilities`：机器人能力中心，查看巡检、安防、访客服务、空间认知、语音、审计等能力。
- `/dashboard/delivery`：交付门禁、客户项目、对象目录、模板和验收材料。
- `/dashboard/voice`：语音状态、音色、播放策略和语音健康。

客户演示时优先走这条顺序：

1. 打开总览，确认系统在线。
2. 打开客户知识库，确认机器人“知道什么”和“依据在哪里”。
3. 打开能力中心，说明机器人能做哪些业务动作、哪些需要审批。
4. 打开现场事件，演示违停、烟火、垃圾桶、机器人故障等场景。
5. 进入对话，演示问路、巡检和知识问答。

## 典型业务场景

| 场景 | 产品行为 |
| --- | --- |
| 游客问路 | 在服务点识别停留和交互意图，解析目的地，给出语音指路；必要时转带路任务。 |
| 游客带路 | 先确认目的地，再检查路线是否可通行，低速引导并记录服务结果。 |
| 车辆违停 | 检测非停车区停车，拍照、附地点、通知保安并归档事件。 |
| 烟火监测 | 接入温度、烟雾或视觉烟火证据，播报风险，通知保安并归档。 |
| 垃圾桶满溢 | 定点检测垃圾桶状态，通知保洁并生成事件记录。 |
| 夜间陌生人 | 识别窗边、角落等重点区域陌生人，拍照并通知保安。 |
| 机器人异常 | 摔倒无法恢复、卡住、电机故障时播报、通知、归档并等待处理。 |
| 突发巡检 | 管理员发起临时任务，系统中断或暂停当前巡检，交接给 runtime。 |

## 知识库和记忆

Askme 把两类记忆分开：

- 客户知识库：园区点位、路线、SOP、设备说明、FAQ，用于回答和证据展示。
- 机器人行为记忆：长期行为、任务偏好、历史运行经验，默认不和客户知识混在一起。

配置建议：

```yaml
memory:
  enabled: true
  customer_knowledge_backend: vector
  robot_behavior_memory_backend: robotmem
  robot_behavior_memory_enabled: false
```

产品原则：

- 过期、冲突、未审批知识不能直接进入回答。
- 回答气泡应展示引用依据。
- 没有证据时，系统应要求确认或拒答。
- 每条知识需要责任人、来源、版本和有效期。

## 语音链路

推荐国产低延迟链路：

```text
实时 ASR -> MiniMax-M2.7-highspeed -> TaskHandoff / SafetyPreflight / runtime arbiter -> MiniMax Speech 2.8 TTS
```

语音体验需要关注：

- 什么时候可以说话：UI 必须显示“正在听 / 正在思考 / 正在播报 / 可打断”。
- 为什么慢：拆分 ASR、LLM、TTS、播放和打断延迟。
- 为什么误触发：InteractionGate 需要结合服务点、声源、视觉、距离、停留和感知 freshness。
- 为什么断断续续：需要检查 TTS 分片、播放缓冲、声卡采样率和回声门限。

常用诊断：

```powershell
python -m askme.cli runtime audio-devices
python -m askme.cli runtime voice-health --json
python -m askme.cli runtime voice-online-smoke
python -m askme.cli runtime sunrise-voice-readiness --json
```

## 现场交付门禁

客户试点前至少要提供这些证据：

- 蓝图交付包：`runtime blueprints --delivery-package`
- 运行 profile：客户现场验证必须使用 `lab` 或 `prod`；`fake`、`sim`、`shadow` 只作为演示、仿真或影子验证证据
- 现场 readiness：`runtime field-readiness --json`
- 语音健康：`runtime voice-health --json`
- 现场事件冒烟：`runtime field-smoke-suite`
- DingTalk 通知预检：`runtime field-notification-preflight`
- 审计完整性：`runtime field-audit-integrity`

示例：

```powershell
python -m askme.cli runtime field-readiness --json
python -m askme.cli runtime field-smoke-suite --json
python -m askme.cli runtime field-audit-integrity --json
```

## 测试

默认 pytest 分片只跑快测；`pyproject.toml` 通过 `-m "not slow"` 排除慢测。
`tests/conftest.py` 会自动把 `tests/scenario_tests/`、`*e2e*` 和 benchmark
测试标为慢测分片。

```powershell
python -m pytest tests -q
python -m pytest tests -m "slow" -q
python -m pytest tests -m "scenario" -q
python -m pytest tests -m "e2e or benchmark" -q
```

常用快速回归：

```powershell
python -m pytest tests/test_blueprints_catalog.py tests/test_cli.py -q
python -m pytest tests/test_capability_center.py tests/test_memory_bridge.py -q
python -m pytest tests/test_api_route_dependency_injection.py -q
```

本地修改蓝图、API、记忆或能力中心后，至少运行：

```powershell
python -m pytest tests/test_package_migration_compat.py -q
```

## 项目结构

```text
askme/
  api/            FastAPI 路由和产品 API 表面
  audit/          审计查询、导出、完整性和复核
  blueprints/     产品运行蓝图
  cli/            操作员 CLI、诊断命令和脚本入口
  cognition/      认知规划、任务理解和上下文
  memory/         客户知识库和机器人行为记忆
  pipeline/       现场事件、空间认知、交付 readiness
  runtime/        Runtime 模块组合和运行服务
  skills/         技能、能力包和技能准入
  static/         Dashboard 前端
  voice/          语音输入、ASR、TTS、播放和诊断
docs/
  PRODUCT.md      产品手册和路线图
  ARCHITECTURE.md 架构说明
  OPERATIONS.md   运维和交付说明
deploy/
  README.md       部署资产、systemd、站点模板和快捷启动说明
docker/
  README.md       Dockerfile 和 Compose 使用说明
```

## 交付边界

可以对客户承诺：

- 支持受控场景下的语音问路、知识问答、巡检任务和现场事件处理。
- 支持试点项目的能力中心、知识库、现场事件和审计闭环。
- 支持通过配置和交付包复制到不同客户项目。

不能在没有现场证据前承诺：

- 无人值守生产运行。
- 任意开放域问答。
- 大模型直接控制机器人硬件。
- 未配置真实传感器、通知和机器人控制网关时的真实处置效果。

## 进一步文档

- [产品手册](docs/PRODUCT.md)
- [架构说明](docs/ARCHITECTURE.md)
- [运维交付](docs/OPERATIONS.md)
- [Warm Session 运维](docs/WARM_SESSIONS.md)
