# 机器人实时语音：产品级对标与当前验收报告

日期：2026-07-18
设备边界：普通音响/麦克风一体设备，本机 Python 运行时，不依赖 ROS2。

## 结论

当前系统已经具备产品级实时语音的主要**控制面**：单一物理音频所有者、AEC 后同源音频分发、流式云端会话、可打断播放、generation 隔离、云端历史 truncate/delete、失败后仅在下一轮建立全新 provider session，以及本地级联兜底。

当前仍不能宣称“目标设备上的产品级流畅度已验收通过”。原因不是代码链路未知，而是还缺两项必须在真实设备、真实房间采集的 20 次 p95/p99 证据：

- 用户开口到物理扬声器停播；
- 用户说完到物理扬声器首个可听声音。

统一报告会把 `measured`、`projected`、`simulated` 分开；预算值和模拟值不会再被误写成实测通过。

## 本轮已经解决的问题

1. `VoiceLoop` 不再通过零散 `getattr` 猜测实时语音能力，实时路径由显式 `RealtimeVoiceFrontendPort` / `RealtimeApprovalPort` 约束。
2. 火山实时通道在初次建连失败、运行中断线或历史回滚不确定时进入 fail-closed；只允许在下一本地回合边界用新 `session_id`、空 `dialog_id` 恢复，不续传半句话。
3. 本地打断先停物理播放，再按真实已播放毫秒数截断云端回复；无法证明截断/删除成功时关闭旧实时会话。
4. 新增统一延迟报告 `scripts/eval/report_voice_latency.py`，产品通过必须同时具备两项真实硬件 p95，且每项至少 20 个样本。
5. 真机验收从 `20 + 20` 扩展为 `20 + 20 + 20`：纯扬声器误触发、真人覆盖说话停播、说完到物理首音。
6. 修复实时初次启动失败后无法恢复、无 realtime 初始化的 AudioAgent 路径异常，以及本轮修改文件的静态类型错误。

## 当前速度证据

### 2026-07-18 本机新跑结果

| 指标 | 结果 | 证据类型 | 能否作为产品验收 |
|---|---:|---|---|
| 本地快速路由 | p50 0.0174 ms / p95 0.0289 ms | measured，进程内 300 次 | 否，只证明本地控制开销 |
| 缓存 PCM 入队 | p50 0.0227 ms / p95 0.0337 ms | measured，进程内 300 次 | 否，只证明播放器前队列开销 |
| 说完到首个 PCM 预算 | p50 750.040 ms / p95 750.063 ms | projected | 否，不是物理首音 |
| 快路径回归 | 181 passed | measured，自动化测试 | 证明行为，不证明声学体验 |

预算 750 ms 的组成是：端点候选静音 300 ms、USB 块收敛 180 ms、语音 lead-in 250 ms、调度预算 20 ms，加上本地路由/队列微小开销。它是优化预算，不是人耳实测。

### 线上级联抽样（非物理端到端）

当前配置的 5 次短请求抽样显示：

- DeepSeek 首内容：热态约 0.38–0.66 s，冷启动约 1.79 s；
- MiniMax TTS 首 PCM：热态约 0.56–0.68 s，冷启动约 1.69 s；
- 由组件相加推算，级联热态“说完到声音”约 1.8–2.3 s，冷态约 4.4 s。

这些是网络组件测量加预算推算，不含目标设备声学首音，因此统一报告仍正确返回 `insufficient_evidence`。自由对话要获得更自然的首音，应优先使用火山端到端 S2S；级联保留为确定性业务和故障兜底。

## 产品级项目怎么降低延迟

### OpenAI Realtime

官方把语音架构区分为 live speech-to-speech 和 STT → LLM → TTS 级联；需要低首音、自然轮次和 barge-in 时优先 live session。WebSocket 客户端打断时必须立即停本地播放，记录实际播放时长，并发送 `conversation.item.truncate(audio_end_ms)`，使服务端历史只保留用户真正听到的部分。轮次可使用 `server_vad` 或根据语义动态等待的 `semantic_vad`。

- [Voice agents](https://developers.openai.com/api/docs/guides/voice-agents)
- [Realtime conversations](https://developers.openai.com/api/docs/guides/realtime-conversations)
- [Realtime VAD](https://developers.openai.com/api/docs/guides/realtime-vad)

### LiveKit Agents

`AgentSession` 是唯一会话编排器，集中管理输入、模型流水线、输出、状态和事件。它把延迟优化拆成动态 endpointing、语义/声学 turn detector、adaptive interruption、误打断恢复和 preemptive generation。默认可提前启动 LLM；提前启动 TTS 更快，但会增加取消时浪费的计算。

- [AgentSession](https://docs.livekit.io/agents/logic/sessions/)
- [Turns and interruptions](https://docs.livekit.io/agents/logic/turns/)
- [Turn-taking tuning](https://docs.livekit.io/agents/logic/turns/tuning/)

### Pipecat

Pipecat 用有方向、可中断的 frame pipeline 组织实时链路：`transport.input → STT → user context → LLM → TTS → transport.output → assistant context`。assistant context 放在输出之后，目的是记录真正播出的内容。S2S provider 使用 `realtime_service_mode=True`，把 provider session 与本地 context 分层；内建指标分别记录 TTFB、TTFA、处理时长和文本聚合时长。WebRTC 自带 AEC/降噪/抖动处理，而普通 WebSocket 音频必须由应用自己补齐。

- [Pipeline and frames](https://docs.pipecat.ai/pipecat/learn/pipeline)
- [External/realtime turn management](https://docs.pipecat.ai/api-reference/server/utilities/turn-management/external-turn-management)
- [Metrics](https://docs.pipecat.ai/pipecat/fundamentals/metrics)
- [Transport choice](https://docs.pipecat.ai/client/concepts/choosing-a-transport)

### 火山端到端实时语音

火山 RealtimeAPI 使用长连接、事件驱动的流式语音交互，适合作为普通自由对话的低延迟主通道。它不能替代本地 AEC、物理停播、机器人动作权限、历史提交裁决或真实设备验收；这些仍由本项目本地控制面负责。

- [火山端到端实时语音大模型 API](https://www.volcengine.com/docs/6561/1594356)

## 产品级 Turn 与 Session 管理模型

产品项目普遍不是用一个 `session_id` 解决全部问题，而是分四层：

1. **媒体会话**：麦克风、扬声器、AEC、播放队列，生命周期最长，尽量持续打开。
2. **Provider realtime session**：可替换的云端执行通道；断线或历史不确定就丢弃，在轮次边界重建。
3. **Conversation session**：本地权威上下文和长期记忆；只提交用户 final 文本及用户实际听到的助手内容。
4. **Turn / generation**：一次用户发言和对应回复；所有音频、取消、truncate、历史提交都必须绑定同一 generation。

推荐状态流：

`listening → user_speaking → endpoint_pending → speculating → admitted → speaking → committed`

异常分支：

- 非普通对话或 transcript/generation 不匹配：`speculating → discarded → provider history delete`；
- 真打断：`speaking → physical stop → response cancel → truncate to played_ms → interrupted`；
- 误打断：`speaking → paused → no valid transcript → resume`；
- provider 故障：`active → quarantined → current turn cascade → next-turn fresh session`。

## 与专业项目相比仍缺什么

| 能力 | 当前状态 | 优先级 |
|---|---|---:|
| 单一物理音频所有者、AEC 后同源分发 | 已完成 | — |
| speculative S2S 隔离、generation 栅栏 | 已完成 | — |
| 物理停播 + 云端 cancel/truncate/delete | 已完成 | — |
| 下一轮 fresh provider session 恢复 | 已完成 | — |
| 统一延迟证据与 20+20+20 验收工具 | 已完成，等待真机数据 | P0 |
| 火山线上凭据下的 S2S p50/p95、长稳和故障注入 | 未完成 | P0 |
| 语义/声学 turn detector 与动态 endpoint | 未完成，当前仍以 VAD/端点策略为主 | P1 |
| false interruption resume | 未完成 | P1 |
| 唯一 Turn/Session ledger，避免多处重复写历史 | 部分完成，所有权仍分散 | P1 |
| 长会话空闲期 compaction 与跨进程持久 session | 部分完成 | P2 |

## 下一阶段停止条件

只有同时满足以下条件，才把状态从“受控试运行”升级为“产品级通过”：

1. 目标设备同一房间完成 20 + 20 + 20：误打断率 0、真人覆盖检测率 ≥95%、停播 p95 ≤250 ms/p99 ≤400 ms、物理首音 p95 ≤1.2 s/p99 ≤1.8 s。
2. 火山 S2S 在目标网络完成多轮 p50/p95/p99、断网恢复和长会话测试，且机器人命令/急停始终由本地链路裁决。
3. 增加语义 turn detector 和误打断恢复后，重复做声学验收。
4. 收敛成唯一 `VoiceTurnLedger`，确保 realtime、级联、runtime bridge 只提交一次历史。
