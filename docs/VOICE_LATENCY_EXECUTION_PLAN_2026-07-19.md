# 语音交互延迟执行计划（2026-07-19）

本计划把延迟优化从“按周完成若干代码改动”改成“按证据闸门放量”。目标是缩短机器人给出**首个有用语义音频**的时间，同时守住急停、审批、轮次一致性、首字完整和声学稳定性。

## 1. 统一目标与指标

主验收指标：

- `physical_first_semantic_audio_ms`：用户最后一个有效语音样本，到扬声器物理输出首个有用语义音频。
- 目标：P95 ≤ 1.2s，P99 ≤ 1.8s。
- 自由对话与机器人控制分开统计；确认音不能冒充有用语义首音。

必须同时报告：

| 指标 | 含义 |
| --- | --- |
| `asr_endpoint_ms` | 最后有效语音样本到 final/安全提前提交 |
| `llm_first_content_ms` | 请求发出到第一个非空文本 token |
| `llm_first_clause_ms` | 请求发出到第一个强结束标点子句 |
| `llm_first_semantic_clause_ms` | 排除“好的/收到”等纯寒暄后的首个有效子句 |
| `tts_provider_first_pcm_ms` | TTS 请求到供应商首 PCM |
| `tts_buffer_commit_ms` | TTS 请求到首批 PCM 进入播放缓冲 |
| `physical_first_nonzero_ms` | TTS 请求到声卡/回采首个非静音样本 |
| `barge_in_physical_stop_ms` | 覆盖说话到扬声器物理停播 |
| `first_word_integrity` | 首字未截断、未出现数字/软起音丢失 |
| `underrun_or_zero_gap` | 首片过小造成的断音或零洞 |

各阶段 P95 不能简单相加后当作端到端 P95；最终结论只取同一轮次的端到端实测。

## 2. 执行顺序

### Gate 0：正确性与可观测性

状态：本轮已完成。

- Conversation Core 统一 Thread/Turn/Generation；同 Thread 至多一个非终态 Turn。
- 跨 local/realtime/runtime 冲突 fail-closed，HTTP 返回 409，不污染旧历史。
- 并发 RAG、认知和 latency trace 使用任务级上下文隔离。
- Conversation Core 降级进入 `/ready`；不就绪返回 HTTP 503，阻止 Kubernetes 继续导流。
- 延迟报告只允许 `measured`、`projected`、`insufficient` 三种证据状态；样本不足不能宣称达标。

### Gate A：无需新云凭据的代码快赢

1. 语义优先短首句
   - 首句优先给结论、动作状态或澄清问题，尽量 10 字以内并使用强结束标点。
   - 安全告警、拒绝和澄清不先说“好的”。
   - 同时测首句合规率、泛化寒暄率和安全场景错误承诺率。

2. 冷启动预热
   - LLM 预热使用真实 `voice_model`、`max_tokens=1`、零温度并完整消费流。
   - 预热任务不阻塞功能 readiness；另设 `latency_warm` 表示是否可以承诺热态 SLA。
   - 失败只记安全诊断，不输出密钥、请求正文或供应商响应正文。
   - 启动与 immediate/pending 运行时 TTS provider 切换都后台预热；替换或 shutdown 取消旧任务，不合作 provider 只允许留下 daemon，并以 0.5 秒总预算有界收割。

3. 高频短语缓存
   - 只预合成具有稳定 `cache_key` 且在运行时真实消费的固定短语。
   - 后台运行时预热必须使用隔离 TTSEngine，不能清空 live TTS buffer。
   - 缓存签名包含 schema、provider、model、voice、speed、volume、pitch、transport、sample rate 等声学参数。
   - 供应商失败后的 fallback 音频不能写进主供应商缓存键。
   - “好的，请跟我来”属于动作承诺，只能在审批通过和 runtime 正式接单后播放。

4. 动态端点
   - 首阶段只加速精确匹配、幂等、纯本地的停止类指令。
   - 急停优先于静音门、InteractionGate、待审批工具、ACK、Memory、LLM 和 S2S。
   - QUICK_REPLY、只读查询和自由对话继续使用 300ms 兜底，直到负样本与真机证据通过。
   - 候选变化、失配、恢复说话和 ASR session 重启必须同时重置静默与稳定计时。

5. TTS 首包实验能力
   - 首片与续片阈值以播放域毫秒配置，不使用误导性的固定 provider 样本数。
   - WebSocket 与 SSE 必须复用同一缓冲状态机。
   - 默认值保持现状；更小首片只通过 A/B profile 启用，并要求 onset、首字和 underrun 测试通过。
   - 当前 44.1kHz 播放域 2400→1600 样本的理论差约 18ms，不能预先宣称能节省 50–100ms。

6. 长尾感知反馈
   - 1.5 秒 thinking fuse 只服务慢轮次；空 keep-alive delta 不取消，首个真实文本/tool payload 或 turn cancel 会原子阻止/停止反馈。
   - ACK 与 thinking 使用独立限频，首个语义 `speak` 会先取消仍在播放的反馈音，避免叠音。
   - thinking/ACK 只属于 `feedback`，绝不能填入 `physical_first_semantic_audio_ms` 或 TTS 首语义音指标。

Gate A 退出条件：聚焦测试与静态检查通过；模型验证至少 20 次；任何收益仍标记为实验室 measured，不等于真机通过。

### Gate B：供应商与参数 A/B

- 使用固定的 [20 句中文语料](../scripts/eval/corpora/tts_zh_20_v1.json)，同一时段按 case 交替测 MiniMax 与火山 TTS，避免按先后顺序引入网络偏差。
- 每条记录 provider、model、transport、voice、sample rate 和冷/热状态。
- 样本数少于 20 时不计算/不发布 P95，不选择胜者。
- 决策优先级：首字完整与无断音 > 物理首音 > 音质盲评 > 成本；败者保留为 fallback。
- 动态端点对抗文本至少 1000 条、录音 partial 轨迹至少 200 条。`0/N` 必须连同 N 报告，不能写成绝对“误截率为零”。

### Gate C：S2S 与真机声学放量

- 火山 S2S：`disabled/split` → `shadow` → `general_chat`，不跨级；中央 policy、durable Turn/Generation 与两阶段 audio release 已完成离线回归，旧一步入口已 fail-closed。下一闸门仍是线上凭据、shadow 隐私/稳定性和真机验收，不能因为单测通过就直接启用 `general_chat`。
- 急停、机器人命令、审批和高风险工具始终由本地级联裁决。
- 目标麦克风/音箱一体设备执行 20+20+20：安静、回声覆盖说话、环境噪声。
- 必测误触发、覆盖说话停播 P95/P99、物理语义首音 P95/P99、断线恢复和供应商回退。
- 只有 Gate C 通过，README 才能从“projected/实验室 measured”改为“产品级通过”。

## 3. 当前状态

| 项目 | 状态 |
| --- | --- |
| 语义优先短首句与验证脚本 | 已实现并用完整 runtime persona 完成 20 样本复测；首个有效子句 P50 1168.4ms / P95 1378.8ms，短句合规率 95%、错误静默率 0%，正确性提升但延迟未达目标 |
| LLM / TTS 非阻塞预热 | LLM 启动预热已加固；TTS 启动与 immediate/pending provider 切换均后台预热，替换/shutdown 有取消与 0.5 秒有界收割 |
| phrase cache / TTS 首续片 / 长尾反馈 | 已实现并回归；隔离引擎后台 prime，缓存签名升级为 v2；默认仍为旧 2400 样本，仅配置显式启用 36/54ms 实验值，尚无真机收益结论。1.5 秒 thinking fuse 已修复空 delta、取消、ACK 限频与语义音叠音竞态，但只改善等待感 |
| 急停优先与安全动态端点 | 已实现并回归；精确急停阈值 150ms，受 160ms 稳定度和至少两次观察约束，普通意图仍为 300ms |
| 统一延迟证据报告 | 已扩展；统一 7 类阶段指标，少于 20 样本不计算 P95，损坏/简略证据 fail-closed，MiniMax/火山只在同 corpus、同 case_id 的 measured 数据上选择延迟胜者 |
| MiniMax / 火山 TTS 在线采集 | MiniMax 在线采集器已实现并完成同语料冷/热 measured；火山 TTS V3 已完成协议 codec、并发安全 WebSocket client、`TTSEngine` backend、配置、health、dashboard、online smoke 和固定 20 句 collector 的离线接线。当前环境缺少 `VOLCENGINE_TTS_*` 凭据，没有火山 online measured 数据，因此不能宣称 Task 5 已完成或已经选出胜者 |
| 火山 S2S split/shadow/general_chat 基础设施 | 真实网络双向接线已存在，默认仍 `disabled`/`split`；中央 `decide_realtime_route` 与 `prepare → durable Turn/Generation → release` 已回归，旧一步放音入口 fail-closed。当前只解锁 shadow 前的软件闸门，尚未解锁线上凭据、隐私、声学和 `general_chat` 放量 |
| 20+20+20 真机工具 | 报告已升级为 fail-closed `askme.full_duplex_hardware.v2`；entry/stopwatch 只算 `manual`，loopback 只算 `render_chain`，均不能满足 required physical gate。物理 stop/first-sound 各需 ≥20 条自动 capture/reference、同 monotonic clock、有效校准、零丢帧的 `physical_acoustic` trial；重叠停播还需独立 `isolated_speaker_monitor`。自动真机采集适配器尚未实现，当前没有目标设备 measured 证据 |
| CI / Issue / PR 模板 | 已存在 |
| MIT LICENSE / 英文 README | 已完成；既有 CI、Issue、PR 模板已复核 |

### 2026-07-19 Task 1 实测校正

旧脚本把第一个任意 stream chunk 当作 TTFT，可能包含空 delta，因此“约 110ms”不能作为首文本或首子句证据。新脚本 [verify_voice_model.py](../scripts/eval/verify_voice_model.py) 使用实际 persona，分开记录首 chunk、首个非空文本、首个强标点子句和首个非寒暄语义子句。

| 20 样本组 | 首文本 P50/P95 | 有效子句 P50/P95 | 首句 ≤10 字 | 错误静默率 |
| --- | --- | --- | --- | --- |
| 完整 runtime prompt，修复前 | 759.0 / 955.5ms | 940.3 / 1104.1ms | 实际可播子句 13/14 ≤10 字 | 30% |
| 完整 runtime prompt，严格短句 + 静默修复 | 961.8 / 1140.7ms | 1168.4 / 1378.8ms | 95% | 0% |

两组是顺序运行而非交替 A/B，网络时段和上下文也不同，不能把延迟差归因于提示词。可确定的是：完整 runtime prompt 已消除这组直接问题的错误 `[SILENT]`，并把短首句合规率提高到 95%；但当前 LLM 子句本身的 P95 已超过端到端 1.2s 目标。后续必须做交替 A/B，并把 provider TTFT、prompt token 体积、短语缓存、端点和 TTS 一起优化；不能再把提示词单独写成“预计提前 200ms”。

### 2026-07-19 Gate A 代码验收

- S2S policy、VoiceLoop、两阶段准入、coordinator、session 与 Conversation Core 聚焦回归：180 passed；修复旧一步绕过后独立复审矩阵 144 passed，并给出 APPROVE。
- StreamProcessor、thinking feedback、AudioAgent 热切换预热与 TTS 相邻回归：182 passed；ruff 通过。
- `askme.full_duplex_hardware.v2`、交互评估器、capture provenance 与统一报告：70 passed；ruff check/format 通过。
- 火山 TTS V3 协议/client/TTSEngine/collector 聚焦回归：113 passed；MiniMax collector 当前专项 13 passed。
- 完成全部修正后的跨模块最终回归：561 passed；`PytestUnraisableExceptionWarning` 按 error 处理，ruff 通过，`git diff --check` 通过。
- 以上 shard 有重叠，不能相加当作唯一测试总数；它们证明软件契约与回归测试通过，不证明目标麦克风/音箱的物理延迟已经达标。

### 2026-07-19 MiniMax TTS 在线 measured

新增 [measure_minimax_tts_latency.py](../scripts/eval/measure_minimax_tts_latency.py) 直接驱动 MiniMax TTS 生成路径并观察 `tts_buffer`，不会启动 sounddevice 或物理扬声器播放。采集器固定读取 [tts_zh_20_v1.json](../scripts/eval/corpora/tts_zh_20_v1.json)，禁用短语缓存，记录 `provider_first_pcm_ms`、`buffer_commit_ms`、`total_synthesis_ms`、model、transport 与 cold/warm 连接标签；MiniMax provider 失败时会抑制 fallback，失败样本不伪造延迟数值。供应商 RPM 较低时使用 `--case-delay-ms` 分散 cold/new-connection 采样；该等待不计入单条样本 latency，但会写入证据文件。

| 模式 | provider 首 PCM P50/P95 | buffer commit P50/P95 | 总合成 P50/P95 | 样本 |
| --- | --- | --- | --- | --- |
| Warm WebSocket 复用 | 270.71 / 376.08ms | 277.01 / 379.32ms | 507.30 / 888.54ms | 20/20 passed |
| Cold 每 case 新连接（4.5s case 间隔） | 631.75 / 2294.78ms | 652.09 / 2314.36ms | 988.03 / 4125.89ms | 20/20 passed |

最新原始证据位于 `artifacts/voice/minimax-tts-warm-20260719T200056Z.json`、`artifacts/voice/minimax-tts-cold-retry-20260719T201344.json`，统一报告位于 `artifacts/voice/voice-latency-report-minimax-warm-cold-retry-20260719.json`。Warm 20 个 case 中首个连接为 `warm_opened`，其余为 `warm_reused`；Cold 使用每 case 新连接并加入 4.5s 间隔以避开供应商高频新连接失败。统一报告退出为 `insufficient_evidence` 是预期行为：这组数据没有物理回采 `physical_first_nonzero_ms`，也缺少 `barge_in_to_speaker_stop_ms`，不能证明目标麦克风/音箱一体设备上的产品级首音达标。

本次并行执行还暴露了一次证据文件同名覆盖事件：较早一轮的汇总报告仍保留，但原始 case JSON 被后续运行替换，因此较早一轮只作为探索性波动参考，不作为最终可审计结论。采集器现已改为默认生成带时间戳和随机后缀的唯一文件名；显式 `--out` 指向既有文件时也会失败，只有操作者主动传入 `--overwrite` 才能覆盖。一次无间隔 cold 重测出现 13/20 passed、7/20 provider failure，这进一步说明“每轮新建 TTS 连接”不应作为产品主路径；后续 provider A/B 必须使用唯一 experiment_id，并采用交替顺序。

### 2026-07-19 火山 TTS V3 接线状态

实现依据是火山引擎[流式语音合成 API V3](https://www.volcengine.com/docs/6561/2532486?lang=zh) 和[双向流式语音合成 WebSocket 说明](https://www.volcengine.com/docs/6561/1329505)。当前代码按 V3 双向 WebSocket 路径接线：协议 codec、并发安全 client、`TTSEngine` backend、配置、health、dashboard、online smoke 和固定 20 句采集器均已具备离线回归入口。默认 backend 仍是 MiniMax；火山 TTS 的 `resource_id` 与 `speaker` 由账号授权决定，不在代码中硬编码。文档和配置中的 `model` 字段在火山 TTS 路径下表示 `X-Api-Resource-Id` 对应的资源/产品选择，而不是额外写入请求体的模型名。

当前环境没有可用的 `VOLCENGINE_TTS_*` 凭据，所以尚未生成火山同语料 online measured 证据。可在拿到账号资源后运行：

```powershell
$env:VOLCENGINE_TTS_API_KEY = "<api-key>"
$env:VOLCENGINE_TTS_RESOURCE_ID = "<resource-id>"
$env:VOLCENGINE_TTS_SPEAKER = "<speaker>"
python scripts/eval/measure_volcengine_tts_latency.py --mode warm --case-delay-ms 4500
python scripts/eval/measure_volcengine_tts_latency.py --mode cold --case-delay-ms 4500
python scripts/eval/report_voice_latency.py `
  --experiment artifacts/voice/<minimax-warm>.json `
  --experiment artifacts/voice/<minimax-cold>.json `
  --experiment artifacts/voice/<volcengine-warm>.json `
  --experiment artifacts/voice/<volcengine-cold>.json `
  --out artifacts/voice/voice-latency-report-tts-ab.json
```

以上命令只测 provider/software boundary，不播放扬声器，也不等于物理首音。若账号仍使用旧授权方式，可改用 `VOLCENGINE_TTS_APP_ID` + `VOLCENGINE_TTS_ACCESS_KEY`；缺 `resource_id` 或 `speaker` 时采集器会 fail-fast。`data/voice/system_control.json` 的运行时持久状态可覆盖 YAML，线上切换 backend 时应通过 control API 或明确更新该状态文件，避免仪表盘显示与实际 TTS backend 不一致。

## 4. 与产品级语音项目的最新对齐

- [OpenAI Realtime conversations](https://developers.openai.com/api/docs/guides/realtime-conversations) 的 WebSocket 打断语义要求客户端停止播放并按实际播放进度截断上下文；本项目对应 `Generation + played_ms + cancel`，不能把未播放全文写成用户已听历史。
- [LiveKit turn tuning](https://docs.livekit.io/agents/logic/turns/tuning/) 把端点、adaptive interruption、preemptive generation 与降噪作为同一个 turn-taking 调优面；因此下一阶段会把普通对话升级为“VAD + partial 稳定度 + 语义完整度”，而不是一刀切缩短 silence。
- [Pipecat context management](https://docs.pipecat.ai/pipecat/learn/context-management) 建议把 assistant context 聚合放在 transport output 之后，以反映实际播出的内容；这与 Conversation Core 的交付后 commit 和被打断 Generation 设计一致。
- [Pipecat metrics](https://docs.pipecat.ai/pipecat/fundamentals/metrics) 和 LiveKit observability 都以 turn 记录多阶段指标；本项目统一报告同样以 Turn/case_id 关联 ASR、LLM、TTS、物理播放与打断，不再相加各阶段 P95。
- [火山端到端实时语音](https://www.volcengine.com/docs/6561/1594360) 继续只作为自由对话低延迟候选；急停、审批、导航和机器人动作保留本地级联裁决。中央 policy、Turn/Generation ledger 与两阶段 audio release 补丁已完成离线回归，但 `general_chat` 仍不应直接启用；必须先完成有凭据的 shadow、隐私确认和真机验收。
- [火山 TTS V3 流式语音合成](https://www.volcengine.com/docs/6561/2532486?lang=zh) 与[双向流式 WebSocket](https://www.volcengine.com/docs/6561/1329505) 供应商路径加入“LLM 文本直流式输入”与现有子句切分流水线的同语料 A/B，避免未经测量地假设哪种切分更快。

## 5. 停止规则

- 首音变快但 `first_semantic_clause` 不变：视为无产品收益。
- 出现任何安全错误承诺、首字截断、误提交或急停被门控：立即回滚实验 profile。
- P95 改善小于测量噪声，或以 underrun/音质显著下降为代价：保持原参数。
- 缺少云凭据、物理回采或目标设备时，只交付可复现的测试入口和 `insufficient` 结论，不伪造 measured 数据。
