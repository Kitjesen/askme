# Warm Session 运维说明

更新时间：2026-07-28

Warm Session 是 Runtime 级后台保温机制。它从机器人 Runtime 启动后开始运行，到 Runtime 停止时退出；它维护“供应商会话始终可用”的产品契约，但不承诺某一条物理 WebSocket、HTTP 连接或云端 session 永远不变。

## 产品边界

| 概念 | 含义 | 生命周期 |
| --- | --- | --- |
| 已识别的 Conversation Thread | 由受信身份或明确客户端持有的逻辑对话，承载上下文和审计轮次 | 可跨断线、重连、供应商切换继续 |
| 匿名语音 Encounter Thread | 公共场景下一段短时连续对话；不能把上一位路人的历史交给下一位 | 默认空闲 25 秒后轮换；环境/旁观者语音以及 VoiceLoop 停止或重启也会结束 encounter |
| Provider Session | LLM/TTS/ASR/S2S 供应商侧的一条临时连接或会话 | 会按 TTL、网络状态、限流、热切换和刷新策略安全轮换 |
| Warm Session Manager | Runtime 内的调度器，负责探针、刷新、退避、预算和 health 快照 | Runtime start 后运行，Runtime stop 时取消并收割后台任务 |

因此，“机器人开机后一直持续”指 Warm Session Manager 一直维护保温循环，不是把一条云端连接保持到永生。物理连接被供应商关闭、网络中断、模型/音色热切换或主动 TTL 刷新时，Manager 会重新打开或刷新。

`voice.interaction_gate.anonymous_encounter_idle_seconds` 控制匿名语音历史的空闲窗口，默认 25 秒。窗口内的连续对话复用同一 Thread；达到边界后旧 Thread 会关闭并创建新 Thread。这个计时器不关闭 LLM/TTS/ASR/S2S 的 Provider Session，也不影响 Warm Session Manager 在机器人开机期间持续刷新连接。

## 当前目标

Warm Session Manager 维护两个目标：

| 目标 | 默认周期 | 行为 | 费用/资源注意 |
| --- | ---: | --- | --- |
| `llm` | 45 秒 | 使用当前热切换后的 LLM client，但固定通过 `health_model=health-probe` 能力别名发一个 `health_probe` 级别的一 token 流式探针 | 可能产生真实模型调用费用；真实业务请求优先，会取消或延后探针 |
| `tts` | 75 秒 | 调用当前 TTS engine 的 `prewarm_provider_session(force_refresh=True)`，不发送文本 | 不合成语音，但可能占用供应商 session、连接数或 RPM |

TTS provider 的本地复用窗当前配置为 90 秒：

- `voice.tts.minimax_ws_idle_timeout_seconds: 90`
- `voice.tts.volcengine_tts_idle_timeout_seconds: 90`

这低于 MiniMax 文档中的 120 秒 idle close，用于降低冷连接长尾。75 秒刷新间隔应短于 90 秒本地复用窗；如果现场网络或供应商策略更激进，应先用延迟证据调整这两个数。

## 配置

`config.yaml` 和 `config.board.yaml` 当前都启用 Warm Session：

```yaml
warm_sessions:
  enabled: true
  shutdown_timeout_seconds: 0.5
  llm:
    enabled: true
    startup_delay_seconds: 0
    refresh_interval_seconds: 45
    timeout_seconds: 20
    initial_backoff_seconds: 2
    max_backoff_seconds: 120
    busy_retry_seconds: 2
    jitter_ratio: 0.1
    max_attempts_per_hour: 80
  tts:
    enabled: true
    startup_delay_seconds: 0.5
    refresh_interval_seconds: 75
    timeout_seconds: 10
    initial_backoff_seconds: 2
    max_backoff_seconds: 60
    busy_retry_seconds: 1
    jitter_ratio: 0.1
    max_attempts_per_hour: 60
```

字段语义：

| 字段 | 说明 |
| --- | --- |
| `enabled` | 总开关。`true/false/yes/no/on/off/1/0` 都按布尔值解析；字符串 `"false"` 会被正确解析为关闭，不会因为非空字符串误启用 |
| `shutdown_timeout_seconds` | Runtime 停止时等待后台保温任务退出的总预算；默认 0.5 秒，超时任务脱离收割，不阻塞关机 |
| `startup_delay_seconds` | Runtime start 后首次尝试前等待时间 |
| `refresh_interval_seconds` | 成功后下一次强制刷新间隔 |
| `timeout_seconds` | 单次保温尝试超时；超时会触发目标取消钩子 |
| `initial_backoff_seconds` / `max_backoff_seconds` | 失败后的指数退避边界 |
| `busy_retry_seconds` | 目标正忙时的中性重试间隔 |
| `jitter_ratio` | 在调度间隔上加入抖动，避免多机器人同秒打供应商 |
| `max_attempts_per_hour` | 每小时最多尝试次数。耗尽后进入 `throttled`，等窗口重置 |

LLM 的 20 秒是冷连接/模型探针的完成预算，不是业务响应 SLA。2026-07-27
在切换 LiteLLM 前的 DeepSeek 直连开发网络中，一次受控冷进程 1-token 探针总耗时 12.23 秒，若沿用
10 秒会在首次保温真正完成前误判超时；同一 client 复用后的真实语音轮次首有效
短句为 971.8ms。单样本只用于校准超时下限，不用于发布 P50/P95 或产品级延迟结论。

未知字段会 fail-fast 报错，避免配置拼写错误被静默忽略。

## Health 和 readiness 语义

`warm_sessions` 模块的 health 快照包含：

- `status`: 模块状态。禁用或 Manager 正在运行时为 `ok`；启用但未运行时为 `degraded`。
- `enabled`: 总开关是否启用。
- `running`: Manager 调度循环是否正在运行。
- `latency_warm`: 仅当 Manager 正常运行且所有配置目标当前均为 `warm` 时为 `true`；它是热态信号，不改变功能 readiness，也不等于已经证明延迟 SLA。
- `manager_status`: `running`、`degraded` 或 `stopped`。
- `targets`: 每个目标的 `status`、尝试/成功/失败/跳过计数、最近状态、最近原因、最近延迟、剩余预算、上次成功距今和下一次尝试倒计时。

目标状态含义：

| 状态 | 含义 | 运维处理 |
| --- | --- | --- |
| `warm` | 最近一次保温成功 | 正常 |
| `warming` | 正在尝试打开或刷新 | 短暂状态，观察是否转 warm |
| `busy` | 真实业务或 TTS 合成占用目标 | 中性状态；真实请求优先，不应直接判故障 |
| `skipped` | 当前目标不可用、未配置或取消 | 看 `last_reason` 判断是否符合部署预期 |
| `degraded` | 最近尝试失败 | 看供应商凭据、网络、限流和目标模块 health |
| `throttled` | 每小时尝试预算耗尽 | 调大预算前先确认是否有配置错误或供应商故障 |

Warm Session 失败不等于 Conversation Thread 丢失。对话上下文归 Conversation Core 管，Provider Session 可重建。 LLM 保温失败也不会触发直连 provider；默认控制面缺失时保持 fail-closed。

## 真实请求优先

LLM 探针使用 `LLMCallContext(purpose="health_probe", request_class="health_probe")`，并通过 cancel token 注册为可抢占任务。真实业务请求开始时会取消正在进行的 warm probe；如果已有真实请求在跑，新的 health probe 会延后或直接中性跳过。

TTS 强制刷新采用 provider-owned adapter：刷新握手期间旧的可用 session 仍保留给真实合成；只有候选 session 建好、且没有真实使用抢先发生时，才替换旧 session。这样避免“为了保温而打断用户正在听的声音”。

## 排障

### 启动后 `warm_sessions.running=false`

检查：

- 运行蓝图是否包含 `warm_sessions` 模块。
- `warm_sessions.enabled` 是否为 `false`。
- 模块依赖 `llm` 和 `voice` 是否已构建。
- health 中 `status` 是否为 `degraded`。

### `llm` 长期 `degraded`

检查：

- LiteLLM `/health/readiness` 是否成功，`health-probe` 别名是否存在并被 AskMe scoped key 授权。
- `LITELLM_BASE_URL`、`LITELLM_VIRTUAL_KEY` 与 `NO_PROXY` 是否正确；不要用 master/provider key 替代 virtual key。
- 供应商是否对 health probe 计费、限流或拒绝短 prompt。
- 真实请求是否持续占满，导致探针被抢占。

### `tts` 长期 `busy`

`busy` 通常表示正在合成或 provider session 正被真实语音使用。若长期不恢复：

- 检查是否有卡住的 TTS worker。
- 检查 MiniMax/Volcengine 是否触发 session/RPM 限流。
- 检查 `max_attempts_per_hour` 是否被过低配置。

### 冷启动仍然慢

Warm Session 只能减少 provider 连接/握手冷启动，不证明物理首音已达标。仍需测：

- ASR final -> LLM 首个有效语义子句。
- TTS provider 首 PCM。
- 播放设备物理 first nonzero audio。
- barge-in -> physical speaker stop。

没有目标硬件的物理声学证据前，不声明产品级延迟达标。

## 验证

代码回归命令：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_warm_sessions.py -q
.\.venv\Scripts\ruff.exe check askme/runtime/warm_sessions.py askme/runtime/warm_session_targets.py askme/runtime/warm_session_tts.py askme/runtime/modules/warm_session_module.py tests/test_warm_sessions.py
```

Runtime 已运行时，可查看 health：

```powershell
curl http://127.0.0.1:8765/health
```

本文档更新时没有启动长期 Runtime 服务，因此 `curl` 示例需要在部署现场或本地运行时启动后验证。
