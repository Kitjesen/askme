# 火山端到端实时语音接入与上线边界（O2.0 / Seeduplex 3.0）

本文描述 AskMe 机器人如何在普通“音响 + 麦克风”一体设备上接入火山引擎豆包端到端实时语音。该方案不依赖 ROS2：本机负责声卡、AEC、VAD、打断、播放代际和安全路由，云端负责端到端语音理解与语音生成。

旧 O2.0 实现依据是火山引擎[RealtimeAPI WebSocket 文档](https://www.volcengine.com/docs/6561/1594356)；新版实现依据是[豆包 Seeduplex 3.0 API](https://www.volcengine.com/docs/6561/2549778?lang=zh)及[全双工接入必读](https://www.volcengine.com/docs/6561/2549732?lang=zh)。

## 当前接入边界

- `volcengine_s2s`：旧 O2.0 `1.2.1.1`，`/api/v3/realtime/dialogue`，使用 App ID + Access Token + `volc.speech.dialog`
- `volcengine_duplex`：Seeduplex 3.0 `1.2.6.1`，`/api/v3/duplex/realtime/dialogue`，仅使用新版控制台 API Key
- 输入：16 kHz、单声道、PCM S16LE，每 20 ms 一帧（640 字节）
- 输出：24 kHz、单声道、PCM S16LE
- `end_smooth_window_ms`：只允许官方范围 500–50000 ms
- SC2.0 `2.2.0.0` 仍失败关闭；它不能借用 O2.0 或 3.0 的协议路径
- 本地级联链路始终保留，火山链路不可用或被隔离时自动回到级联路径

火山端到端语音只处理已经通过本地交互门控的普通对话。机器人动作、工具调用、审批、急停、固定命令和不确定意图继续走本地安全路由及原有级联链路。端到端会话没有机器人硬件执行接口，也不被授权声称动作已经执行。

## 运行结构

```text
普通音响/麦克风一体设备
        |
本地采集 -> AEC -> VAD/交互门控 --------------------+
        |                                           |
        | 普通闲聊且已批准                          | 命令/任务/审批/急停
        v                                           v
火山端到端实时语音                         本地级联 ASR -> LLM -> TTS
        |                                           |
        +-------------- 统一 PCM 播放 --------------+
                               |
                        AEC 播放参考 + 可抢话打断
```

同一份经过 AEC 的麦克风音频进入候选实时会话；火山返回音频先处于隔离状态。VoiceLoop 先调用唯一的中央 `decide_realtime_route`，再让 provider `prepare` 候选（PCM 仍缓冲），只有 Conversation Ledger 已 append+fsync 创建规范 Turn 和 Generation 后才 `release` PCM。冲突、缺少 ledger、Generation 写入失败、取消或代际变化都会在 release 前 fail-closed；旧的一步批准/放音入口只保留为拒绝兼容 shim。这样既能并行准备首包，又不会让云端模型绕过机器人任务、安全审批、急停或会话事实源。

## 凭据与配置

Seeduplex 3.0 需在豆包语音新版控制台开通模型并创建 API Key。旧 ASR 或 O2.0 凭据不能代替该 Key：

```powershell
$env:VOLCENGINE_S2S_API_KEY = "<API Key>"
```

3.0 第一阶段配置：

```yaml
voice:
  realtime:
    enabled: true
    mode: shadow
    provider: volcengine_duplex
    fallback: cascade
    endpoint: wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue
    api_key: ${VOLCENGINE_S2S_API_KEY}
    model: 1.2.6.1
    speaker: zh_male_xiaotian_jupiter_bigtts
    input_mode: audio
    input_sample_rate: 16000
    output_sample_rate: 24000
    output_format: pcm_s16le
    chunk_ms: 20
```

旧 O2.0 从旧控制台取得应用 ID 和 Access Token。凭据只放入运行环境，不写入 YAML、日志、健康接口或提交记录：

```powershell
$env:VOLCENGINE_S2S_APP_ID = "<应用 ID>"
$env:VOLCENGINE_S2S_ACCESS_TOKEN = "<Access Key>"
```

旧 O2.0 第一阶段配置：

```yaml
voice:
  realtime:
    enabled: true
    mode: shadow
    provider: volcengine_s2s
    fallback: cascade
    endpoint: wss://openspeech.bytedance.com/api/v3/realtime/dialogue
    app_id: ${VOLCENGINE_S2S_APP_ID}
    access_token: ${VOLCENGINE_S2S_ACCESS_TOKEN}
    resource_id: volc.speech.dialog
    model: 1.2.1.1
    input_mode: audio
    input_sample_rate: 16000
    output_sample_rate: 24000
    output_format: pcm_s16le
    chunk_ms: 20
    end_smooth_window_ms: 800
```

`shadow` 会建立真实端到端会话并采集延迟、错误和路由数据，但不允许火山音频覆盖当前用户体验。只有在凭据联调、隐私检查和目标设备声学验收通过后，才改为：

```yaml
voice:
  realtime:
    mode: general_chat
```

不要直接从默认的 `split` 跳到 `general_chat`。发布顺序固定为：

1. `split`：现有级联链路作为基线；
2. `shadow`：验证网络、协议、首包延迟、错误率和会话回收；
3. `general_chat` 小流量：只放行普通闲聊，并持续保留级联回退；
4. 扩大流量：必须以真机数据为依据，不以单元测试通过代替声学验收。

## 会话历史的安全隔离

端到端服务可能在本地意图判定完成前生成回复并写入云端会话。provider 音频此时只允许停留在候选缓冲区，不得先播后补账。若本地随后判定该轮是机器人任务、命令、视觉查询、待审批工具、未被寻址的环境语音或其他不允许走端到端的内容，运行时会：

1. 阻止该代际的云端音频进入播放器；
2. O2.0 使用 `ConversationDelete`（事件 514，确认事件 571）；3.0 使用 `conversation.item.delete` 及相关确认删除对应问答；
3. 在删除确认前隔离该实时会话；
4. 删除失败或连接异常时关闭实时通道，并回退到本地级联链路。

`ConversationDelete` 是会话一致性保护，不是机器人动作安全机制。真正的动作权限仍由本地意图、审批、急停和执行层控制。

## 普通音响/麦克风的真机验收

音响与麦克风靠得很近时，扬声器回声是最大风险。云端模型不能替代本地 AEC；没有通过 AEC 就不能把“边播边听”视为专业全双工。

在目标设备、目标驱动和实际房间中至少执行：

- 20 次仅扬声器播放：不得把自己的播报误判为用户抢话；
- 20 次真人覆盖说话：应稳定停止当前播报并接收新一轮；
- 20 次自然说完测试句：记录到扬声器首个可听声音的真实延迟；
- 用户开口到扬声器停止：p95 ≤ 250 ms，p99 ≤ 400 ms；
- 用户说完到物理首音：目标 p95 ≤ 1.2 s，p99 ≤ 1.8 s；
- 被打断的未播放回复不得写入本地助手历史；
- 断网、火山连接失败、队列溢出或 AEC 失效时必须退回安全级联/半双工；
- 本地唤醒、固定命令和急停在云端故障时仍可用。

执行已有真机评估器：

```powershell
python scripts/eval/evaluate_full_duplex_hardware.py `
  --config config.board.yaml `
  --status-source http://127.0.0.1:8765/health `
  --output artifacts/voice/full-duplex-hardware.json
```

任何音频设备、驱动、摆位、机箱结构或房间声学变化都需要重新做 20 + 20 + 20 验收。详尽的 AEC、打断与真实首音标准见 [FULL_DUPLEX_VOICE.md](FULL_DUPLEX_VOICE.md)。

当前交互式评估器的 entry/stopwatch 只用于流程演练和 `manual` 诊断，即使数值很好也会 fail-closed，不能解锁上述门槛。loopback 只标记为 `render_chain`。真正的 v2 物理证据需要后续采集适配器通过 `build_instrumented_trial_evidence()` 写入自动 capture/reference、同 monotonic clock、校准、零丢帧与独立扬声器监听来源；该自动适配器目前尚未实现。

## 目前仍需用真机和线上凭据关闭的差距

单元测试和模拟 WebSocket 可以证明协议、队列、取消、代际隔离及回退逻辑，但不能证明以下生产指标：

- 目标网络到火山服务的真实连接成功率、首包延迟、长会话稳定性和配额余量；
- 一体式音响/麦克风在实际音量、距离和噪声下的回声消除及误打断率；
- 连续会话下的语义断句、犹豫停顿和假打断恢复体验；
- 声卡热插拔和驱动断流已具备本地自动重开、句柄清理、就绪验证与有界退避，但仍需在目标设备验证恢复时间与长稳；火山实时通道运行中断线后只会在**下一本地回合边界**以全新 `session_id`、空 `dialog_id` 恢复，恢复中或失败时整轮走级联，不续传半句话；旧 session 事件按 session/generation 栅栏丢弃；
- 无法证明拒绝回合已从云端历史删除时，旧实时 session 会 fail-closed 关闭，后续仅允许 fresh-session 恢复，避免迟到 provider turn 污染上下文；
- 真实线上异常下 `ConversationDelete` 的确认时延与隔离回退。

中央路由、两阶段 release、Conversation Ledger 和旧一步入口 fail-closed 已完成离线回归，因此软件安全闸门可进入下一步 shadow；这不等于已经具备 `general_chat` 生产条件。当前仓库没有可用的 Seeduplex 3.0 API Key，所以 3.0 尚未验证公网握手、真实事件顺序和首音。当前状态应定义为“具备受控 shadow 试运行条件”，不能定义为“所有专业语音差距已经归零”。完成线上凭据联调与隐私确认、20 + 20 + 20 真机验收、长时稳定性与故障注入后，才能形成生产上线结论。
