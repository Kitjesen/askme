# AskMe 实时端到端语音供应商

AskMe 的 `voice.realtime` 现在使用统一 `RealtimeDialogueSession` 接口接入三条语音到语音链路。实时模型只允许承接本地门控已批准的普通闲聊；机器人动作、工具、审批、视觉查询和急停始终走本地 `ASR → LLM → TTS` 级联。

| selector | 模型 | 鉴权 | WebSocket |
| --- | --- | --- | --- |
| `qwen3_5_omni` | `qwen3.5-omni-flash-realtime` | `DASHSCOPE_API_KEY` + `DASHSCOPE_WORKSPACE_ID` | `wss://{WorkspaceId}.{region}.maas.aliyuncs.com/api-ws/v1/realtime` |
| `volcengine_duplex` | 豆包 Seeduplex 3.0，`1.2.6.1` | `VOLCENGINE_S2S_API_KEY` | `wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue` |
| `volcengine_s2s` | 豆包旧 O2.0，`1.2.1.1` | `VOLCENGINE_S2S_APP_ID` + `VOLCENGINE_S2S_ACCESS_TOKEN` | `wss://openspeech.bytedance.com/api/v3/realtime/dialogue` |

官方接入资料：[Qwen Realtime API](https://help.aliyun.com/zh/model-studio/realtime)、[获取百炼 Workspace ID](https://help.aliyun.com/zh/model-studio/obtain-the-app-id-and-workspace-id)、[Qwen3.5-Omni Flash Realtime](https://help.aliyun.com/zh/model-studio/qwen3-5-omni-flash-realtime)、[豆包 Seeduplex 3.0 API](https://www.volcengine.com/docs/6561/2549778?lang=zh)、[豆包 3.0 接入必读](https://www.volcengine.com/docs/6561/2549732?lang=zh)。

## 在电脑上直接试玩

先复制 `.env.example` 中对应变量到本地 `.env`，不要把密钥写入 YAML 或提交到 Git。列出声卡：

```powershell
.\.venv\Scripts\python.exe scripts\demo\realtime_voice_chat.py --list-devices
```

Qwen3.5-Omni：

```powershell
.\.venv\Scripts\python.exe scripts\demo\realtime_voice_chat.py `
  --provider qwen3_5_omni
```

豆包 Seeduplex 3.0：

```powershell
.\.venv\Scripts\python.exe scripts\demo\realtime_voice_chat.py `
  --provider volcengine_duplex
```

试玩程序使用半双工按键说话：按 Enter 开始录音，再按 Enter 提交本轮。建议戴耳机，避免扬声器声音重新进入麦克风。它绕开机器人动作层，强制关闭工具和硬件执行能力，只用于感受模型理解、音色和响应速度。

Qwen 当前文档要求 workspace 专属地址。中国内地使用 `cn-beijing`，国际站使用 `ap-southeast-1`，API Key 与 workspace 必须属于同一区域。底层适配器仍能识别旧公共 DashScope 地址以便诊断历史部署，但 AskMe 工厂在启用 `qwen3_5_omni` 时会要求 Workspace ID，不把旧地址报告为可用。若旧地址在 `session.created` 后返回 `Access denied, please make sure your account is in good standing.`，应先补齐 Workspace ID，再检查百炼账户余额、欠费状态和模型调用权限；更换提示词或声卡不会解决这一错误。

## 接入 AskMe

Qwen3.5-Omni 配置：

```yaml
voice:
  realtime:
    enabled: true
    mode: shadow
    provider: qwen3_5_omni
    fallback: cascade
    api_key: ${DASHSCOPE_API_KEY}
    workspace_id: ${DASHSCOPE_WORKSPACE_ID}
    region: cn-beijing
    model: qwen3.5-omni-flash-realtime
    speaker: Tina
    input_mode: audio
    input_sample_rate: 16000
    output_sample_rate: 24000
    output_format: pcm_s16le
    chunk_ms: 20
    end_smooth_window_ms: 800
```

豆包 Seeduplex 3.0 配置：

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

豆包 3.0 的 API Key 需要在火山引擎“豆包语音”新版控制台注册、实名认证、开通模型后创建。旧 ASR 凭据以及旧 O2.0 的 App ID/Access Token 不能代替 3.0 API Key。当前仓库没有可用的 `VOLCENGINE_S2S_API_KEY`，因此只能完成离线协议回归，不能声称已经完成豆包 3.0 公网调用。

正式启用不要从 `split` 直接跳到 `general_chat`：

1. `split` 保持当前级联基线；
2. `shadow` 使用真实公网会话采集错误和延迟，但不播放供应商音频；
3. 完成隐私、稳定性和目标声学设备验收后，小流量启用 `general_chat`；
4. 任何实时供应商错误、超时或安全路由不匹配都回退 `cascade`。

## 公网 API 测试口径

供应商直连测试使用同一份 16 kHz、单声道 PCM，并采用 push-to-talk：建立 WebSocket、发送 `session.update`、分块追加音频、显式 commit，然后按事件时间记录输入转写、首段输出文字、首个输出 PCM、完成事件与 usage。这样得到的是“提交输入到客户端收到事件”的 provider-direct 延迟，不包含真人停说判定、操作系统播放缓冲和物理扬声器。

AskMe 集成验收还需另测“真人停说到物理扬声器首个有效语义声音”和“抢话到扬声器物理停播”。两类数字不能混用；公网首包快不代表整机体验已经达到同一延迟。
