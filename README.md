# Askme — 穹沛科技语音 AI 助手

Thunder 机器狗的语音控制大脑。麦克风 → VAD → ASR → 意图路由 → LLM → TTS → 扬声器，全链路低延迟。

---

## 快速启动

```bash
# 文字模式（无需音频设备）
python -m askme

# 语音模式
python -m askme --voice

# 机器人模式（语音 + 机械臂/狗控制）
python -m askme --voice --robot
```

---

## 部署到 aarch64 Linux（sunrise 机器）

### 环境准备

```bash
pip install -r requirements.txt
pip install mcp>=1.0.0 sherpa-onnx pytest-asyncio   # 额外依赖
```

### .env 配置

```env
# LLM（Claude Relay）
LLM_API_KEY=cr_xxx
LLM_BASE_URL=https://cursor.scihub.edu.kg/api/v1

# MiniMax（LLM + TTS，两个都要）
MINIMAX_API_KEY=sk-api-xxx
MINIMAX_GROUP_ID=xxxxxxx

# TTS 音色/情绪
TTS_VOICE_ID=male-qn-qingse
TTS_SPEED=1
TTS_EMOTION=happy
```

### config.yaml 关键参数（远程机器）

```yaml
brain:
  timeout: 60.0          # 默认 30s，远程网络延迟高，必须调大
  voice_model: MiniMax-M2.5-highspeed

voice:
  tts:
    backend: minimax
    sample_rate: 48000   # 必须匹配硬件，见下方"音频双播问题"
    output_device: 2     # 见下方"音频双播问题"
```

---

## ⚠️ 已知坑 & 解决方案

### 1. 音频双播问题（TTS 播放两遍）

**现象**：TTS 语音每句话播放两遍（同一段话连续响两次）。

**根因**：`TTSEngine._playback_loop` 使用 `sd.OutputStream` + callback 方式播放，在 PipeWire 管理的 ALSA 系统（sunrise aarch64）上触发双播。

通过系统性排查确认：
- `aplay -r 24000 -f S16_LE -c 1 /tmp/test.pcm` — 只响一次 ✅
- `sd.play(samples, samplerate=48000)` — 只响一次 ✅
- `sd.OutputStream` + callback — 响两次 ❌（bug 就在这里）

更换 `output_device`（null/1/2）和 `sample_rate`（24000/44100/48000）均无效，也不是 `~/.asoundrc` 路由冲突问题。

**修复**（v4.3.4）：

根因是 MiniMax SSE 协议：`status=1` 事件逐块流式发送音频，`status=2` 最终事件**再次完整发送一遍**所有音频（附带 `extra_info` 元数据）。原代码把两者都累积进 `all_samples`，导致 combined PCM = 2× 实际长度，任何播放器都会响两遍。

修复：在 `_generate_minimax()` SSE 解析循环中检查 `payload["data"]["status"]`，遇到 `status == 2` 直接 `continue` 跳过音频。

`_playback_loop` 同步改为 `subprocess + aplay` 逐句播放（绕过 PortAudio/PipeWire 交互）。

**代码位置**：`askme/voice/tts.py` → `_generate_minimax()`、`_playback_loop()`

---

### 2. 音频设备参考（仅供排查参考，非双播根因）

在 sunrise 机器上 `python3 -c "import sounddevice as sd; print(sd.query_devices())"` 输出：

```
  0 MCP01: USB Audio (hw:1,0)   ← 直接硬件
  1 pipewire                     ← portaudio segfault，不能用
  2 pulse                        ← PipeWire PulseAudio 兼容层
* 3 default                      ← 取决于 ~/.asoundrc
```

- `output_device: 1`（pipewire）：portaudio 崩溃，segfault ❌
- `output_device: null`（system default）：通常可用 ✅

---

### 2. 远程 LLM 超时（streaming 30s 超时）

**现象**：`Request timed out.` 报错，所有模型 fallback 后仍超时。

**根因**：`config.yaml` 默认 `timeout: 30.0`，远程机器到 Relay 服务延迟较高，SSE 流式响应超过 30s。

**修复**：
```yaml
brain:
  timeout: 60.0
```

---

### 3. voice_model 用 MiniMax 但没有 API Key

**现象**：`MiniMax-M2.5-highspeed returned 503`，fallback 到 claude-sonnet 再超时。

**修复**：确保 `.env` 里有 `MINIMAX_API_KEY`，或把 `voice_model` 改为 `claude-opus-4-6`。

---

### 4. `<think>` 块污染对话历史

**现象**：MiniMax M2.5 输出带 `<think>...</think>`，虽然 TTS 不播，但存入了对话历史。

**修复**：`brain_pipeline.py` 的 `_stream_with_tools` 和 `_stream_and_speak` 现在用 `_ThinkFilter` 过滤后的文本作为 `full_response`（v4.3.2 已修）。

---

### 5. 缺失依赖（远程新机器）

```bash
pip install mcp>=1.0.0      # health endpoint 依赖
pip install sherpa-onnx      # ASR/VAD/KWS/TTS 模型
pip install pytest-asyncio   # 异步测试
pip install edge-tts         # TTS 备用后端（可选）
pip install miniaudio        # edge-tts 解码依赖
```

---

### 6. GitHub 在国内远程机器上不可用

**现象**：`git fetch origin` 无响应，GitHub blocked。

**部署方式**：从开发机用 `scp` 直接推文件：
```bash
scp askme/pipeline/brain_pipeline.py sunrise@192.168.66.190:~/askme/askme/pipeline/
scp askme/voice/tts.py sunrise@192.168.66.190:~/askme/askme/voice/
# 注意：tests/tmp/ 目录有 pytest 锁文件，不要 scp 整个 tests/
scp tests/test_*.py sunrise@192.168.66.190:~/askme/tests/
```

---

## 音频设备配置参考

| 设备索引 | 名称 | 适用情况 |
|---------|------|---------|
| `null` | 系统 default | 通常可用 ✅；取决于 ~/.asoundrc 配置 |
| `0` | hw:1,0 直连 | 无 PipeWire 时可用 |
| `1` | pipewire | portaudio 崩溃（segfault）❌ |
| `2` | pulse | PipeWire PulseAudio compat ✅ |

检查当前机器设备：
```python
import sounddevice as sd
print(sd.query_devices())
```

MCP01 USB 音频原生采样率 48000 Hz，`sample_rate` 必须设 48000，否则 ALSA 二次 resample 导致音质下降。

---

## 架构概览

```
麦克风
  └── VAD (silero) → KWS (wake word) → ASR (sherpa-onnx / MiniMax)
                                          │
                                    IntentRouter
                                    ├── SKILL  → SkillDispatcher → SkillExecutor → LLM
                                    ├── ESTOP  → DogSafetyClient
                                    └── GENERAL → BrainPipeline
                                                    ├── MemoryBridge (MemU 向量库)
                                                    ├── LLMClient (MiniMax / Claude Relay)
                                                    │     └── streaming → _ThinkFilter → StreamSplitter
                                                    └── AudioAgent → TTS (MiniMax SSE) → sounddevice
```

## 测试

```bash
# 本地
pytest tests/ -q

# 远程烟雾测试
python3 -c "
import asyncio
from askme.app import AskmeApp
async def t():
    app = AskmeApp()
    r = await app._pipeline.process('你好')
    print(r)
asyncio.run(t())
"
```

---

## 版本历史

| 版本 | 内容 |
|------|------|
| v4.5.0 | 音量/语速控制技能（volume_up/down/reset、speed_up/down/reset）；`create_skill` 工具支持运行时动态创建技能并热加载 |
| v4.4.0 | 语音控制技能：`mute_mic`（闭麦）、`unmute_mic`（开麦）、`stop_speaking`（静音/停止TTS）；E-STOP 同步清空 TTS 队列 |
| v4.3.5 | `OTABridge.status_snapshot()` 死锁修复：内联 `_is_registered()` 检查避免 `threading.Lock` 重入 |
| v4.3.4 | TTS 双播根因修复：跳过 MiniMax SSE status=2 重复音频事件 |
| v4.3.3 | TTS 播放器：`sd.OutputStream` callback → `subprocess+aplay` 逐句播放 |
| v4.3.2 | `<think>` 块不再污染对话历史 |
| v4.3.1 | UX 反馈、线程安全、LLM 并发防护 |
| v4.3.0 | P0/P1 全量 bug 修复 |
| v4.2.0 | MiniMax M2.5 语音 + SkillDispatcher 任务编排 |
| v4.1.0 | LingTu 导航技能 + Runtime 集成 |
