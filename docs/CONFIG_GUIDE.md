# Askme 配置与模块布局总览

> 这是一份"哪里改什么"的对照表。读完后你应该能：
> - 找到任何配置项（`config.yaml` 15 个 section / `.env` 11 个变量）
> - 找到任何子系统的实现层 + Module 装配层
> - 知道改一个行为应该动哪个文件

---

## 1. 配置文件清单

| 文件 | 作用 | 是否提交 git |
|---|---|---|
| **`config.yaml`** | 主配置，15 个 section，**所有非机密参数** | ✅ |
| **`.env`** | 机密 key (LLM/MiniMax/服务地址)，通过 `${VAR}` 注入 yaml | ❌ gitignore |
| **`.env.example`** | `.env` 模板 | ✅ |
| **`SOUL.md`** | 语音人格定义（身份/风格/边界） | ✅ |
| **`askme/config.py`** | 配置加载器：`yaml.safe_load + dotenv + ${VAR}` 解析，单例缓存 | ✅ |

加载方式：

```python
from askme.config import get_config, get_section
cfg = get_config()                # 完整 dict（${VAR} 已解析）
brain = get_section("brain")      # 取某个 section
```

---

## 2. `config.yaml` 15 个 Section（按行号）

| Section | 行号 | 用途 | 关联实现 |
|---|---|---|---|
| `app` | L7 | 名字、版本、`data_dir`、`log_level` | `askme/config.py` |
| **`brain`** ★ | **L17** | **LLM 配置** —— model / API key / fallback / SOUL | `askme/llm/` + `LLMModule` |
| **`memory`** ★ | **L57** | **记忆配置** —— backend (robotmem/mem0/vector) + L3/L4 参数 | `askme/memory/` + `MemoryModule` |
| `conversation` | L91 | L1 工作记忆滑窗 — `history_file`, `max_history=40` | `askme/llm/conversation.py` |
| **`voice`** ★ | **L98** | **语音配置（最长一节，147 行）** —— input_device / mic / asr / vad / tts | `askme/voice/` + `VoiceModule` |
| `vision` | L245 | frame_daemon 接入 | `askme/perception/` |
| `robot` | L259 | 机器人服务地址 (control/safety/runtime) | `askme/robot/` |
| `tools` | L272 | 工具白名单 + 超时 | `askme/tools/` + `ToolsModule` |
| `proactive` | L313 | 主动巡检定时 | `askme/pipeline/proactive_agent.py` |
| `platforms` | L334 | 平台入口（Telegram 等） | `runtime/modules/telegram_module.py` |
| `scheduler` | L351 | 定时任务 | — |
| `swarm` | L358 | 多 agent 协作 | — |
| `ota` | L365 | OTA 升级 | `askme/robot/ota_bridge.py` |
| `health_server` | L389 | HTTP 健康检查 (端口 8765) | `runtime/modules/health_module.py` |
| `runtime` | L397 | NOVA Dog runtime 桥接 | `askme/voice/runtime_bridge.py` |

★ = 三个最重要的 section，90% 的调整在这里。

---

## 3. `.env` 变量清单

```env
# ── LLM Backend (必需) ───────────────────────────
LLM_API_KEY=cr_xxxxxxxx                 # Claude relay key (cr_*)
LLM_BASE_URL=https://cursor.scihub.edu.kg/api/v1

# ── MiniMax (必需) ────────────────────────────────
MINIMAX_API_KEY=eyJhxxx                 # https://platform.minimax.chat/
MINIMAX_GROUP_ID=18xxxx

# ── TTS 音色 (默认值已可用) ──────────────────────
TTS_VOICE_ID=male-qn-qingse
TTS_SPEED=1
TTS_EMOTION=happy

# ── 可选（缺了对应技能"服务未配置"，主链路不影响）
LOCAL_EMBED_URL=http://localhost:8000/v1   # 本地句向量加速记忆检索
NAV_GATEWAY_URL=http://localhost:8088      # LingTu 导航网关
DOG_CONTROL_SERVICE_URL=http://localhost:5080  # 控制服务（站立/坐下）
DOG_SAFETY_SERVICE_URL=http://localhost:5070   # E-STOP
RUNTIME_BEARER_TOKEN=                       # 服务认证
RUNTIME_OPERATOR_ID=askme
ASKME_EDGE_SERVICE_URL=http://localhost:8090
NOVA_DOG_RUNTIME_API_KEY=
TELEGRAM_BOT_TOKEN=                        # @BotFather 申请
OTA_SERVER_URL=                            # OTA 升级服务
NOVA_DOG_SERIAL_NUMBER=                    # OTA 设备序列号
```

`.env` → `config.yaml` 注入示例：

```yaml
brain:
  api_key: ${MINIMAX_API_KEY}    # ← 启动时从 .env 解析
  base_url: ${LLM_BASE_URL}
```

---

## 4. LLM 管理（在哪里）

```
askme/llm/                                ← 实现层（business logic）
├── client.py        18.5KB  ★ LLMClient — chat() / chat_stream(), 超时/重试/fallback chain
├── config.py         3.6KB    LLMConfig dataclass（解耦配置文件）
├── conversation.py  14.2KB    ConversationManager — L1 工作记忆滑窗(40 msgs) + 压缩
└── intent_router.py  8.6KB    IntentRouter — ESTOP(tier-0) → quick_reply → voice_trigger → general

askme/runtime/modules/llm_module.py       ← Module 装配层（Blueprint 用）
└── LLMModule         3.0KB    Out[LLMClient] — 启动时后台 warmup，消除冷启动延迟
```

### `config.yaml [brain]` 字段详解

```yaml
brain:
  api_key: ${MINIMAX_API_KEY}              # 来自 .env
  base_url: https://api.minimax.chat/v1
  model: MiniMax-M2.7-highspeed             # 主力 LLM
  voice_model: MiniMax-M2.7-highspeed       # 语音模式专用（更低 TTFT）
  max_tokens: 0                             # 0 = 模型默认
  temperature: 0.7
  timeout: 30.0
  max_retries: 2
  fallback_models: [MiniMax-M2.5-highspeed] # 主模型故障走这条
  soul_file: SOUL.md                        # 人格文件
  system_prompt: "工业巡检机器人Thunder..."  # SOUL.md 之外的兜底
  user_prefix: "[工业巡检模式：中文口语...]"  # 每条 user msg 前缀
  max_response_chars: 200                   # 语音回复字数上限
  agent_timeout: 120                        # ThunderAgentShell 超时（秒）
```

### "我想改 X" → "改这个文件"

| 你想改 | 改这里 |
|---|---|
| 换 LLM 模型 | `config.yaml [brain.model]` |
| 换 API key | `.env [LLM_API_KEY/MINIMAX_*]`，**绝不提交** |
| 调人格 / 说话风格 | `SOUL.md`（主） + `config.yaml [brain.system_prompt]`（兜底） |
| 改流式输出行为 | `askme/llm/client.py::chat_stream` |
| 改超时/重试 | `config.yaml [brain.timeout/max_retries]` |
| 改对话历史长度 | `config.yaml [conversation.max_history]` |
| 改"哪句话先走 ESTOP / quick_reply / 走 LLM" | `askme/llm/intent_router.py` |
| 改记忆压缩策略 | `askme/llm/conversation.py` |

---

## 5. 语音管理（在哪里）

```
askme/voice/                              ← 实现层（最大目录，19 个 .py 文件）
├── mic_input.py        13.2KB   麦克风输入：48kHz→16kHz 重采样 + HPF + AGC
├── audio_agent.py      32.9KB ★ AudioAgent — 状态机(idle/listening/speaking/muted)
├── audio_router.py      9.5KB   音频路由（多输出端）
├── audio_filter.py      6.1KB   噪声过滤
├── audio_processor.py   6.5KB   音频后处理
├── noise_reduction.py   8.3KB   谱减法降噪
├── vad.py               2.4KB   silero VAD wrapper
├── vad_controller.py    7.4KB   VAD 决策器（最大 30s 守护 + 阈值）
├── kws.py               5.1KB   关键词唤醒
├── asr.py               5.8KB   本地 ASR (sherpa-onnx)
├── asr_manager.py      12.0KB ★ 本地+云并行 ASR，先到先用
├── cloud_asr.py         9.9KB   云端 ASR (DashScope paraformer-realtime-v2)
├── tts.py              43.5KB ★ TTSEngine — MiniMax SSE → sherpa → edge 三级 fallback
├── address_detector.py  6.1KB   "是不是在跟我说话"过滤器
├── stream_splitter.py   6.0KB   流式句子切分（喂给 TTS）
├── punctuation.py       2.0KB   标点恢复
├── runtime_bridge.py    9.5KB   连 NOVA Dog runtime
└── generated_contracts.py        合约自动生成

askme/runtime/modules/voice_module.py     ← Module 装配层
└── VoiceModule         6.6KB    把上面拼起来，对外 provides=("voice","tts","asr")
```

### `config.yaml [voice]` 字段详解（最长一节，L98-244）

```yaml
voice:
  # ── 麦克风 ─────────────────────────────────────
  input_device: null              # null=系统默认；S100P HKMIC=0
  mic_native_rate: 44100          # Win Realtek=44100，HKMIC=48000
  mic_channels: 2
  mic_channel_select: 0           # 多通道选哪一路
  mic_highpass_hz: 80             # 高通滤波截止
  mic_agc_target_rms: 0.1         # AGC 目标响度

  # ── ASR ────────────────────────────────────────
  asr:
    model_dir: models/asr/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30
    sample_rate: 16000
    feature_dim: 80
    num_threads: 2
    rule1_min_trailing_silence: 0.6   # 短句静音判句末（秒）
    rule2_min_trailing_silence: 0.4   # 正常语速静音判句末
    asr_timeout: 10.0                  # 唤醒后等待 ASR 输出超时

  # ── VAD ────────────────────────────────────────
  vad:
    model_path: models/vad/silero_vad.onnx
    threshold: 0.5                # silero 阈值，0.5 = 默认
    min_silence_duration: 0.3
    min_speech_duration: 0.4
    sample_rate: 16000
    buffer_size_seconds: 30

  # ── 噪声 / 回声门限 ─────────────────────────────
  noise_gate_peak: 50             # 麦低于此 peak 跳过 VAD（HKMIC 噪底 12-17）
  echo_gate_peak: 800             # TTS 播报时麦门限（防自激）

  # ── TTS ────────────────────────────────────────
  tts:
    backend: minimax              # minimax / sherpa / edge
    voice_id: ${TTS_VOICE_ID}
    speed: 1.0
    emotion: happy
    fallback_chain: [minimax, sherpa, edge]
```

### "我想改 X" → "改这个文件"

| 你想改 | 改这里 |
|---|---|
| 换麦克风 | `config.yaml [voice.input_device]`（启动日志看可用 index） |
| 改采样率 | `config.yaml [voice.mic_native_rate/mic_channels]` |
| 调 VAD 灵敏度 | `config.yaml [voice.vad.threshold]`（0.3 更敏感，0.7 更迟钝） |
| 换 ASR 模型 | `config.yaml [voice.asr.model_dir]`（指向新 sherpa-onnx 目录） |
| 改"判句末"延迟 | `config.yaml [voice.asr.rule1_min_trailing_silence]`（短句默认 0.6s） |
| 换 TTS 音色 | `.env [TTS_VOICE_ID]` |
| 改 TTS fallback | `askme/voice/tts.py` |
| 改"听到→回答"主循环 | `askme/pipeline/voice_loop.py`（**不在 voice/ 目录！**） |
| 改"是不是在跟我说话"过滤 | `askme/voice/address_detector.py` |

---

## 6. 其他子系统对照表

| 子系统 | 实现层 | Module 装配 | config 章节 | 文件数 |
|---|---|---|---|---|
| **LLM** | `askme/llm/` | `runtime/modules/llm_module.py` | `brain:` | 4 .py |
| **语音** | `askme/voice/` | `runtime/modules/voice_module.py` | `voice:` | 19 .py |
| **记忆** | `askme/memory/` | `runtime/modules/memory_module.py` | `memory:` + `conversation:` | 19 .py |
| **Pipeline** | `askme/pipeline/` (voice_loop, brain_pipeline, planner_agent) | `runtime/modules/pipeline_module.py` | 散在 brain/voice | 12+ |
| **技能** | `askme/skills/` (skill_manager + 41 SKILL.md) | `runtime/modules/skill_module.py` | `tools:` | 7 |
| **工具** | `askme/tools/` (24 个 tool) | `runtime/modules/tools_module.py` | `tools:` | 10 |
| **感知** | `askme/perception/` | `runtime/modules/perception_module.py` | `vision:` | 5 |
| **机器人 IO** | `askme/robot/` (Pulse/SafetyClient/ControlClient/LED/OTA) | `control_module.py` / `led_module.py` / `health_module.py` | `robot:` + `runtime:` | 8 |
| **MCP** | `askme/mcp/` | （独立入口 `python -m askme.mcp.server`） | — | 2+ |
| **主动智能** | `askme/pipeline/proactive_agent.py` + `askme/reaction/` | `runtime/modules/proactive_module.py` + `reaction_module.py` | `proactive:` | — |

---

## 7. 黄金法则速查

| 你想做的事 | 改这里 |
|---|---|
| 换 LLM | `config.yaml [brain.model]` |
| 换 API key | `.env`（不要提交） |
| 调人格 | `SOUL.md` |
| 改对话流式 | `askme/llm/client.py::chat_stream` |
| 加新 tool | `askme/tools/<new>.py` + `tools_module.py` 注册 |
| 加新 skill | `askme/skills/builtin/<name>/SKILL.md` (+ 可选 `contracts_builtin.py`) |
| 调 VAD/ASR | `config.yaml [voice.vad / voice.asr]` |
| 改语音主循环 | `askme/pipeline/voice_loop.py` |
| 调记忆策略 | `config.yaml [memory.episodic]` + `askme/memory/episodic_memory.py` |
| 换 TTS 音色 | `.env [TTS_VOICE_ID]` |
| 改启动模块组合 | `askme/blueprints/<name>.py` |
| 加新 Blueprint | `askme/blueprints/<new>.py` 用 `Runtime.use(M1) + Runtime.use(M2)` |

---

## 8. 配置加载链路

启动时配置如何到达每个 Module：

```
process start
  └─ load_dotenv(".env")               # 把 .env 读进 os.environ
  └─ yaml.safe_load("config.yaml")      # 读主配置
  └─ _resolve_env_vars(...)             # 把 ${VAR} 替换成 os.environ[VAR]
  └─ get_config() → singleton dict      # 缓存
  └─ Blueprint = Runtime.use(LLMModule) + Runtime.use(MemoryModule) + ...
       └─ 每个 Module.__init__ 调 get_section("brain") / get_section("voice")
       └─ 拓扑排序 → start() → 业务运行
```

**重要约束**：只有 `Module.__init__` 应该读 `get_section()`。其他类（LLMClient、AudioAgent…）通过构造参数传入 dataclass 配置（`LLMConfig` / `VoiceConfig`），不直接读 yaml。这样测试可以注入 mock 配置。

---

## 9. 相关文档

- [`README.md`](../README.md) — 安装 / Quick Start / 6 个 Blueprint
- [`SOUL.md`](../SOUL.md) — 人格定义
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — 多进程架构 / DDS / 安全不变量
- [`docs/ASKME_BOUNDARY.md`](ASKME_BOUNDARY.md) — 三层边界（核心/感知/插件）
- [`docs/ARCHITECTURE_PATTERNS.md`](ARCHITECTURE_PATTERNS.md) — Module / Backend Registry / Blueprint 模式
