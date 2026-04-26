# askme

> 穹沛科技四足/人形机器人语音 AI 助手 — **听 → 理解 → 回答 → 说**。
> 支持中英文混合对话、机器人控制、L0–L6 分层记忆、技能扩展、MCP 工具暴露、视觉感知、主动告警。

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-3500+%20passing-brightgreen)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20aarch64-lightgrey)

---

## 目录

- [askme 是什么 / 不是什么](#askme-是什么--不是什么)
- [5 分钟上手](#5-分钟上手)
- [6 个 Blueprint —— 选你需要的](#6-个-blueprint--选你需要的)
- [配置：.env 与 config.yaml](#配置env-与-configyaml)
- [典型使用场景](#典型使用场景)
- [作为 MCP 服务给 Claude Desktop / Code 用](#作为-mcp-服务给-claude-desktop--code-用)
- [扩展：自定义 Skill / Tool](#扩展自定义-skill--tool)
- [测试与调试](#测试与调试)
- [部署到 S100P 机器人](#部署到-s100p-机器人)
- [项目结构](#项目结构)
- [深入文档](#深入文档)
- [常见问题](#常见问题)

---

## askme 是什么 / 不是什么

| ✅ 是 | ❌ 不是 |
|---|---|
| 语音 AI 助手 | 机器人操作系统 |
| 对话流水线 + 记忆 + 技能调度 | 运动控制器（电机/CAN 由 brainstem） |
| 通过 HTTP/gRPC 调用外部能力 | 导航/SLAM（由 LingTu） |
| Voice / Text / MCP 多入口 | 视觉推理（由 frame_daemon） |

核心链路 6 步：

```
麦克风 ──▶ VAD ──▶ ASR ──▶ LLM ──▶ TTS ──▶ 扬声器
          (silero)  (sherpa+云) (MiniMax) (MiniMax→edge)
```

记忆 7 层：`L0 运行态 / L1 工作记忆 / L2 会话 / L3 情节 / L4 向量库 / L5 语义索引 / L6 策略模板`。

---

## 5 分钟上手

### 1. 安装

```bash
git clone https://github.com/Kitjesen/askme.git
cd askme
pip install -e ".[dev]"          # 含 pytest, ruff
```

可选 extras（按需开启）：

```bash
pip install -e ".[dev,robot,robotmem,embed]"
#                ^^^   ^^^^^   ^^^^^^^^   ^^^^^
#                测试  机器人  长期记忆   句向量
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

`.env` 至少填两个就能跑：

```env
LLM_API_KEY=cr_xxxxxxxx               # Claude relay key (cursor.scihub.edu.kg)
MINIMAX_API_KEY=eyJhxxx                # MiniMax LLM + TTS, https://platform.minimax.chat/
MINIMAX_GROUP_ID=18xxxx                # MiniMax 控制台获取
```

其余 key（导航/控制/OTA/Telegram）都是可选，缺了对应技能会回复"服务未配置"，主链路不受影响。

### 3. 跑起来

最小验证（纯文字，无需音频硬件、无需机器人）：

```bash
python -m askme.blueprints.text
> 你好
收到。我是Thunder，穹沛的巡检机器人。等待指令。
```

完整语音对话（需要麦克风+扬声器）：

```bash
python -m askme.blueprints.voice
```

成功标志：日志里看到 `MicInput started`、`VAD threshold=0.5`、`LLM warmup done`。

---

## 6 个 Blueprint —— 选你需要的

Blueprint 是声明式的"功能包"组合（`Runtime.use(A) + Runtime.use(B)`）。所有入口都在 `askme/blueprints/`：

| Blueprint | 模块数 | 用途 | 启动命令 |
|---|---|---|---|
| `text.py` | 6 | 终端文字调试，没有音频依赖 | `python -m askme.blueprints.text` |
| `voice.py` | 7 | 纯语音对话，听+说+记忆+技能 | `python -m askme.blueprints.voice` |
| `voice_perception.py` | 11 | 语音 + 视觉感知 + 反应引擎 | `python -m askme.blueprints.voice_perception` |
| `edge_robot.py` | 17 | **生产配置** — 全量含机器人控制 + LED + 主动巡检 + 健康监控 | `python -m askme.blueprints.edge_robot` |
| `lingtu_voice.py` | 8 | LingTu 项目专用（S100P 部署，含 Telegram） | `python -m askme.blueprints.lingtu_voice` |
| `mcp.py` | — | MCP 服务器模式，给 Claude Desktop/Code 调用 | `python -m askme.mcp.server` |

**怎么选**：

- 第一次跑 → `text`（5 秒就能对话）
- 笔记本测语音 → `voice`
- 装在机器人上 → `edge_robot`
- 想被 Claude Desktop 当作工具 → `mcp`

---

## 配置：`.env` 与 `config.yaml`

### `.env`（机密，git 忽略）

| 变量 | 必需？ | 用途 |
|---|---|---|
| `LLM_API_KEY` | ✅ | Claude relay key（OpenAI 兼容，cursor.scihub.edu.kg） |
| `LLM_BASE_URL` | ✅ | relay endpoint，默认 `https://cursor.scihub.edu.kg/api/v1` |
| `MINIMAX_API_KEY` | ✅ | MiniMax LLM + TTS |
| `MINIMAX_GROUP_ID` | ✅ | MiniMax 账号 group |
| `TTS_VOICE_ID` | 默认 `male-qn-qingse` | TTS 音色 |
| `LOCAL_EMBED_URL` | 可选 | 本地句向量服务，加速记忆检索 |
| `NAV_GATEWAY_URL` | 装在机器人才需要 | LingTu 导航网关 |
| `DOG_CONTROL_SERVICE_URL` | 装在机器人才需要 | 控制服务（站立/坐下/巡检） |
| `DOG_SAFETY_SERVICE_URL` | 装在机器人才需要 | E-STOP 服务 |
| `TELEGRAM_BOT_TOKEN` | 可选 | 启用 Telegram 入口（@BotFather 申请） |
| `OTA_SERVER_URL` | 可选 | OTA 升级服务 |

### `config.yaml`（非机密，git 跟踪）

主要调整项：

```yaml
brain:
  model: MiniMax-M2.7-highspeed     # 主力 LLM
  voice_model: MiniMax-M2.7-highspeed  # 语音模式专用（更低 TTFT）
  max_response_chars: 200            # 语音回复上限（避免播报过长）
  soul_file: SOUL.md                 # 人格定义入口

memory:
  enabled: true
  backend: robotmem                  # mem0 / robotmem / vector 三选一
  embed_model: paraphrase-multilingual-MiniLM-L12-v2

voice:
  input_device: null                 # 麦克风索引（null=系统默认）
  mic_native_rate: 44100             # Windows Realtek 一般 44100，HKMIC 是 48000
  asr:
    model_dir: models/asr/sherpa-onnx-streaming-zipformer-zh-int8-2025-06-30
    rule1_min_trailing_silence: 0.6  # 多久静音算句子结束
  vad:
    threshold: 0.5                   # silero VAD 阈值，0.5 默认
  noise_gate_peak: 50                # 低于此 peak 跳过 VAD（噪底过滤）
  echo_gate_peak: 800                # TTS 播报时的麦克风门限（防自激）
```

完整字段见 `config.yaml`，每行都有注释。

### 模型文件

```bash
python scripts/download_models.py    # 自动下载 sherpa-onnx ASR + silero VAD
```

或手动：
- VAD: https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx → `models/vad/silero_vad.onnx`
- ASR: https://github.com/k2-fsa/sherpa-onnx/releases (找 `streaming-zipformer-zh-int8-2025-06-30`) → 解压到 `models/asr/`

---

## 典型使用场景

### 场景 1：终端文字对话（最快验证）

```bash
python -m askme.blueprints.text
```

```
> 你好
收到。我是Thunder。等待指令。
> 现在几点
[运行时状态] 14:23, ESTOP=False
现在 14:23。
> 站起来
执行中。已发送 stand 命令。
```

### 场景 2：完整语音对话

```bash
python -m askme.blueprints.voice
```

对着麦说 `你好`，系统会：
1. VAD 检测到语音
2. 本地 sherpa-onnx + 云端 DashScope 并行 ASR（先到先用）
3. LLM 流式生成（首字 ~500ms）
4. MiniMax TTS 流式播报，可被打断（barge-in）

### 场景 3：装在 NOVA Dog 机器人上

```bash
# .env 增加
DOG_CONTROL_SERVICE_URL=http://localhost:5080
DOG_SAFETY_SERVICE_URL=http://localhost:5070
NAV_GATEWAY_URL=http://localhost:8088

python -m askme.blueprints.edge_robot
```

支持指令：
- "站起来" / "坐下" → DogControlService
- "去仓库 A" → NAV gateway → LingTu 导航
- "停！" → tier-0 拦截（不走 LLM），直接打 E-STOP
- "检查电池" → HealthModule 读机器人状态

### 场景 4：把 askme 暴露给 Claude Desktop / Code

见下一节 [作为 MCP 服务](#作为-mcp-服务给-claude-desktop--code-用)。

### 场景 5：批量打 LLM 推理性能

```bash
python scripts/bench/bench_ttft.py    # LLM 首字延迟
python scripts/bench/benchmark_all.py # 端到端 P50/P95
```

---

## 作为 MCP 服务给 Claude Desktop / Code 用

askme 实现了 [Model Context Protocol](https://modelcontextprotocol.io/) 服务端。配置后，Claude Desktop / Code 可以直接调用 askme 的工具集（move、scan、vision、robot_api、bash 沙箱等）。

### Claude Desktop 配置

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) 或 `%APPDATA%\Claude\claude_desktop_config.json` (Win)：

```json
{
  "mcpServers": {
    "askme": {
      "command": "python",
      "args": ["-m", "askme.mcp.server"],
      "cwd": "D:/inovxio/tools/askme",
      "env": {
        "LLM_API_KEY": "cr_xxx",
        "MINIMAX_API_KEY": "xxx"
      }
    }
  }
}
```

### Claude Code 配置

`.mcp.json` 在项目根：

```json
{
  "mcpServers": {
    "askme": {
      "command": "askme-mcp"
    }
  }
}
```

启动后在 Claude 里输入 `/mcp` 能看到 askme 暴露的工具。

---

## 扩展：自定义 Skill / Tool

### 加一个 Skill（用户视角的能力，"去仓库A巡检"）

1. 写 `askme/skills/builtin/<name>/SKILL.md`：

   ```markdown
   ---
   name: inspect_warehouse_a
   trigger:
     voice: ["去仓库A", "巡检A区"]
     intent: "warehouse_inspect"
   ---

   去仓库 A 走一圈，记录任何异常。
   ```

2. （可选）加代码契约 `askme/skills/contracts_builtin.py`：

   ```python
   @skill_contract("inspect_warehouse_a")
   async def _(ctx: SkillContext) -> SkillResult:
       await ctx.move_tool.go_to("warehouse_a")
       findings = await ctx.scan_tool.scan_objects()
       return SkillResult.ok(summary=f"巡检完成，发现 {len(findings)} 项")
   ```

3. 跑 `python -m askme.blueprints.voice`，说 `"去仓库A"` 即可触发。

### 加一个 Tool（LLM tool-calling 视角的原子操作）

1. 在 `askme/tools/` 新建 `my_tool.py`，继承 `BaseTool`：

   ```python
   from askme.tools.tool_registry import BaseTool, ToolRegistry

   class GetBatteryTool(BaseTool):
       name = "get_battery"
       description = "读取机器人电池电量百分比"
       parameters = {"type": "object", "properties": {}}

       async def execute(self, **kwargs) -> dict:
           return {"battery_pct": 87}

   def register_battery_tool(registry: ToolRegistry) -> None:
       registry.register(GetBatteryTool())
   ```

2. 在 `askme/runtime/modules/tools_module.py` 的 `build()` 里 `import` 并调用：

   ```python
   from askme.tools.my_tool import register_battery_tool
   register_battery_tool(tools)
   ```

   现有的工具批量注册入口是 `register_builtin_tools(tools, ...)` —— 你的新工具
   可以单独注册或加入 `askme/tools/builtin_tools.py::register_builtin_tools`。

3. LLM 接下来可以调用 `get_battery`。

详细模式见 `docs/ARCHITECTURE_PATTERNS.md`。

---

## 测试与调试

```bash
# 全量测试（~2 分钟，3500+ tests）
python -m pytest -q

# 只跑某个模块
python -m pytest tests/test_voice_loop.py -v

# 看覆盖率
python -m pytest --cov=askme --cov-report=html
open htmlcov/index.html

# Lint
ruff check askme/ tests/
```

### Debug 一次具体对话

```bash
# 打开 trace
LOG_LEVEL=DEBUG python -m askme.blueprints.text

# 看 spans
ls .omc/state/sessions/*/traces/
```

### 麦克风/扬声器自检（S100P）

```bash
python scripts/bench/s100p_audio_check.py
python scripts/bench/check_output_device.py
```

---

## 部署到 S100P 机器人

### 1. 复制代码

```bash
ssh sunrise@192.168.66.190 "mkdir -p ~/askme"
rsync -avz --exclude='.git' --exclude='models/' --exclude='data/' \
  ./ sunrise@192.168.66.190:~/askme/
```

### 2. 安装系统依赖（aarch64）

```bash
ssh sunrise@192.168.66.190
sudo apt-get install libportaudio2 libsndfile1
cd ~/askme
pip install -e ".[dev,robot,robotmem]"
```

注意：`models/asr/`、`models/vad/` 不通过代码同步，单独传。

### 3. systemd 服务

```bash
sudo cp scripts/askme.service /etc/systemd/system/
sudo systemctl enable --now askme.service
journalctl -u askme.service -f
```

### S100P 已知坑

- **声卡顺序会变**：每次启动确认 `card0/card1`，调整 `voice.input_device`
- **HKMIC 48kHz 2ch**：MicInput 自动重采样到 16kHz mono；MCP01 麦克风硬件坏，只能当扬声器
- **legacy askme 抢设备**：`pkill -f "askme --legacy"` 后再启动新 blueprint
- **HuggingFace 直连超时**：用 `HF_ENDPOINT=https://hf-mirror.com` 或预下载模型
- **numpy ABI**：`ele_planner.so` 要 numpy 1.x，需独立 venv `/tmp/venv_np1`

---

## 项目结构

```
askme/
├── blueprints/          # 6 个声明式入口（voice/text/edge_robot/...）
├── runtime/
│   ├── module.py        # Module ABC + In/Out/Required 端口 + 拓扑装配
│   └── modules/         # 17 个 Module：LLM/Memory/Pipeline/Skill/Voice/Text/...
├── llm/                 # LLM 客户端 + 流式 + intent_router (ESTOP→quick→general)
├── memory/              # L1/L2/L3/L4/L5/L6 + reflect + consolidate
├── voice/               # MicInput / VAD / ASR (sherpa+cloud) / TTS (MiniMax+fallback)
├── pipeline/            # BrainPipeline / VoiceLoop / TextLoop / ProactiveAgent
├── perception/          # ChangeDetector / WorldState / VisionBridge
├── reaction/            # 11 条规则的 ReactionEngine
├── skills/              # SkillManager + 41 个 SKILL.md
├── tools/               # 24 个 LLM tool-calling 实现
├── robot/               # Pulse(DDS) / DogSafetyClient / DogControlClient / LED / OTA
├── mcp/                 # MCP 服务器（暴露 tools/resources）
└── schemas/             # 类型化消息 dataclass
```

更详细分层见 `docs/ARCHITECTURE.md`。

---

## 深入文档

| 文件 | 内容 |
|---|---|
| [`SOUL.md`](SOUL.md) | 语音人格定义（身份/风格/边界）— 改人格只动这个文件 |
| [`docs/CONFIG_GUIDE.md`](docs/CONFIG_GUIDE.md) | **配置布局总览** —— `config.yaml` 15 章节 / `.env` 11 变量 / "改 X 动哪里" |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 多进程架构 / DDS topic / 坐标系 / 安全不变量 |
| [`docs/ASKME_BOUNDARY.md`](docs/ASKME_BOUNDARY.md) | 核心/感知/插件三层边界，不该做什么 |
| [`docs/PROACTIVE_INTELLIGENCE_PLAN.md`](docs/PROACTIVE_INTELLIGENCE_PLAN.md) | 主动智能 + ReactionEngine 设计 |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | 关键模式、陷阱、待完成工作 |
| [`docs/LAYER_GAPS.md`](docs/LAYER_GAPS.md) | 分层差距清单（28/28 已关闭） |
| [`docs/ARCHITECTURE_PATTERNS.md`](docs/ARCHITECTURE_PATTERNS.md) | 设计模式参考（Blueprint/Module/Backend） |

---

## 常见问题

**Q: `OSError: PortAudio library not found`**
A: `sudo apt-get install libportaudio2`（Linux）或 `brew install portaudio`（macOS）。

**Q: 启动后没声音 / 麦克风不响应**
A: 跑 `python scripts/bench/s100p_audio_check.py`，确认 `voice.input_device` 索引。Windows Realtek 一般 `null` 即可，S100P HKMIC 一般为 0。

**Q: `models/vad/silero_vad.onnx failed. File doesn't exist`**
A: 跑 `python scripts/download_models.py`，或手动从 silero-vad 仓库下载。

**Q: LLM 响应慢 / 超时**
A: `config.yaml` 里调小 `brain.timeout`，或换更轻的 `voice_model` (`MiniMax-M2.5-highspeed`)。

**Q: 记忆检索慢**
A: 启用 `LOCAL_EMBED_URL`，或换 `memory.backend: robotmem`（默认就是）。

**Q: TTS 没声音但 LLM 有输出**
A: 检查 `MINIMAX_API_KEY` 余额，TTS 自动 fallback 到 sherpa→edge，但 fallback 链中断时静默失败。

**Q: Windows 上 `ValueError: set_wakeup_fd only works in main thread`**
A: askme 已处理 Windows 不兼容；如果还报错，确认在主线程入口跑（不要套到 thread pool）。
