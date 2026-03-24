# Askme

语音 AI 助手。听 → 理解 → 回答 → 说。

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Boundary: [docs/ASKME_BOUNDARY.md](docs/ASKME_BOUNDARY.md)

## What askme is

```
用户说话 → ASR → 理解意图 → LLM + Memory + Skills → TTS → 扬声器
```

askme 是语音 AI 助手，不是机器人操作系统。导航、电机控制、SLAM 是外部服务，askme 通过接口调用它们。

## Quick Start

```bash
# 语音模式（6 模块：听说记答）
python -m askme.blueprints.voice

# 语音 + 感知（10 模块：+ 看 + 主动反应）
python -m askme.blueprints.voice_perception

# 完整机器人（16 模块：+ 硬件控制 + 巡检）
python -m askme.blueprints.edge_robot

# 纯文字（5 模块，SSH/开发用）
python -m askme.blueprints.text

# Legacy CLI
python -m askme --voice
python -m askme --text
```

## Blueprints — 产品 = 文件

```python
# voice.py — 6 modules
voice = (
    Runtime.use(LLMModule)       # 理解语言
    + Runtime.use(MemoryModule)  # 记住上下文
    + Runtime.use(PipelineModule)# 对话流水线
    + Runtime.use(SkillModule)   # 技能调度
    + Runtime.use(VoiceModule)   # 听和说
    + Runtime.use(TextModule)    # 文字交互
)

# voice_perception.py — voice + 4 感知模块
voice_perception = voice + Pulse + Perception + Safety + Reaction

# edge_robot.py — voice_perception + 6 外部插件
edge_robot = voice_perception + Control + LED + Proactive + Health + Executor + Tools
```

加功能 = 加一行。换实现 = config 改一词。拔掉任何一层 = 系统照跑。

## Three-Layer Boundary

| 层 | 模块 | 拔掉后果 |
|---|------|---------|
| **核心**（必须） | LLM, Memory, Pipeline, Skill, Voice, Text | 不能工作 |
| **感知**（可选） | Pulse, Perception, Safety, Reaction | 不能看，但能对话 |
| **插件**（外部） | Control, LED, Proactive, Health, Executor, Tools | 不能控硬件，但能对话+看 |

## Module System

```python
# 声明模块
class MyModule(Module):
    data: In[SomeType]          # 自动从其他模块接
    result: Out[OtherType]      # 自动暴露给其他模块

# 加功能 = 一行
runtime = voice + Runtime.use(MyModule)

# 换实现 = 一词
runtime = runtime.replace(LLMModule, MockLLMModule)
```

- `In[T]` / `Out[T]` / `Required[T]` 类型化端口，自动匹配
- 语义匹配：类型唯一时名字不同也能连
- 拓扑校验：孤立端口、环路、必需依赖缺失 → 构建报错

## Plugin Architecture

```python
# 接口
class LLMBackend(ABC):
    async def chat_stream(self, messages, **kw) -> AsyncIterator: ...

# 注册
@llm_registry.register("minimax")
class MiniMaxLLM(LLMBackend): ...

# 使用（config.yaml）
llm:
  backend: minimax    # 改成 openai / ollama / local
```

7 个接口：LLM, ASR, TTS, Detector, Navigator, Bus, Reaction

## Data Bus — Pulse (CycloneDDS)

```
frame_daemon (5Hz YOLO) → DDS → Pulse (进程内) → asyncio 回调
```

- 零 ROS2 依赖，纯 CycloneDDS
- 0.8ms P50 延迟（S100P 实测）
- `PubSubBase` 抽象，`MockPulse` 测试用

## Performance (S100P)

| 指标 | 数值 |
|------|------|
| LLM TTFT | 500-1800ms (MiniMax) |
| ASR 识别 | 35-245ms |
| TTS 合成 | 40ms (local) |
| Pulse DDS | 0.8ms P50 |
| 感知检测 | 0.04ms |
| 语音全链路 | ~1-2s |

## Repository Layout

```
askme/
├── blueprints/           产品定义（voice / voice_perception / edge_robot / text）
├── interfaces/           ABC 接口 + 注册表（LLM / ASR / TTS / Bus / Reaction）
├── runtime/
│   ├── module.py         Module + In/Out/Required + Runtime.use()
│   ├── registry.py       BackendRegistry（插件架构）
│   └── modules/          16 个声明式 Module
├── llm/                  LLM 调用 + 对话管理 + 意图路由
├── memory/               四层记忆（对话 / 会话 / 经验 / 向量）
├── perception/           感知（YOLO 检测 → 事件 → 场景）
├── pipeline/             对话管线 + 技能调度 + 反应引擎
├── voice/                ASR / TTS / VAD / KWS
├── robot/                Pulse + SafetyClient + ControlClient
├── skills/               41 个技能定义
├── tools/                24 个 LLM 工具
├── schemas/              消息类型 + 反应类型
├── mcp/                  MCP 服务
├── main.py               启动入口
├── cli.py                CLI
└── tui.py                终端 UI
tests/                    1718 tests
docs/
├── ARCHITECTURE.md       系统架构
├── ASKME_BOUNDARY.md     边界定义
├── PROACTIVE_INTELLIGENCE_PLAN.md  主动智能规划
└── LAYER_GAPS.md         层级差距分析
```

## Test Suite

```bash
python -m pytest tests/ -q    # ~2 min, 1718 tests
```
