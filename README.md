# askme

> 语音 AI 助手 for NOVA Dog / Thunder / AXION 机器人 — 听、理解、回答、说。

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-3500+%20passing-brightgreen)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20aarch64-lightgrey)

## 简介

askme 是运行在穹沛科技 NOVA Dog (Thunder) 四足巡检机器人上的语音 AI 助手。核心链路：

```
麦克风 → VAD → ASR → LLM → TTS → 扬声器
```

支持中英文混合对话、机器人控制指令、L0–L4 分层记忆、技能扩展、
MCP 工具暴露、视觉感知、主动告警和多渠道通知。

## 快速开始

```bash
pip install -e ".[dev]"       # 开发安装（含测试工具）
cp .env.example .env           # 填入 LLM / TTS / ASR API keys

# 按场景选择 blueprint：
python -m askme.blueprints.voice              # 纯语音对话
python -m askme.blueprints.text               # 纯文本 / 终端调试
python -m askme.blueprints.voice_perception   # 语音 + 视觉感知 + 主动告警
python -m askme.blueprints.edge_robot         # 全量（含 Thunder 控制 + LED）
python -m askme.blueprints.lingtu_voice       # LingTu 导航部署（S100P）
python -m askme.blueprints.mcp                # MCP 服务器（Claude Desktop / Code 接入）
```

## 架构

askme 是 S100P 机器人 4 进程栈的一部分（另外 3 个：LingTu 导航、frame_daemon 视觉、
brainstem 电机）。进程间通过 **CycloneDDS** (Pulse) + gRPC + REST 解耦。

| 层 | 说明 | 关键文件 |
|---|---|---|
| Blueprint | 声明式 `Runtime.use()` 组合入口（6 个预置） | `askme/blueprints/` |
| Runtime | Module ABC + `In[T]`/`Out[T]` 端口 + 自动装配 + 拓扑排序 | `askme/runtime/module.py` |
| Pipeline | `BrainPipeline` 解耦为 `TurnExecutor` / `StreamProcessor` / `SkillGate` | `askme/pipeline/` |
| Memory | L0 运行态 / L1 工作记忆 / L2 会话 / L3 情节 / L4 向量库 | `askme/memory/` |
| Voice | Mic → VAD → ASR (本地 sherpa-onnx + 云 DashScope) → TTS (MiniMax + Edge 兜底) | `askme/voice/` |
| Robot | DogSafetyClient / DogControlClient / ArmController / LED / Pulse (DDS) | `askme/robot/` |
| Skills | `SkillManager` + `SkillExecutor` + 41 内置 SKILL.md + 合约系统 | `askme/skills/` |
| Tools | 24 个 LLM tool-calling 实现（移动、扫描、视觉、robot API、bash 沙箱…） | `askme/tools/` |
| MCP | 暴露工具 / 资源给外部 Agent（Claude Desktop、Code） | `askme/mcp/` |

## 运行测试

```bash
pip install -e ".[dev]"
python -m pytest -q                          # ~20s，3500+ 测试
python -m pytest tests/test_pipeline_hooks.py -v   # 单模块
python -m pytest --cov=askme --cov-report=html     # 覆盖率报告
```

## 支持平台

- **S100P (sunrise)** — RDK X5, aarch64, Nash BPU 128 TOPS（生产）
- **Linux x86_64** — 服务器部署
- **Windows** — 开发测试（已处理 `add_signal_handler` 不兼容）

## 文档

- [`SOUL.md`](SOUL.md) — 语音人格定义（身份、说话风格、能力边界）
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 多进程系统架构 + DDS topic 规范 + 坐标系
- [`docs/HANDOFF.md`](docs/HANDOFF.md) — 交接文档（关键模式、陷阱、待完成工作）
- [`docs/LAYER_GAPS.md`](docs/LAYER_GAPS.md) — 分层差距清单（已大部分解决）
- [`docs/ARCHITECTURE_PATTERNS.md`](docs/ARCHITECTURE_PATTERNS.md) — 设计模式参考

## License

MIT © 穹沛科技 2025-2026
