# askme

> 语音 AI 助手 for NOVA Dog 机器人 — 听、理解、回答、说。

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-1826%20passing-brightgreen)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20aarch64-lightgrey)

## 简介

askme 是运行在 NOVA Dog 机器人上的语音 AI 助手。核心链路：

```
麦克风 → VAD → ASR → LLM → TTS → 扬声器
```

支持中英文混合对话、机器人控制指令、多层记忆系统和技能扩展。

## 快速开始

```bash
pip install -r requirements.txt
# 填写 .env 中的 API keys
python -m askme.blueprints.voice   # 语音模式
python -m askme.blueprints.text    # 文本模式
```

## 架构

| 层 | 说明 |
|----|------|
| Blueprint | 声明式模块组合入口 (voice / text / edge_robot) |
| Pipeline | Memory → LLM → Tools → TTS |
| Memory | L1 工作记忆 / L2 会话摘要 / L3 情节记忆 / L4 长期向量库 |
| Voice | MicInput → VAD → ASR (local + cloud) → TTS |
| Robot | DogSafetyClient / ControlClient / MoveRobotTool |

## 支持平台

- **S100P (sunrise)** — RDK X5, aarch64, Nash BPU 128 TOPS
- **Windows** — 本地开发测试
- **Linux x86_64** — 服务器部署

## License

MIT © 穹沛科技 2025-2026
