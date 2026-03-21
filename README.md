# Askme — Thunder 工业巡检机器人语音 AI 系统

自主感知 + 语音交互 + 记忆推理 + 联网搜索，全栈 on-device AI。

## 快速启动

```bash
# 语音模式（生产）
python -m askme --legacy

# 文本模式（调试）
python -m askme --legacy --text

# MCP 服务器模式
python -m askme
```

## Sunrise 部署

```bash
# 同步代码到 sunrise（7 秒，tar pipe）
bash scripts/sync_sunrise.sh

# 端到端测试（9 个用例）
bash scripts/e2e_test.sh

# 重启服务
ssh sunrise@192.168.66.190 'tmux kill-session -t askme; \
  tmux new-session -d -s askme bash -c "cd ~/askme && source .venv/bin/activate && \
  export \$(grep -v ^# .env | xargs) && python -m askme --legacy"'
```

## 系统架构

```
感知层 (持续运行)
  Orbbec 相机 30fps → frame_daemon (BPU YOLO 3ms) → ChangeDetector (1Hz)
  → 人出现/消失等事件 → ProactiveAgent 自动响应

决策层 (按需)
  语音输入 → ASR → IntentRouter → 31 个技能 / LLM 对话
  感知事件 → ProactiveAgent → auto_solve

记忆层 (持续积累)
  对话历史 → 会话摘要 → 情景记忆 → qp_memory 长期知识

行动层 (通过 runtime API)
  语音播报 / 视觉理解 / 联网搜索 / 运动控制
```

## 技术栈

| 模块 | 技术 |
|------|------|
| LLM | MiniMax M2.7 highspeed (TTFT ~2s) |
| VLM | 千问 VL (DashScope) |
| ASR | sherpa-onnx + DashScope Paraformer |
| TTS | MiniMax Speech 2.8 HD |
| YOLO | BPU 3ms (Horizon Nash 128 TOPS) |
| 深度 | Orbbec Gemini 335 (毫米精度) |
| 搜索 | Bing + 百度 |
| 记忆 | qp_memory (markdown, DashScope 提取) |

## 技能 (31 启用 / 10 待 runtime)

**自主推理**: solve_problem, agent_task, recall_memory
**视觉搜索**: find_object, find_person, safety_check, check_location, patrol_scan
**语音控制**: volume, speed, mute/unmute, stop_speaking, repeat_last
**信息查询**: get_time, system_status, web_search, list_skills, environment_report
**快速回复**: 你好/谢谢/再见 → 16ms 直回（不走 LLM）

## 关键性能

| 指标 | 数值 |
|------|------|
| 快速回复 (你好) | 16ms |
| LLM TTFT | ~2s |
| YOLO 检测读取 | 0.2ms |
| 帧读取 | 1ms |
| TTS | ~1.6s |
| find_object 端到端 | ~17s |
| 记忆查询 | ~8s |

## 目录结构

```
askme/
├── core/           # 框架层 (@tool @skill, Module, Orchestrator, EventBus)
├── brain/          # 大脑 (LLM, 记忆, 视觉, 意图路由)
├── pipeline/       # 管线 (语音循环, 技能调度, 主动巡逻)
├── agent_shell/    # 自律执行 (ThunderAgentShell, 多轮工具调用)
├── skills/builtin/ # 41 个技能 (SKILL.md 声明式)
├── tools/          # 工具 (视觉, 运动, 搜索, bash)
├── voice/          # 语音 (ASR, TTS, VAD, 降噪)
├── perception/     # 感知 (ChangeDetector, 事件驱动)
├── schemas/        # 数据结构 (Observation, ChangeEvent)
scripts/
├── sync_sunrise.sh         # 一键同步 (7s)
├── e2e_test.sh             # 端到端测试 (9 cases)
├── frame_daemon.py         # 帧+BPU 常驻服务
docs/
├── AUTONOMOUS_ARCHITECTURE.md
├── PROACTIVE_AGENT_V2_ARCHITECTURE.md
├── DECORATOR_ARCHITECTURE.md
├── PRODUCTION_GAPS.md
```

## 架构文档

- [自主运行架构](docs/AUTONOMOUS_ARCHITECTURE.md) — 完整系统图、性能、技能矩阵
- [ProactiveAgent V2](docs/PROACTIVE_AGENT_V2_ARCHITECTURE.md) — 事件驱动感知设计
- [装饰器架构](docs/DECORATOR_ARCHITECTURE.md) — @tool @skill 声明式框架
- [生产差距分析](docs/PRODUCTION_GAPS.md) — 上真机前的待办
