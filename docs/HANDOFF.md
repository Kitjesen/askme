# 项目交接文档 — askme 机器人语音助手

**日期**: 2026-04-04  
**交接人**: Claude Code (Anthropic)  
**接收人**: 下一位开发者  
**当前主分支**: `master`（已合并 `claude/decouple-brainpipeline-eDX9V`）

---

## 一、项目概况

`askme` 是一个部署在 Thunder 巡检机器人上的语音 AI 助手，基于 Python 3.11+，支持：

- 语音识别（Sherpa-ONNX 本地 + 云端 ASR）
- LLM 对话（MiniMax/OpenAI 兼容接口）
- TTS 语音合成（MiniMax + Edge TTS 备选）
- 技能调度（Skill Manager + Agent Shell）
- 视觉感知（YOLOv8 BPU + VisionBridge）
- 机器臂控制（ArmController + Serial Bridge）
- 四足狗控制（DogControlClient）
- MCP 服务器（外部 Agent 接入）
- 主动感知（ProactiveAgent + ChangeDetector + ReactionEngine）

---

## 二、本次完成的工作

### 2.1 架构改进：BrainPipeline 解耦

**背景**: `BrainPipeline` 原本是一个巨型类，内部耦合了 StreamProcessor、SkillGate、TurnExecutor 三个组件，无法单独测试。

**新设计（Protocol 注入）**:

```python
# 测试/自定义实现时直接注入
pipeline = BrainPipeline(
    ...
    stream_processor=my_stream_processor,   # StreamProcessorProtocol
    skill_gate=my_skill_gate,               # SkillGateProtocol
    turn_executor=my_turn_executor,         # TurnExecutorProtocol
)
```

**新增文件**:
- `askme/pipeline/hooks.py` — `PipelineHooks` 生命周期回调系统
- `askme/pipeline/protocols.py` — Protocol 接口定义 + `TurnContext` 不可变上下文
- `askme/pipeline/utils.py` — 公用工具函数

**关键数据类型**:
- `TurnContext(user_text, source, cancel_token, voice_model)` — 每轮对话的不可变快照
- `ToolCallRecord(tool_name, args, result, latency_ms, ...)` — 工具调用结构化记录
- `PipelineHooks` — 注册 pre_turn/post_turn/pre_tool/post_tool/estop 回调

### 2.2 E-STOP 升级

`cancel_token`（`asyncio.Event`）现在从 `BrainPipeline.handle_estop()` 注入，流经：
- `TurnExecutor` → `StreamProcessor` → `SkillGate`

各组件**独立检查** `cancel_token.is_set()`，无需手动协调。调用 `pipeline.reset_estop()` 清除。

### 2.3 测试套件扩展

| 指标 | 之前 | 之后 | 增长 |
|------|------|------|------|
| 测试总数 | 2876 | **3520** | +644 |
| 测试文件数 | ~50 | **~116** | +66 |

**新增测试文件（66个）**涵盖：

| 模块 | 测试文件 | 测试数 |
|------|---------|--------|
| PipelineHooks 回调 | test_pipeline_hooks.py | 39 |
| PolicyStore YAML | test_policy_store.py | 33 |
| CloudASR 接收循环 | test_cloud_asr.py | 29 |
| ReactionEngine 规则 | test_reaction_engine.py | 27 |
| SkillDefinition/SlotSpec | test_skill_model.py | 40 |
| ChangeEvent/Observation | test_schemas.py | 38 |
| RobotMemBackend | test_robotmem_backend.py | 31 |
| ProactiveOrchestrator | test_proactive_base.py | 19 |
| ConfirmationAgent | test_confirm_agent.py | 20 |
| SlotCollectorAgent | test_slot_agent.py | 18 |
| ClarificationSession | test_session_state.py | 25 |
| BrainPipeline E-STOP | test_brain_pipeline_estop.py | 15 |
| FramePipeline | test_frames_pipeline.py | 19 |
| RuntimeModules | test_runtime_modules.py | 15 |
| LedController | test_led_controller.py | 18 |
| PolicyRunner (ONNX) | test_policy_runner.py | 15 |
| MapAdapter | test_map_adapter.py | 23 |
| ExtractionAdapter | test_extraction_adapter.py | 18 |
| MCP 资源/工具 | test_mcp_*.py × 3 | 36 |
| ... 及其他 | ... | ... |

---

## 三、项目结构导图

```
askme/
├── pipeline/           ← 核心管道（BrainPipeline + 3个子组件）
│   ├── brain_pipeline.py    主管道（支持 Protocol 注入）
│   ├── hooks.py             ← 新增：PipelineHooks 回调系统
│   ├── protocols.py         ← 新增：Protocol 接口 + TurnContext
│   ├── stream_processor.py  LLM 流式 + TTS 管道
│   ├── skill_gate.py        技能执行 + 安全门 + AgentShell 路由
│   ├── turn_executor.py     完整轮次编排 + 记忆 + 反思
│   ├── tool_executor.py     工具调用执行（带 hook）
│   ├── voice_loop.py        语音主循环
│   ├── text_loop.py         文本主循环（终端）
│   ├── proactive/           主动交互子系统
│   │   ├── orchestrator.py  ProactiveOrchestrator（三段链）
│   │   ├── clarification_agent.py  类型化槽位澄清
│   │   ├── confirm_agent.py        执行前确认
│   │   ├── slot_agent.py           遗留槽位采集
│   │   └── session_state.py        状态机
│   └── reaction_engine.py   ReactionEngine（规则 + 混合 LLM）
├── memory/
│   ├── policies.py         PolicyStore（YAML 规则 + 模板）
│   ├── robotmem_backend.py RobotMem SDK 包装（L4 向量记忆）
│   └── map_adapter.py      拓扑地图 + KG 适配
├── robot/
│   ├── led_controller.py   LED 状态控制
│   ├── policy_runner.py    ONNX 策略推理
│   └── serial_bridge.py    串口桥接
├── schemas/
│   ├── events.py           ChangeEvent（感知事件）
│   └── observation.py      Detection/Observation（感知帧）
├── skills/
│   ├── skill_model.py      SkillDefinition + SlotSpec
│   └── contracts.py        SkillContract（代码级合约）
├── mcp/                    MCP 服务器（Claude Desktop/Code 接入）
│   ├── server.py
│   ├── tools/              memory_search, robot_*, skill_*
│   └── resources/          perception, robot, health, skills
├── voice/
│   ├── punctuation.py      标点恢复（CT-Transformer）
│   └── generated_contracts.py  Edge Voice 接口契约
└── runtime/
    └── modules/            声明式模块系统
        ├── llm_module.py
        ├── memory_module.py
        ├── health_module.py
        └── led_module.py
```

---

## 四、如何运行测试

```bash
# 安装依赖
pip install -e ".[dev]"

# 运行全套测试（~20秒）
python -m pytest -q

# 只运行特定模块的测试
python -m pytest tests/test_pipeline_hooks.py -v

# 带覆盖率报告
python -m pytest --cov=askme --cov-report=html
```

**注意**: 以下模块需要硬件库（不影响 CI）：
- `sherpa_onnx` — 本地 ASR（测试中已 mock）
- `sounddevice` — 音频采集（测试中已 mock）
- `onnxruntime` — 策略推理（测试中已 mock）

---

## 五、待完成工作

### 5.1 高优先级

| 任务 | 位置 | 说明 |
|------|------|------|
| `ConversationManager` 扩展测试 | `askme/llm/conversation.py` | `add_tool_exchange`, `_trim`, `_strip_orphan_tool_messages` 仅有 6 个测试，342 行代码 |
| `ProactiveAgent` 更深覆盖 | `askme/pipeline/proactive_agent.py` | 44 个测试 / 634 行，`_patrol_tick`、`_detect_anomaly`、`_change_event_loop` 几乎没有测试 |
| `ThunderAgentShell` 子 Agent | `askme/agent_shell/thunder_agent_shell.py` | spawn_child_agent 超时/错误路径 |
| `VoiceLoop` 完整流程 | `askme/pipeline/voice_loop.py` | 目前主要测试 prefers_runtime_bridge 路径 |

### 5.2 中优先级

| 任务 | 说明 |
|------|------|
| 集成测试：语音 → Pipeline → TTS | 端到端 E2E，需要 mock 全栈 |
| `docs/archive/PRODUCTION_GAPS.md` 中列出的生产差距 | 详见该文档（已归档，待重新评估） |
| OTA bridge 远端上报 | `test_ota_bridge_core.py` 已覆盖本地部分 |
| 健康检查 HTTP 端点 | `AskmeHealthServer` 的 `/healthz`、`/metrics` 路由 |

### 5.3 架构改进方向

- **`BrainPipeline` 进一步简化**: 可以将 `_build_system_prompt` 和 `_build_l0_runtime_block` 完全移至 `PromptBuilder`，让 `BrainPipeline` 变成纯委托层
- **`ProactiveAgent` 重构**: 目前 634 行，可以抽取 `PatrolScheduler`、`AnomalyDetector`、`TelemetryPoller` 等
- **MCP 工具测试**: `memory_tools.py`、`voice_tools.py`、`robot_tools.py` 测试已有，但与 MCP context 耦合，可以考虑引入 `AppContext` fixture

---

## 六、关键模式和陷阱

### 6.1 MagicMock 的 `.name` 属性陷阱
```python
# ❌ 错误：MagicMock(name="foo") 只设置 repr，不设置 .name 属性
mock = MagicMock(name="input_obs")

# ✅ 正确：先创建，再赋值
mock = MagicMock()
mock.name = "input_obs"
```

### 6.2 本地 import 的 patch 路径
```python
# 如果函数内部做 from .move_tool import _call_runtime_api
# ❌ 错误
patch("askme.tools.scan_tool._call_runtime_api")
# ✅ 正确：patch 源模块
patch("askme.tools.move_tool._call_runtime_api")
```

### 6.3 asyncio.Event 与 E-STOP
```python
# cancel_token 由 BrainPipeline 持有，通过构造器传入子组件
# E-STOP 触发：
pipeline.handle_estop()   # 调用 cancel_token.set()
# 恢复：
pipeline.reset_estop()    # 调用 cancel_token.clear()
```

### 6.4 工具异常 vs RuntimeError
```python
# 工具中有特殊处理：RuntimeError → 通过 ThreadPoolExecutor 重试
# 测试异常路径时用 OSError，不要用 RuntimeError
vision.find_object = AsyncMock(side_effect=OSError("cam error"))
```

### 6.5 Protocol 注入路径（用于测试）
```python
from askme.pipeline.brain_pipeline import BrainPipeline
from unittest.mock import MagicMock, AsyncMock

pipeline = BrainPipeline(
    llm=MagicMock(), conversation=MagicMock(), memory=MagicMock(),
    tools=MagicMock(), skill_manager=MagicMock(), skill_executor=MagicMock(),
    audio=MagicMock(), splitter=MagicMock(),
    # 三个都传入 → 走注入路径，跳过内部构造
    stream_processor=MagicMock(),
    skill_gate=MagicMock(),
    turn_executor=MagicMock(process=AsyncMock(return_value="ok")),
)
```

---

## 七、依赖安装

```bash
# 开发环境
pip install -e ".[dev]"

# 生产环境（含硬件驱动）
pip install -e ".[prod]"

# 最小安装（仅核心，无硬件库）
pip install -e .
```

---

## 八、联系与资源

- **架构文档**: `docs/ARCHITECTURE.md`、`docs/ARCHITECTURE_PATTERNS.md`
- **配置布局总览**: `docs/CONFIG_GUIDE.md`
- **历史设计稿（已归档）**: `docs/archive/`（PRODUCTION_GAPS、MODULE_MIGRATION_PLAN、LAYER_GAPS、PROACTIVE_AGENT_V2、DECORATOR_ARCHITECTURE、AUTONOMOUS_ARCHITECTURE）
