# Askme — Claude 工作指南

> askme 是面向园区的机器人现场任务与智能交互平台（v4.1.0）。
> 架构细节见 `docs/ARCHITECTURE.md`，产品边界见 `docs/PRODUCT.md`。

## 快速启动

```powershell
cd D:\inovxio\tools\askme
python -m askme.blueprints.presets.edge_robot
```

Dashboard: `http://127.0.0.1:8765/dashboard`

## 常用命令

```powershell
# 测试（默认排除 slow）
python -m pytest tests/ -q

# 全量测试
python -m pytest tests/ -q -m ""

# 静态检查
python -m ruff check askme tests scripts

# 自动修复
python -m ruff check --select I --fix askme tests scripts

# 语音健康
python -m askme runtime voice-health --json

# RAG Trust 评测
python scripts\eval\evaluate_rag_trust_scenarios.py --output artifacts\rag_trust\scenario-evaluation.json

# Voice E2E 评测
python scripts\eval\evaluate_voice_e2e_scenarios.py --output artifacts\voice_e2e\scenario-evaluation.json
```

## 架构依赖方向

```
blueprints → runtime/pipeline/voice_gateway/robot_interaction/api/mcp
  → ports/interfaces → providers → robot/perception/voice/external
```

- `ports` 只定义协议，不能 import provider 或硬件实现
- `providers` 不能依赖 runtime、pipeline、blueprints、api、voice_gateway
- 边界测试：`pytest tests/test_six_layer_package_boundaries.py -q`

## 关键文件

| 文件 | 作用 |
|------|------|
| `askme/robot_interaction/interaction_gate.py` | 交互准入门 |
| `askme/robot_interaction/perception_context.py` | 感知快照归一化 |
| `askme/memory/retrieval/catalog.py` | 知识生命周期事实源 |
| `askme/cognition/active_perception.py` | 缺事实时主动刷新感知 |
| `askme/runtime/task/handoff.py` | TaskHandoff、TaskRun、状态机 |
| `askme/runtime/task/arbiter_client.py` | external/lab contract-only client |
| `askme/runtime/modules/health_module.py` | HTTP/Dashboard wiring |
| `askme/static/dashboard.html` | Voice Mission Center |

## 安全不变量

- 不直接发 motor/gait/arm/serial/cmd_vel
- 不绕过 runtime arbiter 和 SafetyPreflight
- 不用 stale/conflict/unapproved knowledge 驱动高风险任务
- operator action 必须记录 actor、reason、risk acknowledgement
- lab/prod 必须显式启用，默认安全

## 并行开发规则

详见 `docs/MULTI_AGENT_WORKFLOW.md`。核心约束：
- 每个 agent 分配不重叠的 write scope
- 共享文件（`BOUNDARIES.md`、`CODE_MAP.md`、package `__init__.py`）每轮只有一个 owner
- 合并顺序：ports → providers → runtime/pipeline/API
