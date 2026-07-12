# Agent Shell

本目录现在只保留 Agent Profile 治理和历史 AgentShell 兼容入口。旧的
Python ReAct 执行循环已经由 ZeroClaw MCP Agent 取代；Askme 侧只负责
Profile 审核、工具边界、声明式 hook、MCP/TaskHandoff 接入和安全审计。

## 目录职责

- `agent_shell.py`：产品侧中性兼容入口，导出已废弃的 `AgentShell` stub。
- `thunder_agent_shell.py`：历史兼容 stub，保留旧导入路径，新增代码不要直接依赖它。
- `agent_profile.py`：Agent 角色、工具边界、权限模式和执行策略。
- `agent_hooks.py`：Agent 执行前后的钩子、阻断和审计决策。

## 产品边界

本地 AgentShell stub 不执行任务、不调用工具、不派生子 agent。需要 agentic
决策时，走 ZeroClaw MCP Agent，再由 Askme 的受控入口进入：

- 技能审批和能力包开关。
- safety-gated robot API。
- runtime arbiter 的任务接管。
- 审计日志和执行摘要。

## 使用约定

新业务代码不要新增对 `AgentShell.run_task()` 的依赖。旧导入路径仅用于兼容
现有测试、历史配置和已部署脚本：

```python
from askme.agent_shell import AgentShell
```
