# Agent Shell

自主执行 Agent 层，用于把经过安全准入的复杂任务交给大模型工具循环处理。

## 目录职责

- `agent_shell.py`：产品侧中性入口，导出 `AgentShell`。
- `thunder_agent_shell.py`：历史兼容实现文件，保留旧导入路径，新增代码不要直接依赖它。
- `agent_profile.py`：Agent 角色、工具边界、权限模式和执行策略。
- `agent_hooks.py`：Agent 执行前后的钩子、阻断和审计决策。

## 产品边界

Agent Shell 可以读写工作区、调用工具、调用机器人 API 和播报进度，但不能绕过：

- 技能审批和能力包开关。
- safety-gated robot API。
- runtime arbiter 的任务接管。
- 审计日志和执行摘要。

## 使用约定

新代码使用：

```python
from askme.agent_shell import AgentShell
```

旧导入路径仅用于兼容现有测试、历史配置和已部署脚本。
