"""MCP 工具服务蓝图。

用于把 askme 的记忆、技能、认知规划和受控机器人能力暴露给 MCP 客户端。

启动：
    python -m askme.blueprints.presets.mcp
"""

from askme.runtime.core.module import Runtime
from askme.runtime.modules import (
    CognitionModule,
    ControlModule,
    ExecutorModule,
    HealthModule,
    LLMModule,
    MemoryModule,
    MissionModule,
    PipelineModule,
    PulseModule,
    RuntimeHandoffModule,
    SafetyModule,
    SkillModule,
    ToolsModule,
    VoiceModule,
)

mcp = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(PulseModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(MissionModule)
    + Runtime.use(CognitionModule)
    + Runtime.use(RuntimeHandoffModule)
    + Runtime.use(SafetyModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(ExecutorModule)
    + Runtime.use(VoiceModule)
    + Runtime.use(ControlModule)
    + Runtime.use(HealthModule)
)

__all__ = ["mcp"]

if __name__ == "__main__":
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(mcp, "MCP 工具服务")
