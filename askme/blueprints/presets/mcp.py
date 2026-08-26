"""MCP 工具服务蓝图。

用于把 askme 的记忆、技能、认知规划和受控机器人能力暴露给 MCP 客户端。

启动：
    python -m askme.blueprints.presets.mcp
"""

from askme.blueprints.catalog.data import MCP_MODULES

_LABEL = "MCP 工具服务"


def _build_mcp():
    from askme.blueprints.runner.runner import compose_runtime
    from askme.blueprints.runtime_composition import RuntimeHandoffModule
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
        SafetyModule,
        SkillModule,
        ToolsModule,
        VoiceModule,
        WarmSessionModule,
    )

    return compose_runtime(
        (
            LLMModule,
            ToolsModule,
            PulseModule,
            MemoryModule,
            MissionModule,
            CognitionModule,
            RuntimeHandoffModule,
            SafetyModule,
            PipelineModule,
            SkillModule,
            ExecutorModule,
            VoiceModule,
            ControlModule,
            WarmSessionModule,
            HealthModule,
        )
    )


__all__ = ["mcp"]

if __name__ == "__main__":
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(_build_mcp, _LABEL, module_names=MCP_MODULES)
else:
    mcp = _build_mcp()
