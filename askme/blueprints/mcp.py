"""MCP Tool Provider blueprint.

Use this runtime when askme should expose memory, skills, cognition, and
controlled robot capabilities to MCP clients.

Run:
    python -m askme.blueprints.mcp
"""

from askme.runtime.module import Runtime
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
    from askme.blueprints._runner import run_blueprint

    run_blueprint(mcp, "MCP Tool Provider")
