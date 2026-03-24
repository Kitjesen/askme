"""MCP — Model Context Protocol 服务模式.

Exposes robot control, voice I/O, and skills via MCP (stdio).
Used by Claude Desktop/Code as a tool provider.

Usage::

    python -m askme.blueprints.mcp
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import (
    ControlModule,
    ExecutorModule,
    HealthModule,
    LLMModule,
    MemoryModule,
    PipelineModule,
    PulseModule,
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

    run_blueprint(mcp, "MCP")
