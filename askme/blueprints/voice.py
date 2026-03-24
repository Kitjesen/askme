"""Voice — 语音巡检模式.

The default Thunder blueprint: voice AI assistant with perception,
safety gating, skill dispatch, and proactive patrol.

Usage::

    python -m askme.blueprints.voice
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import (
    ExecutorModule,
    HealthModule,
    LLMModule,
    MemoryModule,
    PerceptionModule,
    PipelineModule,
    ProactiveModule,
    PulseModule,
    SafetyModule,
    SkillModule,
    TextModule,
    ToolsModule,
    VoiceModule,
)

voice = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(PulseModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(PerceptionModule)
    + Runtime.use(SafetyModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(ExecutorModule)
    + Runtime.use(VoiceModule)
    + Runtime.use(TextModule)
    + Runtime.use(ProactiveModule)
    + Runtime.use(HealthModule)
)

__all__ = ["voice"]

if __name__ == "__main__":
    from askme.blueprints._runner import run_blueprint

    run_blueprint(voice, "Voice")
