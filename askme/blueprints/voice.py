"""Voice — 纯语音 AI 助手.

核心能力：听、理解、回答、说。6 个模块。

Usage::

    python -m askme.blueprints.voice
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import (
    CognitionModule,
    ExecutorModule,
    HealthModule,
    LLMModule,
    MemoryModule,
    MissionModule,
    PipelineModule,
    RuntimeHandoffModule,
    SkillModule,
    TextModule,
    ToolsModule,
    VoiceModule,
)

voice = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(MissionModule)
    + Runtime.use(CognitionModule)
    + Runtime.use(RuntimeHandoffModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(ExecutorModule)
    + Runtime.use(VoiceModule)
    + Runtime.use(TextModule)
    + Runtime.use(HealthModule)
)

__all__ = ["voice"]

if __name__ == "__main__":
    from askme.blueprints._runner import run_blueprint

    run_blueprint(voice, "Voice")
