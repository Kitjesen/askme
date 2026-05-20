"""语音感知运行时蓝图。

用于语音助手在规划任务前读取新鲜场景事实、交互准入证据和机器人安全状态。

启动：
    python -m askme.blueprints.presets.voice_perception
"""

from askme.runtime.core.module import Runtime
from askme.runtime.modules import (
    CognitionModule,
    ExecutorModule,
    HealthModule,
    LLMModule,
    MemoryModule,
    MissionModule,
    PerceptionModule,
    PipelineModule,
    PulseModule,
    ReactionModule,
    RuntimeHandoffModule,
    SafetyModule,
    SkillModule,
    TextModule,
    ToolsModule,
    VoiceModule,
)

voice_perception = (
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
    + Runtime.use(PulseModule)
    + Runtime.use(PerceptionModule)
    + Runtime.use(SafetyModule)
    + Runtime.use(ReactionModule)
)

__all__ = ["voice_perception"]

if __name__ == "__main__":
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(voice_perception, "语音感知运行时")
