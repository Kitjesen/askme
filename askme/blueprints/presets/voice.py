"""语音任务中心蓝图。

用于麦克风驱动的操作员对话、任务规划和客户语音演示，不直接接管机器人侧 IO。

启动：
    python -m askme.blueprints.presets.voice
"""

from askme.runtime.core.module import Runtime
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
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(voice, "语音任务中心")
