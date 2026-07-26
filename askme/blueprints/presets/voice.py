"""语音任务中心蓝图。

用于麦克风驱动的操作员对话、任务规划和客户语音演示，不直接接管机器人侧 IO。

启动：
    python -m askme.blueprints.presets.voice
"""

from askme.blueprints.catalog.data import VOICE_MODULES

_LABEL = "语音任务中心"


def _build_voice():
    from askme.blueprints.runner.runner import compose_runtime
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

    return compose_runtime(
        (
            LLMModule,
            ToolsModule,
            MemoryModule,
            MissionModule,
            CognitionModule,
            RuntimeHandoffModule,
            PipelineModule,
            SkillModule,
            ExecutorModule,
            VoiceModule,
            TextModule,
            HealthModule,
        )
    )

__all__ = ["voice"]

if __name__ == "__main__":
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(_build_voice, _LABEL, module_names=VOICE_MODULES)
else:
    voice = _build_voice()
