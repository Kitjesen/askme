"""语音感知运行时蓝图。

用于语音助手在规划任务前读取新鲜场景事实、交互准入证据和机器人安全状态。

启动：
    python -m askme.blueprints.presets.voice_perception
"""

from askme.blueprints.catalog.data import VOICE_PERCEPTION_MODULES

_LABEL = "语音感知运行时"


def _build_voice_perception():
    from askme.blueprints.runner.runner import compose_runtime
    from askme.blueprints.runtime_composition import RuntimeHandoffModule
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
        SafetyModule,
        SkillModule,
        TextModule,
        ToolsModule,
        VoiceModule,
        WarmSessionModule,
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
            WarmSessionModule,
            HealthModule,
            PulseModule,
            PerceptionModule,
            SafetyModule,
            ReactionModule,
        )
    )


__all__ = ["voice_perception"]

if __name__ == "__main__":
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(_build_voice_perception, _LABEL, module_names=VOICE_PERCEPTION_MODULES)
else:
    voice_perception = _build_voice_perception()
