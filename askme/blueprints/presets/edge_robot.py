"""园区巡检机器人运行时蓝图。

用于客户试点交付：把语音、感知、现场事件、任务交接、机器人控制适配、
灯光提示和主动监测组合成一个边缘运行时。

启动：
    python -m askme.blueprints.presets.edge_robot
"""

from askme.blueprints.catalog.data import EDGE_ROBOT_MODULES

_LABEL = "园区巡检机器人运行时"


def _build_edge_robot():
    from askme.blueprints.runner.runner import compose_runtime
    from askme.runtime.modules import (
        CognitionModule,
        ControlModule,
        ExecutorModule,
        HealthModule,
        LEDModule,
        LLMModule,
        MemoryModule,
        MissionModule,
        PerceptionModule,
        PipelineModule,
        ProactiveModule,
        PulseModule,
        ReactionModule,
        RuntimeHandoffModule,
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
            ControlModule,
            LEDModule,
            ProactiveModule,
        )
    )


__all__ = ["edge_robot"]

if __name__ == "__main__":
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(_build_edge_robot, _LABEL, module_names=EDGE_ROBOT_MODULES)
else:
    edge_robot = _build_edge_robot()
