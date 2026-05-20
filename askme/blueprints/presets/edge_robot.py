"""园区巡检机器人运行时蓝图。

用于客户试点交付：把语音、感知、现场事件、任务交接、机器人控制适配、
灯光提示和主动监测组合成一个边缘运行时。

启动：
    python -m askme.blueprints.presets.edge_robot
"""

from askme.runtime.core.module import Runtime
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
)

edge_robot = (
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
    + Runtime.use(ControlModule)
    + Runtime.use(LEDModule)
    + Runtime.use(ProactiveModule)
)

__all__ = ["edge_robot"]

if __name__ == "__main__":
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(edge_robot, "园区巡检机器人运行时")
