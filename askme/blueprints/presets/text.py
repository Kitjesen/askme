"""文本运营控制台蓝图。

用于开发调试、知识管理和任务干跑，不依赖麦克风、扬声器或机器人侧 IO。

启动：
    python -m askme.blueprints.presets.text
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
)

text = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(MissionModule)
    + Runtime.use(CognitionModule)
    + Runtime.use(RuntimeHandoffModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(ExecutorModule)
    + Runtime.use(TextModule)
    + Runtime.use(HealthModule)
)

__all__ = ["text"]

if __name__ == "__main__":
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(text, "文本运营控制台")
