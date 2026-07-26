"""文本运营控制台蓝图。

用于开发调试、知识管理和任务干跑，不依赖麦克风、扬声器或机器人侧 IO。

启动：
    python -m askme.blueprints.presets.text
"""

from askme.blueprints.catalog.data import TEXT_MODULES

_LABEL = "文本运营控制台"


def _build_text():
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
            TextModule,
            HealthModule,
        )
    )

__all__ = ["text"]

if __name__ == "__main__":
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(_build_text, _LABEL, module_names=TEXT_MODULES)
else:
    text = _build_text()
