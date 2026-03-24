"""Edge Robot — 完整边缘机器人.

语音 + 感知 + 外部插件。14 个模块。

Usage::

    python -m askme.blueprints.edge_robot
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import (
    ControlModule,
    ExecutorModule,
    HealthModule,
    LEDModule,
    ProactiveModule,
    ToolsModule,
)

from askme.blueprints.voice_perception import voice_perception

edge_robot = (
    voice_perception
    + Runtime.use(ToolsModule)
    + Runtime.use(ExecutorModule)
    + Runtime.use(ControlModule)
    + Runtime.use(LEDModule)
    + Runtime.use(ProactiveModule)
    + Runtime.use(HealthModule)
)

__all__ = ["edge_robot"]

if __name__ == "__main__":
    from askme.blueprints._runner import run_blueprint

    run_blueprint(edge_robot, "Edge Robot")
