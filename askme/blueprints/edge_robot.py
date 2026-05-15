"""Park Patrol Robot Runtime blueprint.

This is the primary customer-pilot runtime: voice, perception, field event
handling, runtime handoff, robot control adapters, indicators, and proactive
monitoring in one edge composition.

Run:
    python -m askme.blueprints.edge_robot
"""

from askme.blueprints.voice_perception import voice_perception
from askme.runtime.module import Runtime
from askme.runtime.modules import (
    ControlModule,
    ExecutorModule,
    HealthModule,
    LEDModule,
    ProactiveModule,
    ToolsModule,
)

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

    run_blueprint(edge_robot, "Park Patrol Robot Runtime")
