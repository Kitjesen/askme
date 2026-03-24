"""Edge Robot — 全功能边缘机器人模式.

Extends voice with hardware control (motors, LED).
Deploy on S100P with robot_mode=True.

Usage::

    python -m askme.blueprints.edge_robot
"""

from askme.runtime.module import Runtime
from askme.runtime.modules import ControlModule, LEDModule

from askme.blueprints.voice import voice

edge_robot = (
    voice
    + Runtime.use(ControlModule)
    + Runtime.use(LEDModule)
)

__all__ = ["edge_robot"]

if __name__ == "__main__":
    from askme.blueprints._runner import run_blueprint

    run_blueprint(edge_robot, "Edge Robot")
