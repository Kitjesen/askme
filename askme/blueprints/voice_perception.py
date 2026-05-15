"""Voice Plus Perception blueprint.

Use this when the voice assistant must reason over fresh scene facts,
interaction-gate evidence, and robot safety state before planning a task.

Run:
    python -m askme.blueprints.voice_perception
"""

from askme.blueprints.voice import voice
from askme.runtime.module import Runtime
from askme.runtime.modules import (
    PerceptionModule,
    PulseModule,
    ReactionModule,
    SafetyModule,
)

voice_perception = (
    voice
    + Runtime.use(PulseModule)
    + Runtime.use(PerceptionModule)
    + Runtime.use(SafetyModule)
    + Runtime.use(ReactionModule)
)

__all__ = ["voice_perception"]

if __name__ == "__main__":
    from askme.blueprints._runner import run_blueprint

    run_blueprint(voice_perception, "Voice Plus Perception")
