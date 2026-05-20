# AskMe Pipeline Package Layout

`askme.pipeline` is split by orchestration responsibility. The root package
keeps compatibility aliases for older imports such as
`askme.pipeline.field_operations`; new code should import from the
responsibility-specific subpackages.

## Subpackages

- `core`: brain pipeline, turn executor, prompt builder, stream processor,
  tool executor, trace, hooks, protocols, and shared utilities.
- `channels`: text and voice loop entrypoints plus command handling and
  external turn recording.
- `field`: field operations, field ingest, incident alerting, site profile,
  readiness, launch readiness, and scenario definitions.
- `skills`: skill dispatcher, skill gate, and planner agent.
- `reactions`: reaction engine, proactive alert agent, and state LED bridge.
- `proactive`: multi-turn clarification and confirmation agents.

## Import Rule

Use canonical imports for new code:

```python
from askme.pipeline.field.field_operations import FieldOperationsService
from askme.pipeline.channels.voice_loop import VoiceLoop
from askme.pipeline.core.brain_pipeline import BrainPipeline
```

Owner subpackages also expose stable product entrypoints:

```python
from askme.pipeline.channels import VoiceLoop
from askme.pipeline.core import BrainPipeline, TurnExecutor
from askme.pipeline.skills import SkillDispatcher, SkillGate
```

Legacy imports remain supported:

```python
from askme.pipeline.field_operations import FieldOperationsService
```
