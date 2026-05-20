# Cognition package

`askme.cognition` owns robot-aware planning state. It is the product brain
boundary between perception input, task planning, short-term working memory,
and world-state facts used by safety preflight and runtime handoff.

- `planning/`: cognitive plan objects, planning sessions, and the planner.
- `world/`: product world-state facts and snapshot service.
- `perception/`: active perception refresh and perception-to-world sync.
- `memory/`: short-term planning memory used during cognition.

## Working Memory

`WorkingMemory` is per-conversation scratch space for the current planning
task. Use it for recent operator utterances, assistant replies, observations,
and focus keys that help the planner complete the active session.

Focus and selected context are scoped by `conversation_session_id` when one is
provided. This prevents one operator/session's short-term planning state from
being injected into another session's plan.

Use `conversation_session_id` for the outer voice/text dialog session. Use
`planning_session_id` for the inner cognition planning loop. Do not reuse the
same `session_id` field for both meanings.

Do not use `WorkingMemory` for long-term persistence, user profile storage,
conversation history, or retrieval memory. Persistent memory belongs in
policy-controlled memory services outside `askme.cognition.memory`.

Minimal planner assembly:

```python
from askme.cognition import CognitivePlanner, WorkingMemory, WorldStateService

world_state = WorldStateService()
working_memory = WorkingMemory(persist_enabled=False)
planner = CognitivePlanner(world_state=world_state, working_memory=working_memory)
```

`voice_gateway` owns conversation/session management and should pass
`conversation_session_id` into cognition. Providers and robot control code must
not call upward into `voice_gateway` or `cognition`; use ports, payloads, or
runtime module assembly instead.

Implementation files live inside those owner subpackages. Root modules such as
`planner.py`, `world_state.py`, `perception_sync.py`, `active_perception.py`,
`working_memory.py`, and `planning_session.py` are compatibility aliases only.

New code should import from the owner package, for example:

```python
from askme.cognition.planning import CognitivePlanner
from askme.cognition.world import WorldStateService
from askme.cognition.perception import CognitionPerceptionSync, normalize_scene_snapshot
```

The package root still exports the public contracts for compatibility while
callers migrate.
