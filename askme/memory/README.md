# Memory package

`askme.memory` is split by product responsibility:

- `core/`: conversation, session, episodic, procedural, admission policy, and memory service orchestration.
- `retrieval/`: knowledge catalog, import, RAG bridge, semantic/vector indexes, map and site knowledge.
- `backends/`: optional external memory providers such as RobotMem and MemPalace.
- `intelligence/`: trend analysis, association graph, extraction adapter, and suggestion strategy.

Product boundary:

- Customer knowledge RAG is the evidence source for answers shown to customers.
  Configure it with `memory.customer_knowledge_backend`; `memory.backend` remains
  a backward-compatible alias.
- Robot behavior memory is separate. Configure it with
  `memory.robot_behavior_memory_backend` and only enable it through
  `memory.robot_behavior_memory_enabled` when the deployment has a clear use
  case and audit policy.

New code should import from the owner package, for example:

```python
from askme.memory.core import EpisodicMemory
from askme.memory.retrieval import MemoryBridge
```

Legacy imports remain available while the rest of the codebase is migrated.
