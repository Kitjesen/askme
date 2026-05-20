# Skills package

`askme.skills` is the customer capability-package layer:

- `core/`: skill model, manager, executor, and generated-skill validation.
- `contracts/`: typed skill contracts, built-in contracts, and field capability routes.
- `governance/`: skill audit log, approval states, package policy, and growth backlog.
- `catalog/`: customer-visible capability center projection.
- `builtin/`: built-in customer, robot, patrol, wayfinding, and safety skill definitions.

New code should import from the owner package, for example:

```python
from askme.skills.core import SkillManager
from askme.skills.contracts import registered_skill_contracts
```

Legacy imports remain available while callers are migrated.
