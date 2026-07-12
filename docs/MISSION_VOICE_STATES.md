# Askme Mission Voice States

Askme is no longer treated as the main mission system. It is a voice and
knowledge service that receives speech, checks whether the speech is allowed in
the current field state, and then forwards allowed commands to ZeroClaw or the
runtime API.

## State Model

| State | User meaning | Voice policy |
| --- | --- | --- |
| `setup` | First-run setup, API/model/device configuration | Visitors are blocked. Operator/admin can continue setup dialogue. |
| `idle` | No active mission | Existing assistant behavior is preserved. Wayfinding, status, and new task requests can enter the agent. |
| `mission_active` | Patrol/reconstruction/field task is running | Free chat is blocked. Only status, pause, safety report, emergency stop, and operator control commands enter the agent. |
| `paused` | Mission is intentionally paused | Free chat is blocked. Continue, stop, status, and safety commands are allowed. Route changes require supervisor/admin. |
| `emergency` | E-stop or safety hold | Safety-only. Resume requires supervisor/admin. |
| `review` | Mission ended; results and reports are being reviewed | Result/report commands are allowed. New mission start is operator-only. |

## Roles

| Role | Meaning |
| --- | --- |
| `visitor` | Public bystander or unauthenticated speaker |
| `operator` | Normal authenticated operator |
| `supervisor` | Safety or site lead |
| `admin` | System administrator |

These roles are only a coarse pre-gate. Runtime RBAC and safety services remain
the final authority for hardware or mission execution.

## Command Classes

Askme classifies a turn before the LLM sees it:

- `status`: state, progress, battery, location, result/report query
- `pause`: pause or hold
- `resume`: continue or resume
- `cancel`: stop mission, cancel, return
- `emergency_stop`: e-stop and urgent stop phrases
- `report_anomaly`: fault, smoke, leak, danger, injury, alarm
- `call_operator`: help, contact operator/admin/safety lead
- `start_mission`: start patrol/reconstruction/mapping
- `route_change`: navigate to / change route
- `wayfinding`: public location questions
- `chat`: everything else

## Runtime Contract

The current code path is:

```python
from askme.robot_interaction import InteractionGate

gate = InteractionGate(config["voice"]["interaction_gate"])
gate.set_mission_context(mission_mode="mission_active", actor_role="visitor")
decision = gate.evaluate("今天天气怎么样", addressed=True)
```

If `decision.should_continue_to_brain` is false, the voice loop must not send the
text to the agent. It can speak `decision.reply` and record the turn as an
environment event when `decision.should_record_environment` is true.

ZeroClaw/Nerva should own the mission state. Askme should only consume the
state snapshot and enforce the voice admission policy.
