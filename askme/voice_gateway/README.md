# askme.voice_gateway

## Purpose

`askme.voice_gateway` owns the unified voice middle layer: stable turn APIs,
session/context routing, quality/latency hooks, and runtime-service switching.
It accepts already-recognized voice/text turns from channels and delegates turn
handling to a runtime bridge through a port.

Upper layers should call `VoiceGatewayService`. Provider or hardware details
must stay outside this package.

## Session Management

`voice_gateway` owns transport-facing conversation routing. Conversation Core
owns the durable Thread/Turn/Generation truth; use `ConversationSessionManager`
as the active gateway projection while that compatibility path is migrated:

- resolve the active conversation session for a voice/text turn;
- assemble turn-local conversation context before calling the runtime bridge;
- pass only explicit session/context values across ports.

Use `conversation_thread_id` for new outer voice/text APIs. During migration,
`conversation_session_id`, `conversation_id`, `chat_session_id`, and
`session_id` are aliases of that same canonical thread ID. Use
`planning_session_id` only for the inner cognition planning loop, and never use
a provider socket/session ID as the conversation thread.

`askme.cognition.memory.WorkingMemory` is not a conversation store. Treat it as
a per-conversation planning scratchpad that expires with the session or task.

Minimal assembly shape:

```python
from askme.cognition import CognitivePlanner, WorkingMemory, WorldStateService
from askme.voice_gateway import ConversationSessionManager, VoiceGatewayService


class EchoBridge:
    def status_snapshot(self):
        return {"enabled": True}

    def handle_voice_text(self, text):
        return {"source": "voice", "text": text}

    def handle_text_input(self, text):
        return {"source": "text", "text": text}


world_state = WorldStateService()
working_memory = WorkingMemory(persist_enabled=False)
planner = CognitivePlanner(world_state=world_state, working_memory=working_memory)

gateway = VoiceGatewayService(
    bridge=EchoBridge(),
    session_manager=ConversationSessionManager(),
)
result = gateway.handle_text_input(
    "inspect area A",
    conversation_thread_id="conv-1",
    include_session=True,
)
```

In production assembly, the bridge is a provider/runtime object that
implements `VoiceTurnBridgePort`. Providers and robot control code receive ports
or payloads; they must not call upward into `voice_gateway` or `cognition`.

## Does Not Own

`voice_gateway` is not the robot interaction policy layer. It does not classify
intent, decide whether the user addressed the robot, choose hardware/providers,
execute tools, or implement ASR/TTS engines.

Interaction decisions belong in `askme.robot_interaction`:

- address detection and bystander filtering: `AddressDetector`;
- wake/ignore/refuse/clarify/defer decisions: `InteractionGate`;
- command/query/quick-reply/voice-trigger routing: `IntentRouter`.

## Public Entrypoints

Use the service facade from channel loops and upper layers:

```python
from askme.voice_gateway import VoiceGatewayService
```

`VoiceGatewayService` owns the stable call shape for:

- `status_snapshot()`;
- `handle_voice_text(text, conversation_thread_id=...)`;
- `handle_text_input(text, conversation_thread_id=...)`;
- `conversation_snapshot(conversation_session_id)`.
- `conversation_context(conversation_session_id, recent_turn_limit=..., max_chars=...)`.

`conversation_context(...)` is the safe assembly surface for prompts and
runtime handoff context. It returns session identity, lifecycle state, active
planning/task IDs, summary, and a bounded recent-turn slice. It is not a long
term memory API.

Current modules:

- `service.py`: `VoiceGatewayService`, the stable voice-middle-layer facade.
- `session.py`: `ConversationSessionManager`, `ConversationSession`, and
  `ConversationTurn`.
- `runtime_bridge.py`: thin legacy facade over provider-owned runtime bridge
  construction.

## Boundary Rules

Allowed dependencies are stable contracts such as `askme.ports` plus local
`askme.voice_gateway.*` modules.

Except for `voice_gateway.runtime_bridge`, which is a legacy compatibility
facade over provider-owned code, files in this package must not import:

- `askme.robot` or `askme.robot.*`;
- `askme.runtime` or `askme.runtime.*`;
- `askme.providers` or `askme.providers.*`;
- `askme.pipeline` or `askme.pipeline.*`;
- `askme.tools` or `askme.tools.*`;
- `askme.mcp` or `askme.mcp.*`;
- `askme.api` or `askme.api.*`;
- `askme.robot_interaction` or `askme.robot_interaction.*`.

New runtime composition should use providers/runtime assembly code rather than
adding provider imports here.

## Common Changes

- Extend `VoiceGatewayService` when channel loops need a stable voice/text turn
  facade method.
- Update status or turn response shaping here only when it is part of the
  voice gateway service contract.
- Keep bridge construction outside this package; inject a
  `VoiceTurnBridgePort` implementation.
- Put address, gate, and intent policy changes in `askme.robot_interaction`.

## Verification

Run the voice gateway lane tests after changing this package:

```powershell
pytest tests\test_voice_runtime_bridge.py tests\test_voice_loop.py tests\test_contract_voice_gate.py tests\test_six_layer_package_boundaries.py -q
```

For README-only boundary edits, run:

```powershell
pytest tests\test_package_migration_compat.py::test_package_readmes_document_current_owner_subpackages -q
```

## Legacy Notes

`voice_gateway.runtime_bridge` is a compatibility facade over provider-owned
runtime bridge code. New runtime composition should use
`askme.providers.build_voice_runtime_bridge()`.

Historical import paths such as `askme.voice.runtime_bridge` and
`askme.voice.orchestration.runtime_bridge` remain compatibility facades. The
explicit compatibility registry lives in `askme.compat.legacy_facades`.
