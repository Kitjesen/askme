# askme.providers

## Purpose

`askme.providers` chooses concrete implementations for a deployment and returns
them behind stable `askme.ports` contracts. It owns deployment selection,
adapter assembly, default provider registration, and codec/edge details that
upper layers should not import directly.

For legacy backend registries under `askme.interfaces`, this package also owns
lower-layer default implementation registration.

## Does Not Own

`askme.providers` is a bottom-layer package. It must not own product workflows,
runtime policy, channel orchestration, MCP/API route behavior, robot
interaction decisions, or tool dispatch.

Do not add business flow here. If a provider needs a new capability, add or
extend a port first, then implement the provider behind that port.

## Public Entrypoints

Middle layers should import provider capabilities from the package facade:

```python
from askme.providers import build_perception, build_voice_runtime_bridge
```

Do not import concrete provider submodules from runtime, pipeline, MCP, API,
voice gateway, or robot interaction code.

Current provider modules:

- `arm_control.py`: mechanical-arm adapter assembly and safety defaults.
- `led.py`: LED controller and status LED bridge providers.
- `perception.py`: vision, interaction-perception, change-monitor,
  scene-intelligence, snapshot, image, and depth helpers.
- `register_defaults.py`: default backend registration for legacy registries.
- `robot_control.py`: robot command/capability dispatch adapter assembly.
- `runtime_executor/`: external runtime HTTP transport and its configured
  factory, exposed behind the runtime executor port.
- `safety.py`: safety/E-STOP adapter assembly.
- `spatial/`: nav-gateway adapters for navigation dispatch, navigation status,
  and LingTu temporal scene-memory queries behind ports.
- `telemetry`: telemetry bus providers.
- `voice.py`: audio frontend/router, ASR/TTS, edge voice I/O, voice runtime
  bridge, and voice profile resolution.
- `voice_runtime.py`: provider-owned runtime turn bridge implementation.

## Boundary Rules

Allowed dependencies:

- `askme.ports`;
- `askme.interfaces` backend registries from registration/factory modules only;
- concrete adapter packages such as `askme.robot`, `askme.perception`,
  `askme.voice`, SDKs, or HTTP clients;
- Python standard library.

Forbidden dependencies:

- `askme.blueprints`;
- `askme.runtime`;
- `askme.pipeline`;
- `askme.api`;
- `askme.mcp`;
- `askme.tools`;
- `askme.voice_gateway`;
- `askme.robot_interaction`.

If choosing between `providers` and `robot`, put concrete robot/service code in
`robot` and provider selection or adapter assembly in `providers`.

## Common Changes

- Add or update `build_*` factory functions when composition needs a configured
  implementation behind a port.
- Update `register_default_provider_backends()` when legacy backend registry
  defaults need to point at provider-owned implementations.
- Keep image/depth/audio codec or edge transport details behind provider helper
  functions when API/MCP code needs the data.
- Export new public factories from `askme.providers.__init__` only when upper
  layers need a stable facade import.

## Verification

Run the providers/ports lane tests after changing this package:

```powershell
pytest tests\test_six_layer_package_boundaries.py tests\test_register_defaults.py tests\test_registry.py tests\test_robot_tools_ext.py -q
```

For README-only boundary edits, run:

```powershell
pytest tests\test_package_migration_compat.py::test_package_readmes_document_current_owner_subpackages -q
```

## Legacy Notes

`register_default_provider_backends()` supports legacy backend registries under
`askme.interfaces`. Startup code may still import
`askme.interfaces.register_defaults`; that compatibility facade delegates to
provider-backed registrations here.

`StateLedBridge` remains exported only as a compatibility surface for
historical pipeline imports. New runtime composition should use
`build_status_led()`.
