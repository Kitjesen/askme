# askme.tools

## Purpose

`askme.tools` contains in-process callable tools used by runtime modules,
skills, MCP handlers, and selected API services. It owns reusable tool
definitions, argument validation, registry metadata, execution-control
contracts, and registration helpers.

Tools are not an external transport. External callers should reach them through
API or MCP unless they embed askme as Python code.

## Does Not Own

Do not put direct hardware, voice, vision, ASR/TTS, camera, scene-intelligence,
or robot-client implementation logic in this package. Tool modules may
coordinate an action, but they should receive concrete behavior through
constructor arguments, registration functions, `askme.ports`,
`askme.providers`, or a service that already owns the boundary.

Transport contracts also stay outside this package:

- HTTP request/response behavior belongs in `askme.api`.
- MCP protocol shape belongs in `askme.mcp`.
- Runtime lifecycle and module wiring belong in `askme.runtime` or pipeline
  composition.

Runtime HTTP access is additionally guarded by static tests. New modules under
`askme.tools` must not open runtime HTTP connections directly with
`urllib.request`, `requests`, `httpx`, or `aiohttp`. Route runtime actions
through injected ports/providers, existing registration wiring, or one of the
explicitly authorized legacy/transport tools:

- `askme.tools.core.builtin_tools`
- `askme.tools.robot.robot_api_tool`
- `askme.tools.robot.runtime_api`
- `askme.tools.spatial.temporal_query_tool`

## Public Entrypoints

Use owner subpackages for new imports:

```python
from askme.tools.core import BaseTool, ToolRegistry, register_builtin_tools
from askme.tools.robot import register_robot_tools
from askme.tools.voice import register_voice_tools
from askme.tools.skills import register_skill_tools
```

The package root preserves these compatibility exports:

```python
from askme.tools import BaseTool, ToolRegistry, register_builtin_tools
```

Current owner directories:

- `core`: tool registry, built-in tools, and execution-control contracts.
- `field`: field event trigger tools.
- `robot`: move, robot API, and robot-specific tool registrations.
- `skills`: skill creation and skill package management tools.
- `spatial`: scan, vision, space, route, and temporal query tools.
- `voice`: mute, unmute, and stop-speaking tools.

## Boundary Rules

- Use canonical imports such as `askme.tools.core.ToolRegistry` or
  `askme.tools.robot.MoveRobotTool` in new code.
- Keep tool payload validation close to the tool class or registration helper.
- Pass concrete clients into tool constructors or registration helpers; do not
  build hardware/provider clients inside generic tool modules.
- Keep transport-specific response formatting out of tool implementations.
- If a tool needs a capability currently provided by a concrete client, add or
  use a port/provider boundary first.

## Common Changes

- Add a new built-in or shared registry behavior under `core/`.
- Add robot-facing actions under `robot/` when they are reusable callable
  actions, not robot-client implementations.
- Add voice command tools under `voice/` only for tool-level control actions
  such as mute, unmute, or stop speaking.
- Add spatial/perception query tools under `spatial/` and route concrete
  perception work through provider/port dependencies.
- Update registration helpers when MCP, runtime, or skills need the new tool in
  a registry.

## Verification

Run the tools/MCP lane tests after changing this package:

```powershell
pytest tests\test_builtin_tools.py tests\test_tool_registry.py tests\test_mcp_tools.py -q
```

For README-only boundary edits, also run:

```powershell
pytest tests\test_package_migration_compat.py::test_package_readmes_document_current_owner_subpackages -q
```

## Legacy Notes

The package root installs compatibility aliases for historical imports such as
`askme.tools.tool_registry`, `askme.tools.move_tool`,
`askme.tools.robot_tools`, and `askme.tools.voice_tools`. Do not add new
implementation modules at those legacy paths. Put new code in the owning
subpackage and update compatibility aliases only through the package facade when
the migration contract requires it.
