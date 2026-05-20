# askme.mcp

## Purpose

`askme.mcp` is the Model Context Protocol boundary for agent clients. It adapts
selected askme capabilities into MCP tools and resources without making agent
clients depend on HTTP route modules or in-process runtime internals.

This package owns:

- the shared `FastMCP` server instance and MCP lifecycle context;
- MCP tool/resource names, descriptions, payload validation, and registration;
- agent-facing health, contract, skill, robot, voice, vision, and memory
  resources;
- translation from MCP calls into existing tools, providers, ports, services,
  and memory boundaries.

## Does Not Own

`askme.mcp` must not implement hardware, voice, vision, robot, memory, runtime,
or product business behavior directly. Delegate reusable behavior to
`askme.tools`, `askme.providers`, `askme.ports`, memory services, skill
services, or API service helpers when that behavior already exists.

MCP is not the HTTP surface. Put HTTP contracts in `askme.api`, MCP protocol
exposure here, and reusable callable behavior in `askme.tools` or a service.

## Public Entrypoints

Start the MCP server through one of these public commands:

```bash
askme-mcp
python -m askme.mcp
python -m askme.mcp.server
python -m askme mcp serve --transport stdio
```

`askme-mcp` is the installed console script declared in `pyproject.toml`; in a
raw checkout where console scripts have not been installed, use one of the
`python -m ...` forms.

Code that needs the server object may import:

```python
from askme.mcp.server import mcp, main
```

`askme.mcp.server.AppContext` is the shared runtime context for MCP tool and
resource handlers. It wires configuration, memory, skills, the tool registry,
providers, and optional robot or voice adapters during the MCP lifespan.

Current owner directories:

- `resources`: MCP resources for contracts, health, perception, robot, and
  skills.
- `tools`: MCP tools for memory, robot, skills, vision, and voice.
- `registration.py`: the explicit `MCP_MODULES` manifest imported during
  startup so decorator registration stays deterministic.

## Boundary Rules

- Use `askme.tools` for callable in-process capabilities and registry-backed
  execution.
- Use `askme.providers` and `askme.ports` for concrete adapters to robot arms,
  voice I/O, perception, scene intelligence, and external systems.
- Use service modules for business rules, schema shaping, and durable product
  contracts.
- Do not import hardware, voice, vision, or robot implementation modules
  directly from MCP handlers unless the provider or port boundary explicitly
  owns that adapter.
- Keep logging on stderr in stdio mode; stdout is the JSON-RPC channel.

## Common Changes

- Add an agent-visible resource under `resources/` when clients need read-only
  state or contract metadata.
- Add an agent-callable action under `tools/` when clients need to invoke an
  existing capability through MCP.
- Extend `AppContext` only when multiple MCP handlers need the same initialized
  dependency.
- Register new MCP modules by adding them to `MCP_MODULES` in
  `registration.py` after the shared `mcp` instance is created.

## Verification

Run the MCP/tools lane tests after changing this package:

```powershell
pytest tests\test_mcp_tools.py tests\test_mcp_memory_tools.py tests\test_mcp_misc_resources.py tests\test_builtin_tools.py tests\test_tool_registry.py -q
```

For README-only boundary edits, also run:

```powershell
pytest tests\test_package_migration_compat.py::test_package_readmes_document_current_owner_subpackages -q
```

## Legacy Notes

`python -m askme.mcp.server` aliases the running `__main__` module back to
`askme.mcp.server` so imported MCP tool/resource modules register on the active
server instance. Keep that compatibility behavior when editing startup code.
