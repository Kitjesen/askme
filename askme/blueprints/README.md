# Runtime Blueprints

`askme.blueprints` defines product runtime packages. A blueprint is not just a
Python module list; it is the contract that product, QA, delivery, and a
customer pilot can inspect before the system is started.

## Package Layout

- `catalog/`: customer-visible metadata, readiness gates, delivery packages,
  and API/CLI payload builders.
- `presets/`: concrete runtime presets such as `text`, `voice`,
  `voice_perception`, `edge_robot`, `mcp`, and `lingtu_voice`.
- `runner/`: shared startup helper used by each preset module.

Historical imports such as `askme.blueprints.voice` are still supported by the
package alias loader. New code should import from `askme.blueprints.presets.*`
or `askme.blueprints.catalog`.

## Responsibilities

Blueprints own product assembly and delivery-facing metadata:

- Choose which runtime modules form a product runtime.
- Expose a named `Runtime` object from each preset.
- Keep customer-visible catalog metadata in sync with the preset.
- Provide readiness gates and validation commands for delivery.
- Start presets through the shared runner when invoked as modules.

Blueprints do not implement module behavior. Module behavior belongs in
`askme.runtime.modules`, and low-level wiring belongs in `askme.runtime.core`.

## Public Entrypoints

Use these imports for new blueprint and delivery code:

```python
from askme.blueprints.catalog import (
    blueprint_delivery_package,
    blueprint_readiness,
    catalog_payload,
    get_blueprint_spec,
    list_blueprints,
    load_blueprint_runtime,
)
from askme.blueprints.presets.text import text
from askme.blueprints.runner.runner import run_blueprint
```

Preset package convenience exports use `*_runtime` names:

```python
from askme.blueprints.presets import text_runtime, voice_runtime
```

Historical imports such as `askme.blueprints.text` remain compatibility
surfaces only.

## Coupling Rules

- A preset must not import another preset. Each file under `presets/` declares
  its own complete Runtime module list.
- Shared behavior belongs in `askme.runtime.modules` or lower-level services,
  not in another blueprint preset.
- Product relationships such as `edge_robot` containing voice and perception
  capabilities are documented in `catalog/data.py`, not encoded through Python
  imports between presets.

## Customer Surfaces

```powershell
python -m askme.cli runtime blueprints --customer-visible
python -m askme.cli runtime blueprints --name park --delivery-package --json
```

```http
GET /api/blueprints
GET /api/blueprints/park
GET /api/blueprints/park/delivery-package
```

`park` is the public alias for `edge_robot`.

## How To Assemble A Runtime

A preset assembles a runtime by adding `Runtime.use(...)` fragments. Keep the
file declarative: imports at the top, one runtime object, `__all__`, and an
optional `__main__` runner.

```python
from askme.runtime.core import Runtime
from askme.runtime.modules import HealthModule, LLMModule, TextModule

text_like = (
    Runtime.use(LLMModule)
    + Runtime.use(TextModule)
    + Runtime.use(HealthModule)
)

__all__ = ["text_like"]

if __name__ == "__main__":
    from askme.blueprints.runner.runner import run_blueprint

    run_blueprint(text_like, "Text-like runtime")
```

Assembly checklist for a new preset:

- Select concrete modules from `askme.runtime.modules`.
- Order modules in the same broad shape as nearby presets: core services,
  mission/cognition/handoff, pipeline/skills/execution, IO, health, then
  product-specific modules.
- Let module `depends_on` and typed ports handle build wiring.
- Add or update catalog metadata for customer-visible blueprints.
- Add validation commands that delivery can run repeatedly.

## Readiness Rules

Blueprint readiness is deliberately stricter than YAML existence:

- Empty dictionaries do not satisfy required configuration.
- Unresolved placeholders such as `${DOG_CONTROL_SERVICE_URL}` do not satisfy
  required configuration.
- Service maps with only empty values do not satisfy required configuration.
- A service with `enabled: false` does not satisfy required configuration.
- External services still require manual smoke evidence before production
  claims, even when static readiness passes.

This prevents a local demo config from being shown as a customer-ready robot
runtime. For example, `edge_robot` requires real or explicitly enabled project
bindings for:

- `voice`
- `perception`
- `field_operations`
- `runtime_handoff`
- `dingding`
- `robot_control`

## Adding A Blueprint

Every customer-visible blueprint must define:

- `deployment_targets`: where it can run.
- `capabilities`: customer-readable business capabilities.
- `scenarios`: acceptance scenarios that can be tested.
- `required_config`: static config gates.
- `external_services`: credentials or systems that need live smoke tests.
- `safety_boundaries`: what the blueprint must never bypass.
- `validation_commands`: repeatable engineering or delivery checks.

The contract is guarded by `tests/test_blueprints_catalog.py`.

## Verification

For documentation-only changes, no code test is required beyond reviewing the
affected Markdown.

For preset or example changes, use the smallest relevant checks:

```powershell
python -m compileall askme/runtime/examples.py
python -m askme.runtime.examples
python -m askme.blueprints.presets.text --preflight --json
python -m askme.cli runtime blueprints --customer-visible
```

Use preflight before full startup when the runtime may open microphone, camera,
robot, or external service connections.
