# askme API Surfaces

The HTTP layer is split by audience. Keep legacy URLs stable while moving new
work into the right surface.

## Product

`askme.api.product` is the customer-facing surface used by Dashboard and
operator workflows. It should use customer language: customer projects, field
events, knowledge, wayfinding, capabilities, voice profiles, missions, and
conversation.

Do not expose internal terms such as arbiter, handoff, runtime advance, or raw
cognition planning in customer copy.

Product routes are customer visible, but they are not a hardware authority and
they are not a production-launch proof. A product route may ask for a task,
show evidence, or explain status; physical execution still has to pass runtime
handoff, safety preflight, and onsite acceptance evidence.

`/api/chat` is the product chat surface. By default it is local-first: it calls
the configured chat handler, which commonly delegates to `TextLoop.process_turn`
for intent routing, cognition planning, skills, LLM, memory evidence, and space
preview. It does not silently mirror the interactive terminal loop's runtime
bridge behavior.

Runtime bridge use from `/api/chat` is explicit. Requests may provide
`runtime_policy=runtime_first` only when the configured handler accepts that
policy; otherwise chat remains `runtime_policy=disabled`. `control_only` is a
reserved policy for runtime-control requests and does not make ordinary chat
runtime-first. `conversation_session_id`, `conversation_id`, and
`chat_session_id` are aliases for the outer dialog session; `planning_session_id`
is only for the inner cognition planning loop.

Knowledge and memory routes must keep customer answers auditable:

- `/api/memory/search` searches approved evidence and returns traceable results.
- `/api/memory/health` exposes the active backend and readiness, not hidden
  implementation assumptions. Its response is locked by
  `MemoryHealthResponse`: customer status, next step, answer evidence rules,
  and robot-behavior-memory isolation must stay visible.
- `/api/knowledge/*` owns preview, import, list, and update flows for the
  Knowledge Console. Routes should stay thin; schema validation, permission
  mapping, and stable error payloads belong in `askme.api.services`.

Runtime blueprint routes are product contracts, not internal implementation
details:

- `/api/blueprints` is the runtime blueprint catalog used by Dashboard and
  delivery tools.
- `/api/blueprints/{name}` resolves canonical names and public aliases such as
  `park`.
- `/api/blueprints/{name}/delivery-package` is the narrow customer handoff
  export. Its response is locked by `BlueprintDeliveryPackageResponse`; every
  successful package must include customer status, next step, acceptance
  boundary, and delivery actions.

Blueprint endpoints can prove that a runtime package is ready for site
validation. They must not be used as production-launch evidence by themselves.

## Admin

`askme.api.admin` is for operators, supervisors, delivery engineers, approvals,
audits, identity readiness, agent profiles, and skill governance.

## Internal

`askme.api.internal` is for runtime, cognition, vision, robot callbacks, device
ingest, and low-level integrations. These routes may keep current `/api/*`
paths during migration, but product UI should not depend on internal concepts.
Internal routes may touch robot/runtime integration boundaries, but they must
not become customer-visible sales or Dashboard copy.

## Platform

`askme.api.platform` owns health, metrics, and system monitoring routes.

## App Factory

New code should import the FastAPI factory from:

```python
from askme.api import create_api_app
```

`askme.health_server` still owns legacy dependency wiring. `askme.api.app`
provides the canonical API boundary while the old implementation is retired in
small, testable steps.

`askme.health_server` must not grow route decorators again. It may construct
dependencies, compatibility helpers, and the FastAPI app, but route ownership
belongs under `askme.api.platform`, `askme.api.product`, `askme.api.admin`, and
`askme.api.internal`. The guard test is
`tests/test_package_migration_compat.py::test_health_server_does_not_declare_inline_route_decorators`.

## External Entry Point

`askme.api` is the HTTP boundary for external clients: Dashboard, operator
tools, delivery scripts, and health checks. Its public entry points are:

- `askme.api.create_api_app`
- `askme.api.create_product_app`
- `askme.api.server.create_api_app`
- the registered FastAPI routes under `/api/*`

API modules own request/response schemas, route audience classification,
permission mapping, and stable customer-readable errors. They should stay thin:
validate input, call services/providers/ports, then return a typed response.

API routes must not directly implement hardware, voice, vision, robot, or
runtime behavior. Physical execution, speech I/O, perception, and device access
belong behind `askme.providers`, `askme.ports`, runtime services, or the
service layer composed by `askme.api.composition`. If an API endpoint needs a
robot, voice, or vision result, it should depend on one of those boundaries
instead of importing `voice_gateway`, `robot_interaction`, camera/vision
implementations, or device-specific modules.

API is distinct from MCP and tools:

- API is the HTTP product/admin/internal/platform surface for humans,
  dashboards, and delivery automation.
- MCP is the agent-facing protocol adapter that exposes selected capabilities
  over Model Context Protocol.
- `askme.tools` is the in-process tool implementation and registry layer used by
  runtime, skills, and MCP.

## Boundary Manifest

The source of truth for API audience boundaries is
`askme.api.composition.API_SURFACES`. Use
`askme.api.composition.api_surface_manifest()` when a page, CLI, test, or
document needs to explain which surface owns customer flows, admin workflows,
internal robot callbacks, or platform health.

The manifest is intentionally code-owned instead of discovered from imports.
Each `route_modules` entry must point at a route module owned by exactly one
surface; `api_surface_module_map()` is the duplicate-assignment guard used by
tests and readiness checks.

`GET /api/surfaces` also returns a route inventory. The inventory classifies
every product-relevant FastAPI route by endpoint module and reports
`unclassified_count`. New route modules must be added to the right
`API_SURFACES.route_modules` entry before the interface is considered
product-ready.

The same endpoint returns `readiness`, a customer-readable gate derived from
the live route inventory. It must stay deterministic and auditable:

- `overall_status=ready` means no route is unclassified and every declared
  surface has registered routes.
- `policy` explains which product rules are currently satisfied, including
  "product UI uses customer-visible routes", "product routes do not allow
  hardware authority", "API surfaces are not production claim sources", and
  "internal robot routes do not drive customer UI".
- `blockers` lists concrete routing gaps that must be fixed before claiming
  the HTTP boundary is ready.
- `release_claim` is the only sentence the Dashboard or delivery report should
  reuse when explaining this boundary to a customer.

When adding a new route module, update `API_SURFACES` with the correct
`customer_visible`, `hardware_authority_allowed`, `production_claim_allowed`,
and `customer_boundary` values. These fields are part of the product contract,
not documentation-only comments.

Any file under `askme/api/routes` that declares FastAPI route decorators must
be listed in exactly one `API_SURFACES.route_modules` tuple. The static guard
is
`tests/test_package_migration_compat.py::test_every_route_module_with_fastapi_decorators_is_surface_classified`.

New route modules should prefer an `APIRouter` factory plus a small compatibility
registrar. For example, `askme.api.routes.monitor` exposes
`create_monitor_router(...)`, and `askme.api.routes.voice` exposes
`create_voice_router(...)`. `askme.api.routes.memory` exposes
`create_memory_router(...)` for customer knowledge and RAG routes, and
`askme.api.routes.governance` exposes `create_governance_router(...)` for
operator/IAM readiness routes. `askme.api.routes.cognition` exposes
`create_cognition_router(...)` for planning/context routes, and
`askme.api.routes.conversation` exposes `create_conversation_router(...)` for
chat and voice-turn routes. `askme.api.routes.runtime` exposes
`create_runtime_router(...)` for TaskRun lifecycle routes, and
`askme.api.routes.space` exposes `create_space_router(...)` for park wayfinding,
guide, and service-point routes. `askme.api.routes.field_admin`,
`askme.api.routes.field_events`, `askme.api.routes.field_product_catalog`,
and `askme.api.routes.field_internal` expose router factories for field
readiness, customer-facing incidents, customer project catalogs, managed
object directories, evidence, notification governance, device ingest, and
runtime callback routes. These modules keep their `register_*_routes(...)`
functions as surface composition hooks. This follows the same route aggregation
pattern used by larger FastAPI applications while keeping the current
dependency injection boundary explicit.

## Response Schema Contract

Every JSON endpoint must publish a concrete response model. Avoid untyped
`dict` or generic `object` responses on product, admin, internal, and platform
JSON APIs because Dashboard, delivery scripts, and customer acceptance reports
need stable machine-readable payloads.

The only accepted OpenAPI no-schema exceptions are non-JSON surfaces:

- `GET /dashboard`: HTML shell.
- `GET /dashboard/{asset_path}`: static asset or HTML fallback.
- `GET /api/field/evidence`: evidence file download.
- `GET /api/runtime/events`: server-sent events stream.

The global guard is
`tests/test_health.py::TestHealthServer::test_openapi_json_response_schema_coverage_allows_only_streaming_and_assets`.
When adding a JSON route, define a response schema under
`askme.api.schemas`, attach it with `response_model=...`, and validate the
actual returned payload in the route or service boundary before responding.

## Request Body Contract

Write routes must reject malformed or non-object JSON before permission checks
and before business dispatch. Customer-facing tools, delivery scripts, and robot
bridges should get a stable `400` response instead of an accidental `500`.

Use one of these patterns:

- `require_json_object(await request.json())` for routes that read JSON directly.
- the injected `optional_json_body(request)` helper for split route modules.

Every such read must convert `ValueError` into a `400` response with the stable
message `JSON object body required`. The guard test is
`tests/test_package_migration_compat.py::test_api_route_json_body_reads_are_object_validated`.
