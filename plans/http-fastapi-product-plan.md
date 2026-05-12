# HTTP / FastAPI Product Refactor Plan

Updated: 2026-05-10

## Product Direction

The HTTP service is already FastAPI. The product problem is not the framework;
the problem is that `askme/health_server.py` has become a monolithic API,
dashboard, telemetry, runtime-control, memory, cognition, mission, and vision
surface.

The target shape is:

```text
askme/
  api/
    server.py
    routes/
      system.py
      monitor.py
      memory.py
      cognition.py
      runtime.py
      mission.py
      vision.py
  services/
  domain/
  adapters/
```

## Migration Rules

1. Keep `askme.health_server.create_health_app` compatible until existing
   runtime modules and tests are migrated.
2. New API code enters through `askme.api`.
3. Split one route family at a time and keep tests green after every family.
4. Do not change endpoint URLs during the refactor.
5. Do not change auth behavior during the refactor.

## Phase 1: System Routes

Move these routes first because they have minimal product state coupling:

- `GET /health`
- `GET /healthz`
- `GET /metrics`
- `GET /metrics/prometheus`
- `GET /trace`

Status: started.

Implemented:

- Added `askme/api/routes/system.py`.
- `create_health_app()` now registers health, metrics, and trace through
  `register_system_routes(...)`.
- Added `askme/api/server.py` as the product-facing app factory alias.

## Phase 2: Monitor Routes

Move:

- `POST /api/chat`
- `GET /api/status`
- `GET /dashboard`
- `GET /api/live`
- `GET /api/conversations`

This phase should introduce a `MonitorRouteDeps` object instead of passing many
loose closures.

## Phase 3: Memory Routes

Move:

- `POST /api/memory/search`
- `POST /api/knowledge/preview`
- `POST /api/knowledge/import`
- `POST /api/knowledge/list`
- `POST /api/knowledge/update`

This phase should make `memory_handler` a typed protocol.

## Phase 4: Runtime / Mission / Cognition

Move the task runtime control plane after memory and monitor are stable:

- runtime context, events, profiles, runs, reports, operator actions
- mission draft/create/list/get/report
- cognition context and plan

These are product-critical and must keep audit, safety, and operator action
behavior unchanged.

## Phase 5: Vision

Move:

- snapshot
- analyze
- captures list/get/delete

Vision can become its own adapter-backed route family once perception contracts
stabilize.

## Acceptance Checks

For every phase:

```powershell
pytest tests/test_health.py tests/test_health_server_metrics.py tests/test_http_probe_server.py -q
ruff check askme/api askme/health_server.py tests/test_health.py tests/test_health_server_metrics.py tests/test_http_probe_server.py
```

For major route families:

```powershell
pytest tests/scenario_tests -q
```
