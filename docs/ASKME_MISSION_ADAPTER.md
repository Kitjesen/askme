# Askme Mission Adapter

Askme now provides a safe mission adapter for industrial dog workflows. It turns
operator text into auditable mission drafts and dry-run submissions, while the
runtime arbiter remains the owner of robot movement, safety checks, and hardware
execution.

## Boundary

Askme may:

- draft high-level inspection, navigation, evidence, status, and emergency intent
  plans;
- dry-run plans locally for operator review;
- submit a reviewed plan only to the configured runtime arbiter endpoint;
- expose mission status and report shells over the local health HTTP server.

Askme must not:

- send direct motor, gait, `cmd_vel`, serial, SDK, or arm commands;
- bypass `dog-safety-service` or `dog-control-service`;
- submit critical safety override requests from the adapter path.

## CLI

```powershell
python -m askme mission draft "inspect area-a" --json
python -m askme mission run "inspect area-a" --json
python -m askme mission run .\mission.json --submit --confirm --json
python -m askme mission report mission-abc123 --server http://127.0.0.1:8765 --json
```

`mission run` is dry-run by default. Live submission requires both
`--submit --confirm` and configuration:

```yaml
runtime:
  mission:
    submit_enabled: true
    base_url: http://127.0.0.1:8088
```

## HTTP

When the text, voice, MCP, or edge runtime includes `HealthModule`, the local
server exposes:

- `POST /api/missions/draft`
- `POST /api/missions`
- `GET /api/missions`
- `GET /api/missions/{mission_id}`
- `GET /api/missions/{mission_id}/report`

Runtime capabilities include `mission_adapter` and component health under
`GET /api/capabilities`.

## Runtime Payload

Live submission is shaped for the runtime `CreateMissionRequest` contract:

- `mission_type`
- `requested_capability`
- `requested_by`
- `channel`
- `robot_id`
- `site_id`
- `priority`
- `approval_required`
- `parameters`

Private adapter metadata keys beginning with `_` are removed before submission.

## Verification

Use the software-only loop before hardware is connected:

```powershell
python -m pytest -q tests/test_mission_service.py tests/test_health.py tests/test_cli.py
python -m askme mission draft "inspect area-a" --json
python -m askme mission run "inspect area-a" --json
```
