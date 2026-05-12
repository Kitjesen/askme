# Askme Scripts Directory

`scripts/` is a curated operator surface. Keep the root quiet: new executable
files should live in one of the buckets below, not beside this README.

## Directory Map

- `scripts/runtime/`
  Runtime-facing helpers used by robot or service processes.
  - `bridges/`: ROS2, frame, rerun, and local embedding bridge processes.
  - `services/`: systemd units and long-running service launchers.
  - `health/`: service and voice health probes.
  - `post_field_runtime_callback.py`: signed runtime-delivery callback helper
    for shadow/lab/robot runtime processes.
- `scripts/eval/`
  Product acceptance and regression scenarios for robot handoff, RAG trust, and
  voice end-to-end behavior.
  - `smoke_field_runtime_roundtrip.py`: FieldIncident -> runtime handoff ->
    signed callback -> archived workflow smoke gate.
- `scripts/dev/`
  Local development, model download, encoding checks, deployment, sync, and
  convenience launchers.
- `scripts/demo/`
  Manual demos and experimental operator utilities that are not product
  acceptance gates.
- `scripts/bench/`
  Benchmarks, latency probes, hardware/audio readiness checks, and generated
  benchmark helper scripts.
- `scripts/e2e/`
  Provider-backed and local end-to-end probes.
- `scripts/artifacts/`
  Non-source media or fixture leftovers that should not become stable operator
  entrypoints.
- `scripts/tools/`
  Low-level helper tools used by scripts or manual maintenance flows.

## Placement Rules

1. Runtime code that a service starts belongs under `scripts/runtime/`.
2. A scenario that proves product behavior belongs under `scripts/eval/`.
3. A developer-only convenience command belongs under `scripts/dev/`.
4. A customer/demo experiment belongs under `scripts/demo/` until it graduates
   into a product scenario.
5. Do not add binaries, captured audio, images, or generated result files to
   script code directories.

## Verification

After moving or adding scripts, update references in `README.md`, `docs/`,
`tests/`, service unit files, and config comments, then run:

```powershell
pytest tests/test_deploy_paths.py tests/test_generated_voice_capability_check.py -q
ruff check scripts tests/test_deploy_paths.py tests/test_generated_voice_capability_check.py
```
