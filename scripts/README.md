# Askme Scripts

`scripts/` is an operator and maintenance surface, not a product package. Keep
business logic in `askme/`; scripts should launch, diagnose, benchmark, or
evaluate existing package behavior.

Keep the root quiet. New executable files should normally live in one of the
documented buckets below instead of beside this README.

## Directory Map

| Bucket | Owns | Does not own | Typical verification |
| --- | --- | --- | --- |
| `scripts/runtime/` | Runtime-side bridges, service launchers, health probes, and signed field runtime callbacks. | Product behavior, API route logic, model policy. | `pytest tests/test_deploy_paths.py tests/test_field_runtime_callbacks.py tests/test_field_runtime_callback_security.py -q` |
| `scripts/eval/` | Deterministic product acceptance scenarios and release gates. | Manual hardware probes, one-off demos. | Run the changed script with `--help`, plus the related scenario tests. |
| `scripts/dev/` | Local development helpers, model download, encoding checks, sync/deploy convenience commands. | Runtime service ownership, customer demos. | `python scripts/dev/<script>.py --help` where supported; use static tests for shell/batch files. |
| `scripts/demo/` | Manual customer/demo utilities and experimental operator flows. | CI gates, production launchers. | Manual only unless the script is explicitly designed for `--help` or dry-run. |
| `scripts/bench/` | Benchmarks, latency probes, audio readiness checks, and benchmark helper scripts. | Pytest unit tests. | `pytest tests/test_performance_benchmarks.py tests/test_generated_voice_capability_check.py -q` for supported benchmark code. |
| `scripts/e2e/` | Provider-backed or hardware-backed end-to-end probes. | Automatic pytest collection. | Manual execution only; document required keys, models, devices, and network. |
| `scripts/audit/` | Secret scanning, key rotation guides, and security audit helpers. | Product APIs and runtime services. | Script-specific `--help` when present. |
| `scripts/artifacts/` | Media, fixtures, and sample files used by scripts. | Executable scripts. | Static structure checks only. |
| `scripts/tools/` | Low-level maintenance helpers used by scripts or manual flows. | Product APIs and runtime services. | Static tests, plus script-specific `--help` when present. |

## Root Files

Root scripts are reserved for cross-cutting CI or performance entrypoints:

- `__init__.py`: package marker so `scripts.eval` and sibling packages are importable.
- `zeroclaw_bridge.py`: ZeroClaw ↔ Askme MCP process orchestration and health monitoring.
- `benchmark_audit_query.py`: audit query benchmark and optimization evidence.
- `benchmark_core_paths.py`: core API/tool/memory path benchmark.
- `check_perf_thresholds.py`: benchmark report threshold gate.
- `check_text_encoding.py`: legacy-compatible text encoding checker entrypoint.

Do not add new root scripts without updating `tests/test_scripts_structure.py`.

## Multi-Agent Lanes

Assign script work by runtime risk, not by filename alone:

| Lane | Write scope | Notes |
| --- | --- | --- |
| Runtime Ops | `scripts/runtime/services`, `scripts/runtime/health` | Owns systemd and operator health probes. Must not change product code. |
| Runtime Bridges | `scripts/runtime/bridges` | Owns long-running ROS2, frame, rerun, embedding, and ingest bridge launchers. Static-check by default. |
| Field Runtime | `scripts/runtime/post_field_runtime_callback.py`, field-related `scripts/eval` | Owns signed callback and field runtime smoke flows. |
| Product Eval | `scripts/eval` | Owns deterministic scenario gates for robot, RAG, voice, dashboard, and field operations. |
| Dev Tools | `scripts/dev`, root `check_text_encoding.py` | Owns local development commands and repository hygiene checks. |
| Performance | root `benchmark_*.py`, `check_perf_thresholds.py`, benchmark-only files in `scripts/bench` | Owns quick/full performance runs and thresholds. |
| Voice/Hardware Manual | hardware and voice probes in `scripts/bench`, `scripts/e2e`, `scripts/demo` | Manual only unless a script has a mock/dry-run path. |
| Demo/Enablement | `scripts/demo`, `scripts/artifacts` | Owns customer demos and presentation/artifact utilities. |

Shared rule: if a script starts a service, touches hardware, uses cloud keys,
or waits for live audio/video, do not add it to automatic CI execution.

## Manual-Only Scripts

Most files named `test_*.py` under `scripts/bench/` and `scripts/e2e/` are
manual smoke probes, not pytest tests. They may require microphones, speakers,
cameras, model files, API keys, network access, or a robot runtime.
`pyproject.toml` keeps `testpaths = ["tests"]`; do not remove that guard.

## Placement Rules

1. Runtime code that a service starts belongs under `scripts/runtime/`.
2. A deterministic scenario that proves product behavior belongs under
   `scripts/eval/`.
3. A developer-only convenience command belongs under `scripts/dev/`.
4. A customer/demo experiment belongs under `scripts/demo/` until it graduates
   into a deterministic product scenario.
5. Benchmarks belong under root performance entrypoints or `scripts/bench/`.
6. Hardware, audio, camera, cloud, and real provider probes must document their
   external requirements and remain manual by default.
7. Do not add binaries, captured audio, generated reports, images, or videos to
   executable script directories; use `scripts/artifacts/` or `artifacts/`.

## Verification

For script structure or documentation changes:

```powershell
pytest tests/test_scripts_structure.py tests/test_scripts_static.py -q
```

For runtime service path changes:

```powershell
pytest tests/test_deploy_paths.py -q
```

For benchmark code changes:

```powershell
pytest tests/test_performance_benchmarks.py tests/test_generated_voice_capability_check.py -q
```

For encoding checker changes:

```powershell
pytest tests/test_text_encoding_check.py -q
```

Do not run manual hardware or cloud scripts in CI unless they have an explicit
mock, dry-run, or fixture-backed mode.
