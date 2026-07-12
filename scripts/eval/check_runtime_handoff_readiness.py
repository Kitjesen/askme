# ruff: noqa: I001
"""Readiness gate for askme robot task handoff demos and lab promotion."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from askme.cognition import WorldStateService  # noqa: E402
from askme.runtime.task.handoff import RuntimeHandoffService  # noqa: E402
from scripts.eval.evaluate_robot_scenarios import (  # noqa: E402
    DEFAULT_REPORT_PATH as DEFAULT_SCENARIO_REPORT_PATH,
    evaluate_scenarios,
    write_report as write_scenario_report,
)


DEFAULT_READINESS_PATH = Path("artifacts/runtime_handoff/readiness.json")
DEFAULT_AUDIT_PATH = Path("artifacts/runtime_handoff/simulation-audit.jsonl")
UNSAFE_PROMOTION_PROFILES = {"lab", "prod", "production"}


def check_readiness(
    *,
    runtime_profile: str = "sim",
    scenario_report_path: Path = DEFAULT_SCENARIO_REPORT_PATH,
    readiness_path: Path = DEFAULT_READINESS_PATH,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    require_audit: bool = False,
    run_scenarios: bool = True,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    normalized_profile = str(runtime_profile or "sim").strip().lower()

    if normalized_profile in UNSAFE_PROMOTION_PROFILES:
        checks.append(
            _check(
                "profile_refuses_unsafe_promotion",
                False,
                {
                    "profile": normalized_profile,
                    "reason": "askme handoff only supports fake/shadow/sim without hardware dispatch",
                },
            )
        )
    else:
        checks.append(_check("profile_refuses_unsafe_promotion", True, {"profile": normalized_profile}))

    world = WorldStateService()
    runtime = RuntimeHandoffService(world_state=world, profile=normalized_profile)
    capabilities = runtime.capabilities()
    profiles = runtime.profiles_payload()
    http_paths = set(capabilities.get("http_paths", []))

    checks.append(_check("hardware_dispatch_disabled", capabilities.get("hardware_dispatch") is False, capabilities))
    checks.append(
        _check(
            "runtime_voice_turn_endpoint_present",
            "POST /api/runtime/voice-turn" in http_paths,
            {"http_paths": sorted(http_paths)},
        )
    )
    checks.append(
        _check(
            "runtime_safety_contracts_present",
            all(
                capabilities.get(key) is True
                for key in (
                    "safety_preflight",
                    "active_perception_requests",
                    "operator_policy",
                    "task_reports",
                )
            ),
            capabilities,
        )
    )
    checks.append(
        _check(
            "runtime_profiles_are_non_hardware",
            all(item.get("hardware_dispatch") is False for item in profiles.get("profiles", [])),
            profiles,
        )
    )

    scenario_payload = _load_or_run_scenarios(
        scenario_report_path,
        run_scenarios=run_scenarios,
    )
    checks.append(
        _check(
            "scenario_evaluation_passed",
            scenario_payload.get("status") == "passed" and scenario_payload.get("failed") == 0,
            {
                "status": scenario_payload.get("status"),
                "passed": scenario_payload.get("passed"),
                "failed": scenario_payload.get("failed"),
                "report_path": str(scenario_report_path),
            },
        )
    )

    audit_ok = audit_path.exists() and audit_path.stat().st_size > 0
    checks.append(
        _check(
            "audit_output_present",
            audit_ok or not require_audit,
            {
                "required": require_audit,
                "path": str(audit_path),
                "exists": audit_path.exists(),
                "size": audit_path.stat().st_size if audit_path.exists() else 0,
            },
        )
    )

    failed = [item for item in checks if not item["passed"]]
    payload = {
        "target": "askme-runtime-handoff-readiness",
        "status": "ok" if not failed else "degraded",
        "runtime_profile": normalized_profile,
        "hardware_dispatch": False,
        "checks": checks,
        "failed_checks": [item["name"] for item in failed],
        "generated_at": time.time(),
    }
    readiness_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["readiness_path"] = str(readiness_path)
    return payload


def _load_or_run_scenarios(path: Path, *, run_scenarios: bool) -> dict[str, Any]:
    if run_scenarios:
        payload = evaluate_scenarios()
        write_scenario_report(payload, path)
        return payload
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
    return {"status": "missing", "failed": 1, "passed": 0}


def _check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "details": details,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-profile", default="sim")
    parser.add_argument("--scenario-report-path", default=str(DEFAULT_SCENARIO_REPORT_PATH))
    parser.add_argument("--readiness-path", default=str(DEFAULT_READINESS_PATH))
    parser.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--require-audit", action="store_true")
    parser.add_argument("--no-run-scenarios", action="store_true")
    args = parser.parse_args(argv)

    payload = check_readiness(
        runtime_profile=args.runtime_profile,
        scenario_report_path=Path(args.scenario_report_path),
        readiness_path=Path(args.readiness_path),
        audit_path=Path(args.audit_path),
        require_audit=args.require_audit,
        run_scenarios=not args.no_run_scenarios,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))  # noqa: T201
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
