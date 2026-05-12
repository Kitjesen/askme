"""Build a customer-facing field-operations demo package.

The package is intentionally offline by default: it proves product behavior,
scenario evidence, archive/notification decisions, and current integration
gaps without claiming that physical robot hardware or production services ran.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.evaluate_field_operations_scenarios import evaluate_scenarios, write_report

DEFAULT_OUTPUT_DIR = Path("artifacts/field_operations/demo")
DEFAULT_SITE_PROFILE = Path("deploy/site-profiles/park-demo.yaml")


async def build_demo_package(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    site_profile_path: Path | None = None,
    with_dashboard_visual: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_report_path = output_dir / "scenario-evaluation.json"
    payload = await evaluate_scenarios()
    write_report(payload, scenario_report_path)

    product_demo = payload.get("product_demo") if isinstance(payload.get("product_demo"), dict) else {}
    dashboard_visual = (
        await asyncio.to_thread(_run_dashboard_visual, output_dir)
        if with_dashboard_visual
        else _skipped_visual()
    )
    site_profile = _site_profile_report(site_profile_path, output_dir)
    guide_path = output_dir / "customer-demo-guide.md"
    guide_path.write_text(
        _render_customer_demo_guide(product_demo, scenario_report_path, dashboard_visual, site_profile),
        encoding="utf-8",
    )
    package = {
        "status": _package_status(payload, dashboard_visual, site_profile),
        "package_name": "askme-field-operations-customer-demo",
        "generated_at": time.time(),
        "output_dir": str(output_dir),
        "scenario_report": str(scenario_report_path),
        "customer_guide": str(guide_path),
        "scenario_status": payload.get("status"),
        "product_demo": {
            "suite_name": product_demo.get("suite_name") or "",
            "demo_ready": bool(product_demo.get("demo_ready", False)),
            "real_integration_ready": bool(product_demo.get("real_integration_ready", False)),
            "customer_scenario_count": int(product_demo.get("customer_scenario_count") or 0),
            "passed": int(product_demo.get("passed") or 0),
            "failed": int(product_demo.get("failed") or 0),
            "blocked_on_real_integrations": list(
                product_demo.get("blocked_on_real_integrations") or []
            ),
        },
        "site_profile": site_profile,
        "dashboard_visual": dashboard_visual,
        "operator_next_steps": [
            "Open the generated customer demo guide.",
            "Run askme runtime field-eval when you need a fresh console checklist.",
            "Run with --with-dashboard-visual before a customer UI review.",
            "Replace fake dispatcher settings with site camera, sensor, DingTalk, MiniMax, and robot runtime credentials before claiming production readiness.",
        ],
    }
    package_path = output_dir / "field-demo-package.json"
    package_path.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    package["package_report"] = str(package_path)
    return package


def _run_dashboard_visual(output_dir: Path) -> dict[str, Any]:
    from scripts.eval.check_dashboard_visual import run

    visual_dir = output_dir / "dashboard-visual"
    result = run(visual_dir)
    summary = {
        "status": result.get("status"),
        "output_dir": str(visual_dir),
        "checks": result.get("checks", []),
        "failures": result.get("failures", []),
    }
    for key in ("reason", "detail", "console_errors", "page_errors", "response_errors"):
        if key in result:
            summary[key] = result.get(key)
    return summary


def _skipped_visual() -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": "run with --with-dashboard-visual to capture dashboard screenshots",
    }


def _site_profile_report(site_profile_path: Path | None, output_dir: Path) -> dict[str, Any]:
    if site_profile_path is None:
        return {"status": "skipped", "reason": "no site profile was provided"}
    from askme.pipeline.field_site_profile import build_site_profile_report

    report = build_site_profile_report(site_profile_path)
    report_path = output_dir / "site-profile-readiness.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": report.get("status"),
        "profile_path": str(site_profile_path),
        "report_path": str(report_path),
        "summary": report.get("summary", {}),
        "readiness": report.get("readiness", {}),
        "errors": report.get("errors", []),
        "warnings": report.get("warnings", []),
    }


def _package_status(
    payload: dict[str, Any],
    dashboard_visual: dict[str, Any],
    site_profile: dict[str, Any],
) -> str:
    if payload.get("status") != "passed":
        return "failed"
    visual_status = dashboard_visual.get("status")
    if visual_status not in {"passed", "skipped"}:
        return "failed"
    site_profile_status = site_profile.get("status")
    if site_profile_status not in {"passed", "skipped"}:
        return "failed"
    return "passed"


def _render_customer_demo_guide(
    product_demo: dict[str, Any],
    scenario_report_path: Path,
    dashboard_visual: dict[str, Any],
    site_profile: dict[str, Any],
) -> str:
    scenarios = product_demo.get("customer_scenarios")
    if not isinstance(scenarios, list):
        scenarios = []
    lines = [
        "# AskMe 园区机器狗场景演示包",
        "",
        "## 演示结论",
        "",
        f"- 演示状态: {'通过' if product_demo.get('demo_ready') else '未通过'}",
        f"- 真实现场接入: {'已接入' if product_demo.get('real_integration_ready') else '未接入'}",
        f"- 场景覆盖: {product_demo.get('passed', 0)}/{product_demo.get('customer_scenario_count', 0)}",
        f"- 场景报告: `{scenario_report_path}`",
        "",
        "## 可演示场景",
        "",
    ]
    for item in scenarios:
        if not isinstance(item, dict):
            continue
        actual = item.get("actual") if isinstance(item.get("actual"), dict) else {}
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        lines.extend([
            f"### {item.get('customer_name') or item.get('name')}",
            "",
            f"- 结果: {'通过' if item.get('passed') else '未通过'}",
            f"- 触发来源: {item.get('trigger_source') or '-'}",
            f"- 机器人动作: {item.get('expected_robot_action') or '-'}",
            f"- 通知对象: {actual.get('notification_group') or item.get('expected_notification') or '-'}",
            f"- 通知投递: {actual.get('delivery_status') or '-'}",
            f"- 事件编号: {evidence.get('event_id') or '-'}",
            f"- 语音播报: {actual.get('voice') or '-'}",
            "",
        ])
    lines.extend([
        "## 仍需真实接入",
        "",
    ])
    gaps = product_demo.get("blocked_on_real_integrations")
    if isinstance(gaps, list):
        for gap in gaps:
            lines.append(f"- {gap}")
    lines.extend([
        "",
        "## UI 证据",
        "",
        f"- Dashboard 截图检查: {dashboard_visual.get('status')}",
    ])
    for check in dashboard_visual.get("checks", []) if isinstance(dashboard_visual.get("checks"), list) else []:
        if isinstance(check, dict):
            lines.append(f"- {check.get('name')}: `{check.get('screenshot')}`")
    lines.extend([
        "",
        "## 站点配置",
        "",
        f"- 配置状态: {site_profile.get('status')}",
        f"- 配置文件: `{site_profile.get('profile_path') or '-'}`",
        f"- 配置报告: `{site_profile.get('report_path') or '-'}`",
    ])
    summary = site_profile.get("summary") if isinstance(site_profile.get("summary"), dict) else {}
    if summary:
        lines.extend([
            f"- 站点: {summary.get('site_name') or '-'}",
            f"- 地图版本: {summary.get('map_version') or '-'}",
            f"- 区域数量: {summary.get('zone_count') or 0}",
            f"- 路引点数量: {summary.get('help_point_count') or 0}",
            f"- 设备数量: {summary.get('device_count') or 0}",
        ])
    errors = site_profile.get("errors") if isinstance(site_profile.get("errors"), list) else []
    if errors:
        lines.append("- 配置错误:")
        lines.extend(f"  - {error}" for error in errors)
    lines.extend([
        "",
        "## 演示口径",
        "",
        "这份演示包证明软件链路、场景决策、通知分组、语音文案、归档和审计字段可以工作。",
        "它不证明真实摄像头、传感器、生产钉钉群、MiniMax 现场音频或机器狗硬件已经完成现场验收。",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for demo package artifacts.",
    )
    parser.add_argument(
        "--with-dashboard-visual",
        action="store_true",
        help="Also run the browser dashboard visual smoke check and capture screenshots.",
    )
    parser.add_argument(
        "--site-profile",
        type=Path,
        default=DEFAULT_SITE_PROFILE,
        help="Field site profile to validate and include in the demo package.",
    )
    args = parser.parse_args(argv)
    package = asyncio.run(
        build_demo_package(
            args.output_dir,
            site_profile_path=args.site_profile,
            with_dashboard_visual=args.with_dashboard_visual,
        )
    )
    print(json.dumps(package, ensure_ascii=False, indent=2))
    return 0 if package["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
