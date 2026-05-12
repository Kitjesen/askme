from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.demo.field_operations_demo import build_demo_package


@pytest.mark.asyncio
async def test_field_operations_demo_package_writes_customer_artifacts(tmp_path: Path) -> None:
    package = await build_demo_package(
        tmp_path,
        site_profile_path=Path("deploy/site-profiles/park-demo.yaml"),
    )

    package_report = tmp_path / "field-demo-package.json"
    scenario_report = tmp_path / "scenario-evaluation.json"
    guide = tmp_path / "customer-demo-guide.md"
    site_profile_report = tmp_path / "site-profile-readiness.json"

    assert package["status"] == "passed"
    assert package["product_demo"]["demo_ready"] is True
    assert package["product_demo"]["real_integration_ready"] is False
    assert package["product_demo"]["customer_scenario_count"] == 10
    assert package["dashboard_visual"]["status"] == "skipped"
    assert package["site_profile"]["status"] == "passed"
    assert package_report.exists()
    assert scenario_report.exists()
    assert guide.exists()
    assert site_profile_report.exists()

    saved_package = json.loads(package_report.read_text(encoding="utf-8"))
    saved_scenarios = json.loads(scenario_report.read_text(encoding="utf-8"))
    guide_text = guide.read_text(encoding="utf-8")

    assert saved_package["package_name"] == "askme-field-operations-customer-demo"
    assert saved_scenarios["product_demo"]["demo_ready"] is True
    assert "AskMe 园区机器狗场景演示包" in guide_text
    assert "车辆违停检测" in guide_text
    assert "真实摄像头/VMS 事件流" in guide_text
    assert "站点配置" in guide_text
    assert "Inovx Demo Park" in guide_text
    assert "它不证明真实摄像头" in guide_text
