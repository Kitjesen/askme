from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.demo.live_field_operations_demo import run_live_demo


def test_live_field_operations_demo_exercises_http_paths(tmp_path: Path) -> None:
    payload = run_live_demo(
        output_dir=tmp_path,
        site_profile=Path("deploy/site-profiles/park-demo.yaml"),
    )

    assert payload["status"] == "passed"
    assert payload["mode"] == "inprocess_http"
    assert payload["accepted"] == payload["scenario_count"] == 5
    assert {item["path"] for item in payload["scenarios"]} == {
        "/api/field/ingest",
        "/api/field/events",
    }
    assert any(item["scenario_id"] == "fire_or_smoke" for item in payload["scenarios"])
    assert any(item["scenario_id"] == "wayfinding_help_point" for item in payload["scenarios"])
    assert payload["events_status"] == 200
    assert payload["events"]["total"] >= 5
    assert payload["devices_status"] == 200
    assert payload["devices"]["summary"]["observed"] >= 3
    assert len(payload["reports"]) >= 5
    voice_texts = [str(item["voice_text"]) for item in payload["scenarios"]]
    assert any("烟雾" in text for text in voice_texts)
    assert any("车辆" in text and "通知安保" in text for text in voice_texts)
    assert any("指路" in text for text in voice_texts)

    report = tmp_path / "live-field-demo.json"
    guide = tmp_path / "live-field-demo.md"
    html = tmp_path / "live-field-demo.html"
    assert report.exists()
    assert guide.exists()
    assert html.exists()
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["status"] == "passed"
    assert "Askme 现场场景验收报告" in guide.read_text(encoding="utf-8")
    html_text = html.read_text(encoding="utf-8")
    assert "Askme 现场场景验收报告" in html_text
    assert "火灾/烟雾异常" in html_text
    assert "车辆违停" in html_text
    assert "本地软件闭环" in html_text
    assert "不是硬件验收" in html_text


def test_live_field_operations_demo_replays_customer_scenario_file(tmp_path: Path) -> None:
    scenario_file = tmp_path / "customer-scenarios.json"
    scenario_file.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "scenario_id": "customer_smoke_sensor",
                        "customer_scene": "客户烟感真实样本",
                        "path": "/api/field/ingest",
                        "device_secret": "smoke-secret",
                        "payload": {
                            "source": "sensor",
                            "device_id": "smoke-warehouse-a",
                            "observed_at": 1777777777.0,
                            "sensor": {"temperature_c": 75, "smoke_level": 0.91},
                            "zone_id": "warehouse-a",
                            "location": "Warehouse A",
                            "image_path": "artifacts/evidence/customer-smoke.jpg",
                        },
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = run_live_demo(
        output_dir=tmp_path / "out",
        site_profile=Path("deploy/site-profiles/park-demo.yaml"),
        scenario_file=scenario_file,
        refresh_scenario_timestamps=True,
    )

    assert payload["status"] == "passed"
    assert payload["scenario_source"] == str(scenario_file)
    assert payload["accepted"] == payload["scenario_count"] == 1
    assert payload["scenarios"][0]["scenario_id"] == "customer_smoke_sensor"
    assert payload["scenarios"][0]["incident_topic"] == "safety.fire_or_smoke"
    html_text = (tmp_path / "out" / "live-field-demo.html").read_text(encoding="utf-8")
    assert "客户烟感真实样本" in html_text
    assert "检测到烟雾或高温异常" in html_text


def test_live_field_operations_demo_rejects_invalid_scenario_file(tmp_path: Path) -> None:
    scenario_file = tmp_path / "bad-scenarios.json"
    scenario_file.write_text(json.dumps({"scenarios": [{"payload": []}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="payload object"):
        run_live_demo(
            output_dir=tmp_path / "out",
            site_profile=Path("deploy/site-profiles/park-demo.yaml"),
            scenario_file=scenario_file,
        )


def test_live_field_operations_demo_accepts_bom_scenario_file(tmp_path: Path) -> None:
    scenario_file = tmp_path / "bom-scenarios.json"
    scenario_file.write_text(
        "\ufeff"
        + json.dumps(
            [
                {
                    "scenario_id": "customer_smoke_sensor",
                    "customer_scene": "BOM 客户样本",
                    "path": "/api/field/ingest",
                    "device_secret": "smoke-secret",
                    "payload": {
                        "source": "sensor",
                        "device_id": "smoke-warehouse-a",
                        "observed_at": 1777777777.0,
                        "sensor": {"temperature_c": 75, "smoke_level": 0.91},
                        "zone_id": "warehouse-a",
                        "location": "Warehouse A",
                        "image_path": "artifacts/evidence/customer-smoke.jpg",
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    payload = run_live_demo(
        output_dir=tmp_path / "out",
        site_profile=Path("deploy/site-profiles/park-demo.yaml"),
        scenario_file=scenario_file,
        refresh_scenario_timestamps=True,
    )

    assert payload["status"] == "passed"
    assert payload["accepted"] == 1
