"""Tests for dashboard monitor service payload assembly."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from askme.api.services.monitor_service import MonitorService


def _service(
    tmp_path: Path,
    *,
    config: dict | None = None,
    command_runner=None,
    conversation_provider=None,
) -> MonitorService:
    return MonitorService(
        config_provider=lambda: config or {},
        project_root=tmp_path,
        conversation_provider=conversation_provider,
        tmp_dir=tmp_path,
        command_runner=command_runner or (lambda _cmd, _timeout: SimpleNamespace(stdout=b"inactive\n")),
    )


def test_system_status_payload_reads_perception_orbbec_and_memory(tmp_path: Path) -> None:
    (tmp_path / "askme_frame_daemon.heartbeat").write_text("98.75", encoding="utf-8")
    (tmp_path / "askme_frame_detections.json").write_text(
        json.dumps(
            {
                "infer_ms": 17,
                "detections": [{"class_id": "person"}, {"class_id": "cart"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "askme_events.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "older"}),
            json.dumps({"event": "latest"}),
        ]),
        encoding="utf-8",
    )
    knowledge_dir = tmp_path / "data" / "qp_memory" / "knowledge"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "site.md").write_text("site", encoding="utf-8")
    (knowledge_dir / "ignore.txt").write_text("ignore", encoding="utf-8")

    service = _service(
        tmp_path,
        command_runner=lambda _cmd, _timeout: SimpleNamespace(stdout=b"active\n"),
    )

    payload = service.system_status_payload(now=100.0)

    assert payload["timestamp"] == 100.0
    assert payload["perception"]["frame_daemon"] == {"alive": True, "age_s": 1.2}
    assert payload["perception"]["detections"] == {
        "count": 2,
        "infer_ms": 17,
        "objects": ["person", "cart"],
    }
    assert payload["perception"]["change_events"] == {
        "total": 2,
        "last": {"event": "latest"},
    }
    assert payload["orbbec_camera"] is True
    assert payload["memory"] == {"knowledge_files": 1}


def test_system_status_payload_degrades_on_missing_or_invalid_sources(tmp_path: Path) -> None:
    (tmp_path / "askme_frame_daemon.heartbeat").write_text("not-a-float", encoding="utf-8")
    (tmp_path / "askme_frame_detections.json").write_text("{invalid", encoding="utf-8")
    (tmp_path / "askme_events.jsonl").write_text("{invalid", encoding="utf-8")

    def failing_runner(_cmd, _timeout):
        raise RuntimeError("systemctl unavailable")

    payload = _service(tmp_path, command_runner=failing_runner).system_status_payload(now=100.0)

    assert payload["perception"]["frame_daemon"] == {"alive": False}
    assert payload["perception"]["detections"] == {"count": 0}
    assert payload["perception"]["change_events"] == {"total": 0}
    assert payload["orbbec_camera"] is False
    assert payload["memory"] == {"knowledge_files": 0}


def test_live_payload_uses_provider_or_empty_default(tmp_path: Path) -> None:
    empty = _service(tmp_path).live_payload()
    live = _service(
        tmp_path,
        conversation_provider=lambda: [{"role": "user", "content": "hello"}],
    ).live_payload()

    assert empty == {"messages": [], "count": 0}
    assert live == {"messages": [{"role": "user", "content": "hello"}], "count": 1}


def test_conversation_history_payload_reads_configured_relative_path(tmp_path: Path) -> None:
    history = [{"role": "assistant", "content": "ready"}]
    (tmp_path / "history.json").write_text(json.dumps(history), encoding="utf-8")

    payload = _service(
        tmp_path,
        config={"conversation": {"history_file": "history.json"}},
    ).conversation_history_payload()

    assert payload == {"messages": history, "count": 1}


def test_conversation_history_payload_returns_empty_for_missing_file(tmp_path: Path) -> None:
    payload = _service(
        tmp_path,
        config={"conversation": {"history_file": "missing.json"}},
    ).conversation_history_payload()

    assert payload == {"messages": [], "count": 0}
