"""HTTP tests for vision routes."""

from fastapi.testclient import TestClient

from askme.api.schemas.vision import (
    VisionAnalyzeResponse,
    VisionCaptureDeleteResponse,
    VisionCaptureDetailResponse,
    VisionCaptureListResponse,
    VisionSnapshotResponse,
)
from askme.health_server import create_health_app
from tests.support.health_snapshots import minimal_runtime_snapshot as _runtime_snapshot


def test_vision_routes_return_unconfigured_status():
    client = TestClient(create_health_app(lambda: _runtime_snapshot()))

    snapshot = client.get("/api/vision/snapshot")
    analyze = client.post("/api/vision/analyze", json={"image_base64": "aGVsbG8="})
    captures = client.get("/api/vision/captures")
    capture = client.get("/api/vision/captures/cap-1")
    deleted = client.delete("/api/vision/captures/cap-1")

    assert snapshot.status_code == 503
    VisionSnapshotResponse.model_validate(snapshot.json())
    assert snapshot.json()["error"] == "vision not configured"
    assert analyze.status_code == 503
    VisionAnalyzeResponse.model_validate(analyze.json())
    assert analyze.json()["error"] == "vision not configured"
    assert captures.status_code == 503
    VisionCaptureListResponse.model_validate(captures.json())
    assert captures.json()["error"] == "image archive not configured"
    assert capture.status_code == 503
    VisionCaptureDetailResponse.model_validate(capture.json())
    assert capture.json()["error"] == "image archive not configured"
    assert deleted.status_code == 503
    VisionCaptureDeleteResponse.model_validate(deleted.json())
    assert deleted.json()["error"] == "image archive not configured"


def test_vision_snapshot_auto_archives_when_archive_handler_is_configured():
    archive_calls: list[tuple[bytes, str, str, int, int]] = []

    async def snapshot_handler():
        return {
            "image_base64": "aGVsbG8=",
            "width": 2,
            "height": 3,
            "timestamp": "2026-05-16T00:00:00Z",
        }

    async def archive_handler(
        image_bytes: bytes,
        label: str,
        description: str,
        width: int,
        height: int,
    ):
        archive_calls.append((image_bytes, label, description, width, height))
        return {"id": "cap-1"}

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            vision_snapshot_handler=snapshot_handler,
            archive_snapshot_handler=archive_handler,
        )
    )

    response = client.get("/api/vision/snapshot")

    assert response.status_code == 200
    payload = response.json()
    VisionSnapshotResponse.model_validate(payload)
    assert payload["capture_id"] == "cap-1"
    assert archive_calls == [(b"hello", "manual", "", 2, 3)]


def test_vision_analyze_requires_image_and_returns_description():
    seen_images: list[str] = []

    async def analyze_handler(image_base64: str) -> str:
        seen_images.append(image_base64)
        return "Image contains one patrol robot."

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            vision_analyze_handler=analyze_handler,
        )
    )

    missing = client.post("/api/vision/analyze", json={})
    ok = client.post("/api/vision/analyze", json={"image_base64": "aGVsbG8="})

    assert missing.status_code == 400
    VisionAnalyzeResponse.model_validate(missing.json())
    assert missing.json()["error"] == "image_base64 required"
    assert ok.status_code == 200
    VisionAnalyzeResponse.model_validate(ok.json())
    assert ok.json()["description"] == "Image contains one patrol robot."
    assert seen_images == ["aGVsbG8="]


def test_vision_analyze_rejects_non_object_json_body_before_dispatch():
    async def analyze_handler(image_base64: str) -> str:
        raise AssertionError(
            f"vision analyze handler should not be called: {image_base64}"
        )

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            vision_analyze_handler=analyze_handler,
        )
    )

    response = client.post("/api/vision/analyze", json=["aGVsbG8="])

    assert response.status_code == 400
    assert response.json()["error"] == "JSON object body required"


def test_vision_archive_routes_filter_limit_get_delete_and_cors():
    deleted_ids: list[str] = []

    async def list_handler():
        return [
            {"id": "cap-1", "label": "manual"},
            {"id": "cap-2", "label": "alarm"},
            {"id": "cap-3", "label": "manual"},
        ]

    async def get_handler(capture_id: str):
        if capture_id == "cap-1":
            return {"id": "cap-1", "image_base64": "aGVsbG8="}
        return None

    async def delete_handler(capture_id: str):
        if capture_id == "cap-1":
            deleted_ids.append(capture_id)
            return True
        return False

    client = TestClient(
        create_health_app(
            lambda: _runtime_snapshot(),
            archive_list_handler=list_handler,
            archive_get_handler=get_handler,
            archive_delete_handler=delete_handler,
        )
    )

    listed = client.get("/api/vision/captures", params={"label": "manual", "limit": 1})
    found = client.get("/api/vision/captures/cap-1")
    missing = client.get("/api/vision/captures/missing")
    deleted = client.delete("/api/vision/captures/cap-1")
    delete_missing = client.delete("/api/vision/captures/missing")
    snapshot_options = client.options("/api/vision/snapshot")
    analyze_options = client.options("/api/vision/analyze")
    captures_options = client.options("/api/vision/captures")
    capture_options = client.options("/api/vision/captures/cap-1")

    assert listed.status_code == 200
    listed_payload = listed.json()
    VisionCaptureListResponse.model_validate(listed_payload)
    assert listed_payload["count"] == 1
    assert listed_payload["captures"][0]["id"] == "cap-1"
    assert found.status_code == 200
    found_payload = found.json()
    VisionCaptureDetailResponse.model_validate(found_payload)
    assert found_payload["image_base64"] == "aGVsbG8="
    assert found.headers["cache-control"] == "no-store"
    assert missing.status_code == 404
    VisionCaptureDetailResponse.model_validate(missing.json())
    assert missing.json()["error"] == "capture not found"
    assert deleted.status_code == 200
    deleted_payload = deleted.json()
    VisionCaptureDeleteResponse.model_validate(deleted_payload)
    assert deleted_payload == {"deleted": True, "capture_id": "cap-1"}
    assert delete_missing.status_code == 404
    VisionCaptureDeleteResponse.model_validate(delete_missing.json())
    assert deleted_ids == ["cap-1"]
    assert snapshot_options.status_code == 204
    assert snapshot_options.headers["access-control-allow-methods"] == "GET, OPTIONS"
    assert analyze_options.status_code == 204
    assert analyze_options.headers["access-control-allow-methods"] == "POST, OPTIONS"
    assert captures_options.status_code == 204
    assert captures_options.headers["access-control-allow-methods"] == "GET, OPTIONS"
    assert capture_options.status_code == 204
    assert capture_options.headers["access-control-allow-methods"] == "GET, DELETE, OPTIONS"
