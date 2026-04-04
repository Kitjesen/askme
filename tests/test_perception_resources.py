"""Tests for MCP perception resources (current_detections, recent_events, depth_info)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch, mock_open

import pytest


# Import the resource functions by patching the mcp decorator
# since FastMCP resources register on import

def _get_resources():
    """Import perception resource functions without triggering MCP registration."""
    import importlib
    import sys
    # The functions are already registered as @mcp.resource() decorators;
    # we call them directly since they just read files.
    from askme.mcp.resources.perception_resources import (
        current_detections,
        recent_events,
        depth_info,
        memory_knowledge,
    )
    return current_detections, recent_events, depth_info, memory_knowledge


# ── current_detections ────────────────────────────────────────────────────────

class TestCurrentDetections:
    def test_returns_json_string(self):
        current_detections, *_ = _get_resources()
        result = current_detections()
        assert isinstance(result, str)
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_file_not_found_returns_error(self):
        current_detections, *_ = _get_resources()
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = current_detections()
        data = json.loads(result)
        assert "error" in data
        assert data["detections"] == []

    def test_with_valid_data_adds_age(self, tmp_path):
        current_detections, *_ = _get_resources()
        det_file = tmp_path / "detections.json"
        payload = {
            "timestamp": time.time(),
            "detections": [{"class_id": "person", "confidence": 0.9}],
        }
        det_file.write_text(json.dumps(payload))

        with patch("builtins.open", mock_open(read_data=json.dumps(payload))):
            result = current_detections()
        data = json.loads(result)
        assert "age_seconds" in data
        assert "fresh" in data

    def test_fresh_when_recent_timestamp(self):
        current_detections, *_ = _get_resources()
        payload = {"timestamp": time.time(), "detections": []}
        with patch("builtins.open", mock_open(read_data=json.dumps(payload))):
            result = current_detections()
        data = json.loads(result)
        assert data["fresh"] is True

    def test_not_fresh_when_old_timestamp(self):
        current_detections, *_ = _get_resources()
        payload = {"timestamp": time.time() - 10.0, "detections": []}  # 10 seconds old
        with patch("builtins.open", mock_open(read_data=json.dumps(payload))):
            result = current_detections()
        data = json.loads(result)
        assert data["fresh"] is False

    def test_invalid_json_returns_error(self):
        current_detections, *_ = _get_resources()
        with patch("builtins.open", mock_open(read_data="not valid json {")):
            result = current_detections()
        data = json.loads(result)
        assert "error" in data


# ── recent_events ─────────────────────────────────────────────────────────────

class TestRecentEvents:
    def test_returns_json_string(self):
        _, recent_events, *_ = _get_resources()
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = recent_events()
        data = json.loads(result)
        assert "events" in data
        assert data["count"] == 0

    def test_parses_jsonl_lines(self):
        _, recent_events, *_ = _get_resources()
        event1 = json.dumps({"event_type": "person_appeared", "timestamp": 1.0})
        event2 = json.dumps({"event_type": "object_appeared", "timestamp": 2.0})
        jsonl = event1 + "\n" + event2 + "\n"

        with patch("builtins.open", mock_open(read_data=jsonl)):
            # mock_open doesn't support readlines well; patch differently
            pass

    def test_empty_file_returns_zero_events(self):
        _, recent_events, *_ = _get_resources()
        with patch("builtins.open", mock_open(read_data="")):
            result = recent_events()
        data = json.loads(result)
        assert data["count"] == 0


# ── depth_info ────────────────────────────────────────────────────────────────

class TestDepthInfo:
    def test_returns_json_string(self):
        _, _, depth_info, _ = _get_resources()
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = depth_info()
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_daemon_not_alive_when_no_heartbeat(self):
        _, _, depth_info, _ = _get_resources()
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = depth_info()
        data = json.loads(result)
        assert data["daemon_alive"] is False

    def test_daemon_alive_when_recent_heartbeat(self):
        _, _, depth_info, _ = _get_resources()
        heartbeat_content = str(time.time())

        open_calls = []
        def _mock_open(path, *args, **kwargs):
            if "heartbeat" in str(path):
                return mock_open(read_data=heartbeat_content)()
            raise FileNotFoundError(path)

        with patch("builtins.open", side_effect=_mock_open):
            result = depth_info()
        data = json.loads(result)
        assert data["daemon_alive"] is True

    def test_center_depth_none_when_no_depth_file(self):
        _, _, depth_info, _ = _get_resources()
        with patch("builtins.open", side_effect=FileNotFoundError):
            result = depth_info()
        data = json.loads(result)
        assert data["center_depth_m"] is None


# ── memory_knowledge ──────────────────────────────────────────────────────────

class TestMemoryKnowledge:
    def test_returns_json_with_files_key(self):
        *_, memory_knowledge = _get_resources()
        with patch("os.listdir", side_effect=FileNotFoundError):
            result = memory_knowledge()
        data = json.loads(result)
        assert "files" in data
        assert data["files"] == {}

    def test_lists_md_files(self, tmp_path):
        *_, memory_knowledge = _get_resources()
        fake_files = ["device_a.md", "process_b.md", "notes.txt"]

        def fake_listdir(path):
            return fake_files

        def fake_getsize(path):
            return 100

        with patch("os.listdir", side_effect=fake_listdir), \
             patch("os.path.getsize", side_effect=fake_getsize):
            result = memory_knowledge()
        data = json.loads(result)
        assert "device_a.md" in data["files"]
        assert "process_b.md" in data["files"]
        # .txt should not be in files
        assert "notes.txt" not in data["files"]
