"""Tests for the standardised error response format."""

import json

from askme.errors import (
    ROBOT_NOT_CONNECTED,
    SKILL_NOT_FOUND,
    error_response,
)


class TestErrorResponse:
    def test_basic_format(self):
        result = error_response(ROBOT_NOT_CONNECTED, "Robot arm not connected")
        data = json.loads(result)
        assert "error" in data
        assert data["error"]["code"] == "robot_not_connected"
        assert data["error"]["message"] == "Robot arm not connected"
        assert "details" not in data["error"]

    def test_with_details(self):
        result = error_response(
            SKILL_NOT_FOUND,
            "Skill 'foo' not found",
            {"available": ["bar", "baz"]},
        )
        data = json.loads(result)
        assert data["error"]["code"] == "skill_not_found"
        assert data["error"]["details"]["available"] == ["bar", "baz"]

    def test_unicode_preserved(self):
        result = error_response("test", "机器人未连接")
        data = json.loads(result)
        assert "机器人" in data["error"]["message"]
