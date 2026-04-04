"""Tests for robot MCP resources (robot_status, robot_joint_info, robot_safety_config)."""

from __future__ import annotations

import json
import math
from unittest.mock import patch, MagicMock

import pytest


def _get_resources():
    from askme.mcp.resources.robot_resources import (
        robot_status,
        robot_joint_info,
        robot_safety_config,
    )
    return robot_status, robot_joint_info, robot_safety_config


# ── robot_status ──────────────────────────────────────────────────────────────

class TestRobotStatus:
    def test_returns_json(self):
        robot_status, *_ = _get_resources()
        with patch("askme.config.get_section",
                   return_value={"enabled": True, "simulate": False}):
            result = robot_status()
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_contains_enabled_field(self):
        robot_status, *_ = _get_resources()
        with patch("askme.config.get_section",
                   return_value={"enabled": True, "simulate": False}):
            result = robot_status()
        data = json.loads(result)
        assert "enabled" in data

    def test_contains_simulate_field(self):
        robot_status, *_ = _get_resources()
        with patch("askme.config.get_section",
                   return_value={"simulate": True}):
            result = robot_status()
        data = json.loads(result)
        assert "simulate" in data

    def test_default_when_empty_config(self):
        robot_status, *_ = _get_resources()
        with patch("askme.config.get_section",
                   return_value={}):
            result = robot_status()
        data = json.loads(result)
        assert data["enabled"] is False  # default
        assert data["simulate"] is True   # default


# ── robot_joint_info ──────────────────────────────────────────────────────────

class TestRobotJointInfo:
    def test_valid_arm_joint(self):
        _, robot_joint_info, _ = _get_resources()
        result = robot_joint_info("0")
        data = json.loads(result)
        assert data["joint_id"] == 0
        assert data["type"] == "arm"

    def test_valid_finger_joint(self):
        _, robot_joint_info, _ = _get_resources()
        result = robot_joint_info("6")
        data = json.loads(result)
        assert data["joint_id"] == 6
        assert data["type"] == "finger"

    def test_padding_joint(self):
        _, robot_joint_info, _ = _get_resources()
        result = robot_joint_info("10")
        data = json.loads(result)
        assert data["type"] == "padding"

    def test_invalid_joint_id_string(self):
        _, robot_joint_info, _ = _get_resources()
        result = robot_joint_info("abc")
        data = json.loads(result)
        assert "error" in data

    def test_negative_joint_id(self):
        _, robot_joint_info, _ = _get_resources()
        result = robot_joint_info("-1")
        data = json.loads(result)
        assert "error" in data

    def test_out_of_range_joint_id(self):
        _, robot_joint_info, _ = _get_resources()
        result = robot_joint_info("16")
        data = json.loads(result)
        assert "error" in data

    def test_arm_limits_pi(self):
        _, robot_joint_info, _ = _get_resources()
        result = robot_joint_info("0")
        data = json.loads(result)
        assert data["limit_max_rad"] == math.pi
        assert data["limit_min_rad"] == -math.pi

    def test_joint_name_in_response(self):
        _, robot_joint_info, _ = _get_resources()
        result = robot_joint_info("0")
        data = json.loads(result)
        assert data["name"] == "shoulder_pan"

    def test_last_valid_joint(self):
        _, robot_joint_info, _ = _get_resources()
        result = robot_joint_info("15")
        data = json.loads(result)
        assert data["joint_id"] == 15
        assert "error" not in data


# ── robot_safety_config ───────────────────────────────────────────────────────

class TestRobotSafetyConfig:
    def test_returns_json(self):
        *_, robot_safety_config = _get_resources()
        mock_defaults = {"estop_words": ["停", "急停"]}
        with patch("askme.mcp.resources.robot_resources._DEFAULT_CONFIG",
                   mock_defaults, create=True), \
             patch("askme.robot.safety._DEFAULT_CONFIG", mock_defaults, create=True):
            result = robot_safety_config()
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_contains_estop_keywords(self):
        *_, robot_safety_config = _get_resources()
        mock_defaults = {"estop_words": ["停", "急停", "estop"]}
        with patch("askme.robot.safety._DEFAULT_CONFIG", mock_defaults):
            result = robot_safety_config()
        data = json.loads(result)
        assert "estop_keywords" in data

    def test_contains_joint_limits(self):
        *_, robot_safety_config = _get_resources()
        mock_defaults = {"estop_words": []}
        with patch("askme.robot.safety._DEFAULT_CONFIG", mock_defaults):
            result = robot_safety_config()
        data = json.loads(result)
        assert "arm_joint_limits_rad" in data
