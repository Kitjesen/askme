"""Tests for NavStatusTool — navigation status query tool."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from askme.skills.skill_manager import SkillManager
from askme.tools.builtin_tools import NavStatusTool


class TestNavStatusTool:
    def test_no_odometry_returns_clear_final_voice_reply(self):
        navigation = MagicMock()
        navigation.status.return_value = {
            "state": "IDLE",
            "has_odometry": False,
            "reason_codes": ["odometry_missing"],
            "readiness": {
                "localization_ready": False,
                "blockers": ["odometry_missing"],
            },
        }

        result = NavStatusTool(navigation_client=navigation).execute()

        assert result == "定位未就绪，暂时无法获取当前位置。"
        assert "0.0" not in result

    def test_ready_odometry_returns_concise_coordinates(self):
        navigation = MagicMock()
        navigation.status.return_value = {
            "state": "IDLE",
            "has_odometry": True,
            "odometry": {"x": 1.234, "y": -2.345, "frame_id": "map"},
        }

        result = NavStatusTool(navigation_client=navigation).execute()

        assert result == "当前位置：map 坐标系，横坐标 1.23 米，纵坐标 -2.35 米。"
        assert not result.startswith("{")

    def test_not_configured(self, monkeypatch):
        """NAV_GATEWAY_URL が未設定のとき '未配置' を含む文字列を返す。"""
        monkeypatch.delenv("NAV_GATEWAY_URL", raising=False)
        tool = NavStatusTool()
        result = tool.execute()
        assert "未配置" in result

    def test_configured_success(self, monkeypatch):
        """A successful status response is converted to a concise voice reply."""
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:9000")

        nav_data = {"status": "idle", "position": {"x": 1.0, "y": 2.0}}
        encoded = json.dumps(nav_data).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = encoded
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            tool = NavStatusTool()
            result = tool.execute()

        assert result == "当前位置：地图 坐标系，横坐标 1.00 米，纵坐标 2.00 米。"

    def test_configured_failure(self, monkeypatch):
        """urlopen が例外を投げたとき '查询失败' を含む文字列を返す。"""
        monkeypatch.setenv("NAV_GATEWAY_URL", "http://localhost:9000")

        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            tool = NavStatusTool()
            result = tool.execute()

        assert "查询失败" in result


def test_nav_query_is_enabled_read_only_status_skill(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import askme.skills.skill_manager as skill_manager_module

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(skill_manager_module, "_DATA_DIR", data_dir)
    monkeypatch.setattr(
        skill_manager_module,
        "_SETTINGS_FILE",
        data_dir / "skills_settings.json",
    )
    manager = SkillManager(project_dir=tmp_path)
    manager.load()

    skill = manager.get("nav_query")

    assert skill is not None
    assert skill.enabled is True
    assert skill.execution == "read_only_tool"
    assert skill.safety_level == "normal"
    assert skill.tools_section.strip() == "nav_status"
