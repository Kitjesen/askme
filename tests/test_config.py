"""Tests for askme.config module."""

import os
from pathlib import Path

import pytest

from askme.config import get_config, get_section, project_root

CONFIG_PATH = Path("config.yaml")


class TestGetConfig:
    def test_returns_dict(self):
        cfg = get_config(reload=True)
        assert isinstance(cfg, dict)

    def test_app_section_exists(self):
        cfg = get_config(reload=True)
        assert "app" in cfg
        assert cfg["app"]["name"] == "askme"

    def test_env_vars_resolved(self):
        """Env vars set in conftest should be substituted into config."""
        cfg = get_config(reload=True)
        brain = cfg.get("brain", {})
        assert brain.get("api_key") == "sk-test-key"

    def test_tts_sample_rate(self):
        """TTS sample_rate should be an integer."""
        cfg = get_config(reload=True)
        sr = cfg.get("voice", {}).get("tts", {}).get("sample_rate")
        assert isinstance(sr, (int, float))

    def test_phrase_cache_dir_is_expanded_to_an_absolute_path(self):
        cfg = get_config(reload=True)
        cache_dir = Path(cfg["voice"]["tts"]["phrase_cache_dir"])

        assert cache_dir.is_absolute()
        assert cache_dir == Path.home() / ".cache" / "askme" / "voice_phrases"

    @pytest.mark.skipif(os.name != "nt", reason="Windows audio override")
    def test_windows_audio_profile_uses_verified_realtek_route(self):
        cfg = get_config(reload=True)
        voice = cfg["voice"]

        assert voice["input_device"] == 1
        assert voice["input_transport"] == "sounddevice"
        assert voice["mic_native_rate"] == 44100
        assert voice["mic_channels"] == 1
        assert voice["tts"]["output_device"] == 3
        assert voice["tts"]["output_transport"] == "sounddevice"

    def test_julong_deployment_identity_and_scope(self):
        cfg = get_config(reload=True)

        assert cfg["brain"]["persona"]["robot_name"] == "小算"
        assert cfg["brain"]["persona"]["customer_name"] == "聚龙科创e谷"
        assert cfg["voice"]["address_detection"]["names"] == ["小算"]
        assert cfg["voice"]["interaction_gate"]["wake_terms"] == ["小算"]
        assert cfg["voice"]["kws"]["keywords"] == [
            "x iǎo s uàn :2.0 #0.20 @小算"
        ]
        assert Path(cfg["field_operations"]["site_profile_path"]) == (
            project_root()
            / "deploy"
            / "site-profiles"
            / "julong-tech-e-valley.yaml"
        )
        assert cfg["field_operations"]["robot_name"] == "小算"
        assert cfg["space_cognition"]["park_id"] == "julong-tech-e-valley"
        assert cfg["space_cognition"]["routes"] == []

    def test_customer_knowledge_uses_inspectable_vector_backend_by_default(self):
        """Product default stays local and inspectable; optional memory SDKs are not blockers."""
        cfg = get_config(reload=True)
        memory = cfg.get("memory", {})

        assert memory["backend"] == "vector"
        assert memory["customer_knowledge_backend"] == "vector"
        assert memory["mempalace_fallback_backend"] == "vector"
        assert memory["auto_backend_order"][:2] == ["mempalace", "vector"]
        assert memory["robot_behavior_memory_backend"] == "robotmem"
        assert memory["robot_behavior_memory_enabled"] is False

    def test_customer_editable_config_sections_are_readable(self):
        """High-touch deployment config must not contain mojibake text."""
        text = CONFIG_PATH.read_text(encoding="utf-8")
        high_touch_lines = []
        for line in text.splitlines():
            if any(
                marker in line
                for marker in (
                    "voice_profiles",
                    "label:",
                    "telegram",
                    "lingtu_url",
                    "site_map",
                    "main-road-1",
                    "guide-01",
                    "north-window-01",
                    "simulate:",
                    "MiniMax",
                    "Edge backend",
                    "local\"",
                )
            ):
                high_touch_lines.append(line)
        high_touch_text = "\n".join(high_touch_lines)

        mojibake_markers = [
            "鈥",
            "鈫",
            "鐜",
            "鍥",
            "浣",
            "璇",
            "涓",
            "绌",
            "褰",
            "瀹",
            "鍖",
            "娓",
            "璧",
            "楗",
        ]
        for marker in mojibake_markers:
            assert marker not in high_touch_text
        assert not any("\ue000" <= char <= "\uf8ff" for char in text)

        cfg = get_config(reload=True)
        profiles = cfg["voice"]["tts"]["voice_profiles"]
        assert profiles["patrol_default"]["label"] == "巡检播报"
        assert profiles["visitor_friendly"]["label"] == "游客服务"
        assert "待现场标定" in cfg["field_operations"]["site_map"]["zones"][
            "julong-guide-01"
        ]["name"]
        assert cfg["field_operations"]["site_map"]["zones"]["julong-patrol-01"][
            "type"
        ] == "patrol_checkpoint"
        assert "确认执行" in cfg["tools"]["confirmation_phrases"]
        assert "取消执行" in cfg["tools"]["rejection_phrases"]

    def test_project_root_is_valid(self):
        root = project_root()
        assert root.exists()
        assert (root / "config.yaml").exists()


class TestGetSection:
    def test_known_section(self):
        brain = get_section("brain")
        assert isinstance(brain, dict)
        assert "model" in brain

    def test_missing_section_returns_empty(self):
        result = get_section("nonexistent_section_xyz")
        assert result == {}
