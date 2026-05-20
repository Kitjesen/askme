"""Tests for askme.config module."""

from pathlib import Path

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

    def test_customer_knowledge_uses_mempalace_with_vector_fallback(self):
        """Product default should use MemPalace for customer RAG, not robot behavior memory."""
        cfg = get_config(reload=True)
        memory = cfg.get("memory", {})

        assert memory["backend"] == "mempalace"
        assert memory["customer_knowledge_backend"] == "mempalace"
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
        assert cfg["field_operations"]["site_map"]["zones"]["guide-01"]["name"] == "游客中心路引点"
        assert cfg["field_operations"]["site_map"]["zones"]["north-window-01"]["name"] == "北侧一层窗户"
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
