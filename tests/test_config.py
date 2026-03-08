"""Tests for askme.config module."""

from askme.config import get_config, get_section, project_root


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
