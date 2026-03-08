"""Tests for config validation and feature flags."""

import pytest
from askme.config import validate_config, _apply_feature_flags


class TestValidateConfig:
    def test_missing_api_key(self):
        """Empty api_key should produce a validation error."""
        config = {"brain": {"api_key": "", "base_url": "https://api.deepseek.com"}}
        errors = validate_config(config)
        assert any("api_key" in e for e in errors)

    def test_valid_minimal_config(self):
        """Config with required fields should pass."""
        config = {
            "brain": {"api_key": "sk-test", "base_url": "https://api.deepseek.com"},
        }
        errors = validate_config(config)
        assert len(errors) == 0

    def test_voice_section_no_key_needed(self):
        """Edge TTS requires no API key -- voice section should pass validation."""
        config = {
            "brain": {"api_key": "sk-test", "base_url": "https://api.deepseek.com"},
            "voice": {"tts": {"voice": "zh-CN-YunxiNeural"}},
        }
        errors = validate_config(config)
        assert len(errors) == 0


class TestFeatureFlags:
    def test_robot_off(self, monkeypatch):
        """ASKME_FEATURE_ROBOT=0 should disable robot."""
        monkeypatch.setenv("ASKME_FEATURE_ROBOT", "0")
        config = {"robot": {"enabled": True}}
        _apply_feature_flags(config)
        assert config["robot"]["enabled"] is False

    def test_robot_on(self, monkeypatch):
        """ASKME_FEATURE_ROBOT=1 should enable robot."""
        monkeypatch.setenv("ASKME_FEATURE_ROBOT", "1")
        config = {"robot": {"enabled": False}}
        _apply_feature_flags(config)
        assert config["robot"]["enabled"] is True

    def test_voice_off_removes_section(self, monkeypatch):
        """ASKME_FEATURE_VOICE=0 should remove voice section."""
        monkeypatch.setenv("ASKME_FEATURE_VOICE", "0")
        config = {"voice": {"tts": {}}}
        _apply_feature_flags(config)
        assert "voice" not in config

    def test_voice_on_creates_section(self, monkeypatch):
        """ASKME_FEATURE_VOICE=1 should ensure voice section exists."""
        monkeypatch.setenv("ASKME_FEATURE_VOICE", "1")
        config = {}
        _apply_feature_flags(config)
        assert "voice" in config

    def test_no_flag_no_change(self, monkeypatch):
        """No env var set should leave config unchanged."""
        monkeypatch.delenv("ASKME_FEATURE_ROBOT", raising=False)
        monkeypatch.delenv("ASKME_FEATURE_VOICE", raising=False)
        config = {"robot": {"enabled": True}}
        _apply_feature_flags(config)
        assert config["robot"]["enabled"] is True

    def test_memory_flag(self, monkeypatch):
        """ASKME_FEATURE_MEMORY=0 should disable memory."""
        monkeypatch.setenv("ASKME_FEATURE_MEMORY", "0")
        config = {"memory": {"enabled": True}}
        _apply_feature_flags(config)
        assert config["memory"]["enabled"] is False
