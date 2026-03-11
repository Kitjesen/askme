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

    def test_ota_enabled_requires_server_url(self):
        """OTA bridge config must include server_url when enabled."""
        config = {
            "brain": {"api_key": "sk-test", "base_url": "https://api.deepseek.com"},
            "ota": {"enabled": True, "device": {}},
        }
        errors = validate_config(config)
        assert any("ota.server_url" in e for e in errors)

    def test_ota_disabled_allows_empty_server_url(self):
        """Disabled OTA bridge should not fail validation."""
        config = {
            "brain": {"api_key": "sk-test", "base_url": "https://api.deepseek.com"},
            "ota": {"enabled": False, "server_url": ""},
        }
        errors = validate_config(config)
        assert len(errors) == 0


_VALID_BASE = {"brain": {"api_key": "sk-test", "base_url": "https://api.example.com"}}


class TestValidateConfigNumericRules:
    """Coverage for the numeric/range validation rules added to validate_config."""

    def test_brain_timeout_negative_is_error(self):
        cfg = {**_VALID_BASE, "brain": {**_VALID_BASE["brain"], "timeout": -1}}
        errors = validate_config(cfg)
        assert any("brain.timeout" in e for e in errors)

    def test_brain_timeout_zero_is_error(self):
        cfg = {**_VALID_BASE, "brain": {**_VALID_BASE["brain"], "timeout": 0}}
        errors = validate_config(cfg)
        assert any("brain.timeout" in e for e in errors)

    def test_brain_timeout_positive_ok(self):
        cfg = {**_VALID_BASE, "brain": {**_VALID_BASE["brain"], "timeout": 30}}
        assert validate_config(cfg) == []

    def test_brain_timeout_non_numeric_is_error(self):
        cfg = {**_VALID_BASE, "brain": {**_VALID_BASE["brain"], "timeout": "fast"}}
        errors = validate_config(cfg)
        assert any("brain.timeout" in e for e in errors)

    def test_brain_max_retries_eleven_is_error(self):
        cfg = {**_VALID_BASE, "brain": {**_VALID_BASE["brain"], "max_retries": 11}}
        errors = validate_config(cfg)
        assert any("brain.max_retries" in e for e in errors)

    def test_brain_max_retries_zero_ok(self):
        cfg = {**_VALID_BASE, "brain": {**_VALID_BASE["brain"], "max_retries": 0}}
        assert validate_config(cfg) == []

    def test_brain_max_retries_ten_ok(self):
        cfg = {**_VALID_BASE, "brain": {**_VALID_BASE["brain"], "max_retries": 10}}
        assert validate_config(cfg) == []

    def test_brain_model_empty_string_is_error(self):
        cfg = {**_VALID_BASE, "brain": {**_VALID_BASE["brain"], "model": "  "}}
        errors = validate_config(cfg)
        assert any("brain.model" in e for e in errors)

    def test_brain_model_absent_is_ok(self):
        assert validate_config(_VALID_BASE) == []

    def test_conversation_max_history_too_small_is_error(self):
        cfg = {**_VALID_BASE, "conversation": {"max_history": 5}}
        errors = validate_config(cfg)
        assert any("conversation.max_history" in e for e in errors)

    def test_conversation_max_history_too_large_is_error(self):
        cfg = {**_VALID_BASE, "conversation": {"max_history": 201}}
        errors = validate_config(cfg)
        assert any("conversation.max_history" in e for e in errors)

    def test_conversation_max_history_boundary_ok(self):
        for val in (10, 100, 200):
            cfg = {**_VALID_BASE, "conversation": {"max_history": val}}
            assert validate_config(cfg) == [], f"Expected no errors for max_history={val}"

    def test_health_port_below_1024_is_error(self):
        cfg = {**_VALID_BASE, "health_server": {"port": 999}}
        errors = validate_config(cfg)
        assert any("health_server.port" in e for e in errors)

    def test_health_port_above_65535_is_error(self):
        cfg = {**_VALID_BASE, "health_server": {"port": 70000}}
        errors = validate_config(cfg)
        assert any("health_server.port" in e for e in errors)

    def test_health_port_valid_ok(self):
        cfg = {**_VALID_BASE, "health_server": {"port": 8765}}
        assert validate_config(cfg) == []

    def test_health_port_non_integer_is_error(self):
        cfg = {**_VALID_BASE, "health_server": {"port": "http"}}
        errors = validate_config(cfg)
        assert any("health_server.port" in e for e in errors)

    def test_tools_invalid_safety_level_is_error(self):
        cfg = {**_VALID_BASE, "tools": {"general_chat_max_safety_level": "extreme"}}
        errors = validate_config(cfg)
        assert any("general_chat_max_safety_level" in e for e in errors)

    def test_tools_valid_safety_levels_ok(self):
        for level in ("normal", "dangerous", "critical"):
            cfg = {**_VALID_BASE, "tools": {"general_chat_max_safety_level": level}}
            assert validate_config(cfg) == [], f"Expected no errors for level={level}"

    def test_empty_config_does_not_raise(self):
        errors = validate_config({})
        # Should return errors (missing api_key, base_url) but never KeyError
        assert isinstance(errors, list)
        assert any("api_key" in e for e in errors)


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
