"""Tests for LLMConfig — pure data class with validation."""

from __future__ import annotations

from askme.llm.config import LLMConfig


class TestDefaults:
    def test_default_model(self):
        cfg = LLMConfig(api_key="k")
        assert cfg.model == "MiniMax-M2.7-highspeed"

    def test_default_base_url(self):
        cfg = LLMConfig(api_key="k")
        assert "minimax" in cfg.base_url

    def test_default_temperature(self):
        assert LLMConfig(api_key="k").temperature == 0.7

    def test_default_fallback_models_empty(self):
        assert LLMConfig(api_key="k").fallback_models == []

    def test_separate_instances_do_not_share_fallback_list(self):
        a = LLMConfig(api_key="k")
        b = LLMConfig(api_key="k")
        a.fallback_models.append("extra")
        assert b.fallback_models == []


class TestValidate:
    def test_valid_config_returns_no_errors(self):
        cfg = LLMConfig(api_key="sk-test", model="gpt-4", temperature=0.5, timeout=10.0)
        assert cfg.validate() == []

    def test_empty_api_key_is_error(self):
        cfg = LLMConfig(api_key="")
        errors = cfg.validate()
        assert any("api_key" in e for e in errors)

    def test_empty_model_is_error(self):
        cfg = LLMConfig(api_key="k", model="")
        errors = cfg.validate()
        assert any("model" in e for e in errors)

    def test_temperature_below_zero_is_error(self):
        cfg = LLMConfig(api_key="k", temperature=-0.1)
        errors = cfg.validate()
        assert any("temperature" in e for e in errors)

    def test_temperature_above_two_is_error(self):
        cfg = LLMConfig(api_key="k", temperature=2.1)
        errors = cfg.validate()
        assert any("temperature" in e for e in errors)

    def test_temperature_at_boundaries_ok(self):
        for t in (0.0, 2.0):
            cfg = LLMConfig(api_key="k", temperature=t)
            assert not any("temperature" in e for e in cfg.validate())

    def test_zero_timeout_is_error(self):
        cfg = LLMConfig(api_key="k", timeout=0.0)
        errors = cfg.validate()
        assert any("timeout" in e for e in errors)

    def test_negative_timeout_is_error(self):
        cfg = LLMConfig(api_key="k", timeout=-1.0)
        errors = cfg.validate()
        assert any("timeout" in e for e in errors)

    def test_negative_max_retries_is_error(self):
        cfg = LLMConfig(api_key="k", max_retries=-1)
        errors = cfg.validate()
        assert any("max_retries" in e for e in errors)

    def test_zero_max_retries_is_ok(self):
        cfg = LLMConfig(api_key="k", max_retries=0)
        assert not any("max_retries" in e for e in cfg.validate())

    def test_empty_base_url_is_error(self):
        cfg = LLMConfig(api_key="k", base_url="")
        errors = cfg.validate()
        assert any("base_url" in e for e in errors)

    def test_multiple_errors_accumulated(self):
        cfg = LLMConfig(api_key="", model="", timeout=-1.0, base_url="")
        errors = cfg.validate()
        assert len(errors) >= 3


class TestValidateAndWarn:
    def test_returns_true_when_valid(self):
        cfg = LLMConfig(api_key="k", model="m")
        assert cfg.validate_and_warn() is True

    def test_returns_false_when_invalid(self):
        cfg = LLMConfig(api_key="")
        assert cfg.validate_and_warn() is False


class TestFromCfg:
    def test_reads_all_fields(self):
        brain_cfg = {
            "api_key": "sk-abc",
            "base_url": "https://custom.api/v1",
            "model": "my-model",
            "max_tokens": 1024,
            "temperature": 0.3,
            "timeout": 15.0,
            "max_retries": 3,
            "fallback_models": ["a", "b"],
            "minimax_api_key": "mm-key",
            "minimax_base_url": "https://mm.api/v1",
        }
        cfg = LLMConfig.from_cfg(brain_cfg)
        assert cfg.api_key == "sk-abc"
        assert cfg.base_url == "https://custom.api/v1"
        assert cfg.model == "my-model"
        assert cfg.max_tokens == 1024
        assert cfg.temperature == 0.3
        assert cfg.timeout == 15.0
        assert cfg.max_retries == 3
        assert cfg.fallback_models == ["a", "b"]
        assert cfg.minimax_api_key == "mm-key"
        assert cfg.minimax_base_url == "https://mm.api/v1"

    def test_empty_dict_uses_defaults(self):
        cfg = LLMConfig.from_cfg({})
        assert cfg.api_key == ""
        assert cfg.model == "MiniMax-M2.7-highspeed"
        assert cfg.fallback_models == []

    def test_partial_override(self):
        cfg = LLMConfig.from_cfg({"api_key": "sk-x", "model": "custom"})
        assert cfg.api_key == "sk-x"
        assert cfg.model == "custom"
        assert cfg.timeout == 30.0  # default
