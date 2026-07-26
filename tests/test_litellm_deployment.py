from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_litellm_sidecar_is_version_pinned_and_loopback_bound() -> None:
    compose = yaml.safe_load(
        (ROOT / "docker" / "docker-compose.litellm.yml").read_text(encoding="utf-8")
    )

    proxy = compose["services"]["litellm"]

    assert proxy["image"] == (
        "ghcr.io/berriai/litellm-database:v1.93.0"
        "@sha256:72360d8bd5602faa49be5098a8ac3dd069d9fb74503d6bd014242d96dc753e43"
    )
    assert proxy["ports"] == ["127.0.0.1:${LITELLM_PORT:-4000}:4000"]
    assert "./litellm-config.yaml:/app/config.yaml:ro" in proxy["volumes"]
    assert proxy["environment"]["NO_DOCS"] == "True"
    assert proxy["environment"]["NO_REDOC"] == "True"
    assert proxy["environment"]["LITELLM_MODE"] == "PRODUCTION"
    assert proxy["read_only"] is True
    assert proxy["environment"]["LITELLM_NON_ROOT"] == "true"
    assert proxy["environment"]["LITELLM_MIGRATION_DIR"] == "/app/migrations"
    assert proxy["environment"]["PRISMA_BINARY_CACHE_DIR"] == "/app/cache/prisma"
    assert proxy["environment"]["XDG_CACHE_HOME"] == "/app/cache"
    assert {"/tmp", "/app/migrations", "/app/cache"} <= {
        item.split(":", 1)[0] for item in proxy["tmpfs"]
    }


def test_litellm_model_config_keeps_provider_secrets_out_of_source() -> None:
    config = yaml.safe_load((ROOT / "docker" / "litellm-config.yaml").read_text(encoding="utf-8"))

    models = {item["model_name"]: item["litellm_params"] for item in config["model_list"]}

    assert {
        "voice-fast",
        "voice-quality",
        "robot-action",
        "memory-compact",
        "health-probe",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
        "MiniMax-M2.7-highspeed",
    } <= models.keys()
    assert models["voice-fast"]["model"] == "openai/deepseek-v4-flash"
    assert models["voice-quality"]["model"] == "openai/deepseek-v4-pro"
    assert models["robot-action"]["model"] == "openai/deepseek-v4-flash"
    assert models["memory-compact"]["model"] == "openai/deepseek-v4-flash"
    assert models["health-probe"]["model"] == "openai/deepseek-v4-flash"
    assert models["deepseek-v4-flash"]["api_key"] == "os.environ/DEEPSEEK_API_KEY"
    assert models["deepseek-v4-pro"]["api_key"] == "os.environ/DEEPSEEK_API_KEY"
    assert models["MiniMax-M2.7-highspeed"]["api_key"] == "os.environ/MINIMAX_API_KEY"
    assert config["litellm_settings"]["num_retries"] == 0
    assert config["litellm_settings"]["set_verbose"] is False
    assert config["litellm_settings"]["json_logs"] is True
    assert config["general_settings"]["disable_error_logs"] is True
    assert config["general_settings"]["turn_off_message_logging"] is True
    assert config["general_settings"]["redact_user_api_key_info"] is True
    assert config["router_settings"]["fallbacks"] == [
        {"voice-fast": ["voice-quality"]},
        {"memory-compact": ["voice-quality"]},
        {"deepseek-v4-flash": ["deepseek-v4-pro"]},
    ]
    assert "fallbacks" not in config["litellm_settings"]


def test_askme_litellm_preset_uses_scoped_key_and_single_routing_owner() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    preset = config["brain"]["provider_presets"]["litellm"]

    assert preset == {
        "api_key": "${LITELLM_VIRTUAL_KEY}",
        "base_url": "${LITELLM_BASE_URL}",
        "model": "voice-fast",
        "voice_model": "voice-fast",
        "max_retries": 0,
        "fallback_models": [],
        "minimax_api_key": "",
    }

    board = yaml.safe_load((ROOT / "config.board.yaml").read_text(encoding="utf-8"))
    assert board["brain"]["provider_presets"]["litellm"]["model"] == "voice-fast"
    assert board["brain"]["provider_presets"]["litellm"]["voice_model"] == "voice-fast"
