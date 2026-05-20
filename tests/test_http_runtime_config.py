from askme.api.services.http_runtime_config import (
    api_documentation_urls,
    bool_config,
    conversation_runtime_settings,
    path_prefix_config,
)


def test_api_documentation_urls_are_disabled_by_default() -> None:
    assert api_documentation_urls({}, env={}) == {
        "docs_url": None,
        "redoc_url": None,
        "openapi_url": None,
    }


def test_api_documentation_urls_support_env_prefix_override() -> None:
    urls = api_documentation_urls(
        {"api": {"openapi_enabled": False, "docs_prefix": "/ignored"}},
        env={
            "ASKME_OPENAPI_ENABLED": "true",
            "ASKME_API_DOCS_PREFIX": "internal-api",
        },
    )

    assert urls == {
        "docs_url": "/internal-api/docs",
        "redoc_url": "/internal-api/redoc",
        "openapi_url": "/internal-api/openapi.json",
    }


def test_api_documentation_urls_support_root_prefix() -> None:
    urls = api_documentation_urls(
        {"api": {"docs_enabled": True, "docs_prefix": "/"}},
        env={},
    )

    assert urls == {
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": "/openapi.json",
    }


def test_bool_and_path_prefix_config_are_operator_tolerant() -> None:
    assert bool_config("enabled") is True
    assert bool_config("off", default=True) is False
    assert bool_config("maybe", default=True) is True
    assert path_prefix_config("api/", default="/api") == "/api"
    assert path_prefix_config("/", default="/api") == ""


def test_conversation_runtime_settings_match_health_server_defaults() -> None:
    settings = conversation_runtime_settings({})

    assert settings.chat_timeout_s == 30.0
    assert settings.chat_max_concurrency == 8
    assert settings.chat_slow_threshold_ms == 2000.0
    assert settings.chat_diagnostics_history_limit == 20
    assert settings.runtime_voice_turn_timeout_s == 30.0


def test_conversation_runtime_settings_parse_and_clamp_values() -> None:
    settings = conversation_runtime_settings(
        {
            "conversation": {
                "chat_timeout_s": 0,
                "chat_max_concurrency": 0,
                "chat_slow_threshold_ms": "not-a-number",
                "chat_diagnostics_history_limit": "3",
                "runtime_voice_turn_timeout_s": -1,
            }
        }
    )

    assert settings.chat_timeout_s is None
    assert settings.chat_max_concurrency == 1
    assert settings.chat_slow_threshold_ms == 2000.0
    assert settings.chat_diagnostics_history_limit == 3
    assert settings.runtime_voice_turn_timeout_s is None
