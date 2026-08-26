from __future__ import annotations

import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _construct_compose_sequence(loader, node):
    return loader.construct_sequence(node, deep=True)


for _compose_tag in ("!override", "!reset"):
    yaml.SafeLoader.add_constructor(_compose_tag, _construct_compose_sequence)


def _dotenv_keys(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def test_product_configs_route_llm_through_litellm_by_default() -> None:
    expected = {
        "provider": "litellm",
        "api_key": "${LITELLM_VIRTUAL_KEY}",
        "base_url": "${LITELLM_BASE_URL}",
        "model": "voice-fast",
        "voice_model": "voice-fast",
        "health_model": "health-probe",
        "max_retries": 0,
        "fallback_models": [],
        "minimax_api_key": "",
    }

    for filename in ("config.yaml", "config.board.yaml"):
        config = yaml.safe_load((ROOT / filename).read_text(encoding="utf-8"))
        brain = config["brain"]

        assert {key: brain.get(key) for key in expected} == expected


def test_chinese_readme_matches_fail_closed_vlm_and_zeroclaw_status() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "板卡 profile 只启用本地摄像头/YOLO 感知" in readme
    assert "板卡 profile `config.board.yaml` 会显式开启两者" not in readme
    assert "ZeroClaw v0.1.7 容器尚未接通 AskMe MCP" in readme
    assert "启动 ZeroClaw 进程不等于 MCP 集成可用" in readme


def test_askme_images_embed_an_explicit_runtime_config_contract() -> None:
    for filename in ("Dockerfile", "docker/Dockerfile.askme"):
        dockerfile = (ROOT / filename).read_text(encoding="utf-8")
        assert "COPY config.yaml /app/config.yaml" in dockerfile
        assert "ENV ASKME_CONFIG_PATH=/app/config.yaml" in dockerfile


def test_health_server_host_is_environment_driven_and_loopback_by_default() -> None:
    for filename in ("config.yaml", "config.board.yaml"):
        config = yaml.safe_load((ROOT / filename).read_text(encoding="utf-8"))
        assert config["health_server"]["host"] == "${ASKME_HEALTH_HOST}"


def test_product_cloud_vlm_is_fail_closed_behind_a_future_litellm_alias() -> None:
    expected = {
        "vlm_enabled": False,
        "vlm_backend": "openai",
        "vlm_api_key": "${VISION_LITELLM_VIRTUAL_KEY}",
        "vlm_base_url": "${LITELLM_BASE_URL}",
        "vlm_model": "vision-scene",
    }

    for filename in ("config.yaml", "config.board.yaml"):
        config = yaml.safe_load((ROOT / filename).read_text(encoding="utf-8"))
        vision = config["vision"]
        assert {key: vision.get(key) for key in expected} == expected
        assert "minimaxi.com" not in str(vision)


def test_application_env_template_uses_only_scoped_litellm_credentials() -> None:
    values = _dotenv_keys(ROOT / ".env.example")

    assert values["LITELLM_BASE_URL"] == "http://127.0.0.1:4000/v1"
    assert values["LITELLM_VIRTUAL_KEY"] == ""
    assert values["NO_PROXY"] == "127.0.0.1,localhost"
    assert "LLM_API_KEY" not in values
    assert "LLM_BASE_URL" not in values
    assert "DEEPSEEK_API_KEY" not in values

    docker_values = _dotenv_keys(ROOT / "docker" / ".env.example")
    assert docker_values["LITELLM_VIRTUAL_KEY"] == ""
    assert docker_values["LITELLM_BASE_URL"] == "http://127.0.0.1:4000/v1"
    assert docker_values["ASKME_CONTROL_API_KEY"] == ""
    assert "ASKME_LITELLM_BASE_URL" not in docker_values
    assert "LLM_API_KEY" not in docker_values
    assert "LLM_BASE_URL" not in docker_values
    assert "DEEPSEEK_API_KEY" not in docker_values


def _compose_command(
    *files: str,
    env: dict[str, str],
    config_args: tuple[str, ...] = ("config", "--quiet"),
) -> subprocess.CompletedProcess[str]:
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed")
    try:
        version = subprocess.run(
            ["docker", "compose", "version"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("Docker Compose CLI is not responsive")
    if version.returncode != 0:
        pytest.skip("Docker Compose plugin is not installed")
    command = ["docker", "compose"]
    for env_file in ("docker/.env.example", "docker/litellm.env.example"):
        command.extend(["--env-file", env_file])
    for compose_file in files:
        command.extend(["-f", compose_file])
    command.extend(config_args)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _valid_compose_environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "LITELLM_MASTER_KEY": "sk-test-master-A7m4Q9x2K8v6R3c5",
            "LITELLM_VIRTUAL_KEY": "sk-test-askme-B8n5R2w9K4x7T1c6",
            "ZEROCLAW_LITELLM_VIRTUAL_KEY": "sk-test-zeroclaw-C9p6T3x0V5y8M2d7",
            "LITELLM_SALT_KEY": "salt-test-D0q7V4y1N6z9K3c8R2m5",
            "LITELLM_DATABASE_PASSWORD": "db-test-E1r8W5z2P7x0M4d9T3n6",
            "ASKME_CONTROL_API_KEY": "test-control-key",
            "DEEPSEEK_API_KEY": "sk-test-provider",
            "DEEPSEEK_BASE_URL": "https://provider.invalid/v1",
        }
    )
    return env


def _default_compose_environment() -> dict[str, str]:
    env = _valid_compose_environment()
    env.pop("ZEROCLAW_LITELLM_VIRTUAL_KEY")
    return env


@pytest.mark.parametrize(
    "compose_files",
    [
        ("docker/docker-compose.litellm.yml",),
        ("docker/docker-compose.yml",),
        ("docker/docker-compose.yml", "docker/docker-compose.prod.yml"),
        ("docker-compose.yml",),
    ],
)
def test_compose_entrypoints_render_without_a_running_docker_daemon(
    compose_files: tuple[str, ...],
) -> None:
    result = _compose_command(*compose_files, env=_default_compose_environment())

    assert result.returncode == 0, result.stderr or result.stdout


def test_production_compose_removes_development_host_ports() -> None:
    result = _compose_command(
        "docker/docker-compose.yml",
        "docker/docker-compose.prod.yml",
        env=_default_compose_environment(),
        config_args=("config", "--format", "json"),
    )

    assert result.returncode == 0, result.stderr or result.stdout
    rendered = json.loads(result.stdout)
    assert not rendered["services"]["askme"].get("ports")
    assert "zeroclaw" not in rendered["services"]


def test_product_compose_rejects_empty_scoped_virtual_keys() -> None:
    env = _valid_compose_environment()
    env["LITELLM_VIRTUAL_KEY"] = ""

    result = _compose_command("docker/docker-compose.yml", env=env)

    assert result.returncode != 0
    assert "LITELLM_VIRTUAL_KEY" in (result.stderr + result.stdout)


def test_product_compose_rejects_empty_remote_control_key() -> None:
    env = _valid_compose_environment()
    env["ASKME_CONTROL_API_KEY"] = ""

    result = _compose_command("docker/docker-compose.yml", env=env)

    assert result.returncode != 0
    assert "ASKME_CONTROL_API_KEY" in (result.stderr + result.stdout)


def test_product_compose_injects_supported_optional_runtime_environment() -> None:
    compose = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8"))
    environment = compose["services"]["askme"]["environment"]

    assert {
        key: environment[key]
        for key in (
            "NAV_GATEWAY_URL",
            "DOG_CONTROL_SERVICE_URL",
            "DOG_SAFETY_SERVICE_URL",
            "ASKME_EDGE_SERVICE_URL",
            "NOVA_DOG_RUNTIME_API_KEY",
            "RUNTIME_BEARER_TOKEN",
            "RUNTIME_OPERATOR_ID",
        )
    } == {
        "NAV_GATEWAY_URL": "${NAV_GATEWAY_URL:-}",
        "DOG_CONTROL_SERVICE_URL": "${DOG_CONTROL_SERVICE_URL:-}",
        "DOG_SAFETY_SERVICE_URL": "${DOG_SAFETY_SERVICE_URL:-}",
        "ASKME_EDGE_SERVICE_URL": "${ASKME_EDGE_SERVICE_URL:-}",
        "NOVA_DOG_RUNTIME_API_KEY": "${NOVA_DOG_RUNTIME_API_KEY:-}",
        "RUNTIME_BEARER_TOKEN": "${RUNTIME_BEARER_TOKEN:-}",
        "RUNTIME_OPERATOR_ID": "${RUNTIME_OPERATOR_ID:-askme}",
    }

    values = _dotenv_keys(ROOT / "docker" / ".env.example")
    for key in ("NAV_GATEWAY_URL", "DOG_CONTROL_SERVICE_URL", "DOG_SAFETY_SERVICE_URL"):
        assert values[key] == ""
    for unsupported in (
        "TTS_VOICE_ID",
        "TTS_SPEED",
        "TTS_EMOTION",
        "OTA_SERVER_URL",
        "NOVA_DOG_SERIAL_NUMBER",
        "ROBOT_SERIAL_PORT",
    ):
        assert unsupported not in values
        assert unsupported not in environment


def test_minimax_tts_and_optional_litellm_provider_keys_use_separate_env_names() -> None:
    application_env = _dotenv_keys(ROOT / "docker" / ".env.example")
    control_env = _dotenv_keys(ROOT / "docker" / "litellm.env.example")
    sidecar = yaml.safe_load(
        (ROOT / "docker" / "docker-compose.litellm.yml").read_text(encoding="utf-8")
    )["services"]["litellm"]

    assert "MINIMAX_API_KEY" in application_env
    assert "MINIMAX_API_KEY" not in control_env
    assert control_env["LITELLM_MINIMAX_PROVIDER_API_KEY"] == ""
    assert sidecar["environment"]["MINIMAX_API_KEY"] == ("${LITELLM_MINIMAX_PROVIDER_API_KEY:-}")


@pytest.mark.parametrize(
    "name",
    ["LITELLM_MASTER_KEY", "LITELLM_SALT_KEY", "LITELLM_DATABASE_PASSWORD"],
)
def test_control_plane_env_template_values_cannot_pass_the_startup_gate(name: str) -> None:
    from askme.llm.core.key_policy import KeyPolicyError, validate_litellm_key_policy

    environment = _valid_compose_environment()
    template = _dotenv_keys(ROOT / "docker" / "litellm.env.example")
    environment[name] = template[name]

    with pytest.raises(KeyPolicyError, match=name):
        validate_litellm_key_policy(environment, require_application=False)


def test_product_compose_starts_askme_behind_the_litellm_readiness_gate() -> None:
    compose = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert services["litellm"]["extends"] == {
        "file": "docker-compose.litellm.yml",
        "service": "litellm",
    }
    assert services["litellm-db"]["extends"] == {
        "file": "docker-compose.litellm.yml",
        "service": "litellm-db",
    }
    assert services["askme"]["depends_on"]["litellm"]["condition"] == "service_healthy"
    assert (
        services["askme"]["depends_on"]["litellm-key-policy"]["condition"]
        == "service_completed_successfully"
    )
    environment = services["askme"]["environment"]
    assert environment["LITELLM_BASE_URL"] == "http://litellm:4000/v1"
    assert environment["LITELLM_VIRTUAL_KEY"] == ("${LITELLM_VIRTUAL_KEY:?set LITELLM_VIRTUAL_KEY}")
    assert environment["ASKME_HEALTH_HOST"] == "0.0.0.0"
    assert environment["ASKME_CONTROL_API_KEY"] == (
        "${ASKME_CONTROL_API_KEY:?set ASKME_CONTROL_API_KEY}"
    )
    assert environment["NO_PROXY"] == "${NO_PROXY:-127.0.0.1,localhost,litellm}"
    assert "LLM_API_KEY" not in environment
    assert "LLM_BASE_URL" not in environment


def test_product_compose_healthcheck_uses_component_readiness() -> None:
    compose = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8"))
    command = " ".join(compose["services"]["askme"]["healthcheck"]["test"])

    assert "/ready" in command
    assert "/healthz" not in command


@pytest.mark.parametrize("component", ["llm", "memory"])
def test_ready_endpoint_fails_closed_for_degraded_runtime_components(component: str) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from askme.api.routes.health import register_health_routes
    from askme.api.services.health_service import HealthService

    health_service = HealthService()
    health_service.register(component, lambda: {"status": "degraded"})
    app = FastAPI()
    register_health_routes(app, health_service, routes=("ready",))

    response = TestClient(app).get("/ready")

    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["components"][component]["status"] == "degraded"


def test_product_compose_has_hardened_fail_closed_key_policy_gate() -> None:
    compose = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    policy = services["litellm-key-policy"]

    assert policy["entrypoint"] == [
        "python",
        "-m",
        "askme.llm.key_policy",
    ]
    assert policy["restart"] == "no"
    assert policy["network_mode"] == "none"
    assert policy["read_only"] is True
    assert policy["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in policy["security_opt"]
    assert policy["environment"] == {
        "LITELLM_MASTER_KEY": "${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY}",
        "LITELLM_VIRTUAL_KEY": "${LITELLM_VIRTUAL_KEY:?set LITELLM_VIRTUAL_KEY}",
        "LITELLM_SALT_KEY": "${LITELLM_SALT_KEY:?set LITELLM_SALT_KEY}",
        "LITELLM_DATABASE_PASSWORD": (
            "${LITELLM_DATABASE_PASSWORD:?set LITELLM_DATABASE_PASSWORD}"
        ),
    }

    experimental = services["litellm-zeroclaw-key-policy"]
    assert experimental["profiles"] == ["experimental-zeroclaw"]
    assert experimental["entrypoint"] == [
        "python",
        "-m",
        "askme.llm.key_policy",
        "--require-zeroclaw",
    ]
    assert experimental["environment"]["ZEROCLAW_LITELLM_VIRTUAL_KEY"] == (
        "${ZEROCLAW_LITELLM_VIRTUAL_KEY:-}"
    )
    assert experimental["environment"]["LITELLM_SALT_KEY"] == (
        "${LITELLM_SALT_KEY:?set LITELLM_SALT_KEY}"
    )
    assert experimental["environment"]["LITELLM_DATABASE_PASSWORD"] == (
        "${LITELLM_DATABASE_PASSWORD:?set LITELLM_DATABASE_PASSWORD}"
    )
    assert (
        services["zeroclaw"]["depends_on"]["litellm-zeroclaw-key-policy"]["condition"]
        == "service_completed_successfully"
    )


def test_default_product_profile_excludes_experimental_zeroclaw() -> None:
    result = _compose_command(
        "docker/docker-compose.yml",
        env=_default_compose_environment(),
        config_args=("config", "--format", "json"),
    )

    assert result.returncode == 0, result.stderr or result.stdout
    services = json.loads(result.stdout)["services"]
    assert "askme" in services
    assert "zeroclaw" not in services
    assert "litellm-zeroclaw-key-policy" not in services


def test_root_compose_uses_the_same_litellm_control_plane() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    assert services["litellm"]["extends"]["service"] == "litellm"
    assert services["litellm-db"]["extends"]["service"] == "litellm-db"
    askme = services["askme"]
    assert askme["depends_on"]["litellm"]["condition"] == "service_healthy"
    assert (
        askme["depends_on"]["litellm-key-policy"]["condition"] == "service_completed_successfully"
    )
    assert services["litellm-key-policy"]["entrypoint"] == [
        "python",
        "-m",
        "askme.llm.key_policy",
    ]
    assert services["litellm-key-policy"]["environment"] == {
        "LITELLM_MASTER_KEY": "${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY}",
        "LITELLM_VIRTUAL_KEY": "${LITELLM_VIRTUAL_KEY:?set LITELLM_VIRTUAL_KEY}",
        "LITELLM_SALT_KEY": "${LITELLM_SALT_KEY:?set LITELLM_SALT_KEY}",
        "LITELLM_DATABASE_PASSWORD": (
            "${LITELLM_DATABASE_PASSWORD:?set LITELLM_DATABASE_PASSWORD}"
        ),
    }
    assert askme["environment"]["LITELLM_BASE_URL"] == "http://litellm:4000/v1"
    assert askme["environment"]["LITELLM_VIRTUAL_KEY"] == (
        "${LITELLM_VIRTUAL_KEY:?set LITELLM_VIRTUAL_KEY}"
    )
    assert "DEEPSEEK_API_KEY" not in askme["environment"]
    assert "DEEPSEEK_BASE_URL" not in askme["environment"]


def test_zeroclaw_routes_only_through_litellm_robot_action() -> None:
    compose = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8"))
    zeroclaw = compose["services"]["zeroclaw"]
    environment = zeroclaw["environment"]

    assert environment["ZEROCLAW_PROVIDER"] == "custom:http://litellm:4000/v1"
    assert environment["ZEROCLAW_MODEL"] == "robot-action"
    assert environment["ZEROCLAW_API_KEY"] == ("${ZEROCLAW_LITELLM_VIRTUAL_KEY:-}")
    assert "MINIMAX_API_KEY" not in environment
    assert zeroclaw["profiles"] == ["experimental-zeroclaw"]
    assert zeroclaw["depends_on"]["litellm"]["condition"] == "service_healthy"
    assert (
        zeroclaw["depends_on"]["litellm-zeroclaw-key-policy"]["condition"]
        == "service_completed_successfully"
    )
    assert "./zeroclaw/config.toml:/root/.zeroclaw/config.toml:ro" in zeroclaw["volumes"]

    config = tomllib.loads(
        (ROOT / "docker" / "zeroclaw" / "config.toml").read_text(encoding="utf-8")
    )
    assert config["default_provider"] == "custom:http://litellm:4000/v1"
    assert config["default_model"] == "robot-action"
    assert config["reliability"]["provider_retries"] == 0
    assert config["reliability"]["fallback_providers"] == []
    assert config["reliability"]["api_keys"] == []


def test_local_zeroclaw_setup_uses_litellm_scoped_key() -> None:
    script = (ROOT / "scripts" / "dev" / "setup_zeroclaw.py").read_text(encoding="utf-8")

    assert '"minimax-cn"' not in script
    assert "minimax_api_key" not in script
    assert '"robot-action"' in script
    assert 'f"custom:{base_url}"' in script
    assert "provider_retries" in script


def test_sidecar_bootstrap_and_product_stack_share_project_and_network() -> None:
    standalone = yaml.safe_load(
        (ROOT / "docker" / "docker-compose.litellm.yml").read_text(encoding="utf-8")
    )
    product = yaml.safe_load((ROOT / "docker" / "docker-compose.yml").read_text(encoding="utf-8"))
    root = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))

    assert standalone["name"] == product["name"] == root["name"] == "askme-litellm"
    for compose in (standalone, product, root):
        assert "askme-net" in compose["networks"]
        assert "askme-net" in compose["services"]["litellm"]["networks"]
        assert "askme-net" in compose["services"]["litellm-db"]["networks"]
    assert "askme-net" in product["services"]["askme"]["networks"]
    assert "askme-net" in root["services"]["askme"]["networks"]


def test_standalone_sidecar_blocks_weak_control_plane_secrets_before_startup() -> None:
    compose = yaml.safe_load(
        (ROOT / "docker" / "docker-compose.litellm.yml").read_text(encoding="utf-8")
    )
    services = compose["services"]
    policy = services["litellm-key-policy"]

    assert policy["entrypoint"] == [
        "python",
        "-m",
        "askme.llm.key_policy",
        "--control-plane-only",
    ]
    assert policy["network_mode"] == "none"
    assert policy["read_only"] is True
    assert policy["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in policy["security_opt"]
    assert policy["environment"] == {
        "LITELLM_MASTER_KEY": "${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY}",
        "LITELLM_SALT_KEY": "${LITELLM_SALT_KEY:?set LITELLM_SALT_KEY}",
        "LITELLM_DATABASE_PASSWORD": (
            "${LITELLM_DATABASE_PASSWORD:?set LITELLM_DATABASE_PASSWORD}"
        ),
    }
    for service_name in ("litellm-db", "litellm"):
        assert (
            services[service_name]["depends_on"]["litellm-key-policy"]["condition"]
            == "service_completed_successfully"
        )


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
        "health_model": "health-probe",
        "max_retries": 0,
        "fallback_models": [],
        "minimax_api_key": "",
    }

    board = yaml.safe_load((ROOT / "config.board.yaml").read_text(encoding="utf-8"))
    assert board["brain"]["provider_presets"]["litellm"]["model"] == "voice-fast"
    assert board["brain"]["provider_presets"]["litellm"]["voice_model"] == "voice-fast"
