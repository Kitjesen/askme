from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUNRISE_ASKME_DIR = "/home/sunrise/data/inovxio/askme"
LEGACY_ASKME_DIR = "/home/sunrise/askme"


def read(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_production_service_defaults_to_sunrise_data_path() -> None:
    service = read("deploy/askme.service")

    assert f"WorkingDirectory={SUNRISE_ASKME_DIR}" in service
    assert f"Environment=ASKME_DIR={SUNRISE_ASKME_DIR}" in service
    assert "EnvironmentFile=-/etc/default/askme" in service
    assert "askme.blueprints.presets.edge_robot" in service
    assert LEGACY_ASKME_DIR not in service


def test_sunrise_service_files_default_to_sunrise_data_path() -> None:
    service_files = [
        "scripts/runtime/services/askme.service",
        "scripts/runtime/services/askme-frame-daemon.service",
        "scripts/runtime/services/brainstem-ros2-bridge.service",
        "scripts/runtime/services/rerun-bridge.service",
    ]

    for relpath in service_files:
        service = read(relpath)
        assert f"Environment=ASKME_DIR={SUNRISE_ASKME_DIR}" in service
        assert "EnvironmentFile=-/etc/default/askme" in service
        assert LEGACY_ASKME_DIR not in service


def test_deploy_scripts_default_to_sunrise_data_path_with_env_override() -> None:
    install = read("deploy/install.sh")
    sync = read("scripts/dev/sync_sunrise.sh")
    agentic = read("scripts/dev/deploy_agentic_shell.sh")

    assert f'ASKME_DIR="${{ASKME_DIR:-{SUNRISE_ASKME_DIR}}}"' in install
    assert f'REMOTE_DIR="${{REMOTE_DIR:-{SUNRISE_ASKME_DIR}}}"' in sync
    assert f'RPATH="${{RPATH:-{SUNRISE_ASKME_DIR}}}"' in agentic


def test_sync_sunrise_never_pushes_remote_secrets_or_device_config() -> None:
    sync = read("scripts/dev/sync_sunrise.sh")

    assert "--exclude='.env'" in sync
    assert "--exclude='config.yaml'" in sync
    assert "for f in pyproject.toml requirements.txt README.md" in sync
    assert "$LOCAL_DIR/prompts/SOUL.md" in sync
    assert "$REMOTE_DIR/prompts/SOUL.md" in sync
    assert "config.yaml" not in "pyproject.toml requirements.txt README.md"


def test_deploy_surface_documents_assets_and_keeps_helpers_portable() -> None:
    readme = read("deploy/README.md")
    quickstart_sh = read("deploy/quickstart.sh")
    quickstart_bat = read("deploy/quickstart.bat")

    for token in (
        "`askme.service`",
        "`install.sh`",
        "`quickstart.sh`",
        "`quickstart.bat`",
        "`site-profiles/`",
        "`customer-project-templates/`",
        "`delivery-resources/`",
        "`security/`",
    ):
        assert token in readme

    assert "docker/docker-compose.yml" in quickstart_sh
    assert "docker\\docker-compose.yml" in quickstart_bat
    assert "D:\\inovxio" not in quickstart_bat
    assert "%~dp0.." in quickstart_bat


def test_docker_surface_documents_compose_entrypoints() -> None:
    readme = read("docker/README.md")

    for token in (
        "`docker-compose.yml`",
        "`docker-compose.prod.yml`",
        "`Dockerfile.askme`",
        "`Dockerfile.zeroclaw`",
        "`docker-entrypoint.sh`",
        "docker compose --env-file docker/.env -f docker/docker-compose.yml up -d",
    ):
        assert token in readme


def test_deployment_guide_uses_repo_root_docker_env_and_compose_file() -> None:
    guide = read("docs/DEPLOYMENT.md")

    assert "cp docker/.env.example docker/.env" in guide
    assert "vi docker/.env" in guide
    assert "--env-file docker/.env" in guide
    assert "-f docker/docker-compose.yml" in guide
    assert "--env-file .env" not in guide
    assert "cp .env .env.backup" not in guide
