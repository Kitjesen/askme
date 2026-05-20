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
