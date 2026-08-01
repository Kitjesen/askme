from pathlib import Path

import yaml

from askme.runtime.deployment_preflight import run_edge_robot_preflight

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_default_compose_keeps_edge_robot_model_mount() -> None:
    compose = yaml.safe_load(_read("docker/docker-compose.yml"))
    askme = compose["services"]["askme"]

    assert "../models:/app/models:ro" in askme["volumes"]
    assert "ASKME_BLUEPRINT" not in askme["environment"]


def test_linux_edge_override_maps_audio_device_and_host_audio_group() -> None:
    compose = yaml.safe_load(_read("docker/docker-compose.edge-linux.yml"))
    askme = compose["services"]["askme"]

    assert askme["devices"] == [
        "${ASKME_AUDIO_DEVICE:-/dev/snd}:/dev/snd",
    ]
    assert askme["group_add"] == [
        "${ASKME_AUDIO_GID:?set ASKME_AUDIO_GID to the host audio group GID}",
    ]


def test_container_entrypoint_preflights_before_starting_edge_robot() -> None:
    entrypoint = _read("docker/docker-entrypoint.sh")

    preflight = "python -m askme.runtime.deployment_preflight"
    runtime = "exec python -m askme.blueprints.presets.edge_robot"
    assert preflight in entrypoint
    assert runtime in entrypoint
    assert entrypoint.index(preflight) < entrypoint.index(runtime)


def test_clean_runtime_without_models_is_blocked_before_startup(tmp_path: Path) -> None:
    config = yaml.safe_load(_read("config.yaml"))

    payload = run_edge_robot_preflight(
        config,
        root=tmp_path,
        audio_probe=lambda _cfg: {
            "ok": True,
            "errors": [],
            "input": {"index": 0},
            "output": {"index": 1},
        },
    )

    assert payload["status"] == "blocked"
    assert payload["ready"] is False
    assert any(error.startswith("ASR missing ") for error in payload["errors"])
    assert any(error.startswith("VAD missing ") for error in payload["errors"])
    assert any(error.startswith("KWS missing ") for error in payload["errors"])
