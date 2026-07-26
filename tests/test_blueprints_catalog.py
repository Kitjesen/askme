from __future__ import annotations

import ast
import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from askme.runtime.module import Runtime

from askme.blueprints import (
    BLUEPRINTS,
    blueprint_configuration_summary,
    blueprint_delivery_package,
    blueprint_readiness,
    catalog_payload,
    get_blueprint_spec,
    inspect_blueprint,
    list_blueprints,
    load_blueprint_runtime,
    load_runtime_blueprint_for_modes,
    resolve_runtime_blueprint_for_modes,
)
from askme.blueprints.runner.runner import _preflight_payload


@pytest.mark.parametrize("spec", BLUEPRINTS)
def test_blueprint_catalog_matches_runtime_modules(spec) -> None:
    inspection = inspect_blueprint(spec.name)

    assert inspection["valid"] is True
    assert inspection["modules"] == list(spec.modules)
    assert inspection["duplicate_modules"] == []
    assert inspection["missing_declared_modules"] == []
    assert inspection["undeclared_runtime_modules"] == []


@pytest.mark.parametrize(
    ("voice_mode", "robot_mode", "blueprint_name"),
    [
        (False, False, "text"),
        (False, True, "text"),
        (True, False, "voice"),
        (True, True, "edge_robot"),
    ],
)
def test_runtime_mode_flags_resolve_through_blueprint_catalog(
    voice_mode: bool,
    robot_mode: bool,
    blueprint_name: str,
) -> None:
    assert (
        resolve_runtime_blueprint_for_modes(
            voice_mode=voice_mode,
            robot_mode=robot_mode,
        )
        == blueprint_name
    )
    assert (
        load_runtime_blueprint_for_modes(
            voice_mode=voice_mode,
            robot_mode=robot_mode,
        )
        is load_blueprint_runtime(blueprint_name)
    )


def test_blueprint_catalog_payload_is_customer_readable() -> None:
    payload = catalog_payload(config={"voice": {}, "perception": {}, "interaction_gate": {}, "cognition": {}, "runtime_handoff": {}})
    edge = next(item for item in payload["items"] if item["name"] == "edge_robot")

    assert payload["summary"]["blueprint_count"] >= 6
    assert payload["summary"]["customer_visible_count"] >= 3
    assert payload["summary"]["valid_count"] == payload["summary"]["blueprint_count"]
    assert "configuration_incomplete_count" in payload["summary"]
    assert edge["title"] == "园区巡检机器人运行时"
    assert "访客问路和带路服务" in edge["scenarios"]
    assert "runtime_handoff" in edge["modules"]
    assert "大模型和语音层不能直接控制硬件。" in edge["safety_boundaries"]
    assert edge["inspection"]["startup_command"] == "python -m askme.blueprints.presets.edge_robot"
    assert edge["readiness"]["status"] == "configuration_incomplete"
    assert "field_operations" in edge["readiness"]["missing_config"]
    assert edge["delivery_package"]["package_id"] == "blueprint.edge_robot"
    assert edge["delivery_package"]["deliverables"]["scenario_acceptance"]
    visible_text = repr(payload["items"])
    assert "ASR provider" not in visible_text
    assert "LLM provider" not in visible_text
    assert "TTS provider" not in visible_text
    voice = next(item for item in payload["items"] if item["name"] == "voice")
    assert "语音识别服务" in voice["external_services"]
    assert "大模型服务" in voice["external_services"]
    assert "语音合成服务" in voice["external_services"]


def test_blueprint_configuration_summary_lists_customer_blockers() -> None:
    summary = blueprint_configuration_summary(config={})

    assert summary["blueprint_count"] >= 6
    assert summary["configuration_incomplete_count"] >= 1
    assert summary["ready_for_validation_blueprints"] == []
    edge = next(
        item
        for item in summary["configuration_incomplete_blueprints"]
        if item["name"] == "edge_robot"
    )
    assert "robot_control" in edge["missing_config"]


def test_customer_visible_blueprints_have_delivery_contract() -> None:
    offenders: list[str] = []
    for spec in list_blueprints(customer_visible=True):
        prefix = f"{spec.name}:"
        if len(spec.deployment_targets) < 1:
            offenders.append(f"{prefix} missing deployment targets")
        if len(spec.capabilities) < 3:
            offenders.append(f"{prefix} should list concrete customer capabilities")
        if len(spec.scenarios) < 2:
            offenders.append(f"{prefix} should list acceptance scenarios")
        if len(spec.external_services) < 1:
            offenders.append(f"{prefix} missing external service dependencies")
        if len(spec.safety_boundaries) < 2:
            offenders.append(f"{prefix} missing safety boundaries")
        if len(spec.validation_commands) < 1:
            offenders.append(f"{prefix} missing validation commands")

    assert offenders == []


def test_blueprint_readiness_reports_config_and_validation_gates() -> None:
    missing = blueprint_readiness("voice", config={})
    ready = blueprint_readiness(
        "voice",
        config={
            "voice": {
                "asr": {"model_dir": "models/asr/local"},
                "tts": {"provider": "minimax", "voice_id": "customer-default"},
            },
            "llm": {"provider": "fake"},
            "memory": {"enabled": True, "backend": "vector"},
            "dashboard": {"enabled": True, "host": "127.0.0.1"},
        },
    )

    assert missing["status"] == "configuration_incomplete"
    assert missing["production_ready"] is False
    assert missing["missing_config"] == ["voice.asr", "voice.tts", "llm", "memory", "dashboard"]
    assert missing["config_evidence"][2]["paths_checked"] == ["llm", "brain"]
    assert ready["status"] == "ready_for_validation"
    assert ready["missing_config"] == []
    assert ready["gates"][0]["gate_id"] == "runtime_composition"
    assert ready["gates"][1]["gate_id"] == "required_config"
    assert ready["gates"][2]["gate_id"] == "external_services"
    assert ready["gates"][3]["gate_id"] == "validation_commands"
    assert "tests/test_voice_loop.py" in " ".join(ready["validation_commands"])


def test_blueprint_readiness_uses_product_config_aliases() -> None:
    voice_ready = blueprint_readiness(
        "voice",
        config={
            "voice": {
                "asr": {"model_dir": "models/asr/local"},
                "tts": {"provider": "minimax", "voice_id": "customer-default"},
            },
            "brain": {"provider": "minimax", "model": "MiniMax-M2.7-highspeed"},
            "memory": {"enabled": True, "backend": "vector"},
            "health_server": {"enabled": True, "host": "127.0.0.1"},
        },
    )
    edge_ready = blueprint_readiness(
        "edge_robot",
        config={
            "voice": {
                "asr": {"model_dir": "models/asr/local"},
                "tts": {"provider": "minimax", "voice_id": "customer-default"},
            },
            "perception": {
                "interaction_provider": {
                    "enabled": True,
                    "paths": {"pose_gaze": "artifacts/perception/pose_gaze.json"},
                }
            },
            "field_operations": {
                "dingtalk_webhooks": {"security": "https://example.invalid/dingtalk"},
            },
            "runtime_handoff": {"enabled": True, "profile": "lab"},
            "runtime": {
                "dog_control": {
                    "enabled": True,
                    "base_url": "http://dog-control.local",
                    "bearer_token": "test-token",
                }
            },
        },
    )

    assert voice_ready["status"] == "ready_for_validation"
    assert voice_ready["missing_config"] == []
    assert voice_ready["config_evidence"][2]["matched_path"] == "brain"
    assert voice_ready["config_evidence"][4]["matched_path"] == "health_server"
    assert edge_ready["status"] == "ready_for_validation"
    assert edge_ready["missing_config"] == []
    assert edge_ready["config_evidence"][4]["matched_path"] == "field_operations.dingtalk_webhooks"
    assert edge_ready["config_evidence"][5]["matched_path"] == "runtime.dog_control"


def test_edge_robot_fake_runtime_profile_blocks_customer_site_validation() -> None:
    result = blueprint_readiness(
        "edge_robot",
        config={
            "voice": {
                "asr": {"model_dir": "models/asr/local"},
                "tts": {"provider": "minimax", "voice_id": "customer-default"},
            },
            "perception": {
                "interaction_provider": {
                    "enabled": True,
                    "paths": {"pose_gaze": "artifacts/perception/pose_gaze.json"},
                }
            },
            "field_operations": {
                "dingtalk_webhooks": {"security": "https://example.invalid/dingtalk"},
            },
            "runtime_handoff": {"enabled": True, "profile": "fake"},
            "runtime": {
                "dog_control": {
                    "enabled": True,
                    "base_url": "http://dog-control.local",
                    "bearer_token": "test-token",
                }
            },
        },
    )
    package = blueprint_delivery_package(
        "edge_robot",
        config={
            "voice": {
                "asr": {"model_dir": "models/asr/local"},
                "tts": {"provider": "minimax", "voice_id": "customer-default"},
            },
            "perception": {
                "interaction_provider": {
                    "enabled": True,
                    "paths": {"pose_gaze": "artifacts/perception/pose_gaze.json"},
                }
            },
            "field_operations": {
                "dingtalk_webhooks": {"security": "https://example.invalid/dingtalk"},
            },
            "runtime_handoff": {"enabled": True, "profile": "fake"},
            "runtime": {
                "dog_control": {
                    "enabled": True,
                    "base_url": "http://dog-control.local",
                    "bearer_token": "test-token",
                }
            },
        },
    )

    assert result["status"] == "runtime_profile_not_site_ready"
    assert result["runtime_profile"]["profile"] == "fake"
    assert result["runtime_profile"]["allowed_for_site_validation"] == ["lab", "prod"]
    runtime_gate = next(gate for gate in result["gates"] if gate["gate_id"] == "runtime_profile")
    assert runtime_gate["status"] == "fail"
    assert "fake" in runtime_gate["message"]
    assert package["status"] == "demo_or_shadow_only"
    assert package["customer_status"] == "仅可演示或影子验证"
    assert package["customer_next_step"] == "切换 runtime_handoff.profile 到 lab 或 prod 后再进入客户现场验证。"


def test_blueprint_readiness_rejects_demo_placeholders_and_disabled_services() -> None:
    result = blueprint_readiness(
        "edge_robot",
        config={
            "voice": {
                "asr": {"model_dir": "models/asr/local"},
                "tts": {"provider": "minimax", "voice_id": "customer-default"},
            },
            "perception": {
                "interaction_provider": {
                    "enabled": True,
                    "paths": {"pose_gaze": "artifacts/perception/pose_gaze.json"},
                }
            },
            "field_operations": {
                "dingtalk_webhooks": {
                    "security": "${ASKME_DINGTALK_SECURITY_WEBHOOK}",
                    "cleaning": "",
                    "operations": "",
                },
            },
            "runtime_handoff": {"enabled": True, "profile": "fake"},
            "runtime": {
                "dog_control": {
                    "enabled": False,
                    "base_url": "http://dog-control.local",
                }
            },
        },
    )

    assert result["status"] == "configuration_incomplete"
    assert "dingding" in result["missing_config"]
    assert "robot_control" in result["missing_config"]


def test_blueprint_delivery_package_is_actionable_for_customer_pilot() -> None:
    package = blueprint_delivery_package(
        "park",
        config={
            "voice": {
                "asr": {"model_dir": "models/asr/local"},
                "tts": {"provider": "minimax", "voice_id": "customer-default"},
            },
            "perception": {
                "interaction_provider": {
                    "enabled": True,
                    "paths": {"pose_gaze": "artifacts/perception/pose_gaze.json"},
                }
            },
            "field_operations": {
                "dingtalk_webhooks": {"security": "https://example.invalid/dingtalk"},
            },
            "runtime_handoff": {"enabled": True, "profile": "lab"},
            "runtime": {
                "dog_control": {
                    "enabled": True,
                    "base_url": "http://dog-control.local",
                    "bearer_token": "test-token",
                }
            },
        },
    )

    assert package["status"] == "ready_for_site_validation"
    assert package["release_boundary"].startswith("可用于实验室或客户试点验证")
    assert package["acceptance_boundary"] == package["release_boundary"]
    assert package["customer_status"] == "可进入现场验证"
    assert package["customer_next_step"] == "运行现场验证用例，并归档客户可查证据。"
    assert package["delivery_actions"] == [
        "运行现场验证用例。",
        "归档语音、通知、机器人运行和客户复核证据。",
        "签收前复核安全边界和人工接管方案。",
    ]
    assert package["deliverables"]["runtime_composition"]["status"] == "ready"
    assert package["deliverables"]["external_service_checklist"]
    assert any(
        step["step"] == "generate_site_env_template"
        for step in package["handoff_steps"]
    )
    assert any(
        item["customer_scenario"] == "访客问路和带路服务"
        for item in package["deliverables"]["scenario_acceptance"]
    )
    assert "如果运行组合门禁失败，停止交付。" in package["stop_conditions"]


def test_blueprint_delivery_package_blocks_missing_config() -> None:
    package = blueprint_delivery_package("voice", config={})

    assert package["status"] == "missing_configuration"
    assert package["customer_status"] == "运行配置未补齐"
    assert "voice.asr" in package["customer_next_step"]
    assert package["delivery_actions"][0].startswith("补齐运行配置：")
    assert package["deliverables"]["configuration_checklist"][0]["status"] == "missing"
    assert any("voice.asr" in item for item in package["stop_conditions"])


def test_mcp_blueprint_uses_real_server_entrypoint() -> None:
    package = blueprint_delivery_package(
        "mcp",
        config={
            "mcp": {"enabled": True, "transport": "stdio"},
            "tools": {"enabled": True, "registry": "default"},
            "memory": {"enabled": True, "backend": "vector"},
            "skills": {"enabled": True, "directory": "askme/skills"},
            "runtime_handoff": {"enabled": True, "profile": "fake"},
        },
    )

    assert get_blueprint_spec("mcp").startup_command == "python -m askme.mcp.server"
    assert package["startup_command"] == "python -m askme.mcp.server"
    assert package["operator_runbook"]["start"] == "python -m askme.mcp.server"
    assert package["status"] == "ready_for_site_validation"


@pytest.mark.parametrize("spec", BLUEPRINTS)
def test_blueprint_startup_commands_use_real_entrypoints(spec) -> None:
    if spec.name == "mcp":
        assert spec.startup_command == "python -m askme.mcp.server"
        return

    assert spec.startup_command == f"python -m askme.blueprints.presets.{spec.name}"
    assert spec.startup_command != f"python -m askme.blueprints.{spec.name}"


@pytest.mark.parametrize(
    "startup_command",
    [spec.startup_command for spec in BLUEPRINTS],
)
def test_blueprint_startup_commands_resolve_executable_modules(startup_command: str) -> None:
    module_name = _module_name_from_python_m(startup_command)
    result = subprocess.run(
        [sys.executable, "-B", "-m", module_name, "--help"],
        cwd=".",
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Traceback" not in result.stderr
    assert result.stdout.strip() or result.stderr.strip()


@pytest.mark.parametrize(
    "startup_command",
    [
        spec.startup_command
        for spec in BLUEPRINTS
        if spec.startup_command.startswith("python -m askme.blueprints.")
    ],
)
def test_blueprint_startup_commands_support_no_io_preflight(startup_command: str) -> None:
    module_name = _module_name_from_python_m(startup_command)
    result = subprocess.run(
        [sys.executable, "-B", "-m", module_name, "--preflight", "--json"],
        cwd=".",
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    raw = result.stdout or result.stderr
    # Log output may include a prefix; extract JSON from the last line
    if "INFO:" in raw:
        raw = raw.split("INFO:")[-1].strip()
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert payload["opens_runtime_io"] is False
    assert payload["module_count"] == len(payload["modules"])
    assert payload["modules"]


@pytest.mark.parametrize(
    "startup_command",
    [
        spec.startup_command
        for spec in BLUEPRINTS
        if spec.startup_command.startswith("python -m askme.blueprints.presets.")
    ],
)
@pytest.mark.parametrize("cli_args", [("--help",), ("--preflight", "--json")])
def test_blueprint_preset_cli_paths_do_not_import_runtime_io_modules(
    startup_command: str,
    cli_args: tuple[str, ...],
) -> None:
    module_name = _module_name_from_python_m(startup_command)
    probe = (
        "import json, runpy, sys; "
        f"sys.argv = [{module_name!r}, *{list(cli_args)!r}]; "
        f"runpy.run_module({module_name!r}, run_name='__main__'); "
        "bad = sorted(name for name in sys.modules "
        "if name == 'askme.runtime.modules' "
        "or name.startswith('askme.runtime.modules.') "
        "or name == 'askme.voice.output.tts' "
        "or name.startswith('askme.voice.output.tts.')); "
        "print(json.dumps(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=".",
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    bad_modules = json.loads(result.stdout.strip().splitlines()[-1])
    assert bad_modules == []


def test_blueprint_runner_preflight_payload_matches_catalog_inspection() -> None:
    runtime = load_blueprint_runtime("text")
    inspection = inspect_blueprint("text")

    payload = _preflight_payload(runtime, "Text runtime")

    assert payload == {
        "ok": True,
        "label": "Text runtime",
        "module_count": inspection["module_count"],
        "modules": inspection["modules"],
        "duplicates": [],
        "opens_runtime_io": False,
    }


def _module_name_from_python_m(command: str) -> str:
    parts = shlex.split(command)
    assert len(parts) >= 3
    assert parts[0] == "python"
    assert parts[1] == "-m"
    return parts[2]


def test_blueprint_aliases_and_runtime_loading() -> None:
    assert get_blueprint_spec("park").name == "edge_robot"
    assert get_blueprint_spec("lingtu").name == "lingtu_voice"

    runtime = load_blueprint_runtime("park")

    assert isinstance(runtime, Runtime)
    assert runtime.module_names()[-1] == "proactive"


def test_blueprint_preset_package_does_not_shadow_submodule_imports() -> None:
    module = importlib.import_module("askme.blueprints.presets.edge_robot")
    legacy_module = importlib.import_module("askme.blueprints.edge_robot")
    package = importlib.import_module("askme.blueprints.presets")

    assert isinstance(module, ModuleType)
    assert isinstance(legacy_module, ModuleType)
    assert hasattr(module, "edge_robot")
    assert hasattr(package, "edge_robot_runtime")


def test_blueprint_presets_do_not_import_each_other() -> None:
    preset_dir = Path("askme/blueprints/presets")
    violations: list[str] = []
    for path in sorted(preset_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name.startswith("askme.blueprints.presets."):
                    violations.append(f"{path}:{node.lineno} imports {module_name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("askme.blueprints.presets."):
                        violations.append(f"{path}:{node.lineno} imports {alias.name}")

    assert violations == []


def test_customer_visible_blueprint_filter() -> None:
    customer_names = {item.name for item in list_blueprints(customer_visible=True)}
    internal_names = {item.name for item in list_blueprints(customer_visible=False)}

    assert {"voice", "voice_perception", "edge_robot", "lingtu_voice"} <= customer_names
    assert {"text", "mcp"} <= internal_names
