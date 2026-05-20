from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def test_new_owner_packages_and_legacy_facades_share_objects() -> None:
    from askme.voice.runtime_bridge import VoiceRuntimeBridge as VoiceAliasBridge

    from askme.interaction.intent_router import IntentRouter as LegacyIntentRouter
    from askme.llm.intent_router import IntentRouter as LlmIntentRouter
    from askme.ports import RobotControlPort
    from askme.robot.dog.control_client import DogControlClient
    from askme.robot_interaction import IntentRouter, RobotInteractionService
    from askme.voice.orchestration.runtime_bridge import (
        VoiceRuntimeBridge as LegacyVoiceRuntimeBridge,
    )
    from askme.voice_gateway import VoiceGatewayService, VoiceRuntimeBridge

    router = IntentRouter()
    interaction = RobotInteractionService(router)
    gateway = VoiceGatewayService()

    assert LegacyIntentRouter is IntentRouter
    assert LlmIntentRouter is IntentRouter
    assert LegacyVoiceRuntimeBridge is VoiceRuntimeBridge
    assert VoiceAliasBridge is VoiceRuntimeBridge
    assert interaction.router is router
    assert gateway.status_snapshot()["enabled"] is False
    assert isinstance(DogControlClient({}), RobotControlPort)


def test_voice_gateway_public_api_does_not_promote_provider_runtime_bridge() -> None:
    import askme.voice_gateway as voice_gateway
    import askme.voice_gateway.runtime_bridge as runtime_bridge_facade
    from askme.providers.voice_runtime import VoiceRuntimeBridge as ProviderVoiceRuntimeBridge

    assert "VoiceRuntimeBridge" not in voice_gateway.__all__
    assert voice_gateway.VoiceRuntimeBridge is ProviderVoiceRuntimeBridge
    assert runtime_bridge_facade.__all__ == ["VoiceRuntimeBridge"]
    assert not hasattr(runtime_bridge_facade, "requests")
    assert not hasattr(runtime_bridge_facade, "time")


def test_voice_gateway_service_delegates_to_runtime_bridge_contract() -> None:
    from askme.voice_gateway import VoiceGatewayService

    class Bridge:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def status_snapshot(self) -> dict[str, Any]:
            return {"enabled": True, "circuit_open": False}

        def handle_voice_text(self, text: str) -> dict[str, Any] | None:
            self.calls.append(("voice", text))
            return {"handled": True, "channel": "voice", "text": text}

        def handle_text_input(self, text: str) -> dict[str, Any] | None:
            self.calls.append(("text", text))
            return {"handled": True, "channel": "text", "text": text}

    bridge = Bridge()
    gateway = VoiceGatewayService(bridge)

    assert gateway.status_snapshot()["enabled"] is True
    assert gateway.handle_voice_text("hello") == {
        "handled": True,
        "channel": "voice",
        "text": "hello",
    }
    assert gateway.handle_text_input("hi") == {
        "handled": True,
        "channel": "text",
        "text": "hi",
    }
    assert bridge.calls == [("voice", "hello"), ("text", "hi")]


def test_middle_layers_do_not_import_hardware_or_device_bindings() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    layer_roots = [
        repo_root / "askme" / "voice_gateway",
        repo_root / "askme" / "robot_interaction",
    ]
    forbidden = (
        "askme.robot.dog",
        "askme.robot.arm",
        "sounddevice",
        "cv2",
        "fastapi",
    )

    violations: list[str] = []
    for root in layer_roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden:
                if pattern in text:
                    violations.append(f"{path.relative_to(repo_root)} imports {pattern}")

    assert violations == []


def test_providers_stay_below_product_and_runtime_layers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    provider_root = repo_root / "askme" / "providers"
    forbidden = (
        "askme.api",
        "askme.blueprints",
        "askme.pipeline",
        "askme.runtime",
        "askme.robot_interaction",
        "askme.skills",
        "askme.tools",
        "askme.voice_gateway",
    )

    violations: list[str] = []
    for path in provider_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if any(
                    module_name == item or module_name.startswith(f"{item}.")
                    for item in forbidden
                ):
                    violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == item or alias.name.startswith(f"{item}.")
                        for item in forbidden
                    ):
                        violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    assert violations == []


def test_runtime_layer_does_not_import_blueprints() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runtime_root = repo_root / "askme" / "runtime"
    forbidden = "askme.blueprints"

    violations: list[str] = []
    for path in runtime_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name == forbidden or module_name.startswith(f"{forbidden}."):
                    violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == forbidden or alias.name.startswith(f"{forbidden}."):
                        violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    assert violations == []


def test_upper_layers_import_provider_facade_not_provider_submodules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checked_roots = [
        repo_root / "askme" / "api",
        repo_root / "askme" / "runtime",
        repo_root / "askme" / "pipeline",
        repo_root / "askme" / "mcp",
        repo_root / "askme" / "tools",
    ]
    checked_files = [
        repo_root / "askme" / "health_server.py",
    ]
    compatibility_exceptions = {
        repo_root / "askme" / "pipeline" / "reactions" / "register_defaults.py",
    }
    violations: list[str] = []

    checked_paths = [path for root in checked_roots for path in root.rglob("*.py")]
    checked_paths.extend(path for path in checked_files if path.exists())
    for path in checked_paths:
        if path in compatibility_exceptions:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name.startswith("askme.providers."):
                    violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("askme.providers."):
                        violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    assert violations == []


def test_tools_runtime_http_touchpoints_stay_in_compat_fallback_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tools_root = repo_root / "askme" / "tools"
    allowed_files = {
        Path("askme/tools/core/builtin_tools.py"),
        Path("askme/tools/robot/move_tool.py"),
        Path("askme/tools/robot/robot_api_tool.py"),
        Path("askme/tools/robot/runtime_api.py"),
        Path("askme/tools/spatial/scan_tool.py"),
        Path("askme/tools/spatial/temporal_query_tool.py"),
    }
    runtime_tokens = (
        "DOG_CONTROL_SERVICE_URL",
        "DOG_NAV_SERVICE_URL",
        "NAV_GATEWAY_URL",
        "/api/v1/control/",
        "/api/v1/memory/temporal",
        "/api/v1/navigation/",
    )

    violations: list[str] = []
    for path in sorted(tools_root.rglob("*.py")):
        rel_path = path.relative_to(repo_root)
        if rel_path in allowed_files:
            continue
        text = path.read_text(encoding="utf-8")
        for token in runtime_tokens:
            if token in text:
                violations.append(
                    f"{rel_path} references {token}; route robot/navigation runtime "
                    "access through askme.ports + askme.providers or an approved "
                    "compat fallback file"
                )

    assert violations == []


def test_mcp_runtime_adapter_does_not_construct_provider_implementations() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "askme" / "mcp" / "runtime_adapter.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = (
        "askme.providers",
        "askme.robot",
        "askme.voice.input",
        "askme.voice.output",
        "askme.perception",
        "cv2",
        "numpy",
        "sounddevice",
    )
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if any(
                module_name == item or module_name.startswith(f"{item}.")
                for item in forbidden
            ):
                violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == item or alias.name.startswith(f"{item}.")
                    for item in forbidden
                ):
                    violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    assert violations == []


def test_ports_do_not_depend_on_provider_implementations() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    port_root = repo_root / "askme" / "ports"
    forbidden = (
        "askme.providers",
        "askme.robot",
        "askme.runtime",
        "requests",
        "sounddevice",
        "cv2",
    )

    violations: list[str] = []
    for path in port_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if any(
                    module_name == item or module_name.startswith(f"{item}.")
                    for item in forbidden
                ):
                    violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == item or alias.name.startswith(f"{item}.")
                        for item in forbidden
                    ):
                        violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    assert violations == []


def test_interfaces_do_not_depend_on_lower_layer_implementations() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    interface_root = repo_root / "askme" / "interfaces"
    forbidden = (
        "askme.robot",
        "askme.voice",
        "askme.perception",
        "askme.runtime",
    )

    violations: list[str] = []
    for path in interface_root.rglob("*.py"):
        if path.name == "register_defaults.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if any(
                    module_name == item or module_name.startswith(f"{item}.")
                    for item in forbidden
                ):
                    violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == item or alias.name.startswith(f"{item}.")
                        for item in forbidden
                    ):
                        violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    assert violations == []


def test_interface_default_registration_delegates_concrete_backends() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    interface_path = repo_root / "askme" / "interfaces" / "register_defaults.py"
    provider_path = repo_root / "askme" / "providers" / "register_defaults.py"
    reaction_path = repo_root / "askme" / "pipeline" / "reactions" / "register_defaults.py"

    interface_imports: set[str] = set()
    tree = ast.parse(interface_path.read_text(encoding="utf-8"), filename=str(interface_path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            interface_imports.add(node.module)
        elif isinstance(node, ast.Import):
            interface_imports.update(alias.name for alias in node.names)

    forbidden_interface_imports = {
        "askme.llm.core.client",
        "askme.voice.input.asr",
        "askme.voice.input.cloud_asr",
        "askme.voice.output.tts",
        "askme.robot.telemetry.pulse",
        "askme.robot.telemetry.mock_pulse",
        "askme.perception.change_detector",
        "askme.pipeline.reactions.reaction_engine",
    }

    assert interface_imports.isdisjoint(forbidden_interface_imports)
    assert "askme.providers.register_defaults" in interface_imports
    assert "askme.pipeline.reactions.register_defaults" in interface_imports

    provider_source = provider_path.read_text(encoding="utf-8")
    reaction_source = reaction_path.read_text(encoding="utf-8")
    assert "askme.voice.input.asr" in provider_source
    assert "askme.voice.output.tts" in provider_source
    assert "askme.perception.change_detector" in provider_source
    assert "askme.robot.telemetry.pulse" in provider_source
    assert "askme.pipeline.reactions.reaction_engine" not in provider_source
    assert "askme.pipeline.reactions.reaction_engine" in reaction_source


def test_runtime_and_pipeline_use_ports_for_provider_backed_capabilities() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checked_files = [
        repo_root / "askme" / "runtime" / "modules" / "control_module.py",
        repo_root / "askme" / "runtime" / "modules" / "safety_module.py",
        repo_root / "askme" / "runtime" / "modules" / "led_module.py",
        repo_root / "askme" / "runtime" / "modules" / "perception_module.py",
        repo_root / "askme" / "runtime" / "modules" / "pulse_module.py",
        repo_root / "askme" / "runtime" / "modules" / "pipeline_module.py",
        repo_root / "askme" / "runtime" / "modules" / "proactive_module.py",
        repo_root / "askme" / "runtime" / "modules" / "reaction_module.py",
        repo_root / "askme" / "runtime" / "modules" / "text_module.py",
        repo_root / "askme" / "runtime" / "modules" / "voice_module.py",
        repo_root / "askme" / "agent_shell" / "thunder_agent_shell.py",
        repo_root / "askme" / "pipeline" / "channels" / "text_loop.py",
        repo_root / "askme" / "pipeline" / "channels" / "voice_loop.py",
        repo_root / "askme" / "pipeline" / "core" / "brain_pipeline.py",
        repo_root / "askme" / "pipeline" / "field" / "field_operations.py",
        repo_root / "askme" / "pipeline" / "core" / "prompt_builder.py",
        repo_root / "askme" / "pipeline" / "core" / "stream_processor.py",
        repo_root / "askme" / "pipeline" / "core" / "tool_executor.py",
        repo_root / "askme" / "pipeline" / "core" / "turn_executor.py",
        repo_root / "askme" / "pipeline" / "skills" / "skill_gate.py",
        repo_root / "askme" / "pipeline" / "skills" / "skill_dispatcher.py",
        repo_root / "askme" / "tools" / "robot" / "robot_tools.py",
        repo_root / "askme" / "tools" / "voice" / "voice_tools.py",
        repo_root / "askme" / "mcp" / "tools" / "robot_tools.py",
        repo_root / "askme" / "contracts" / "adapters.py",
    ]
    forbidden_imports = (
        "askme.robot.dog.control_client",
        "askme.robot.dog.safety_client",
        "askme.robot.indicators.led_controller",
        "askme.robot.indicators.state_led_bridge",
        "askme.robot.arm.arm_controller",
        "askme.robot.telemetry.pulse",
        "askme.robot.telemetry.mock_pulse",
        "askme.perception.vision_bridge",
        "askme.perception.interaction_provider",
        "askme.perception.change_detector",
        "askme.voice.input.asr_manager",
        "askme.voice.orchestration.audio_agent",
        "askme.voice.output.audio_router",
        "askme.voice.output.tts",
        "askme.voice.output.voice_profiles",
        "askme.voice_gateway.runtime_bridge",
        "askme.providers.voice_runtime",
        "askme.voice.input.address_detector",
        "askme.voice.interaction.interaction_gate",
        "askme.voice.interaction.perception_context",
    )

    violations: list[str] = []
    for path in checked_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name in forbidden_imports:
                    violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
                if module_name == "askme.voice_gateway" and any(
                    alias.name == "VoiceRuntimeBridge" for alias in node.names
                ):
                    violations.append(
                        f"{path.relative_to(repo_root)} imports VoiceRuntimeBridge from {module_name}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden_imports:
                        violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

        if "_vision_cfg" in path.read_text(encoding="utf-8"):
            violations.append(f"{path.relative_to(repo_root)} reads VisionBridge private config")

    assert violations == []


def test_dog_control_dispatch_execute_does_not_construct_provider() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "askme" / "tools" / "core" / "builtin_tools.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DogControlDispatchTool":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "execute":
                    for child in ast.walk(item):
                        if isinstance(child, ast.ImportFrom):
                            module_name = child.module or ""
                            if module_name == "askme.providers":
                                violations.append(
                                    f"{path.relative_to(repo_root)} imports {module_name} in execute"
                                )
                        elif isinstance(child, ast.Call):
                            func = child.func
                            if isinstance(func, ast.Name) and func.id == "build_robot_control":
                                violations.append(
                                    f"{path.relative_to(repo_root)} calls build_robot_control in execute"
                                )

    assert violations == []


def test_scan_tool_does_not_reuse_move_tool_private_runtime_helper() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "askme" / "tools" / "spatial" / "scan_tool.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name in {"askme.tools.move_tool", "askme.tools.robot.move_tool"}:
                violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"askme.tools.move_tool", "askme.tools.robot.move_tool"}:
                    violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    assert violations == []


def test_tools_runtime_http_direct_calls_stay_in_authorized_modules() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tools_root = repo_root / "askme" / "tools"
    authorized_http_modules = {
        repo_root / "askme" / "tools" / "core" / "builtin_tools.py",
        repo_root / "askme" / "tools" / "robot" / "robot_api_tool.py",
        repo_root / "askme" / "tools" / "robot" / "runtime_api.py",
        repo_root / "askme" / "tools" / "spatial" / "temporal_query_tool.py",
    }
    forbidden_import_roots = {"urllib.request", "requests", "httpx", "aiohttp"}
    forbidden_call_roots = {"requests", "httpx", "aiohttp"}
    urllib_request_names = {"Request", "urlopen"}

    violations: list[str] = []
    for path in tools_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                imports_direct_http = module_name in forbidden_import_roots
            elif isinstance(node, ast.Import):
                imports_direct_http = any(
                    alias.name == item or alias.name.startswith(f"{item}.")
                    for alias in node.names
                    for item in forbidden_import_roots
                )
            else:
                imports_direct_http = False

            if imports_direct_http and path not in authorized_http_modules:
                violations.append(
                    f"{path.relative_to(repo_root)} imports direct HTTP client at line {node.lineno}"
                )

            if isinstance(node, ast.Call):
                func = node.func
                uses_direct_http = False
                if isinstance(func, ast.Attribute):
                    value = func.value
                    if isinstance(value, ast.Attribute):
                        uses_direct_http = (
                            isinstance(value.value, ast.Name)
                            and value.value.id == "urllib"
                            and value.attr == "request"
                            and func.attr in urllib_request_names
                        )
                    elif isinstance(value, ast.Name):
                        uses_direct_http = (
                            value.id in forbidden_call_roots
                            or (value.id == "request" and func.attr in urllib_request_names)
                        )
                elif isinstance(func, ast.Name):
                    uses_direct_http = func.id in urllib_request_names

                if uses_direct_http and path not in authorized_http_modules:
                    violations.append(
                        f"{path.relative_to(repo_root)} calls direct HTTP client at line {node.lineno}"
                    )

    assert violations == []


def test_shared_ota_metrics_do_not_depend_on_robot_package() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checked_roots = [
        repo_root / "askme" / "runtime",
        repo_root / "askme" / "llm",
        repo_root / "askme" / "voice" / "orchestration",
        repo_root / "askme" / "memory" / "core",
        repo_root / "askme" / "skills" / "core",
    ]
    forbidden = {
        "askme.robot.ota_bridge",
        "askme.robot.telemetry.ota_bridge",
    }

    violations: list[str] = []
    for root in checked_roots:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    if module_name in forbidden:
                        violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in forbidden:
                            violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    assert violations == []


def test_mcp_arm_wiring_uses_provider_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checked_files = [
        repo_root / "askme" / "mcp" / "context.py",
        repo_root / "askme" / "mcp" / "server.py",
        repo_root / "askme" / "mcp" / "tools" / "robot_tools.py",
    ]
    forbidden = {
        "askme.robot.arm.arm_controller",
        "askme.robot.arm_controller",
    }
    provider_import_found = False
    violations: list[str] = []

    for path in checked_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name in forbidden:
                    violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
                if module_name == "askme.providers" and any(
                    alias.name == "build_arm_control" for alias in node.names
                ):
                    provider_import_found = True
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden:
                        violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    assert violations == []
    assert provider_import_found


def test_mcp_robot_resources_delegate_dependencies_to_resource_surface() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    resource_paths = [
        repo_root / "askme" / "mcp" / "resources" / "health_resources.py",
        repo_root / "askme" / "mcp" / "resources" / "perception_resources.py",
        repo_root / "askme" / "mcp" / "resources" / "robot_resources.py",
        repo_root / "askme" / "mcp" / "resources" / "skill_resources.py",
    ]
    forbidden = {
        "askme.config",
        "askme.providers",
        "askme.skills.core.skill_manager",
    }
    surface_path = repo_root / "askme" / "mcp" / "resource_surface.py"
    surface_imports: set[tuple[str, str]] = set()
    violations: list[str] = []

    for path in resource_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if any(
                    module_name == item or module_name.startswith(f"{item}.")
                    for item in forbidden
                ):
                    violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if any(
                        alias.name == item or alias.name.startswith(f"{item}.")
                        for item in forbidden
                    ):
                        violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    surface_tree = ast.parse(surface_path.read_text(encoding="utf-8"), filename=str(surface_path))
    for node in ast.walk(surface_tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            surface_imports.update((module_name, alias.name) for alias in node.names)

    assert violations == []
    assert ("askme.config", "get_config") in surface_imports
    assert ("askme.config", "get_section") in surface_imports
    assert ("askme.providers", "get_arm_safety_defaults") in surface_imports
    assert ("askme.skills.core.skill_manager", "SkillManager") in surface_imports


def test_mcp_voice_and_perception_wiring_use_provider_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checked_files = [
        repo_root / "askme" / "mcp" / "context.py",
        repo_root / "askme" / "mcp" / "server.py",
        repo_root / "askme" / "mcp" / "tools" / "voice_tools.py",
    ]
    forbidden = {
        "askme.perception.scene_intelligence",
        "askme.perception.vision_bridge",
        "askme.voice.input.asr",
        "askme.voice.input.mic_input",
        "askme.voice.input.vad",
        "askme.voice.output.tts",
    }
    provider_imports: set[str] = set()
    violations: list[str] = []

    for path in checked_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name in forbidden:
                    violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
                if module_name == "askme.providers":
                    provider_imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in forbidden:
                        violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    assert violations == []
    assert {
        "build_edge_voice_io",
        "build_perception",
        "build_scene_intelligence",
    } <= provider_imports


def test_cli_voice_entry_points_use_provider_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "askme" / "cli.py"
    forbidden = {
        "askme.voice.orchestration.audio_agent",
        "askme.voice.output.tts",
    }
    provider_imports: set[str] = set()
    violations: list[str] = []

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name in forbidden:
                violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
            if module_name == "askme.providers":
                provider_imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in forbidden:
                    violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")

    assert violations == []
    assert {"build_audio_frontend", "build_tts_provider"} <= provider_imports


def test_health_server_vision_code_uses_provider_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "askme" / "health_server.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    provider_imports: set[str] = set()
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"cv2", "numpy"}:
                    violations.append(f"{path.relative_to(repo_root)} imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name in {"cv2", "numpy"}:
                violations.append(f"{path.relative_to(repo_root)} imports {module_name}")
            if module_name == "askme.providers":
                provider_imports.update(alias.name for alias in node.names)

    assert violations == []
    assert {"analyze_image_base64", "capture_snapshot_payload"} <= provider_imports


def test_mcp_perception_resources_keep_depth_codec_in_provider() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    perception_path = repo_root / "askme" / "mcp" / "resources" / "perception_resources.py"
    surface_path = repo_root / "askme" / "mcp" / "resource_surface.py"
    perception_tree = ast.parse(
        perception_path.read_text(encoding="utf-8"),
        filename=str(perception_path),
    )
    surface_tree = ast.parse(surface_path.read_text(encoding="utf-8"), filename=str(surface_path))
    provider_import_found = False
    violations: list[str] = []

    for node in ast.walk(perception_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"numpy", "askme.providers"}:
                    violations.append(
                        f"{perception_path.relative_to(repo_root)} imports {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name in {"numpy", "askme.providers"}:
                violations.append(
                    f"{perception_path.relative_to(repo_root)} imports {module_name}"
                )

    for node in ast.walk(surface_tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name == "askme.providers" and any(
                alias.name == "read_depth_info" for alias in node.names
            ):
                provider_import_found = True

    assert violations == []
    assert provider_import_found
