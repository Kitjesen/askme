from __future__ import annotations

import ast
from pathlib import Path

from askme.robot_interaction import (
    AddressDetector,
    IntentRouter,
    InteractionGate,
    RobotInteractionService,
)
from askme.voice_gateway import (
    ConversationSessionManager,
    VoiceGatewayService,
)

ROOT = Path(__file__).resolve().parents[1]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _assert_no_forbidden_imports(
    package: str,
    forbidden_roots: set[str],
    *,
    allow: dict[str, set[str]] | None = None,
) -> None:
    package_dir = ROOT / package.replace(".", "/")
    allowed_by_file = allow or {}
    failures: list[str] = []
    for path in package_dir.glob("*.py"):
        imports = _imported_modules(path)
        allowed = allowed_by_file.get(path.name, set())
        for module in sorted(imports):
            if module in allowed:
                continue
            if any(module == root or module.startswith(f"{root}.") for root in forbidden_roots):
                failures.append(f"{path.relative_to(ROOT)} imports {module}")

    assert failures == []


def test_voice_gateway_does_not_import_interaction_or_execution_layers() -> None:
    _assert_no_forbidden_imports(
        "askme.voice_gateway",
        {
            "askme.api",
            "askme.mcp",
            "askme.pipeline",
            "askme.providers",
            "askme.robot",
            "askme.robot_interaction",
            "askme.runtime",
            "askme.tools",
        },
        allow={"runtime_bridge.py": {"askme.providers"}},
    )


def test_robot_interaction_does_not_import_voice_gateway_or_execution_layers() -> None:
    _assert_no_forbidden_imports(
        "askme.robot_interaction",
        {
            "askme.api",
            "askme.mcp",
            "askme.pipeline",
            "askme.providers",
            "askme.robot",
            "askme.runtime",
            "askme.tools",
            "askme.voice",
            "askme.voice_gateway",
        },
    )


def test_package_facades_expose_boundary_services() -> None:
    assert VoiceGatewayService.__name__ == "VoiceGatewayService"
    assert ConversationSessionManager.__name__ == "ConversationSessionManager"
    assert AddressDetector.__name__ == "AddressDetector"
    assert InteractionGate.__name__ == "InteractionGate"
    assert IntentRouter.__name__ == "IntentRouter"
    assert RobotInteractionService.__name__ == "RobotInteractionService"
