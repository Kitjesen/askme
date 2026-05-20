from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_repository_layout_doc_tracks_confusing_roots() -> None:
    text = (ROOT / "docs" / "REPOSITORY_LAYOUT.md").read_text(encoding="utf-8")

    assert "Same-level folders do not mean same-level architecture authority" in text
    for path in (
        "askme/",
        "scripts/",
        "prompts/",
        "data/",
        "models/",
        "artifacts/",
    ):
        assert f"`{path}`" in text


def test_askme_package_readme_classifies_every_top_level_package_dir() -> None:
    package_root = ROOT / "askme"
    text = (package_root / "README.md").read_text(encoding="utf-8")

    ignored = {"__pycache__"}
    directories = {
        path.name
        for path in package_root.iterdir()
        if path.is_dir() and path.name not in ignored
    }

    missing = sorted(name for name in directories if f"`{name}/`" not in text)

    assert "Product Composition" in text
    assert "Voice And Interaction" in text
    assert "Contracts And Boundaries" in text
    assert "Provider And Edge Implementations" in text
    assert "External Surfaces" in text
    assert missing == []


def test_compatibility_and_parking_dirs_are_marked_as_not_new_code_homes() -> None:
    text = (ROOT / "askme" / "README.md").read_text(encoding="utf-8")

    for marker in (
        "`compat/` | compatibility only",
        "`interaction/` | compatibility only",
        "`data/` | parking only",
    ):
        assert marker in text


def test_legacy_interaction_package_stays_facade_only() -> None:
    interaction_root = ROOT / "askme" / "interaction"
    violations: list[str] = []

    for path in sorted(interaction_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            if isinstance(node, ast.ImportFrom) and (
                node.module == "askme.robot_interaction"
                or node.module.startswith("askme.robot_interaction.")
            ):
                continue
            if isinstance(node, ast.Assign) and all(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                continue
            violations.append(f"{path.relative_to(ROOT)} contains {type(node).__name__}")

    assert violations == []


def test_non_compat_code_does_not_import_legacy_interaction_facade() -> None:
    allowed = {
        ROOT / "tests" / "test_package_migration_compat.py",
        ROOT / "tests" / "test_six_layer_package_boundaries.py",
    }
    violations: list[str] = []

    for root in (ROOT / "askme", ROOT / "tests"):
        for path in sorted(root.rglob("*.py")):
            if path in allowed or (ROOT / "askme" / "interaction") in path.parents:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    if module_name == "askme.interaction" or module_name.startswith(
                        "askme.interaction."
                    ):
                        violations.append(f"{path.relative_to(ROOT)} imports {module_name}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "askme.interaction" or alias.name.startswith(
                            "askme.interaction."
                        ):
                            violations.append(f"{path.relative_to(ROOT)} imports {alias.name}")

    assert violations == []


def test_non_compat_code_does_not_import_llm_intent_router_alias() -> None:
    allowed = {
        ROOT / "tests" / "test_package_migration_compat.py",
        ROOT / "tests" / "test_six_layer_package_boundaries.py",
    }
    violations: list[str] = []

    for root in (ROOT / "askme", ROOT / "scripts", ROOT / "tests"):
        for path in sorted(root.rglob("*.py")):
            if path in allowed:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    if module_name == "askme.llm.intent_router":
                        violations.append(f"{path.relative_to(ROOT)} imports {module_name}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "askme.llm.intent_router":
                            violations.append(f"{path.relative_to(ROOT)} imports {alias.name}")

    assert violations == []


def test_prompt_assets_stay_outside_prompt_registry_package() -> None:
    package_prompt_docs = {path.name for path in (ROOT / "askme" / "prompts").glob("*.md")}

    assert (ROOT / "prompts" / "SOUL.md").is_file()
    assert package_prompt_docs <= {"README.md"}


def test_package_local_data_is_not_importable_runtime_state() -> None:
    package_data = ROOT / "askme" / "data"

    assert not (package_data / "__init__.py").exists()
    assert list(package_data.glob("*.py")) == []
    assert (ROOT / "data").is_dir()


def test_cognition_owner_subpackages_are_not_empty_facades() -> None:
    cognition_root = ROOT / "askme" / "cognition"
    expected = {
        "memory": {"working_memory.py"},
        "perception": {"active_perception.py", "perception_sync.py"},
        "planning": {"planner.py", "planning_session.py"},
        "world": {"world_state.py"},
    }

    for package, files in expected.items():
        existing = {path.name for path in (cognition_root / package).glob("*.py")}
        assert files <= existing


def test_cognition_legacy_root_modules_alias_owner_subpackages() -> None:
    import importlib

    pairs = {
        "askme.cognition.active_perception": "askme.cognition.perception.active_perception",
        "askme.cognition.perception_sync": "askme.cognition.perception.perception_sync",
        "askme.cognition.planner": "askme.cognition.planning.planner",
        "askme.cognition.planning_session": "askme.cognition.planning.planning_session",
        "askme.cognition.working_memory": "askme.cognition.memory.working_memory",
        "askme.cognition.world_state": "askme.cognition.world.world_state",
    }

    for legacy, owner in pairs.items():
        assert importlib.import_module(legacy) is importlib.import_module(owner)
