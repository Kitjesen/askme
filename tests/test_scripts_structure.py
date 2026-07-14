from __future__ import annotations

import re
from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib

SCRIPTS_ROOT = Path("scripts")

ALLOWED_ROOT_FILES = {
    "README.md",
    "__init__.py",
    "benchmark_audit_query.py",
    "benchmark_core_paths.py",
    "check_perf_thresholds.py",
    "check_text_encoding.py",
    "zeroclaw_bridge.py",
}


def test_scripts_root_stays_quiet() -> None:
    root_files = {path.name for path in SCRIPTS_ROOT.iterdir() if path.is_file()}

    assert root_files == ALLOWED_ROOT_FILES


def test_scripts_readme_documents_every_top_level_bucket() -> None:
    readme = (SCRIPTS_ROOT / "README.md").read_text(encoding="utf-8")
    documented = set(re.findall(r"`scripts/([^`/]+)/`", readme))
    actual = {
        path.name
        for path in SCRIPTS_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__"
    }

    assert actual <= documented
    assert documented <= actual


def test_scripts_readme_documents_root_files_and_manual_collection_guard() -> None:
    readme = (SCRIPTS_ROOT / "README.md").read_text(encoding="utf-8")

    for filename in sorted(ALLOWED_ROOT_FILES - {"README.md"}):
        assert f"`{filename}`" in readme
    assert "manual" in readme.lower()
    assert "testpaths" in readme


def test_pytest_collection_stays_limited_to_tests_directory() -> None:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
