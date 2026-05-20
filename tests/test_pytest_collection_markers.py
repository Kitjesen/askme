from __future__ import annotations

from pathlib import Path

from tests.conftest import _pytest_marker_names_for_path


def test_marks_scenario_tests_as_slow_scenario_shard() -> None:
    markers = _pytest_marker_names_for_path(
        Path("tests/scenario_tests/test_voice_e2e_evaluation.py")
    )

    assert markers == {"e2e", "scenario", "slow"}


def test_marks_e2e_test_files_as_slow_e2e_shard() -> None:
    markers = _pytest_marker_names_for_path(Path("tests/test_agent_task_e2e.py"))

    assert markers == {"e2e", "slow"}


def test_marks_benchmark_test_files_as_slow_benchmark_shard() -> None:
    markers = _pytest_marker_names_for_path(Path("tests/test_performance_benchmarks.py"))

    assert markers == {"benchmark", "slow"}


def test_leaves_regular_unit_tests_unmarked() -> None:
    markers = _pytest_marker_names_for_path(Path("tests/test_scripts_structure.py"))

    assert markers == set()


def test_docs_document_copyable_pytest_partition_commands() -> None:
    text = Path("docs/README.md").read_text(encoding="utf-8")

    for command in (
        'python -m pytest tests -q',
        'python -m pytest tests -m "slow" -q',
        'python -m pytest tests -m "scenario" -q',
        'python -m pytest tests -m "e2e" -q',
        'python -m pytest tests -m "benchmark" -q',
        'python -m pytest tests -m "e2e or benchmark" -q',
    ):
        assert command in text
