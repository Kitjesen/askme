"""Contract tests for the MemPalace smoke-test script."""

from unittest.mock import patch

import pytest

from scripts.dev.mempalace_smoke import build_parser, run_smoke


@pytest.mark.asyncio
async def test_smoke_reports_missing_package(tmp_path):
    args = build_parser().parse_args([
        "--palace",
        str(tmp_path / "palace"),
        "--data-dir",
        str(tmp_path / "data"),
    ])

    with patch("scripts.dev.mempalace_smoke.importlib.util.find_spec", return_value=None):
        result = await run_smoke(args)

    assert result["ok"] is False
    assert result["code"] == "mempalace_not_installed"
    assert result["installed"] is False
    assert "pip install" in result["message"]
