"""Tests for PolicyRunner — model loading, is_loaded, infer, close."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from askme.robot.policy_runner import PolicyRunner


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_runner_with_mock_session(action_dim: int = 16) -> PolicyRunner:
    """Create PolicyRunner with a mocked onnxruntime session."""
    runner = PolicyRunner.__new__(PolicyRunner)
    runner.model_path = Path("/fake/model.onnx")
    runner.obs_dim = 285
    runner.action_dim = action_dim

    # Build mock session with properly set .name attributes on inputs/outputs
    mock_input = MagicMock()
    mock_input.name = "input_obs"
    mock_output = MagicMock()
    mock_output.name = "output_action"

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.run.return_value = [np.zeros((1, action_dim), dtype=np.float32)]
    runner._session = mock_session
    return runner


# ── Init — model loading ──────────────────────────────────────────────────────

class TestModelLoading:
    def test_is_loaded_false_when_onnx_not_available(self, tmp_path):
        with patch.dict("sys.modules", {"onnxruntime": None}):
            runner = PolicyRunner(str(tmp_path / "model.onnx"))
        assert runner.is_loaded is False

    def test_is_loaded_false_when_file_not_found(self, tmp_path):
        runner = PolicyRunner(str(tmp_path / "nonexistent.onnx"))
        assert runner.is_loaded is False

    def test_is_loaded_true_with_valid_session(self):
        runner = _make_runner_with_mock_session()
        assert runner.is_loaded is True

    def test_model_path_stored(self, tmp_path):
        runner = PolicyRunner(str(tmp_path / "model.onnx"))
        assert runner.model_path == tmp_path / "model.onnx"

    def test_obs_dim_stored(self, tmp_path):
        runner = PolicyRunner(str(tmp_path / "model.onnx"), obs_dim=512)
        assert runner.obs_dim == 512

    def test_action_dim_stored(self, tmp_path):
        runner = PolicyRunner(str(tmp_path / "model.onnx"), action_dim=8)
        assert runner.action_dim == 8


# ── infer ─────────────────────────────────────────────────────────────────────

class TestInfer:
    def test_raises_when_not_loaded(self):
        runner = PolicyRunner.__new__(PolicyRunner)
        runner._session = None
        runner.obs_dim = 285
        runner.action_dim = 16
        with pytest.raises(RuntimeError, match="not loaded"):
            runner.infer(np.zeros(285))

    def test_1d_input_reshaped_to_2d(self):
        runner = _make_runner_with_mock_session()
        runner.infer(np.zeros(285))  # 1D input
        called_input = runner._session.run.call_args[0][1]["input_obs"]
        assert called_input.ndim == 2

    def test_returns_action_dim_array(self):
        runner = _make_runner_with_mock_session(action_dim=16)
        result = runner.infer(np.zeros((1, 285)))
        assert result.shape == (16,)

    def test_trims_oversized_output(self):
        runner = _make_runner_with_mock_session(action_dim=4)
        runner._session.run.return_value = [np.ones((1, 10), dtype=np.float32)]
        result = runner.infer(np.zeros((1, 285)))
        assert result.shape == (4,)

    def test_pads_undersized_output(self):
        runner = _make_runner_with_mock_session(action_dim=8)
        runner._session.run.return_value = [np.ones((1, 4), dtype=np.float32)]
        result = runner.infer(np.zeros((1, 285)))
        assert result.shape == (8,)

    def test_input_cast_to_float32(self):
        runner = _make_runner_with_mock_session()
        runner.infer(np.zeros(285, dtype=np.float64))
        called_input = runner._session.run.call_args[0][1]["input_obs"]
        assert called_input.dtype == np.float32

    def test_2d_input_not_reshaped(self):
        runner = _make_runner_with_mock_session()
        runner.infer(np.zeros((1, 285)))
        called_input = runner._session.run.call_args[0][1]["input_obs"]
        assert called_input.shape == (1, 285)


# ── close ─────────────────────────────────────────────────────────────────────

class TestClose:
    def test_close_clears_session(self):
        runner = _make_runner_with_mock_session()
        runner.close()
        assert runner._session is None
        assert runner.is_loaded is False

    def test_close_when_no_session_no_crash(self):
        runner = PolicyRunner.__new__(PolicyRunner)
        runner._session = None
        runner.obs_dim = 285
        runner.action_dim = 16
        runner.close()  # should not raise
