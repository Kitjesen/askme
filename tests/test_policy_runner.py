"""Tests for PolicyRunner — ONNX policy inference for robot control."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from askme.robot.policy_runner import PolicyRunner


def _make_runner(session=None, model_path="fake.onnx"):
    """Build a PolicyRunner with a mocked ONNX session."""
    runner = PolicyRunner.__new__(PolicyRunner)
    runner.model_path = Path(model_path)
    runner.obs_dim = 285
    runner.action_dim = 16
    runner._session = session
    return runner


class TestInit:
    def test_not_loaded_when_model_missing(self, tmp_path):
        runner = PolicyRunner.__new__(PolicyRunner)
        runner.model_path = tmp_path / "missing.onnx"
        runner.obs_dim = 285
        runner.action_dim = 16
        runner._session = None
        runner._load_model()
        assert runner.is_loaded is False

    def test_not_loaded_when_onnxruntime_unavailable(self, tmp_path):
        model = tmp_path / "model.onnx"
        model.write_bytes(b"fake")
        runner = PolicyRunner.__new__(PolicyRunner)
        runner.model_path = model
        runner.obs_dim = 285
        runner.action_dim = 16
        runner._session = None
        with patch.dict("sys.modules", {"onnxruntime": None}):
            runner._load_model()
        assert runner.is_loaded is False

    def test_is_loaded_when_session_set(self):
        runner = _make_runner(session=MagicMock())
        assert runner.is_loaded is True

    def test_is_not_loaded_when_session_none(self):
        runner = _make_runner(session=None)
        assert runner.is_loaded is False


class TestInfer:
    def _make_session(self, output: np.ndarray) -> MagicMock:
        """Build a mock session that returns the given output array."""
        session = MagicMock()
        inp = MagicMock()
        inp.name = "obs"
        out = MagicMock()
        out.name = "action"
        session.get_inputs.return_value = [inp]
        session.get_outputs.return_value = [out]
        session.run.return_value = [output]
        return session

    def test_raises_when_not_loaded(self):
        runner = _make_runner(session=None)
        with pytest.raises(RuntimeError, match="not loaded"):
            runner.infer(np.zeros(285, dtype=np.float32))

    def test_1d_input_reshaped(self):
        output = np.ones(16, dtype=np.float32)
        session = self._make_session(output)
        runner = _make_runner(session=session)
        runner.infer(np.zeros(285))
        # session.run should have been called with shape (1, 285)
        call_input = session.run.call_args[0][1]["obs"]
        assert call_input.shape == (1, 285)

    def test_2d_input_passed_as_is(self):
        output = np.ones(16, dtype=np.float32)
        session = self._make_session(output)
        runner = _make_runner(session=session)
        runner.infer(np.zeros((1, 285)))
        call_input = session.run.call_args[0][1]["obs"]
        assert call_input.shape == (1, 285)

    def test_output_trimmed_to_action_dim(self):
        output = np.arange(20, dtype=np.float32)  # 20 > 16
        session = self._make_session(output)
        runner = _make_runner(session=session)
        result = runner.infer(np.zeros(285))
        assert result.shape == (16,)
        assert result[0] == 0.0
        assert result[15] == 15.0

    def test_output_padded_to_action_dim(self):
        output = np.ones(10, dtype=np.float32)  # 10 < 16
        session = self._make_session(output)
        runner = _make_runner(session=session)
        result = runner.infer(np.zeros(285))
        assert result.shape == (16,)
        assert result[9] == 1.0
        assert result[10] == 0.0  # padded

    def test_output_exact_action_dim(self):
        output = np.full(16, 2.5, dtype=np.float32)
        session = self._make_session(output)
        runner = _make_runner(session=session)
        result = runner.infer(np.zeros(285))
        assert result.shape == (16,)
        assert np.allclose(result, 2.5)

    def test_input_cast_to_float32(self):
        output = np.ones(16, dtype=np.float32)
        session = self._make_session(output)
        runner = _make_runner(session=session)
        runner.infer(np.zeros(285, dtype=np.float64))
        call_input = session.run.call_args[0][1]["obs"]
        assert call_input.dtype == np.float32

    def test_returns_1d_array(self):
        output = np.ones(16, dtype=np.float32)
        session = self._make_session(output)
        runner = _make_runner(session=session)
        result = runner.infer(np.zeros(285))
        assert result.ndim == 1


class TestClose:
    def test_close_clears_session(self):
        runner = _make_runner(session=MagicMock())
        runner.close()
        assert runner._session is None
        assert runner.is_loaded is False

    def test_close_noop_when_not_loaded(self):
        runner = _make_runner(session=None)
        runner.close()  # should not raise
        assert runner._session is None
