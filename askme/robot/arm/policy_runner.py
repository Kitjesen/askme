"""
Policy runner for askme robot control.

Loads an ONNX reinforcement-learning policy model and runs inference
to produce continuous joint actions from observation histories.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PolicyRunner:
    """Load and run an ONNX policy model for robot control.

    The policy takes an observation history (flattened) and produces
    a continuous action vector (joint angle deltas or targets).
    """

    def __init__(
        self,
        model_path: str | Path,
        obs_dim: int = 285,
        action_dim: int = 16,
    ) -> None:
        """
        Args:
            model_path: Path to the ONNX model file.
            obs_dim: Observation vector dimension.
            action_dim: Action vector dimension.
        """
        self.model_path = Path(model_path)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._session: Any | None = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the ONNX model via onnxruntime."""
        try:
            import onnxruntime as ort

            if not self.model_path.is_file():
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            # Prefer CPU to keep robot control deterministic
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            input_info = self._session.get_inputs()
            output_info = self._session.get_outputs()
            logger.info(
                "PolicyRunner loaded: %s (inputs=%d, outputs=%d)",
                self.model_path.name,
                len(input_info),
                len(output_info),
            )
        except ImportError:
            logger.error("onnxruntime is not installed. Policy inference unavailable.")
            self._session = None
        except Exception as exc:
            logger.error("Failed to load policy model: %s", exc)
            self._session = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model was loaded successfully."""
        return self._session is not None

    def infer(self, obs_history: np.ndarray) -> np.ndarray:
        """Run inference on an observation history.

        Args:
            obs_history: A numpy array of shape (1, obs_dim) or (obs_dim,).

        Returns:
            Action vector of shape (action_dim,).

        Raises:
            RuntimeError: If the model is not loaded.
        """
        if self._session is None:
            raise RuntimeError("Policy model is not loaded.")

        # Ensure batch dimension
        if obs_history.ndim == 1:
            obs_history = obs_history.reshape(1, -1)

        obs_history = obs_history.astype(np.float32)

        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name

        results = self._session.run(
            [output_name],
            {input_name: obs_history},
        )

        action = results[0].flatten()

        # Trim or pad to expected action_dim
        if action.shape[0] > self.action_dim:
            action = action[: self.action_dim]
        elif action.shape[0] < self.action_dim:
            action = np.pad(action, (0, self.action_dim - action.shape[0]))

        return action

    def close(self) -> None:
        """Release the ONNX runtime session."""
        if self._session is not None:
            # onnxruntime sessions don't have an explicit close,
            # but we clear the reference for GC.
            self._session = None
            logger.info("PolicyRunner session released.")
