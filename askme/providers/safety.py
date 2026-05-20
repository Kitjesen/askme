"""Robot safety provider adapters."""

from __future__ import annotations

from typing import Any

from askme.ports import SafetyPort
from askme.robot.dog.safety_client import DogSafetyClient


def build_safety(
    config: dict[str, Any] | None = None,
    *,
    pulse: Any = None,
) -> SafetyPort:
    """Build the configured robot-safety implementation."""
    return DogSafetyClient(config, pulse=pulse)


__all__ = ["build_safety"]
