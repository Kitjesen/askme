"""LED state controller for Thunder robot.

Defines the LedStateKind enum and LedController protocol.
Two concrete implementations are provided:

- NullLedController: no-op (default, safe for dev/test)
- HttpLedController: POSTs to dog-control-service when the LED API is defined

Usage
-----
Wire a controller via config:

  led_controller = HttpLedController(base_url="http://localhost:5080")
  # or the safe default:
  led_controller = NullLedController()

Then pass it to StateLedBridge which polls the system state and calls
led_controller.set_state() when the resolved LED state changes.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Protocol

logger = logging.getLogger(__name__)


class LedStateKind(Enum):
    """Visual state of the Thunder status LED.

    Priority (highest first) when multiple conditions are true:
      ESTOP > MUTED > AGENT_TASK > SPEAKING > PROCESSING > LISTENING > IDLE
    """

    IDLE = "idle"             # green slow blink  — waiting for wake word
    LISTENING = "listening"   # blue fast blink   — active listening
    PROCESSING = "processing" # yellow breathing  — LLM thinking
    SPEAKING = "speaking"     # blue constant     — TTS playback
    MUTED = "muted"           # white constant    — mic muted
    AGENT_TASK = "agent_task" # orange breathing  — long background task
    ESTOP = "estop"           # red constant      — emergency stop


class LedController(Protocol):
    """Protocol that all LED controller implementations must satisfy."""

    def set_state(self, state: LedStateKind) -> None:
        """Drive the hardware/API to the given LED state.

        Must be non-blocking and safe to call from any thread.
        Implementations should log errors and swallow exceptions — a failed
        LED update must never affect the robot control path.
        """
        ...


class NullLedController:
    """No-op LED controller.  Default when no hardware interface is configured."""

    def set_state(self, state: LedStateKind) -> None:
        logger.debug("[LED] %s (NullLedController — no hardware)", state.value)


class HttpLedController:
    """LED controller that POSTs to dog-control-service.

    Ready for when the runtime contracts define an LED endpoint.
    Endpoint shape (assumed):
      POST /api/v1/control/led  {"state": "<LedStateKind.value>"}

    Falls back silently when the service is not reachable — a missing LED
    update is never a critical failure.
    """

    _PATH = "/api/v1/control/led"

    def __init__(self, base_url: str, timeout: float = 1.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def set_state(self, state: LedStateKind) -> None:
        import json
        import threading
        import urllib.request

        def _post() -> None:
            try:
                url = f"{self._base_url}{self._PATH}"
                body = json.dumps({"state": state.value}).encode()
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=self._timeout)
                logger.debug("[LED] HTTP set_state=%s → %s", state.value, url)
            except Exception as exc:
                logger.debug("[LED] HTTP set_state failed (non-critical): %s", exc)

        # Fire-and-forget in a daemon thread — never block the caller
        threading.Thread(target=_post, daemon=True, name="led-http").start()
