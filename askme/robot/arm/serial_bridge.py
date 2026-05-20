"""
Serial bridge for robot communication.

Sends 16-dim action vectors to the robot over a serial (UART) connection,
and reads state feedback. Supports a simulation mode that logs actions
to the console instead of requiring real hardware.
"""

from __future__ import annotations

import logging
import struct
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SerialBridge:
    """Communicate with the robot arm over serial or in simulation mode."""

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 115200,
        simulate: bool = True,
    ) -> None:
        """
        Args:
            port: Serial port name (e.g. 'COM3' on Windows, '/dev/ttyUSB0' on Linux).
            baudrate: Serial baud rate.
            simulate: If True, no real hardware connection is made; actions are logged.
        """
        self._port = port
        self._baudrate = baudrate
        self._simulate = simulate
        self._serial: Any | None = None
        self._connected = False

        # Simulated state tracking
        self._sim_state: np.ndarray = np.zeros(16, dtype=np.float32)

    @property
    def is_connected(self) -> bool:
        """Whether the bridge has an active connection (or is simulating)."""
        return self._connected

    def connect(self) -> bool:
        """Open the serial connection.

        Returns:
            True if connected successfully (or simulation mode active).
        """
        if self._simulate:
            logger.info("[SIM] Serial bridge connected (simulation mode)")
            self._connected = True
            return True

        try:
            import serial  # pyserial
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=1.0,
            )
            self._connected = True
            logger.info("Serial bridge connected: %s @ %d", self._port, self._baudrate)
            return True
        except ImportError:
            logger.error("pyserial is not installed. Cannot connect to robot.")
            return False
        except Exception as exc:
            logger.error("Serial connection failed: %s", exc)
            return False

    def disconnect(self) -> None:
        """Close the serial connection."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        self._connected = False
        logger.info("Serial bridge disconnected.")

    def send_action(self, action: np.ndarray) -> bool:
        """Send a 16-dim action vector to the robot.

        The action is packed as 16 float32 values (64 bytes) with a
        header byte (0xAA) and a simple checksum byte.

        Args:
            action: A numpy array of shape (16,).

        Returns:
            True if sent successfully.
        """
        if not self._connected:
            logger.warning("Cannot send action: not connected.")
            return False

        action = action.astype(np.float32)
        if action.shape[0] != 16:
            logger.error("Action must be 16-dim, got %d", action.shape[0])
            return False

        if self._simulate:
            self._sim_state = action.copy()
            angles_str = ", ".join(f"{a:.3f}" for a in action[:6])
            grip_str = ", ".join(f"{a:.3f}" for a in action[6:10])
            logger.info("[SIM] Action -> arm=[%s] grip=[%s]", angles_str, grip_str)
            return True

        # Pack: header(1) + 16 floats(64) + checksum(1) = 66 bytes
        try:
            payload = struct.pack("<16f", *action)
            checksum = sum(payload) & 0xFF
            packet = bytes([0xAA]) + payload + bytes([checksum])
            self._serial.write(packet)
            return True
        except Exception as exc:
            logger.error("Failed to send action: %s", exc)
            return False

    def get_state(self) -> np.ndarray | None:
        """Read the current robot state from serial.

        Returns:
            A 16-dim numpy array of joint angles, or None on failure.
        """
        if not self._connected:
            return None

        if self._simulate:
            return self._sim_state.copy()

        try:
            # Request state: send 0xBB command byte
            self._serial.write(bytes([0xBB]))
            # Read response: header(1) + 16 floats(64) + checksum(1)
            data = self._serial.read(66)
            if len(data) < 66 or data[0] != 0xBB:
                return None
            payload = data[1:65]
            state = np.array(struct.unpack("<16f", payload), dtype=np.float32)
            return state
        except Exception as exc:
            logger.error("Failed to read state: %s", exc)
            return None
