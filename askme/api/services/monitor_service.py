"""Monitor payload helpers for dashboard-facing API routes."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ConfigProvider = Callable[[], dict[str, Any]]
ConversationProvider = Callable[[], list[dict[str, Any]]]
CommandRunner = Callable[[Sequence[str], float], Any]


class MonitorService:
    """Build product monitor payloads without coupling routes to local IO."""

    def __init__(
        self,
        *,
        config_provider: ConfigProvider,
        project_root: Path,
        conversation_provider: ConversationProvider | None = None,
        tmp_dir: Path | str = Path("/tmp"),
        knowledge_dir: Path | None = None,
        command_runner: CommandRunner | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config_provider = config_provider
        self._project_root = Path(project_root)
        self._conversation_provider = conversation_provider
        self._tmp_dir = Path(tmp_dir)
        self._knowledge_dir = knowledge_dir or self._project_root / "data" / "qp_memory" / "knowledge"
        self._command_runner = command_runner or _run_command
        self._logger = logger or logging.getLogger(__name__)

    def system_status_payload(self, *, now: float | None = None) -> dict[str, Any]:
        """Return the dashboard status payload used by ``/api/status``."""
        current_time = time.time() if now is None else float(now)
        status: dict[str, Any] = {"timestamp": current_time}
        status["perception"] = self._perception_payload(current_time)
        status["orbbec_camera"] = self._orbbec_camera_active()
        status["memory"] = self._memory_payload()
        return status

    def live_payload(self) -> dict[str, Any]:
        """Return in-memory conversation history for live dashboard updates."""
        if self._conversation_provider is None:
            return {"messages": [], "count": 0}
        messages = self._conversation_provider()
        return {"messages": messages, "count": len(messages)}

    def conversation_history_payload(
        self,
        *,
        conversation_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Return persisted conversation history from configured storage."""
        cfg = self._config_provider()
        conversation_cfg = cfg.get("conversation", {}) if isinstance(cfg, dict) else {}
        raw_path = (
            conversation_cfg.get("history_file", "data/conversation_history.json")
            if isinstance(conversation_cfg, dict)
            else "data/conversation_history.json"
        )
        history_file = Path(str(raw_path))
        if not history_file.is_absolute():
            history_file = self._project_root / history_file
        if history_file.exists():
            history = json.loads(history_file.read_text(encoding="utf-8"))
        else:
            history = []

        if isinstance(history, list):
            return {"messages": history, "count": len(history)}

        sessions = history.get("sessions") if isinstance(history, dict) else None
        if not isinstance(sessions, dict):
            return {"messages": [], "count": 0}

        session_id = str(conversation_session_id or "").strip()
        messages = sessions.get(session_id, [])
        if not isinstance(messages, list):
            messages = []
        available_session_ids = sorted(
            str(candidate)
            for candidate, candidate_messages in sessions.items()
            if str(candidate) and isinstance(candidate_messages, list)
        )
        payload: dict[str, Any] = {
            "messages": messages,
            "count": len(messages),
            "available_session_ids": available_session_ids,
        }
        if session_id:
            payload["conversation_session_id"] = session_id
        return payload

    def _perception_payload(self, now: float) -> dict[str, Any]:
        perception: dict[str, Any] = {}
        heartbeat_path = self._tmp_dir / "askme_frame_daemon.heartbeat"
        try:
            heartbeat = float(heartbeat_path.read_text(encoding="utf-8").strip())
            perception["frame_daemon"] = {
                "alive": now - heartbeat < 3.0,
                "age_s": round(now - heartbeat, 1),
            }
        except (FileNotFoundError, OSError, ValueError):
            perception["frame_daemon"] = {"alive": False}

        detections_path = self._tmp_dir / "askme_frame_detections.json"
        try:
            detections_payload = json.loads(detections_path.read_text(encoding="utf-8"))
            detections = detections_payload.get("detections", [])
            perception["detections"] = {
                "count": len(detections),
                "infer_ms": detections_payload.get("infer_ms", 0),
                "objects": [item["class_id"] for item in detections],
            }
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            perception["detections"] = {"count": 0}

        events_path = self._tmp_dir / "askme_events.jsonl"
        try:
            if events_path.exists():
                lines = events_path.read_text(encoding="utf-8").splitlines()
                perception["change_events"] = {"total": len(lines)}
                if lines:
                    perception["change_events"]["last"] = json.loads(lines[-1].strip())
            else:
                perception["change_events"] = {"total": 0}
        except Exception as exc:
            self._logger.debug("change-event status unavailable: %s", exc)
            perception["change_events"] = {"total": 0}
        return perception

    def _orbbec_camera_active(self) -> bool:
        try:
            result = self._command_runner(["systemctl", "is-active", "orbbec-camera"], 3.0)
            stdout = getattr(result, "stdout", b"")
            if isinstance(stdout, bytes):
                stdout = stdout.decode()
            return str(stdout).strip() == "active"
        except Exception as exc:
            self._logger.debug("orbbec-camera status unavailable: %s", exc)
            return False

    def _memory_payload(self) -> dict[str, Any]:
        try:
            if self._knowledge_dir.is_dir():
                files = [item for item in self._knowledge_dir.iterdir() if item.suffix == ".md"]
                return {"knowledge_files": len(files)}
        except Exception as exc:
            self._logger.debug("knowledge memory status unavailable: %s", exc)
        return {"knowledge_files": 0}


def _run_command(command: Sequence[str], timeout_s: float) -> Any:
    return subprocess.run(
        list(command),
        capture_output=True,
        timeout=timeout_s,
    )
