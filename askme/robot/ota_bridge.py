"""Async OTA bridge for registering askme with the OTA platform."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import platform
import socket
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

from askme.config import project_root

logger = logging.getLogger(__name__)


class OTABridgeAuthError(RuntimeError):
    """Raised when persisted OTA credentials are no longer accepted."""


class OTABridgeMetrics:
    """Thread-safe runtime metrics collected for OTA telemetry uploads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._started_at = time.time()
            self._conversation_count = 0
            self._llm_call_count = 0
            self._llm_success_count = 0
            self._llm_failure_count = 0
            self._llm_total_latency_ms = 0.0
            self._llm_last_latency_ms: float | None = None
            self._llm_last_mode: str | None = None
            self._llm_last_model: str | None = None
            # Rolling window of last 100 LLM latencies (ms) for percentile calc
            self._llm_latency_window: deque[float] = deque(maxlen=100)
            self._skill_run_count = 0
            self._skill_success_count = 0
            self._skill_failure_count = 0
            # Per-skill stats: {skill_name: {calls, success, failure, total_ms, last_ms}}
            self._skill_stats: dict[str, dict[str, Any]] = {}
            self._voice_state: dict[str, Any] = {
                "mode": "text",
                "enabled": False,
                "input_ready": False,
                "output_ready": False,
                "asr_available": False,
                "vad_available": False,
                "kws_available": False,
                "wake_word_enabled": False,
                "tts_backend": None,
                "last_listen_started_at": None,
                "last_input_at": None,
                "last_input_chars": 0,
                "last_error": None,
                "last_error_at": None,
            }

    def record_conversation_turn(self) -> None:
        with self._lock:
            self._conversation_count += 1

    def record_llm_call(
        self,
        duration_s: float,
        *,
        success: bool,
        mode: str | None = None,
        model: str | None = None,
    ) -> None:
        latency_ms = round(max(duration_s, 0.0) * 1000.0, 2)
        with self._lock:
            self._llm_call_count += 1
            if success:
                self._llm_success_count += 1
            else:
                self._llm_failure_count += 1
            self._llm_total_latency_ms += latency_ms
            self._llm_last_latency_ms = latency_ms
            self._llm_latency_window.append(latency_ms)
            if mode:
                self._llm_last_mode = mode
            if model:
                self._llm_last_model = model

    def record_skill_execution(
        self,
        *,
        success: bool,
        skill_name: str = "",
        duration_s: float = 0.0,
    ) -> None:
        latency_ms = round(max(duration_s, 0.0) * 1000.0, 2)
        with self._lock:
            self._skill_run_count += 1
            if success:
                self._skill_success_count += 1
            else:
                self._skill_failure_count += 1
            if skill_name:
                s = self._skill_stats.setdefault(
                    skill_name,
                    {"calls": 0, "success": 0, "failure": 0,
                     "total_ms": 0.0, "last_ms": None},
                )
                s["calls"] += 1
                s["total_ms"] += latency_ms
                s["last_ms"] = latency_ms
                if success:
                    s["success"] += 1
                else:
                    s["failure"] += 1

    def update_voice_state(self, **updates: Any) -> None:
        with self._lock:
            self._voice_state.update(updates)

    def mark_voice_listen_started(self) -> None:
        self.update_voice_state(last_listen_started_at=_iso_utc_now())

    def mark_voice_input(self, text: str) -> None:
        self.update_voice_state(
            last_input_at=_iso_utc_now(),
            last_input_chars=len(text),
            last_error=None,
            last_error_at=None,
        )

    def mark_voice_error(self, error: str) -> None:
        self.update_voice_state(
            last_error=str(error),
            last_error_at=_iso_utc_now(),
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            llm_call_count = self._llm_call_count
            skill_run_count = self._skill_run_count
            llm_average = (
                round(self._llm_total_latency_ms / llm_call_count, 2)
                if llm_call_count
                else None
            )
            skill_success_rate = (
                round(self._skill_success_count / skill_run_count, 4)
                if skill_run_count
                else None
            )
            latency_percentiles = _compute_percentiles(list(self._llm_latency_window))
            per_skill = {
                name: {
                    **stats,
                    "avg_ms": round(stats["total_ms"] / stats["calls"], 2)
                    if stats["calls"]
                    else None,
                }
                for name, stats in self._skill_stats.items()
            }
            return {
                "uptime_seconds": round(max(time.time() - self._started_at, 0.0), 2),
                "conversation_count": self._conversation_count,
                "llm": {
                    "call_count": llm_call_count,
                    "success_count": self._llm_success_count,
                    "failure_count": self._llm_failure_count,
                    "last_latency_ms": self._llm_last_latency_ms,
                    "average_latency_ms": llm_average,
                    "p50_latency_ms": latency_percentiles.get("p50"),
                    "p95_latency_ms": latency_percentiles.get("p95"),
                    "p99_latency_ms": latency_percentiles.get("p99"),
                    "last_mode": self._llm_last_mode,
                    "last_model": self._llm_last_model,
                },
                "skills": {
                    "run_count": skill_run_count,
                    "success_count": self._skill_success_count,
                    "failure_count": self._skill_failure_count,
                    "success_rate": skill_success_rate,
                    "per_skill": per_skill,
                },
                "voice_pipeline": dict(self._voice_state),
            }


def _compute_percentiles(values: list[float]) -> dict[str, float | None]:
    """Return p50/p95/p99 for a sorted list of latency samples."""
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def _pct(p: float) -> float:
        idx = (p / 100.0) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return round(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac, 2)

    return {"p50": _pct(50), "p95": _pct(95), "p99": _pct(99)}


_GLOBAL_OTA_METRICS = OTABridgeMetrics()


def get_ota_runtime_metrics() -> OTABridgeMetrics:
    """Return the process-wide default OTA metrics collector."""
    return _GLOBAL_OTA_METRICS


class OTABridge:
    """Register askme with OTA and upload heartbeat plus telemetry in the background."""

    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        metrics: OTABridgeMetrics | None = None,
        voice_status_provider: Callable[[], dict[str, Any]] | None = None,
        app_name: str = "askme",
        app_version: str = "",
        voice_mode: bool = False,
        robot_mode: bool = False,
    ) -> None:
        cfg = config or {}
        device_cfg = cfg.get("device", {})
        self.enabled = bool(cfg.get("enabled", False))
        self._server_url = str(cfg.get("server_url", "")).strip().rstrip("/")
        self._product = str(
            cfg.get("product", device_cfg.get("product", "inovxio-dog"))
        ).strip() or "inovxio-dog"
        self._channel = str(cfg.get("channel", "stable")).strip() or "stable"
        raw_tags = cfg.get("tags", device_cfg.get("tags", []))
        self._tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
        self._serial_number = _clean_optional(
            cfg.get("serial_number", device_cfg.get("serial_number"))
        )
        self._robot_id = _clean_optional(
            cfg.get("robot_id", device_cfg.get("robot_id"))
        )
        self._site_id = _clean_optional(
            cfg.get("site_id", device_cfg.get("site_id"))
        )
        self._timeout_s = max(1.0, float(cfg.get("timeout", 10.0)))
        self._heartbeat_interval_s = max(5.0, float(cfg.get("heartbeat_interval", 60.0)))
        self._telemetry_interval_s = max(5.0, float(cfg.get("telemetry_interval", 60.0)))
        self._retry_interval_s = max(
            5.0,
            float(
                cfg.get(
                    "registration_retry_interval",
                    min(self._heartbeat_interval_s, self._telemetry_interval_s),
                )
            ),
        )
        self._state_path = _resolve_path(
            cfg.get("state_file", "data/ota_bridge_state.json")
        )
        self._metrics = metrics or get_ota_runtime_metrics()
        self._voice_status_provider = voice_status_provider
        self._app_name = app_name or "askme"
        self._app_version = str(app_version).strip()
        package_name = str(cfg.get("package_name", self._app_name)).strip()
        self._voice_mode = voice_mode
        self._robot_mode = robot_mode
        self._current_versions = {
            package_name or self._app_name: self._app_version or "unknown",
        }
        self._http_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._session = requests.Session()
        self._task: asyncio.Task[None] | None = None
        # Created lazily in start() so __init__ is safe to call from sync code
        # (creating asyncio.Event() outside a running loop is deprecated in 3.12+).
        self._stop_event: asyncio.Event | None = None
        self._device_id: str | None = None
        self._device_token: str | None = None
        self._registered_at: str | None = None
        self._connection_state = "disabled" if not self.enabled else "stopped"
        self._last_error: str | None = None
        self._last_registration_attempt_at: str | None = None
        self._last_heartbeat_at: str | None = None
        self._last_telemetry_at: str | None = None
        self._load_state()

        if self.enabled and not self._server_url:
            logger.warning("OTABridge enabled but server_url is empty; disabling bridge")
            self.enabled = False
            self._connection_state = "disabled"

    def start(self) -> asyncio.Task[None] | None:
        """Start the OTA bridge background task."""
        if not self.enabled:
            return None
        if self._task is not None and not self._task.done():
            return self._task
        # Clear a dead task so a fresh one can be created. Without this,
        # an unexpected exception in _run() would permanently stop the bridge
        # because start() would return the already-done task on every call.
        if self._task is not None and self._task.done():
            self._task = None
        self._set_connection_state("starting", clear_error=True)
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name="askme-ota-bridge")
        return self._task

    async def stop(self) -> None:
        """Stop the OTA bridge background task."""
        task = self._task
        if task is None:
            self._session.close()
            if self.enabled:
                self._set_connection_state("stopped", clear_error=True)
            return

        if self._stop_event is not None:
            self._stop_event.set()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._session.close()
            if self.enabled:
                self._set_connection_state("stopped", clear_error=True)

    def status_snapshot(self) -> dict[str, Any]:
        """Return OTA bridge registration and connectivity state."""
        with self._state_lock:
            return {
                "enabled": self.enabled,
                "state": self._connection_state,
                "registered": bool(self._device_id and self._device_token),
                "device_id": self._device_id,
                "registered_at": self._registered_at,
                "last_registration_attempt_at": self._last_registration_attempt_at,
                "last_heartbeat_at": self._last_heartbeat_at,
                "last_telemetry_at": self._last_telemetry_at,
                "last_error": self._last_error,
                "server_url": self._server_url,
                "product": self._product,
                "channel": self._channel,
                "serial_number": self._serial_number,
                "robot_id": self._robot_id,
                "site_id": self._site_id,
                "task_running": bool(self._task and not self._task.done()),
            }

    async def _run(self) -> None:
        next_heartbeat = 0.0
        next_telemetry = 0.0
        logger.info(
            "OTA bridge started (server=%s, product=%s, channel=%s)",
            self._server_url,
            self._product,
            self._channel,
        )

        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if not self._is_registered():
                    registered = await self._ensure_registered()
                    if not registered:
                        await self._sleep_or_stop(self._retry_interval_s)
                        continue
                    now = time.monotonic()
                    next_heartbeat = now
                    next_telemetry = now

                if now >= next_heartbeat:
                    await self._send_heartbeat()
                    next_heartbeat = now + self._heartbeat_interval_s

                if now >= next_telemetry:
                    await self._send_telemetry()
                    next_telemetry = now + self._telemetry_interval_s

                sleep_for = max(
                    1.0,
                    min(next_heartbeat, next_telemetry) - time.monotonic(),
                )
                await self._sleep_or_stop(sleep_for)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._set_connection_state("degraded", error=f"runtime error: {exc}")
            logger.error("OTA bridge stopped after unexpected error: %s", exc, exc_info=True)
        finally:
            logger.info("OTA bridge stopped")

    async def _ensure_registered(self) -> bool:
        if self._is_registered():
            return True

        self._mark_registration_attempt()
        payload = self._build_registration_payload()
        logger.info(
            "Registering askme with OTA server (product=%s, channel=%s)",
            self._product,
            self._channel,
        )
        try:
            response = await self._post_json(
                "/devices/register",
                payload,
                require_token=False,
            )
        except requests.RequestException as exc:
            self._set_connection_state("degraded", error=f"registration failed: {exc}")
            logger.warning("OTA registration failed: %s", exc)
            return False
        except ValueError as exc:
            self._set_connection_state(
                "degraded",
                error=f"registration returned invalid payload: {exc}",
            )
            logger.warning("OTA registration returned invalid payload: %s", exc)
            return False

        device_id = _clean_optional(response.get("device_id") or response.get("id"))
        device_token = _clean_optional(response.get("device_token"))
        if not device_id or not device_token:
            self._set_connection_state(
                "degraded",
                error="registration response missing device credentials",
            )
            logger.warning("OTA registration response missing credentials: %s", response)
            return False

        self._set_registration(
            device_id=device_id,
            device_token=device_token,
            registered_at=_clean_optional(response.get("registered_at")) or _iso_utc_now(),
        )
        self._set_connection_state("connected", clear_error=True)
        logger.info("OTA bridge registered successfully as %s", device_id)
        return True

    async def _send_heartbeat(self) -> None:
        if not self._is_registered():
            return

        payload = {
            "device_id": self._device_id,
            "current_versions": self._current_versions,
            "ip_address": _get_ip_address(),
            "system_info": self._build_system_info(),
        }

        try:
            response = await self._post_json(
                "/agent/heartbeat",
                payload,
                require_token=True,
            )
        except OTABridgeAuthError as exc:
            self._set_connection_state("auth_error", error=str(exc))
            logger.warning("OTA heartbeat rejected persisted credentials: %s", exc)
            self._clear_registration()
            return
        except requests.RequestException as exc:
            self._set_connection_state("degraded", error=f"heartbeat failed: {exc}")
            logger.warning("OTA heartbeat failed: %s", exc)
            return
        except ValueError as exc:
            self._set_connection_state(
                "degraded",
                error=f"heartbeat returned invalid payload: {exc}",
            )
            logger.warning("OTA heartbeat returned invalid payload: %s", exc)
            return

        self._mark_heartbeat()

        pending_configs = response.get("pending_configs") or []
        if pending_configs:
            logger.info(
                "OTA heartbeat returned %d pending configs; askme bridge does not apply them",
                len(pending_configs),
            )

        diag_command = response.get("diag_command")
        if diag_command:
            logger.info(
                "OTA heartbeat returned diag command %s; askme bridge does not execute it",
                diag_command,
            )

    async def _send_telemetry(self) -> None:
        if not self._is_registered():
            return

        metrics = self._metrics.snapshot()
        voice_status = (
            self._voice_status_provider()
            if self._voice_status_provider is not None
            else {
                "mode": "voice" if self._voice_mode else "text",
                "pipeline_ok": not self._voice_mode,
            }
        )
        payload = {
            "device_id": self._device_id,
            "collected_at": _iso_utc_now(),
            "custom_metrics": {
                "service_name": self._app_name,
                "service_version": self._app_version or "unknown",
                "conversation_count": metrics["conversation_count"],
                "uptime_seconds": metrics["uptime_seconds"],
                "llm_latency_ms": metrics["llm"],
                "skill_success_rate": metrics["skills"]["success_rate"],
                "skill_stats": metrics["skills"],
                "voice_pipeline_status": voice_status,
                "robot_id": self._robot_id,
                "site_id": self._site_id,
            },
        }

        try:
            await self._post_json(
                "/telemetry/report",
                payload,
                require_token=True,
            )
        except OTABridgeAuthError as exc:
            self._set_connection_state("auth_error", error=str(exc))
            logger.warning("OTA telemetry rejected persisted credentials: %s", exc)
            self._clear_registration()
        except requests.RequestException as exc:
            self._set_connection_state("degraded", error=f"telemetry failed: {exc}")
            logger.warning("OTA telemetry upload failed: %s", exc)
        except ValueError as exc:
            self._set_connection_state(
                "degraded",
                error=f"telemetry returned invalid payload: {exc}",
            )
            logger.warning("OTA telemetry returned invalid payload: %s", exc)
        else:
            self._mark_telemetry()

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        require_token: bool,
    ) -> dict[str, Any]:
        # OTA HTTP calls are low-frequency (heartbeat every ~60s) and complete
        # quickly (<200ms).  Running them directly avoids asyncio.to_thread
        # executor deadlocks observed on Python 3.13/Windows under pytest-asyncio.
        return self._post_json_sync(path, payload, require_token)

    def _post_json_sync(
        self,
        path: str,
        payload: dict[str, Any],
        require_token: bool,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._device_id:
            headers["X-Device-ID"] = self._device_id
        if require_token:
            if not self._device_id or not self._device_token:
                raise OTABridgeAuthError("missing OTA device credentials")
            headers["X-Device-Token"] = self._device_token

        url = f"{self._server_url}{path}"
        with self._http_lock:
            response = self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout_s,
            )

        if require_token and response.status_code in {401, 403, 404}:
            raise OTABridgeAuthError(f"HTTP {response.status_code} for {path}")

        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"OTA endpoint {path} returned non-object payload: {data!r}")
        return data

    def _build_registration_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "product": self._product,
            "tags": list(self._tags),
            "channel": self._channel,
            "system_info": self._build_system_info(),
            "ip_address": _get_ip_address(),
        }
        if self._serial_number:
            payload["serial_number"] = self._serial_number
        hardware_info = _build_hardware_info()
        if hardware_info:
            payload["hardware_info"] = hardware_info
        return payload

    def _build_system_info(self) -> dict[str, Any]:
        system_info = {
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "arch": platform.machine(),
            "python_version": platform.python_version(),
            "service_name": self._app_name,
            "service_version": self._app_version or "unknown",
            "voice_mode": self._voice_mode,
            "robot_mode": self._robot_mode,
        }
        if self._robot_id:
            system_info["robot_id"] = self._robot_id
        if self._site_id:
            system_info["site_id"] = self._site_id
        return system_info

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return

        try:
            with open(self._state_path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load OTA bridge state from %s: %s", self._state_path, exc)
            return

        device_id = _clean_optional(data.get("device_id"))
        device_token = _clean_optional(data.get("device_token"))
        if not device_id or not device_token:
            logger.warning("OTA bridge state is incomplete; re-registering on next start")
            return

        self._device_id = device_id
        self._device_token = device_token
        self._registered_at = _clean_optional(data.get("registered_at"))

    def _set_connection_state(
        self,
        state: str,
        *,
        error: str | None = None,
        clear_error: bool = False,
    ) -> None:
        with self._state_lock:
            self._connection_state = state
            if clear_error:
                self._last_error = None
            elif error is not None:
                self._last_error = str(error)

    def _mark_registration_attempt(self) -> None:
        with self._state_lock:
            self._last_registration_attempt_at = _iso_utc_now()
            self._connection_state = "registering"
            self._last_error = None

    def _mark_heartbeat(self) -> None:
        with self._state_lock:
            self._last_heartbeat_at = _iso_utc_now()
            self._connection_state = "connected"
            self._last_error = None

    def _mark_telemetry(self) -> None:
        with self._state_lock:
            self._last_telemetry_at = _iso_utc_now()
            self._connection_state = "connected"
            self._last_error = None

    def _save_state(self) -> None:
        payload = {
            "device_id": self._device_id,
            "device_token": self._device_token,
            "registered_at": self._registered_at,
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("Failed to persist OTA bridge state to %s: %s", self._state_path, exc)

    def _set_registration(
        self,
        *,
        device_id: str,
        device_token: str,
        registered_at: str,
    ) -> None:
        with self._state_lock:
            self._device_id = device_id
            self._device_token = device_token
            self._registered_at = registered_at
        self._save_state()

    def _clear_registration(self) -> None:
        with self._state_lock:
            self._device_id = None
            self._device_token = None
            self._registered_at = None
        self._save_state()

    def _is_registered(self) -> bool:
        with self._state_lock:
            return bool(self._device_id and self._device_token)

    async def _sleep_or_stop(self, delay_s: float) -> None:
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay_s)
        except TimeoutError:
            return


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


def _build_hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {}

    mac_int = uuid.getnode()
    if mac_int:
        mac_hex = f"{mac_int:012x}"
        info["mac_address"] = ":".join(mac_hex[i : i + 2] for i in range(0, 12, 2))

    if sys.platform.startswith("linux"):
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            try:
                with open(cpuinfo, encoding="utf-8") as fh:
                    for line in fh:
                        if line.lower().startswith("serial"):
                            info["cpu_serial"] = line.split(":", 1)[1].strip()
                            break
            except OSError:
                pass

    return info


def _get_ip_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0)
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "unknown"


def _iso_utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
