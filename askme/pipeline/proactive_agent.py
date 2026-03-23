"""ProactiveAgent — autonomous patrol, anomaly detection, event monitoring, and proactive reporting."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import Any, TYPE_CHECKING
from urllib import error, parse, request

from askme.pipeline.alert_dispatcher import AlertDispatcher

if TYPE_CHECKING:
    from askme.memory.episodic_memory import EpisodicMemory
    from askme.llm.client import LLMClient
    from askme.perception.vision_bridge import VisionBridge
    from askme.voice.audio_agent import AudioAgent

logger = logging.getLogger(__name__)

# Sensitivity presets for anomaly detection prompts
_SENSITIVITY = {
    "low": "只关注危险情况（火灾、泄漏、人员倒地）",
    "medium": "关注明显变化（人员增减、物品移动、设备状态）",
    "high": "关注所有变化（包括细微差异）",
}

# ── Telemetry event → spoken alert templates ──
_ALERT_TEMPLATES: dict[str, str] = {
    "navigation.stall_detected": "注意，我好像被卡住了，速度接近零已超过{stall_duration_s:.0f}秒，请检查周围是否有障碍物。",
    "navigation.stall_cleared": "已恢复移动，继续执行任务。",
    "navigation.arrival": "已到达目标位置，距离目标{distance_remaining_m:.1f}米。",
    "navigation.milestone": "巡检进度{progress_pct}%，已通过第{waypoint_current}个航点，共{waypoint_total}个。",
    "mission.failed": "任务执行失败。",
    "mission.completed": "任务已完成。",
    "mission.canceled": "任务已取消。",
}

# Topics that warrant spoken alerts (subset of all telemetry topics)
_ALERT_TOPICS = set(_ALERT_TEMPLATES.keys())

# Minimum seconds between the same topic being spoken
_PER_TOPIC_COOLDOWN_S = 30.0

_ANOMALY_PROMPT = """\
Compare these two sets of object detection results from a YOLO monitoring system.

Previous scans:
{history}

Current scan:
{current}

Detection threshold: {sensitivity}

Output EXACTLY one of:
- ANOMALY|brief Chinese description (max 20 chars) if objects changed significantly
- NORMAL if no significant change

Examples:
- ANOMALY|新增一把椅子
- ANOMALY|检测到新物体
- NORMAL"""


class ProactiveAgent:
    """Background agent that autonomously patrols, detects anomalies, and reports.

    Runs as an ``asyncio.Task`` alongside the main VoiceLoop/TextLoop.
    Uses VisionBridge for scene capture, LLMClient (lightweight model) for
    anomaly judgment, AudioAgent for TTS alerts, and EpisodicMemory for logging.

    Config dict expected keys (under ``proactive``)::

        enabled: bool             - master switch (default False)
        patrol_interval: int      - seconds between scans (default 60)
        alert_cooldown: int       - min seconds between TTS alerts (default 30)
        judge_model: str          - model for anomaly judgment (default Haiku)
        sensitivity: str          - low / medium / high (default medium)
        scene_history_size: int   - past scenes to keep (default 5)
        auto_tasks: list          - periodic task definitions (optional)
    """

    def __init__(
        self,
        *,
        vision: VisionBridge | None,
        audio: AudioAgent,
        episodic: EpisodicMemory | None,
        llm: LLMClient,
        config: dict[str, Any],
    ) -> None:
        pro_cfg = config.get("proactive", {})

        self._enabled: bool = pro_cfg.get("enabled", False)
        self._patrol_interval: float = float(pro_cfg.get("patrol_interval", 120))
        # Adaptive scheduling
        self._base_interval: float = self._patrol_interval
        self._last_anomaly_time: float = 0.0
        self._consecutive_normal: int = 0
        self._alert_cooldown: float = float(pro_cfg.get("alert_cooldown", 30))
        self._judge_model: str = pro_cfg.get(
            "judge_model",
            config.get("brain", {}).get("voice_model", "MiniMax-M2.7-highspeed"),
        )
        sensitivity_key = pro_cfg.get("sensitivity", "medium")
        self._sensitivity_text: str = _SENSITIVITY.get(sensitivity_key, _SENSITIVITY["medium"])
        self._history_size: int = int(pro_cfg.get("scene_history_size", 5))
        self._auto_solve: bool = pro_cfg.get("auto_solve", True)
        # Callback to trigger autonomous problem-solving on anomaly
        # Set via set_solve_callback() after construction
        self._solve_callback: Any = None
        # ChangeDetector event file (Phase 1 event-driven perception)
        _cd_cfg = pro_cfg.get("change_detector", {})
        self._change_event_file: str = _cd_cfg.get("event_file", "/tmp/askme_events.jsonl")
        self._change_events_enabled: bool = _cd_cfg.get("enabled", False)
        # When change_detector is active, relax patrol interval
        self._patrol_interval_with_events: float = float(
            pro_cfg.get("patrol_interval_with_events", 300)
        )

        self._vision = vision
        self._audio = audio
        self._episodic = episodic
        self._llm = llm
        self._robot_id: str | None = config.get("robot", {}).get("robot_id")

        # Multi-channel alert dispatcher
        self._alert_dispatcher = AlertDispatcher(
            voice=audio,
            config=pro_cfg.get("alerts", {}),
            robot_id=self._robot_id,
            robot_name=config.get("robot", {}).get("robot_name", "Thunder"),
        )

        # State
        self._scene_history: deque[str] = deque(maxlen=self._history_size)
        self._last_alert_time: float = 0.0
        self._tick_count: int = 0

        # Event monitor
        self._telemetry_hub_url: str | None = pro_cfg.get("telemetry_hub_url") or None
        self._event_poll_interval: float = float(pro_cfg.get("event_poll_interval", 5))
        self._telemetry_api_key: str | None = pro_cfg.get("telemetry_api_key")
        self._last_event_id: str | None = None
        self._seen_event_ids: set[str] = set()
        self._topic_last_spoken: dict[str, float] = {}

        # Auto-tasks
        self._auto_tasks: list[dict[str, Any]] = []
        for task_def in pro_cfg.get("auto_tasks", []):
            self._auto_tasks.append({
                "name": task_def.get("name", "unnamed"),
                "interval": float(task_def.get("interval", 300)),
                "prompt": task_def.get("prompt", ""),
                "last_run": 0.0,
            })

    def set_solve_callback(self, callback: Any) -> None:
        """Wire the autonomous solve callback (called with anomaly description)."""
        self._solve_callback = callback

    def _adaptive_interval(self) -> float:
        """Calculate patrol interval based on time-of-day and anomaly history.

        Rules:
        - Night (22:00-06:00): 5x base interval (less frequent)
        - Peak hours (09-11, 14-16): 0.5x base (more frequent)
        - After anomaly: 0.25x base for 5 min, then gradually recover
        - 10+ consecutive normal scans: 1.5x base (relax)
        """
        import datetime
        hour = datetime.datetime.now().hour
        base = self._base_interval

        # Time-of-day multiplier
        if 22 <= hour or hour < 6:
            multiplier = 5.0  # night: relax
        elif (9 <= hour < 11) or (14 <= hour < 16):
            multiplier = 0.5  # peak: vigilant
        else:
            multiplier = 1.0  # normal

        # Post-anomaly acceleration
        if self._last_anomaly_time > 0:
            since_anomaly = time.monotonic() - self._last_anomaly_time
            if since_anomaly < 300:  # within 5 min of anomaly
                multiplier = 0.25  # high alert
            elif since_anomaly < 600:  # 5-10 min
                multiplier = min(multiplier, 0.5)

        # Consecutive normal relaxation
        if self._consecutive_normal >= 10:
            multiplier = max(multiplier, 1.5)

        # When ChangeDetector is active, relax patrol (events handle real-time)
        if self._change_events_enabled:
            base = self._patrol_interval_with_events

        interval = base * multiplier
        return max(30.0, min(600.0, interval))  # clamp 30s-10min

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run patrol loop until *stop_event* is set. Launched as background task."""
        if not self._enabled:
            logger.info("[Proactive] Disabled in config.")
            return

        logger.info(
            "[Proactive] Started — interval=%ds, judge=%s, sensitivity=%s, telemetry=%s",
            self._patrol_interval, self._judge_model, self._sensitivity_text[:10],
            "on" if self._telemetry_hub_url else "off",
        )

        # Launch parallel tasks
        event_task = asyncio.create_task(self._event_monitor_loop(stop_event))
        change_task = asyncio.create_task(self._change_event_loop(stop_event))

        try:
            while not stop_event.is_set():
                try:
                    await self._patrol_tick()
                except Exception as exc:
                    logger.warning("[Proactive] Tick error: %s", exc)

                # Auto-tasks
                try:
                    await self._process_auto_tasks()
                except Exception as exc:
                    logger.warning("[Proactive] Auto-task error: %s", exc)

                # Sleep until next tick — adaptive interval
                interval = self._adaptive_interval()
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=interval)
                    break  # stop_event was set
                except asyncio.TimeoutError:
                    pass  # normal — time for next tick
        finally:
            event_task.cancel()
            change_task.cancel()
            for t in (event_task, change_task):
                try:
                    await t
                except asyncio.CancelledError:
                    pass

        logger.info("[Proactive] Stopped.")

    # ------------------------------------------------------------------
    # Patrol tick
    # ------------------------------------------------------------------

    async def _patrol_tick(self) -> None:
        """Single patrol heartbeat: capture → compare → alert."""
        if not self._vision or not self._vision.available:
            return

        self._tick_count += 1
        current_scene = await self._vision.describe_scene()

        # Retry once on empty (VLM relay can be flaky)
        if not current_scene:
            await asyncio.sleep(2.0)
            current_scene = await self._vision.describe_scene()
        if not current_scene:
            logger.info("[Proactive] Tick #%d: scene capture failed, skipping", self._tick_count)
            return

        logger.info("[Proactive] Tick #%d scene: %s", self._tick_count, current_scene[:60])

        # Log to episodic memory
        if self._episodic:
            self._episodic.log("perception", f"巡检扫描: {current_scene}")

        # Anomaly detection (needs at least one prior observation)
        anomaly = await self._detect_anomaly(current_scene)
        if anomaly:
            self._last_anomaly_time = time.monotonic()
            self._consecutive_normal = 0
            logger.warning("[Proactive] ANOMALY: %s (next scan in %.0fs)", anomaly, self._adaptive_interval())

            # Capture snapshot of the anomaly scene
            image_path: str | None = None
            if self._vision:
                image_path = await self._vision.save_snapshot(label=f"anomaly_{self._tick_count}")

            if self._episodic:
                ctx = {"image_path": image_path} if image_path else {}
                self._episodic.log("perception", f"异常: {anomaly}", context=ctx)

            await self._speak_alert(
                f"巡检异常：{anomaly}",
                severity="warning",
                topic="patrol.anomaly",
                payload={"anomaly": anomaly, "image_path": image_path},
            )

            # Auto-solve: trigger solve_problem skill on detected anomaly
            if self._auto_solve and self._solve_callback is not None:
                logger.info("[Proactive] Auto-solving anomaly: %s", anomaly[:60])
                try:
                    await self._solve_callback(f"巡检发现异常：{anomaly}。请分析原因并尝试解决。")
                except Exception as exc:
                    logger.warning("[Proactive] Auto-solve failed: %s", exc)
        else:
            self._consecutive_normal += 1
            # Periodic normal report (every 5 ticks) — also save a baseline snapshot
            if self._tick_count % 5 == 0:
                if self._vision:
                    await self._vision.save_snapshot(label=f"patrol_{self._tick_count}")
                await self._speak_alert(f"巡检正常，第{self._tick_count}次扫描完成。", topic="patrol.normal")

        self._scene_history.append(current_scene)

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    async def _detect_anomaly(self, current_scene: str) -> str | None:
        """Compare current scene with history via lightweight LLM. Returns description or None."""
        if not self._scene_history:
            return None  # First scan, no baseline

        history_text = "\n".join(
            f"[{i + 1}] {s}" for i, s in enumerate(self._scene_history)
        )

        prompt = _ANOMALY_PROMPT.format(
            history=history_text,
            current=current_scene,
            sensitivity=self._sensitivity_text,
        )

        try:
            response = await asyncio.wait_for(
                self._llm.chat(
                    [{"role": "user", "content": prompt}],
                    model=self._judge_model,
                    temperature=0.1,
                ),
                timeout=10.0,
            )
            response = response.strip()
            if response.startswith("ANOMALY|"):
                return response[8:].strip()
        except asyncio.TimeoutError:
            logger.debug("[Proactive] Anomaly check timed out")
        except Exception as exc:
            logger.debug("[Proactive] Anomaly check failed: %s", exc)

        return None

    # ------------------------------------------------------------------
    # TTS alert (conflict-safe)
    # ------------------------------------------------------------------

    async def _speak_alert(self, message: str, *, severity: str = "info", topic: str = "", payload: dict[str, Any] | None = None) -> None:
        """Dispatch an alert via multi-channel dispatcher (voice + webhook + IM)."""
        now = time.monotonic()
        if now - self._last_alert_time < self._alert_cooldown:
            logger.debug("[Proactive] Alert suppressed by cooldown: %s", message[:30])
            return

        sent = await asyncio.to_thread(
            self._alert_dispatcher.dispatch,
            message,
            severity=severity,
            topic=topic,
            payload=payload,
        )

        self._last_alert_time = now
        logger.info("[Proactive] Alert dispatched via %s: %s", sent, message[:40])

    # ------------------------------------------------------------------
    # Auto-tasks
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ChangeDetector event consumer (Phase 1 event-driven perception)
    # ------------------------------------------------------------------

    async def _change_event_loop(self, stop_event: asyncio.Event) -> None:
        """Consume events from ChangeDetector's JSONL file. Reacts in ~2s."""
        if not self._change_events_enabled:
            logger.info("[Proactive] Change event consumer disabled.")
            return

        import json as _json
        from askme.schemas.events import ChangeEvent

        logger.info("[Proactive] Change event consumer started: %s", self._change_event_file)

        # Seek to end — skip historical events from previous runs
        file_pos = 0
        try:
            file_pos = os.path.getsize(self._change_event_file)
        except FileNotFoundError:
            pass

        while not stop_event.is_set():
            try:
                events = await asyncio.to_thread(
                    self._read_change_events, file_pos,
                )
                if events:
                    new_pos, parsed = events
                    file_pos = new_pos
                    for event in parsed:
                        await self._handle_change_event(event)
            except Exception as exc:
                logger.debug("[Proactive] Change event read error: %s", exc)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=2.0)
                break
            except asyncio.TimeoutError:
                pass

    def _read_change_events(self, file_pos: int) -> tuple[int, list[Any]] | None:
        """Read new lines from event JSONL file since file_pos. Blocking."""
        import json as _json
        from askme.schemas.events import ChangeEvent

        try:
            with open(self._change_event_file, "r", encoding="utf-8") as f:
                f.seek(file_pos)
                new_lines = f.readlines()
                new_pos = f.tell()
        except FileNotFoundError:
            return None

        if not new_lines:
            return None

        parsed = []
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = _json.loads(line)
                parsed.append(ChangeEvent.from_dict(data))
            except Exception:
                continue

        if not parsed:
            return None
        return new_pos, parsed

    async def _handle_change_event(self, event: Any) -> None:
        """Process a single change event — alert, log, optionally auto-solve."""
        description = event.description_zh()
        logger.info("[Proactive] Change event: %s (importance=%.2f)", description, event.importance)

        # Log to episodic memory
        if self._episodic:
            self._episodic.log("perception", f"感知事件: {description}")

        # Person events → immediate voice alert
        if event.is_person_event:
            await self._speak_alert(
                description,
                severity="warning" if event.importance >= 0.7 else "info",
                topic="change.person",
                payload={"event": event.to_dict()},
            )

        # High importance + auto_solve → trigger problem solving
        if (
            self._auto_solve
            and self._solve_callback is not None
            and event.importance >= 0.7
        ):
            logger.info("[Proactive] Auto-solving change event: %s", description)
            try:
                await self._solve_callback(f"感知系统检测到变化：{description}。请判断是否需要处理。")
            except Exception as exc:
                logger.warning("[Proactive] Auto-solve from change event failed: %s", exc)

    # ------------------------------------------------------------------
    # Telemetry event monitor
    # ------------------------------------------------------------------

    async def _event_monitor_loop(self, stop_event: asyncio.Event) -> None:
        """Background loop: poll telemetry-hub for alert-worthy events → TTS."""
        if not self._telemetry_hub_url:
            logger.info("[Proactive] Event monitor disabled — no telemetry_hub_url")
            return

        logger.info("[Proactive] Event monitor started — polling every %ds", self._event_poll_interval)

        while not stop_event.is_set():
            try:
                await self._poll_telemetry_events()
            except Exception as exc:
                logger.warning("[Proactive] Event poll error: %s", exc)

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._event_poll_interval)
                break
            except asyncio.TimeoutError:
                pass

    async def _poll_telemetry_events(self) -> None:
        """Fetch recent events from telemetry-hub and speak alerts."""
        events = await asyncio.to_thread(self._fetch_events)
        if not events:
            return

        now = time.monotonic()

        for event in events:
            event_id = event.get("event_id", "")
            topic = event.get("topic", "")

            # Skip events we've already processed (use set for safe dedup)
            if event_id in self._seen_event_ids:
                continue
            self._seen_event_ids.add(event_id)

            # Only alert on recognized topics
            if topic not in _ALERT_TOPICS:
                continue

            # Per-topic cooldown
            last_spoken = self._topic_last_spoken.get(topic, 0.0)
            if now - last_spoken < _PER_TOPIC_COOLDOWN_S:
                continue

            # Build spoken message from template
            event_payload = event.get("payload", {})
            event_severity = event.get("severity", "info")
            message = self._format_alert(topic, event_payload)
            if message:
                logger.info("[Proactive] Alert event: %s [%s] → %s", topic, event_severity, message[:40])
                await self._speak_alert(
                    message,
                    severity=event_severity,
                    topic=topic,
                    payload=event_payload,
                )
                self._topic_last_spoken[topic] = now

                # Log to episodic memory
                if self._episodic:
                    self._episodic.log("alert", f"主动告警[{event_severity}]: {message}")

        # Track last event_id to avoid re-processing
        if events:
            self._last_event_id = events[-1].get("event_id")
            # Bound the dedup set to prevent unbounded memory growth
            if len(self._seen_event_ids) > 1000:
                self._seen_event_ids.clear()

    def _fetch_events(self) -> list[dict[str, Any]]:
        """Synchronous HTTP fetch of recent telemetry events."""
        query: dict[str, str] = {
            "limit": "10",
            "stream": "navigation.alert",
        }
        if self._robot_id:
            query["robot_id"] = self._robot_id
        url = f"{self._telemetry_hub_url}/api/events?{parse.urlencode(query)}"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._telemetry_api_key:
            headers["Authorization"] = f"Bearer {self._telemetry_api_key}"

        req = request.Request(url, headers=headers, method="GET")
        try:
            with request.urlopen(req, timeout=3) as response:
                body = response.read().decode("utf-8")
                data = json.loads(body) if body else {}
                return data.get("events", [])
        except (error.HTTPError, error.URLError, TimeoutError):
            return []

    @staticmethod
    def _format_alert(topic: str, payload: dict[str, Any]) -> str | None:
        """Format a telemetry event into a spoken Chinese message."""
        template = _ALERT_TEMPLATES.get(topic)
        if not template:
            return None
        try:
            return template.format(**payload)
        except (KeyError, ValueError):
            # Fallback: return template with placeholders stripped
            return template.split("，")[0] + "。"

    # ------------------------------------------------------------------
    # Auto-tasks
    # ------------------------------------------------------------------

    async def _process_auto_tasks(self) -> None:
        """Run any auto-tasks whose interval has elapsed."""
        now = time.monotonic()
        for task in self._auto_tasks:
            if not task["prompt"]:
                continue
            if now - task["last_run"] < task["interval"]:
                continue

            logger.info("[Proactive] Running auto-task: %s", task["name"])

            scene = ""
            if self._vision and self._vision.available:
                try:
                    scene = await self._vision.describe_scene()
                except Exception:
                    pass

            content = f"当前视野: {scene}\n任务: {task['prompt']}" if scene else task["prompt"]

            try:
                response = await asyncio.wait_for(
                    self._llm.chat(
                        [{"role": "user", "content": content}],
                        model=self._judge_model,
                    ),
                    timeout=10.0,
                )
                await self._speak_alert(response.strip())
                task["last_run"] = now
            except Exception as exc:
                logger.warning("[Proactive] Auto-task '%s' failed: %s", task["name"], exc)
