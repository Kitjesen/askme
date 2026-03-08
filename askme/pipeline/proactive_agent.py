"""ProactiveAgent — autonomous patrol, anomaly detection, and proactive reporting."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from askme.brain.episodic_memory import EpisodicMemory
    from askme.brain.llm_client import LLMClient
    from askme.brain.vision_bridge import VisionBridge
    from askme.voice.audio_agent import AudioAgent

logger = logging.getLogger(__name__)

# Sensitivity presets for anomaly detection prompts
_SENSITIVITY = {
    "low": "只关注危险情况（火灾、泄漏、人员倒地）",
    "medium": "关注明显变化（人员增减、物品移动、设备状态）",
    "high": "关注所有变化（包括细微差异）",
}

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
        self._patrol_interval: float = float(pro_cfg.get("patrol_interval", 60))
        self._alert_cooldown: float = float(pro_cfg.get("alert_cooldown", 30))
        self._judge_model: str = pro_cfg.get(
            "judge_model",
            config.get("brain", {}).get("voice_model", "claude-haiku-4-5-20251001"),
        )
        sensitivity_key = pro_cfg.get("sensitivity", "medium")
        self._sensitivity_text: str = _SENSITIVITY.get(sensitivity_key, _SENSITIVITY["medium"])
        self._history_size: int = int(pro_cfg.get("scene_history_size", 5))

        self._vision = vision
        self._audio = audio
        self._episodic = episodic
        self._llm = llm

        # State
        self._scene_history: deque[str] = deque(maxlen=self._history_size)
        self._last_alert_time: float = 0.0
        self._tick_count: int = 0

        # Auto-tasks
        self._auto_tasks: list[dict[str, Any]] = []
        for task_def in pro_cfg.get("auto_tasks", []):
            self._auto_tasks.append({
                "name": task_def.get("name", "unnamed"),
                "interval": float(task_def.get("interval", 300)),
                "prompt": task_def.get("prompt", ""),
                "last_run": 0.0,
            })

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self, stop_event: asyncio.Event) -> None:
        """Run patrol loop until *stop_event* is set. Launched as background task."""
        if not self._enabled:
            logger.info("[Proactive] Disabled in config.")
            return

        logger.info(
            "[Proactive] Started — interval=%ds, judge=%s, sensitivity=%s",
            self._patrol_interval, self._judge_model, self._sensitivity_text[:10],
        )

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

            # Sleep until next tick (interruptible)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._patrol_interval)
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # normal — time for next tick

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
            logger.warning("[Proactive] ANOMALY: %s", anomaly)
            if self._episodic:
                self._episodic.log("perception", f"异常: {anomaly}")
            await self._speak_alert(f"巡检异常：{anomaly}")
        else:
            # Periodic normal report (every 5 ticks)
            if self._tick_count % 5 == 0:
                await self._speak_alert(f"巡检正常，第{self._tick_count}次扫描完成。")

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

    async def _speak_alert(self, message: str) -> None:
        """Speak an alert via TTS, respecting cooldown and busy state."""
        now = time.monotonic()
        if now - self._last_alert_time < self._alert_cooldown:
            logger.debug("[Proactive] Alert suppressed by cooldown: %s", message[:30])
            return

        # Don't interrupt if main loop is speaking
        if self._audio.is_busy:
            logger.info("[Proactive] TTS busy, skipping alert: %s", message[:40])
            return

        self._audio.start_playback()
        self._audio.speak(message)
        await asyncio.to_thread(self._audio.wait_speaking_done)
        self._audio.stop_playback()

        self._last_alert_time = now
        logger.info("[Proactive] Alert spoken: %s", message[:40])

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
