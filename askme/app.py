"""
Askme interactive application facade over the blueprint module system.

Usage::

    from askme.app import AskmeApp
    import asyncio

    app = AskmeApp(voice_mode=False)
    asyncio.run(app.run())
"""

from __future__ import annotations

import logging
from typing import Any

from askme import __version__ as ASKME_VERSION
from askme.config import get_config
from askme.runtime.profiles import legacy_profile_for

logger = logging.getLogger(__name__)


def _select_blueprint(*, voice_mode: bool, robot_mode: bool):
    """Select the appropriate blueprint based on mode flags."""
    if voice_mode and robot_mode:
        from askme.blueprints.edge_robot import edge_robot
        return edge_robot
    if voice_mode:
        from askme.blueprints.voice import voice
        return voice
    from askme.blueprints.text import text
    return text


class AskmeApp:
    """Thin facade that exposes the blueprint-assembled runtime through the legacy API.

    All callers (cli.py, tui.py, main.py, mcp/server.py, tests) access
    the same attribute names they always have: ``conversation``, ``audio``,
    ``pipeline``, ``_text_loop``, ``_voice_loop``, ``skill_manager``, etc.

    Internally these are now routed to the appropriate Module inside the
    built ``RuntimeApp``.
    """

    def __init__(self, *, voice_mode: bool = False, robot_mode: bool = False) -> None:
        self.cfg = get_config()
        self._app_name = self.cfg.get("app", {}).get("name", "askme")
        self._app_version = self.cfg.get("app", {}).get("version") or ASKME_VERSION
        self._setup_logging()

        self.voice_mode = voice_mode
        self.robot_mode = robot_mode

        # Profile for introspection — same object as the old system produced
        self.profile = legacy_profile_for(
            voice_mode=voice_mode, robot_mode=robot_mode,
        )
        self.mode = self.profile

        # Select and synchronously store the blueprint; build is deferred
        # to ensure it can be done in an async context, but for backward
        # compatibility we build eagerly here (no await needed — the
        # Module.build() methods are all synchronous).
        self._blueprint = _select_blueprint(
            voice_mode=voice_mode, robot_mode=robot_mode,
        )

        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're inside an event loop (unlikely for init, but guard it).
            # Caller must await _async_build() before using the app.
            self._runtime_app = None  # type: ignore[assignment]
            self._needs_build = True
        else:
            self._runtime_app = asyncio.get_event_loop().run_until_complete(
                self._blueprint.build(self.cfg),
            )
            self._needs_build = False

    async def _ensure_built(self) -> None:
        """Build the runtime app if not yet built (async-in-async guard)."""
        if self._needs_build:
            self._runtime_app = await self._blueprint.build(self.cfg)
            self._needs_build = False

    # ── Main lifecycle ────────────────────────────────────────────────

    async def run(self) -> None:
        """Start the runtime and enter the selected interactive loop."""
        await self._ensure_built()
        await self._runtime_app.start()
        try:
            if self.voice_mode:
                vl = self._voice_loop
                if vl is None:
                    raise RuntimeError("voice profile is missing the voice loop")
                await vl.run()
            else:
                await self._text_loop.run()
        finally:
            await self._runtime_app.stop()

    async def shutdown(self) -> None:
        """Gracefully stop the assembled runtime."""
        if self._runtime_app is not None:
            await self._runtime_app.stop()

    # ── Compatibility shim: self.runtime ──────────────────────────────
    # tui.py calls ``app.runtime.start()`` directly.

    @property
    def runtime(self):
        """Expose the RuntimeApp as ``self.runtime`` for backward compatibility.

        Supports ``.start()``, ``.stop()`` and attribute access into modules.
        """
        return self._runtime_app

    # ── Compatibility properties: delegate to modules ─────────────────

    def _mod(self, name: str):
        """Get a module by name, or None if not present."""
        if self._runtime_app is None:
            return None
        return self._runtime_app.modules.get(name)

    # -- Memory
    @property
    def conversation(self):
        mod = self._mod("memory")
        return getattr(mod, "conversation", None)

    @property
    def session_memory(self):
        mod = self._mod("memory")
        return getattr(mod, "session_memory", None)

    @property
    def memory(self):
        mod = self._mod("memory")
        return getattr(mod, "memory_bridge", None)

    @property
    def episodic(self):
        mod = self._mod("memory")
        return getattr(mod, "episodic", None)

    @property
    def memory_system(self):
        mod = self._mod("memory")
        return getattr(mod, "memory_system", None)

    # -- LLM
    @property
    def llm(self):
        mod = self._mod("llm")
        return getattr(mod, "client", None)

    @property
    def ota_metrics(self):
        mod = self._mod("llm")
        return getattr(mod, "ota_metrics", None)

    # -- Tools
    @property
    def tools(self):
        mod = self._mod("tools")
        return getattr(mod, "registry", None)

    # -- Voice
    @property
    def audio(self):
        # Voice module's audio if present, else text module's fallback
        mod = self._mod("voice")
        if mod is not None:
            return getattr(mod, "audio", None)
        # TextModule creates its own AudioAgent in text-only mode
        mod = self._mod("text")
        return getattr(mod, "_text_audio", None)

    @audio.setter
    def audio(self, value):
        """Allow tui.py to set app.audio = SilentAudio()."""
        # Store on the voice module if present, else stash as override
        mod = self._mod("voice")
        if mod is not None:
            mod.audio = value
        # Also patch the pipeline's _audio so TUI estop/speak works
        pipeline = self.pipeline
        if pipeline is not None:
            pipeline._audio = value
        # Stash for direct access
        self._audio_override = value

    @property
    def audio_router(self):
        mod = self._mod("voice")
        return getattr(mod, "audio_router", None)

    @property
    def router(self):
        mod = self._mod("voice")
        if mod is not None:
            return getattr(mod, "router", None)
        mod = self._mod("text")
        return getattr(mod, "_router", None)

    @property
    def voice_runtime_bridge(self):
        mod = self._mod("voice")
        if mod is not None:
            return getattr(mod, "voice_runtime_bridge", None)
        mod = self._mod("text")
        return getattr(mod, "_voice_runtime_bridge", None)

    # -- Pipeline
    @property
    def pipeline(self):
        mod = self._mod("pipeline")
        return getattr(mod, "brain_pipeline", None)

    @property
    def _pipeline(self):
        return self.pipeline

    # -- Safety
    @property
    def dog_safety(self):
        mod = self._mod("safety")
        return getattr(mod, "client", None)

    @property
    def _dog_safety(self):
        return self.dog_safety

    # -- Control
    @property
    def dog_control(self):
        mod = self._mod("control")
        return getattr(mod, "client", None)

    # -- Skills
    @property
    def skill_manager(self):
        mod = self._mod("skill")
        return getattr(mod, "skill_manager", None)

    @property
    def skill_executor(self):
        mod = self._mod("skill")
        return getattr(mod, "skill_executor", None)

    @property
    def dispatcher(self):
        mod = self._mod("skill")
        return getattr(mod, "skill_dispatcher", None)

    @property
    def _dispatcher(self):
        return self.dispatcher

    # -- Executor (agent shell)
    @property
    def agent_shell(self):
        mod = self._mod("executor")
        return getattr(mod, "shell", None)

    @property
    def executor(self):
        return self.agent_shell

    # -- Loops
    @property
    def _text_loop(self):
        mod = self._mod("text")
        return getattr(mod, "text_loop", None)

    @property
    def _voice_loop(self):
        mod = self._mod("voice")
        return getattr(mod, "voice_loop", None)

    @property
    def _commands(self):
        mod = self._mod("text")
        return getattr(mod, "commands", None)

    # -- Address detection
    @property
    def _address_detector(self):
        mod = self._mod("voice")
        return getattr(mod, "address_detector", None)

    # -- Proactive
    @property
    def _proactive(self):
        mod = self._mod("proactive")
        return getattr(mod, "agent", None)

    @property
    def _supervisor(self):
        return self._proactive

    # -- Perception
    @property
    def vision(self):
        mod = self._mod("perception")
        return getattr(mod, "vision_bridge", None)

    @property
    def _change_detector(self):
        mod = self._mod("perception")
        return getattr(mod, "change_detector", None)

    @property
    def _change_monitor(self):
        return self._change_detector

    # -- Pulse
    @property
    def pulse(self):
        mod = self._mod("pulse")
        return getattr(mod, "bus", None)

    @property
    def telemetry(self):
        return self.pulse

    # -- Health server
    @property
    def health_server(self):
        mod = self._mod("health")
        return getattr(mod, "server", None)

    @property
    def diagnostics_server(self):
        return self.health_server

    # -- LED bridge
    @property
    def _led_bridge(self):
        mod = self._mod("led")
        return getattr(mod, "led_bridge", None)

    @property
    def _indicators(self):
        return self._led_bridge

    # -- World state (not a module attribute in the old system either)
    @property
    def _world_state(self):
        return None  # WorldState was never wired in the old assembly

    # -- Misc legacy aliases
    @property
    def qp_memory(self):
        return None  # qp_memory init is not in blueprints yet

    @property
    def arm_controller(self):
        return None  # arm_controller is not in blueprints yet

    # ── Logging ───────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        """Configure logging from config."""
        level_str = self.cfg.get("app", {}).get("log_level", "INFO")
        level = getattr(logging, level_str.upper(), logging.INFO)
        fmt = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        datefmt = "%H:%M:%S"
        logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

        log_file = self.cfg.get("app", {}).get("log_file")
        if log_file:
            handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            logging.getLogger().addHandler(handler)

    # ── Snapshot methods ──────────────────────────────────────────────

    def health_snapshot(self) -> dict[str, object]:
        """Return the compact HTTP health payload."""
        from askme.health_server import build_health_snapshot
        from askme.robot.runtime_health import merge_voice_pipeline_status

        ota = self.ota_metrics
        llm = self.llm
        sm = self.skill_manager

        ota_snap = ota.snapshot() if ota else {}
        audio_obj = getattr(self, "_audio_override", None) or self.audio
        voice_status = {}
        if audio_obj is not None and hasattr(audio_obj, "status_snapshot"):
            voice_status = audio_obj.status_snapshot()
        voice_status = merge_voice_pipeline_status(
            voice_status,
            ota_snap.get("voice_pipeline", {}),
        )
        vrb = self.voice_runtime_bridge

        return build_health_snapshot(
            app_name=self._app_name,
            app_version=self._app_version,
            model_name=llm.model if llm else "unknown",
            metrics_snapshot=ota_snap,
            active_skills=[s.name for s in sm.get_enabled()] if sm else [],
            voice_status=voice_status,
            ota_status=None,
            voice_bridge=vrb.status_snapshot() if vrb else None,
        )

    def metrics_snapshot(self) -> dict[str, object]:
        """Return the latest runtime metrics snapshot."""
        ota = self.ota_metrics
        return ota.snapshot() if ota else {}

    def capabilities_snapshot(self) -> dict[str, Any]:
        """Return the runtime/profile/component capability view."""
        sm = self.skill_manager
        contracts = sm.get_contracts() if sm else []
        openapi_doc = sm.openapi_document() if sm else {"info": {"title": "", "version": ""}, "paths": {}}

        # Build component health from modules
        components: dict[str, dict[str, Any]] = {}
        if self._runtime_app is not None:
            for name, mod in self._runtime_app.modules.items():
                components[name] = {
                    "health": mod.health(),
                    "capabilities": mod.capabilities(),
                }

        return {
            "app": {
                "name": self._app_name,
                "version": self._app_version,
                "voice_mode": self.voice_mode,
                "robot_mode": self.robot_mode,
            },
            "profile": self.profile.snapshot(),
            "components": components,
            "skills": {
                "count": len(sm.get_all()) if sm else 0,
                "enabled_count": len(sm.get_enabled()) if sm else 0,
                "contract_count": len(contracts),
                "code_contract_count": sum(
                    1 for c in contracts if c.source == "code"
                ),
                "legacy_contract_count": sum(
                    1 for c in contracts if c.source != "code"
                ),
                "catalog": sm.get_contract_catalog() if sm else [],
            },
            "openapi": {
                "title": openapi_doc["info"]["title"],
                "version": openapi_doc["info"]["version"],
                "path_count": len(openapi_doc["paths"]),
            },
        }

    def _voice_status_snapshot(self) -> dict[str, object]:
        """Compatibility adapter for legacy callers."""
        from askme.robot.runtime_health import merge_voice_pipeline_status

        ota = self.ota_metrics
        ota_snap = ota.snapshot() if ota else {}
        audio_obj = getattr(self, "_audio_override", None) or self.audio
        live_status = {}
        if audio_obj is not None and hasattr(audio_obj, "status_snapshot"):
            live_status = audio_obj.status_snapshot()
        return merge_voice_pipeline_status(
            live_status,
            ota_snap.get("voice_pipeline", {}),
        )
