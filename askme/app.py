"""
Askme interactive application facade over the runtime assembly layer.

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
from askme.runtime import build_legacy_runtime, legacy_profile_for

logger = logging.getLogger(__name__)


class AskmeApp:
    """Thin facade that exposes the assembled runtime through the legacy API."""

    def __init__(self, *, voice_mode: bool = False, robot_mode: bool = False) -> None:
        self.cfg = get_config()
        self._app_name = self.cfg.get("app", {}).get("name", "askme")
        self._app_version = self.cfg.get("app", {}).get("version") or ASKME_VERSION
        self._setup_logging()

        profile = legacy_profile_for(
            voice_mode=voice_mode,
            robot_mode=robot_mode,
        )
        self.runtime = build_legacy_runtime(
            cfg=self.cfg,
            app_name=self._app_name,
            app_version=self._app_version,
            profile=profile,
            robot_requested=robot_mode,
        )
        self.profile = self.runtime.profile
        self.voice_mode = self.runtime.voice_mode
        self.robot_mode = self.runtime.robot_mode

        for attr, value in self.runtime.services.bindings().items():
            setattr(self, attr, value)

    async def run(self) -> None:
        """Start the profile runtime and enter the selected interactive loop."""
        await self.runtime.start()
        try:
            if self.voice_mode:
                if self._voice_loop is None:
                    raise RuntimeError("voice profile is missing the voice loop")
                await self._voice_loop.run()
            else:
                await self._text_loop.run()
        finally:
            await self.runtime.stop()

    async def shutdown(self) -> None:
        """Gracefully stop the assembled runtime."""
        await self.runtime.stop()

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

    def health_snapshot(self) -> dict[str, object]:
        """Return the compact HTTP health payload."""
        return self.runtime.health_snapshot()

    def metrics_snapshot(self) -> dict[str, object]:
        """Return the latest runtime metrics snapshot."""
        return self.runtime.metrics_snapshot()

    def capabilities_snapshot(self) -> dict[str, Any]:
        """Return the runtime/profile/component capability view."""
        return self.runtime.capabilities_snapshot()

    def _voice_status_snapshot(self) -> dict[str, object]:
        """Compatibility adapter for legacy callers."""
        return self.runtime.voice_status_snapshot()
